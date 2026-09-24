"""Lightweight neural network policy: preprocessed_state → action.

Implements the imitation learning framework from:
  "Safety-Guaranteed Imitation Learning from NMPC for Spacecraft CPO" (2026).

Training loss (Eq. 18):
  L = λ_imit * L_imit + λ_cbf * L_cbf + λ_clf * L_clf

where:
  L_imit = MSE(π_NN(x), u_expert)                        — behaviour cloning
  L_cbf  = E[max(0, −H_dot(x, π_NN(x)))²]               — CBF penalty (Eq. 16)
  L_clf  = E[max(0,  V_dot(x, π_NN(x)))²]               — CLF penalty (Eq. 17)

L_cbf / L_clf are optional: pass torch-callable cbf_fn / clf_fn to PolicyTrainer.
Each callable must accept (state_batch, action_batch) as float tensors and return
a scalar-per-sample tensor (positive = constraint satisfied).

This module stays environment-agnostic — the certificates themselves live with
their environment (hypercrl/envs/space_cbf_clf.py for SatDynEnv,
hypercrl/envs/mujoco/half_cheetah_safe.py for HalfCheetahSafeEnv), which is
also where the numpy/cvxpy versions the runtime QP filter uses are defined.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """4-layer MLP with LayerNorm + ReLU + Dropout, linear output.

    Architecture matches Section IV-A of the paper (256 neurons, 4 layers).
    Input:  preprocessed + normalised state (same as dynamics model input).
    Output: normalised action (unbounded linear); caller clips to physical bounds.

    No final Tanh: expert normalised targets can reach ±u_max/a_std (e.g. ±1.6),
    which saturates tanh and kills gradients for exactly the safety-critical states
    where the expert commands full torque to avoid the KOZ.  The physical-space
    clip happens in NNPolicyAgent.act() after denormalisation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple = (256, 256, 256, 256),
        dropout: float = 0.1,
        n_tasks: int = 0,
    ) -> None:
        super().__init__()

        # n_tasks > 0 appends a one-hot task id to the input.  Needed when the
        # policy trains on several tasks at once (expert replay): a faulted
        # thruster needs a different action at the same state, so without the
        # task id the replayed tasks are averaged together.
        self.n_tasks = n_tasks
        dims = [state_dim + n_tasks, *hidden_dims]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], action_dim))
        # No Tanh — linear head so full-torque targets are representable.

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, task_id=None) -> torch.Tensor:
        if self.n_tasks:
            if task_id is None:
                raise ValueError("task-conditioned PolicyNet requires task_id")
            if torch.is_tensor(task_id) and task_id.dim() == 2:
                onehot = task_id.to(x)                      # already one-hot rows
            else:
                onehot = torch.zeros(x.shape[0], self.n_tasks, dtype=x.dtype, device=x.device)
                onehot[:, int(task_id)] = 1.0
            x = torch.cat([x, onehot], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PolicyTrainer:
    """Trains a PolicyNet to clone an MPC expert with optional CBF/CLF losses.

    Parameters
    ----------
    policy:
        The PolicyNet to train.
    hparams:
        Namespace with at least: device, policy_lr, policy_bs,
        policy_train_iters, policy_lambda_imit, policy_lambda_cbf,
        policy_lambda_clf.
    cbf_fn:
        Optional torch-callable (state_B, action_B) → scalar_B.
        Returns H_dot per sample; positive means the CBF constraint is met.
    clf_fn:
        Optional torch-callable (state_B, action_B) → scalar_B.
        Returns V_dot per sample; negative means the CLF constraint is met.
    """

    def __init__(
        self,
        policy: PolicyNet,
        hparams,
        cbf_fn: Optional[Callable] = None,
        clf_fn: Optional[Callable] = None,
    ) -> None:
        self.policy = policy
        self.device = hparams.device

        self.lambda_imit     = getattr(hparams, "policy_lambda_imit",     1.0)
        self._lambda_cbf_base = getattr(hparams, "policy_lambda_cbf",     0.0)
        self._lambda_clf_base = getattr(hparams, "policy_lambda_clf",     0.0)
        self.lambda_cbf      = self._lambda_cbf_base
        self.lambda_clf      = self._lambda_clf_base
        self.lambda_ramp      = getattr(hparams, "policy_lambda_ramp",   2.0)
        self.lambda_max       = getattr(hparams, "policy_lambda_max",    1.0)
        self.n_iters          = getattr(hparams, "policy_train_iters",  1000)
        self.bs               = getattr(hparams, "policy_bs",           128)
        self._dagger_iter     = 0
        self._dagger_n_iter   = getattr(hparams, "dagger_n_iter",       5)

        self.cbf_fn = cbf_fn
        self.clf_fn = clf_fn

        # Safety-prioritised sampling: samples where the expert's filter had
        # to step in (buffer tags) or that lie near the KOZ boundary
        # (margin_fn, set per task by the experiment driver) are drawn up to
        # `safety_oversample`x more often during training.  1.0 = uniform.
        self.margin_fn: Optional[Callable] = None   # state_norm (B, D) -> θ-margin [deg] (B,)
        self.safety_oversample  = getattr(hparams, "policy_safety_oversample", 1.0)
        self.safety_margin_deg  = getattr(hparams, "policy_safety_margin_deg", 15.0)

        # CBF loss effectiveness (baseline_33: loss_cbf was 0.0000 for the
        # whole run — training states are all safe, so the zero-margin hinge
        # never fired and λ_cbf multiplied an exact zero):
        #  * cbf_eps_train: hinge fires below this condition margin, creating
        #    gradient in the approach corridor before outright violation;
        #  * boundary_sampler: maps a real state batch to synthetic states
        #    resampled through the KOZ corridor — the CBF penalty is
        #    self-supervised (needs no expert label), so it can be shaped on
        #    a designed state distribution instead of the (all-safe) data.
        self.cbf_eps_train    = getattr(hparams, "policy_cbf_eps_train", 0.0)
        self.boundary_sampler: Optional[Callable] = None
        # Control-feasibility mask for the boundary penalty: excludes states
        # where no action in the box can satisfy the condition (ḣ≈0 regime) —
        # they have (near-)zero gradient through u and only inflate the loss.
        self.cbf_feasible_fn: Optional[Callable] = None

        # Output head weight (and all biases/LayerNorm params) are excluded
        # from weight decay: decoupled decay on the final Linear directly
        # shrinks predicted action magnitude, which fights against matching
        # large expert corrective torques (see u_pred_norm_max staying far
        # below u_target_norm_max in baseline_16 diagnostics).
        output_linear = None
        for m in policy.net:
            if isinstance(m, nn.Linear):
                output_linear = m
        decay, no_decay = [], []
        for p in policy.parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or p is output_linear.weight:
                no_decay.append(p)
            else:
                decay.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": 1e-4},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=getattr(hparams, "policy_lr", 1e-4),
        )

        self._step = 0  # global step counter for TensorBoard

        # Per-task DAGGER buffer: (preprocessed+normalised state, PHYSICAL
        # expert action).  Actions are stored unnormalised and only scaled by
        # the collector's *current* a_mu/a_std when the train set is built —
        # normalising at collection time froze stale statistics into old
        # buffer entries as the collector stats kept updating.
        #
        # DAGGER rollout transitions live ONLY here — they are never added to
        # the dynamics collector.  Feeding mixed-policy rollouts to the
        # dynamics model shifted its training distribution away from the
        # expert's trajectories (80% of rows by the end of baseline_18) and
        # measurably degraded the MPC expert that plans through it; the
        # confirmed-good pure-MBRL run trained on expert/random data only.
        self._dag_states: list = []
        self._dag_actions_phys: list = []
        # 1.0 where the expert's safety filter corrected the label's action
        # (read from mpc_agent.safety_filter right after act()) — these are
        # the states that carry the avoidance knowledge; aligned with
        # _dag_states/_dag_actions_phys.
        self._dag_filter_active: list = []

        # Expert replay: {task_id: (states_norm, actions_norm)} for finished
        # tasks, relabelled by the MPC expert through the hypernetwork's
        # weights for that task.  Survives reset_per_task by design.
        self.n_tasks = getattr(policy, "n_tasks", 0)
        self.replay_n = getattr(hparams, "policy_replay_n", 256)
        self._replay: dict = {}

    def _onehot(self, task_id: int, n_rows: int) -> torch.Tensor:
        oh = torch.zeros(n_rows, max(self.n_tasks, 1))
        oh[:, int(task_id)] = 1.0
        return oh

    def refresh_replay(self, mpc_agent, collector, task_id: int, env_for_task) -> None:
        """Relabel stored states of every finished task with the MPC expert.

        The expert plans with the hypernetwork's weights for that task, so the
        labels are only correct if the hypernetwork still remembers it — which
        is exactly the property under test.  States come from the collector, so
        nothing is simulated in the old environment.
        """
        for j in range(task_id):
            x_all, _ = collector.get_dataset(j)
            x_all = x_all.tensors[0]
            idx = torch.randperm(x_all.shape[0])[: self.replay_n]
            x_norm = x_all[idx]

            x_mu, x_std, a_mu, a_std = collector.norm(j)
            x_raw = x_norm * x_std.flatten().cpu() + x_mu.flatten().cpu()

            mpc_agent.cache_hnet(j)
            env_j = env_for_task(j)
            if hasattr(env_j, "get_safety_filter"):
                mpc_agent.set_safety_filter(env_j.get_safety_filter())
            acts = []
            for row in x_raw:
                u = mpc_agent.act(row.numpy(), task_id=j).detach().cpu().flatten()
                acts.append(u)
            u_phys = torch.stack(acts)
            u_norm = (u_phys - a_mu.flatten().cpu()) / a_std.flatten().cpu()
            self._replay[j] = (x_norm, u_norm)
            print(f"  [replay] task {j} relabelled ({x_norm.shape[0]} states)")

    # ------------------------------------------------------------------

    def reset_per_task(self) -> None:
        """Reset all per-task DAGGER state at the start of every new task.

        Clears the rollout buffer and restarts the kappa/lambda curriculum so
        each task gets a fresh DAGGER pass.  Policy weights are NOT reset —
        they keep benefiting from prior tasks.
        """
        self._dagger_iter = 0
        self._dag_states.clear()
        self._dag_actions_phys.clear()
        self._dag_filter_active.clear()
        self.lambda_cbf = self._lambda_cbf_base
        self.lambda_clf = self._lambda_clf_base

    def _make_policy_train_set(self, dynamics_train_set, collector, task_id):
        """Build the policy train set: expert BC base + DAGGER buffer.

        Returns ``(dataset, sample_weights)`` — the weights implement
        safety-prioritised sampling (see _sample_weights) and are passed to
        train() as its DataLoader sampler weights.

        The passed dynamics dataset is all-expert by construction — the main
        loop executes the pure (filtered) MPC agent and DAGGER rollouts never
        add to the collector.  DAGGER buffer actions are physical u_mpc —
        normalised here with the collector's *current* per-task statistics so
        they stay on the same scale as the base rows.
        """
        from torch.utils.data import TensorDataset as _TDS
        base = dynamics_train_set
        dyn_x, dyn_u = base.tensors[0], base.tensors[1]

        # Row order matters: _sample_weights tags the LAST len(tags) rows as the
        # DAGGER buffer, so replayed rows go first and DAGGER rows stay last.
        xs, us, ts = [], [], []
        for j, (rx, ru) in sorted(self._replay.items()):
            xs.append(rx); us.append(ru); ts.append(self._onehot(j, rx.shape[0]))
        xs.append(dyn_x); us.append(dyn_u); ts.append(self._onehot(task_id, dyn_x.shape[0]))

        tags = None
        if self._dag_states:
            dag_x = torch.cat(self._dag_states, dim=0)             # (Nd, proc_dim)
            dag_u = torch.stack(self._dag_actions_phys, dim=0)     # (Nd, act_dim)
            try:
                _, _, a_mu, a_std = collector.norm(task_id)
                dag_u = (dag_u - a_mu.flatten().to(dag_u.dtype)) \
                    / a_std.flatten().to(dag_u.dtype)
            except (KeyError, AttributeError, TypeError):
                pass  # no norms (normalize_xu=False) — physical labels match the base
            xs.append(dag_x); us.append(dag_u); ts.append(self._onehot(task_id, dag_x.shape[0]))
            tags = torch.tensor(self._dag_filter_active, dtype=torch.float32)

        dataset = _TDS(torch.cat(xs, dim=0), torch.cat(us, dim=0), torch.cat(ts, dim=0))
        return dataset, self._sample_weights(dataset.tensors[0], tags=tags)

    def _sample_weights(self, x_all: torch.Tensor,
                        tags: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Safety-prioritised sampling weights, one per row of x_all.

        weight = 1 + (K−1)·criticality with K = safety_oversample and
        criticality ∈ [0, 1] the max of two signals:
          * θ-margin proximity (margin_fn, all rows): 1 at the KOZ boundary,
            fading linearly to 0 at safety_margin_deg — covers the approach
            corridor even where the filter stayed idle;
          * expert-filter intervention tags (last len(tags) rows = DAGGER
            buffer): 1 where the QP corrected the label's action.

        Returns None when oversampling is off (K ≤ 1) or nothing can grade
        the rows — train() then samples uniformly.
        """
        K = float(self.safety_oversample)
        if K <= 1.0 or (self.margin_fn is None and tags is None):
            return None

        n = x_all.shape[0]
        crit = torch.zeros(n)
        if self.margin_fn is not None:
            with torch.no_grad():
                margin_deg = self.margin_fn(x_all).flatten().cpu()
            crit = (1.0 - margin_deg / self.safety_margin_deg).clamp(0.0, 1.0)
        if tags is not None and len(tags) > 0:
            crit[n - len(tags):] = torch.maximum(crit[n - len(tags):], tags)

        return 1.0 + (K - 1.0) * crit

    # ------------------------------------------------------------------

    def train(self, dataset, writer=None, sample_weights=None, task_id=0) -> float:
        """Run one training phase; return mean total loss.

        sample_weights:
            Optional per-row weights (len(dataset),) for a
            WeightedRandomSampler — safety-critical rows are drawn
            proportionally more often.  None = uniform shuffling.
        """
        # A dataset smaller than one batch yields no batches (drop_last=True),
        # and an empty one doesn't even construct a DataLoader — happens right
        # after policy_train_start when the BC set excludes the random phase
        # and few expert rows exist yet.  Skip instead of crashing.
        if len(dataset) < self.bs:
            print(f"  [policy] skipped training — {len(dataset)} rows < batch size {self.bs}")
            return 0.0
        if sample_weights is not None and len(sample_weights) == len(dataset):
            sampler = WeightedRandomSampler(sample_weights,
                                            num_samples=len(dataset),
                                            replacement=True)
            loader = DataLoader(dataset, batch_size=self.bs, sampler=sampler,
                                drop_last=True)
        else:
            loader = DataLoader(dataset, batch_size=self.bs, shuffle=True,
                                drop_last=True)
        it = iter(loader)

        self.policy.train()
        total = 0.0
        u_pred_norm_max   = 0.0
        u_target_norm_max = 0.0

        for i in range(self.n_iters):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            x, u_expert, tid = batch
            x        = x.to(self.device)
            u_expert = u_expert.to(self.device)
            tid      = tid.to(self.device)

            u_pred = self.policy(x, task_id=tid)

            # The CBF/CLF terms below are built for ONE task's actuator, so
            # they may only score rows of the task currently being trained.
            cur = (tid.argmax(dim=1) == task_id) if self.n_tasks else torch.ones(
                x.shape[0], dtype=torch.bool, device=x.device)

            with torch.no_grad():
                u_pred_norm_max   = max(u_pred_norm_max,   float(u_pred.abs().max()))
                u_target_norm_max = max(u_target_norm_max, float(u_expert.abs().max()))

            # --- imitation loss ---
            # Weighted by expert action magnitude: plain MSE averages over a
            # batch dominated by small routine actions, so the network
            # regresses toward the mean and under-predicts the rare large
            # corrective torques used near the KOZ.  Up-weighting samples by
            # ‖u_expert‖ keeps those safety-critical actions from being
            # drowned out.  Capped: the CEM expert is stochastic and extreme
            # normalised targets (‖u‖ > 10σ observed) are partly noise —
            # unbounded weights would let those outliers dominate the batch.
            imit_weight = 1.0 + u_expert.norm(dim=1, keepdim=True).clamp(max=5.0)
            loss_imit = (imit_weight * (u_pred - u_expert) ** 2).mean()
            loss = self.lambda_imit * loss_imit

            # --- CBF loss (Eq. 16) ---
            # Hinge at cbf_eps_train (not 0): requires a margin on the CBF
            # condition, so gradients exist in the approach corridor and the
            # learned condition has robustness headroom for the dt=0.1
            # discretization — a zero hinge never fired on the all-safe
            # training states (baseline_33).
            cbf_viol_frac = torch.zeros(1)
            cbf_mean_margin = torch.zeros(1)
            cbf_synth_viol_frac = torch.zeros(1)
            if self.cbf_fn is not None and self.lambda_cbf > 0.0 and cur.any():
                h_dot = self.cbf_fn(x[cur], u_pred[cur])
                loss_cbf = torch.mean(torch.clamp(self.cbf_eps_train - h_dot, min=0.0) ** 2)
                loss = loss + self.lambda_cbf * loss_cbf
                with torch.no_grad():
                    cbf_viol_frac   = (h_dot < self.cbf_eps_train).float().mean()
                    cbf_mean_margin = h_dot.mean()

                # Synthetic boundary states: same penalty, evaluated on states
                # resampled through the KOZ corridor.  Self-supervised — the
                # policy's own action at the synthetic state is penalised, no
                # expert label involved.
                if self.boundary_sampler is not None:
                    x_bnd = self.boundary_sampler(x[cur])
                    # Keep only control-feasible states — infeasible ones
                    # (ḣ≈0 near the boundary) have no gradient through u and
                    # pin the loss/viol-frac at a floor (baseline_34: ~0.6).
                    if self.cbf_feasible_fn is not None:
                        x_bnd = x_bnd[self.cbf_feasible_fn(x_bnd)]
                    if x_bnd.shape[0] > 0:
                        u_bnd = self.policy(x_bnd, task_id=task_id)
                        h_dot_bnd = self.cbf_fn(x_bnd, u_bnd)
                        loss_cbf_bnd = torch.mean(
                            torch.clamp(self.cbf_eps_train - h_dot_bnd, min=0.0) ** 2)
                        loss = loss + self.lambda_cbf * loss_cbf_bnd
                        with torch.no_grad():
                            cbf_synth_viol_frac = (h_dot_bnd < self.cbf_eps_train).float().mean()
                    else:
                        loss_cbf_bnd = torch.zeros(1)
                else:
                    loss_cbf_bnd = torch.zeros(1)
            else:
                loss_cbf = torch.zeros(1)
                loss_cbf_bnd = torch.zeros(1)

            # --- CLF loss (Eq. 17) ---
            clf_viol_frac = torch.zeros(1)
            if self.clf_fn is not None and self.lambda_clf > 0.0 and cur.any():
                v_dot = self.clf_fn(x[cur], u_pred[cur])
                loss_clf = torch.mean(torch.clamp(v_dot, min=0.0) ** 2)
                loss = loss + self.lambda_clf * loss_clf
                with torch.no_grad():
                    clf_viol_frac = (v_dot > 0).float().mean()
            else:
                loss_clf = torch.zeros(1)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total += loss.item()
            self._step += 1

            if writer is not None and self._step % 200 == 0:
                writer.add_scalar("policy/loss_imit",         loss_imit.item(),        self._step)
                writer.add_scalar("policy/loss_cbf",          loss_cbf.item(),         self._step)
                writer.add_scalar("policy/loss_cbf_boundary", loss_cbf_bnd.item(),     self._step)
                writer.add_scalar("policy/loss_clf",          loss_clf.item(),         self._step)
                writer.add_scalar("policy/loss_total",        loss.item(),             self._step)
                writer.add_scalar("policy/cbf_viol_frac",     cbf_viol_frac.item(),    self._step)
                writer.add_scalar("policy/cbf_synth_viol_frac", cbf_synth_viol_frac.item(), self._step)
                writer.add_scalar("policy/cbf_mean_margin",   cbf_mean_margin.item(),  self._step)
                writer.add_scalar("policy/clf_viol_frac",     clf_viol_frac.item(),    self._step)
                # Output range — u_pred_norm_max > 1.0 confirms the linear head
                # can represent full-torque targets (impossible with old Tanh).
                writer.add_scalar("policy/u_pred_norm_max",   u_pred_norm_max,         self._step)
                writer.add_scalar("policy/u_target_norm_max", u_target_norm_max,       self._step)

        mean_loss = total / self.n_iters
        logger.info("policy training — mean loss %.5f over %d iters", mean_loss, self.n_iters)
        print(f"  [policy] mean loss: {mean_loss:.5f}  "
              f"u_pred_max={u_pred_norm_max:.3f}  u_target_max={u_target_norm_max:.3f}")
        return mean_loss

    # ------------------------------------------------------------------

    def dagger_update(
        self,
        env,
        mpc_agent,
        collector,
        task_id: int,
        preprocess_fn,          # fn(raw_obs_np) -> preprocessed+normalised tensor (1, dim)
        n_rollout: int = 5,
        max_ep_steps: int = 1000,
        writer=None,
        a_mu: Optional[np.ndarray] = None,
        a_std: Optional[np.ndarray] = None,
        filter_rollouts: bool = False,
        bc_skip_first_n: int = 0,
        student_frac: float = 0.0,
    ) -> None:
        """One DAGGER refinement iteration (Algorithm 1, lines 7-20).

        Rolls out the mixed policy κ·π* + (1-κ)·π_NN in the environment,
        labels visited states with the MPC expert into the policy buffer
        (never into the dynamics collector), retrains the policy on
        expert base + buffer with the CBF-CLF loss, and ramps λ_CBF / λ_CLF
        for the next iteration (curriculum).

        Parameters
        ----------
        env : gymnasium.Env
            Live environment (reset() / step() interface).
        mpc_agent :
            Expert with .act(obs, task_id) → tensor in physical action space.
        collector :
            DataCollector — read-only here: supplies the expert BC base and
            the normalisation statistics for the DAGGER labels.
        task_id : int
        preprocess_fn :
            Converts a raw numpy obs (state_dim,) to the preprocessed+normalised
            float tensor (1, proc_dim) on self.device.
        n_rollout : int
            Number of rollout episodes per DAGGER iteration.
        max_ep_steps : int
            Truncate episodes at this many steps.
        writer : SummaryWriter or None
        a_mu, a_std : np.ndarray or None
            Per-task action normalisation statistics (shape: (action_dim,)).
            When provided, the policy tanh output (normalised space) is
            denormalised to physical space before mixing with u_mpc.
            If None, the policy output is used as-is (only correct when
            normalize_xu=False).
        filter_rollouts : bool
            If True, pass the mixed action through the expert's safety filter
            before stepping the env (pre-baseline_22 behaviour).  Default
            False: DAGGER must sample the *learner's* state distribution —
            filtered rollouts never enter the near-KOZ region, so the buffer
            contains no avoidance labels and the raw policy stays unsafe
            (baseline_21: expert data 99.9% filter-inactive, unfiltered
            validation KOZ violations flat across all 10 iterations).
            Rollouts run in a dedicated eval env, so violations here are
            training signal, not mission failures.
        bc_skip_first_n : int
            Exclude the task's first N collector transitions from the BC base
            (pass init_rand_steps): their "labels" are RandomAgent noise, not
            expert actions — in baseline_27 they were 27% of the base with
            physically impossible torques up to 4x the actuator box, and the
            magnitude-weighted imitation loss up-weighted them further.
        student_frac : float
            Fraction of rollout episodes run with the PURE NN policy (κ=0)
            regardless of the curriculum.  κ-mixed rollouts are dragged to
            safety by the expert component and almost never reach the KOZ
            corridor (baseline_33: rollout_koz == 0 in 18/20 iterations), so
            the buffer contained no failure states to learn avoidance from.
            Student episodes harvest exactly those states, with expert labels.
        """
        # Expert BC base for retraining after the rollout.  All-expert by
        # construction: the random phase is sliced off (bc_skip_first_n) and
        # the rollout below never writes to the collector (see the buffer
        # comment in __init__).  Also refreshes the per-task norms that
        # _make_policy_train_set uses for the DAGGER labels.
        _expert_train_set, _ = collector.get_dataset(task_id,
                                                     skip_first_n=bc_skip_first_n)

        self._dagger_iter += 1
        kappa = max(0.0, 1.0 - self._dagger_iter / max(1, self._dagger_n_iter))
        print(f"  [dagger iter {self._dagger_iter}] κ={kappa:.2f}  "
              f"λ_cbf={self.lambda_cbf:.2e}  λ_clf={self.lambda_clf:.2e}")

        self.policy.eval()
        new_pairs = 0
        rollout_koz = 0
        n_label_filtered = 0
        nn_vs_expert_diffs: list = []
        n_student = int(round(student_frac * n_rollout))
        self._last_rollout_kappas: list = []

        for ep_i in range(n_rollout):
            # Student episodes (pure NN, κ=0) come first; the rest follow the
            # curriculum κ.  See student_frac in the docstring.
            kappa_ep = 0.0 if ep_i < n_student else kappa
            self._last_rollout_kappas.append(kappa_ep)
            obs, _ = env.reset()
            mpc_agent.reset()
            steps = 0

            while steps < max_ep_steps:
                # Expert action in physical space
                with torch.no_grad():
                    u_mpc_t = mpc_agent.act(obs, task_id=task_id)
                u_mpc = u_mpc_t.detach().cpu().numpy().flatten()

                # Tag: did the expert's QP have to correct THIS label?  Read
                # immediately after act() — the execution filter below (only
                # when filter_rollouts=True) reuses the same SafetyFilter
                # instance and would overwrite last_was_active.
                _sf_lbl = getattr(mpc_agent, 'safety_filter', None)
                label_filtered = 1.0 if getattr(_sf_lbl, 'last_was_active', False) else 0.0

                # NN action: preprocess+normalised state → linear output
                with torch.no_grad():
                    x_proc = preprocess_fn(obs)                    # (1, proc_dim)
                    u_nn_t = self.policy(x_proc, task_id=task_id)  # (1, action_dim)
                u_nn = u_nn_t.cpu().numpy().flatten()

                # Denormalise NN output to physical space; clip before mixing
                # so an untrained linear head cannot dominate u_mix.
                if a_mu is not None and a_std is not None:
                    u_nn = u_nn * a_std + a_mu
                u_nn = np.clip(u_nn, env.action_space.low, env.action_space.high)

                nn_vs_expert_diffs.append(float(np.linalg.norm(u_nn - u_mpc)))

                # Mixed policy in physical space (κ=0 for student episodes)
                u_mix = kappa_ep * u_mpc + (1.0 - kappa_ep) * u_nn
                u_mix = np.clip(u_mix, env.action_space.low, env.action_space.high)

                # Execute the RAW mixed action (Algorithm 1, line 9).  DAGGER
                # exists to correct covariate shift, so rollouts must visit the
                # states the raw policy actually reaches — including near-KOZ
                # states, which is where the expert's avoidance labels come
                # from.  filter_rollouts=True restores the old safe-rollout
                # behaviour (see docstring for why that starves the buffer).
                if filter_rollouts:
                    _sf = getattr(mpc_agent, 'safety_filter', None)
                    u_exec = (np.asarray(_sf.filter(obs.flatten(), u_mix), dtype=np.float32)
                              if _sf is not None else u_mix)
                else:
                    u_exec = u_mix

                obs_next, _, terminated, truncated, info = env.step(
                    u_exec.reshape(env.action_space.shape)
                )
                if info.get("keep_out_violation"):
                    rollout_koz += 1

                # Policy buffer ONLY — rollout transitions are deliberately
                # kept out of the dynamics collector (see __init__ comment).
                # Store (preprocessed state, PHYSICAL expert label);
                # normalisation happens in _make_policy_train_set with the
                # collector's stats current at training time.
                self._dag_states.append(x_proc.detach().cpu())
                self._dag_actions_phys.append(torch.tensor(u_mpc, dtype=torch.float32))
                self._dag_filter_active.append(label_filtered)
                n_label_filtered += int(label_filtered)

                new_pairs += 1
                obs = obs_next
                steps += 1
                if terminated or truncated:
                    break

        nn_vs_expert_mean = float(np.mean(nn_vs_expert_diffs)) if nn_vs_expert_diffs else 0.0
        nn_vs_expert_max  = float(np.max(nn_vs_expert_diffs))  if nn_vs_expert_diffs else 0.0
        label_filt_frac = n_label_filtered / max(new_pairs, 1)
        print(f"  [dagger] {new_pairs} new pairs  rollout_koz={rollout_koz}  "
              f"label_filtered={label_filt_frac:.1%}  "
              f"||u_nn-u_mpc|| mean={nn_vs_expert_mean:.3f}  max={nn_vs_expert_max:.3f}")

        # Retrain on: expert BC base + DAGGER expert buffer, with
        # safety-prioritised sampling (filter-intervened + near-KOZ rows).
        policy_train_set, sample_w = self._make_policy_train_set(
            _expert_train_set, collector, task_id)
        n_dyn = _expert_train_set.tensors[0].shape[0]
        n_dag = len(self._dag_states)
        bc_loss = self.train(policy_train_set, writer=writer, sample_weights=sample_w)

        # Curriculum: scale λ_CBF and λ_CLF by lambda_ramp each iteration up
        # to lambda_max (Algorithm 1, line 19 uses doubling = ramp 2.0).
        self.lambda_cbf = min(self.lambda_cbf * self.lambda_ramp, self.lambda_max)
        self.lambda_clf = min(self.lambda_clf * self.lambda_ramp, self.lambda_max)

        if writer is not None:
            writer.add_scalar("dagger/iter",                   self._dagger_iter,    self._step)
            writer.add_scalar("dagger/kappa",                  kappa,                self._step)
            writer.add_scalar("dagger/lambda_cbf",             self.lambda_cbf,      self._step)
            writer.add_scalar("dagger/lambda_clf",             self.lambda_clf,      self._step)
            writer.add_scalar("dagger/new_pairs",              new_pairs,            self._step)
            # KOZ hits during the (unfiltered) rollouts: nonzero early is GOOD
            # — it means the buffer now contains near-KOZ states with expert
            # avoidance labels; it should trend to 0 as the policy improves.
            writer.add_scalar("dagger/rollout_koz",            rollout_koz,          self._step)
            # Fraction of this iteration's labels the expert's QP corrected —
            # the "filter had to step in" states the sampler prioritises.
            writer.add_scalar("dagger/rollout_label_filtered_frac", label_filt_frac,  self._step)
            if sample_w is not None:
                writer.add_scalar("policy/sample_weight_mean", float(sample_w.mean()), self._step)
            writer.add_scalar("dagger/nn_vs_expert_mean",      nn_vs_expert_mean,    self._step)
            writer.add_scalar("dagger/nn_vs_expert_max",       nn_vs_expert_max,     self._step)
            writer.add_scalar("dagger/bc_loss_after_training", bc_loss,              self._step)
            # dagger/ namespace (policy-step axis) — the same quantities are
            # also logged as policy/train_set_n_* from the periodic training
            # block on the env-step axis; sharing one tag zigzagged the plot.
            writer.add_scalar("dagger/train_set_n_dyn",        n_dyn,                self._step)
            writer.add_scalar("dagger/train_set_n_dag",        n_dag,                self._step)

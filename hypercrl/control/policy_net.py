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
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    ) -> None:
        super().__init__()

        dims = [state_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], action_dim))
        # No Tanh — linear head so full-torque targets are representable.

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        if not self._dag_states:
            return base, self._sample_weights(base.tensors[0], tags=None)

        dag_x = torch.cat(self._dag_states, dim=0)             # (Nd, proc_dim)
        dag_u = torch.stack(self._dag_actions_phys, dim=0)     # (Nd, act_dim)

        try:
            _, _, a_mu, a_std = collector.norm(task_id)
            dag_u = (dag_u - a_mu.flatten().to(dag_u.dtype)) \
                / a_std.flatten().to(dag_u.dtype)
        except (KeyError, AttributeError, TypeError):
            pass  # no norms (normalize_xu=False) — physical labels match the base

        dyn_x, dyn_u, dyn_xtt = base.tensors
        Nd = dag_x.shape[0]
        dag_dummy = torch.zeros(Nd, dyn_xtt.shape[1])

        dataset = _TDS(
            torch.cat([dyn_x, dag_x],       dim=0),
            torch.cat([dyn_u, dag_u],       dim=0),
            torch.cat([dyn_xtt, dag_dummy], dim=0),
        )
        tags = torch.tensor(self._dag_filter_active, dtype=torch.float32)
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

    def train(self, dataset, writer=None, sample_weights=None) -> float:
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

            x, u_expert, _ = batch
            x        = x.to(self.device)
            u_expert = u_expert.to(self.device)

            u_pred = self.policy(x)

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
            if self.cbf_fn is not None and self.lambda_cbf > 0.0:
                h_dot = self.cbf_fn(x, u_pred)
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
                    x_bnd = self.boundary_sampler(x)
                    # Keep only control-feasible states — infeasible ones
                    # (ḣ≈0 near the boundary) have no gradient through u and
                    # pin the loss/viol-frac at a floor (baseline_34: ~0.6).
                    if self.cbf_feasible_fn is not None:
                        x_bnd = x_bnd[self.cbf_feasible_fn(x_bnd)]
                    if x_bnd.shape[0] > 0:
                        u_bnd = self.policy(x_bnd)
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
            if self.clf_fn is not None and self.lambda_clf > 0.0:
                v_dot = self.clf_fn(x, u_pred)
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
                    u_nn_t = self.policy(x_proc)                   # (1, action_dim)
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


# ---------------------------------------------------------------------------
# Torch-differentiable CBF factories
# ---------------------------------------------------------------------------

def make_cheetah_cbf_fn(
    zones: List[Tuple[float, float]],
    x_mu: torch.Tensor,
    x_std: torch.Tensor,
    a_mu: torch.Tensor,
    a_std: torch.Tensor,
    alpha: float = 1.0,
    x_accel_gain: float = 0.5,
) -> Callable:
    """Return a differentiable CBF fn for HalfCheetahSafeEnv.

    The stored training states are preprocessed + normalised.  Layout after
    preprocessing (19 dims):
        [0]  z_pos
        [1]  cos(root_angle)
        [2]  sin(root_angle)
        [3-8] joint angles
        [9]  qvel[0]  ← global x-velocity  (what the CBF needs)
        [10-17] remaining velocities
        [18] x_pos   ← global x-position   (what the CBF needs)

    Both x_vel and x_pos are further normalised by (x_mu, x_std); the action
    is normalised by (a_mu, a_std).  This factory captures those statistics so
    the returned fn can denormalise on the fly.

    H_dot (linear in u, from HalfCheetahKeepOutCBF.H_dot_expr):
        left  approach: -(s @ u_phys) - alpha * x_vel
        right approach:  (s @ u_phys) + alpha * x_vel
    where s = x_accel_gain / n_actions (uniform sensitivity).
    """
    # Flatten: finalize() stores norms as (1, dim) after .T — index into dim axis.
    x_mu_f = x_mu.flatten()
    x_std_f = x_std.flatten()
    xvel_mu  = float(x_mu_f[9])
    xvel_std = float(x_std_f[9])
    xpos_mu  = float(x_mu_f[18])
    xpos_std = float(x_std_f[18])

    def cbf_fn(state_norm: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
        dev = state_norm.device

        # Denormalise the two quantities the CBF depends on.
        x_vel = state_norm[:, 9]  * xvel_std + xvel_mu   # global x-velocity
        x_pos = state_norm[:, 18] * xpos_std + xpos_mu   # global x-position

        # Denormalise action to physical space (a_mu/a_std are (1,n_act) — flatten to (n_act,)).
        u_phys = action_norm * a_std.flatten().to(dev) + a_mu.flatten().to(dev)

        n_act = u_phys.shape[1]
        s     = x_accel_gain / n_act
        s_u   = s * u_phys.sum(dim=1)          # s @ u  (uniform s)

        h_dot_left  = -s_u - alpha * x_vel
        h_dot_right =  s_u + alpha * x_vel

        min_h_dot = torch.full_like(x_vel, float("inf"))
        for x_min, x_max in zones:
            x_min_t = torch.tensor(x_min, dtype=x_vel.dtype, device=dev)
            x_max_t = torch.tensor(x_max, dtype=x_vel.dtype, device=dev)

            on_left  = (x_pos <= x_min_t).float()
            on_right = (x_pos >= x_max_t).float()
            in_zone  = 1.0 - on_left - on_right

            h_dot = (
                on_left  * h_dot_left
                + on_right * h_dot_right
                + in_zone  * torch.minimum(h_dot_left, h_dot_right)
            )
            min_h_dot = torch.minimum(min_h_dot, h_dot)

        return min_h_dot   # positive ⇒ constraint met; negative ⇒ violation

    return cbf_fn


def make_space_cbf_components(
    x_mu: torch.Tensor,
    x_std: torch.Tensor,
    inertia,               # (3,3) array-like
    gamma: float = 0.5,
) -> Callable:
    """Affine decomposition of the CBF condition for SatDynEnv (Eq. 5-7).

    Returns fn(state_norm) → (c0, b) with

        Ḣ(x,u) + γH(x) = c0(x) + b(x)·u_raw ,   u_raw ∈ [-1, 1]³.

    The affine-in-u structure is what makes the closed-form safety head and
    the feasibility test possible (best case over the box = c0 + Σ|bᵢ|).

    State layout (collector-normalised, 13-dim):
        [0:4]  q_e (error quaternion, scalar-first)
        [4:7]  omega_norm  (physical: * 5 rad/s)
        [7]    theta_margin_norm  (physical: (v+1)*3π/4 - π/2)
        [8]    theta_norm         (physical: (v+1)*π/2)
        [9:12] rel_avoid_b  (unit direction: (avoid_b - boresight) / |…|)
    """
    import numpy as _np
    _PI = math.pi
    _SCALE_TORQUE = 2.0
    _SCALE_OMEGA  = 5.0
    _U_MAX = _SCALE_TORQUE * math.sqrt(3.0)

    I_np    = _np.array(inertia, dtype=_np.float64)
    I_inv_np = _np.linalg.inv(I_np)
    I_t     = torch.tensor(I_np,    dtype=torch.float32)
    I_inv_t = torch.tensor(I_inv_np, dtype=torch.float32)
    _bore   = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    x_mu_f = x_mu.flatten()
    x_std_f = x_std.flatten()

    def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1],
            a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2],
            a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0],
        ], dim=1)

    def components(state_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dev  = state_norm.device
        I    = I_t.to(dev)
        Ii   = I_inv_t.to(dev)
        bore = _bore.to(dev).unsqueeze(0)          # (1, 3)

        # Undo collector normalisation → env-normalised obs
        obs = state_norm * x_std_f.to(dev) + x_mu_f.to(dev)

        # Physical quantities
        omega  = obs[:, 4:7] * _SCALE_OMEGA                            # (B, 3) rad/s
        h      = (obs[:, 7:8] + 1.0) * (3.0*_PI/4.0) - _PI/2.0       # (B, 1) rad
        theta  = (obs[:, 8:9] + 1.0) * (_PI / 2.0)                    # (B, 1) rad
        rel_av = obs[:, 9:12]                                           # (B, 3)

        # Recover avoid_in_b from stored rel_avoid_b
        half_sin = (theta / 2.0).sin()                                  # (B, 1)
        av_b = bore + 2.0 * half_sin * rel_av
        av_b = av_b / av_b.norm(dim=1, keepdim=True).clamp(min=1e-9)   # (B, 3)

        # Singularity guard: the ḧ terms divide by sin θ but their numerators
        # (e.g. ω·dc_perp_dt ~ |ω|²) do NOT vanish as θ → 0 or π, so near the
        # poles the quotient explodes as |ω|²/sin θ.  With the loss squaring
        # it, a single near-pole sample in a batch reached ~1e22 in run
        # baseline_17 and destroyed the policy weights.  θ ≈ π (pointing
        # directly away from the avoid vector) is the SAFEST attitude, so
        # zeroing the rate terms there is exact in the limit; θ ≈ 0 is deep
        # inside the KOZ where H ≈ h < 0 keeps the γH penalty active.
        sin_t_raw = theta.sin().squeeze(1)                              # (B,)
        valid   = sin_t_raw > 0.05                                      # ~3° from poles
        sin_t   = sin_t_raw.clamp(min=0.05)                             # safe denominator
        cos_t   = theta.cos().squeeze(1)                                # (B,)
        bore_b  = bore.expand_as(av_b)

        # c_perp = boresight × avoid_in_b,  |c_perp| = sin(theta)
        c_perp = _cross(bore_b, av_b)                                   # (B, 3)

        # h_dot = -ω · c_perp / sin(theta)
        h_dot = -(omega * c_perp).sum(dim=1) / sin_t                   # (B,)
        h_dot = torch.where(valid, h_dot, torch.zeros_like(h_dot))
        h     = h.squeeze(1)                                            # (B,)

        # H(x) = h + |ḣ|·ḣ / (2·U_MAX)
        H_val = h + h_dot.abs() * h_dot / (2.0 * _U_MAX)              # (B,)

        # Drift: ω_dot_f = I⁻¹(−ω × Iω)
        Iw           = omega @ I.T                                      # (B, 3)
        omega_dot_f  = (-_cross(omega, Iw)) @ Ii.T                     # (B, 3)

        # dc_perp/dt = boresight × (−ω × avoid_in_b)
        dc_perp_dt = _cross(bore_b, -_cross(omega, av_b))              # (B, 3)

        # ḧ drift = −(ω_dot_f·c_perp + ω·dc_perp_dt)/sinθ − ḣ²·cosθ/sinθ
        num       = (omega_dot_f * c_perp).sum(dim=1) + (omega * dc_perp_dt).sum(dim=1)
        hdd_drift = -num / sin_t - h_dot * cos_t * h_dot / sin_t         # (B,)
        hdd_drift = torch.where(valid, hdd_drift, torch.zeros_like(hdd_drift))

        # ḧ linear-in-u coefficient: g = −(I⁻¹ c_perp)·τ_scale / sinθ
        # Physical torque = u_raw * _SCALE_TORQUE, so g already absorbs τ_scale.
        g_hdd = -(c_perp @ Ii.T) * _SCALE_TORQUE / sin_t.unsqueeze(1)  # (B, 3)
        g_hdd = torch.where(valid.unsqueeze(1), g_hdd, torch.zeros_like(g_hdd))

        # condition = ḣ + |ḣ|/U·ḧ_drift + γH  +  (|ḣ|/U·g)·u_raw
        c0 = h_dot + h_dot.abs() / _U_MAX * hdd_drift + gamma * H_val   # (B,)
        b  = (h_dot.abs() / _U_MAX).unsqueeze(1) * g_hdd                # (B, 3)
        return c0, b

    return components


def make_space_cbf_fn(
    x_mu: torch.Tensor,
    x_std: torch.Tensor,
    a_mu: torch.Tensor,
    a_std: torch.Tensor,
    inertia,               # (3,3) array-like
    gamma: float = 0.5,
) -> Callable:
    """Differentiable H_dot(x,u) + γH(x) for SatDynEnv (paper Eq. 5-7).

    Returns the full CBF condition value per sample:
        positive ⇒ constraint satisfied (no penalty)
        negative ⇒ violation (loss term penalises this squared)

    Thin wrapper over make_space_cbf_components (condition is affine in u).
    """
    components = make_space_cbf_components(x_mu, x_std, inertia, gamma)
    a_mu_f, a_std_f = a_mu.flatten(), a_std.flatten()

    def cbf_fn(state_norm: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
        dev = state_norm.device
        c0, b = components(state_norm)
        u_raw = action_norm * a_std_f.to(dev) + a_mu_f.to(dev)         # (B, 3)
        return c0 + (b * u_raw).sum(dim=1)   # ≥ 0 ⇒ safe

    return cbf_fn


def make_space_cbf_feasible_fn(
    x_mu: torch.Tensor,
    x_std: torch.Tensor,
    inertia,
    gamma: float = 0.5,
    eps: float = 0.0,
) -> Callable:
    """Control-feasibility mask: can ANY action in [-1,1]³ reach the margin?

    best case over the box = c0 + Σᵢ|bᵢ|.  States failing this (typically
    ḣ ≈ 0 near the boundary, where control authority through the
    relative-degree-2 barrier vanishes) are excluded from the boundary CBF
    penalty — no gradient can fix them, they only inflate the loss floor
    (baseline_34: cbf_synth_viol_frac pinned at ~0.6).
    """
    components = make_space_cbf_components(x_mu, x_std, inertia, gamma)

    def feasible_fn(state_norm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            c0, b = components(state_norm)
            return (c0 + b.abs().sum(dim=1)) >= eps

    return feasible_fn




def make_space_margin_fn(x_mu: torch.Tensor, x_std: torch.Tensor) -> Callable:
    """θ-margin [deg] per sample from a collector-normalised spaceEnv batch.

    Used by PolicyTrainer._sample_weights to grade how safety-critical each
    training row is: obs[7] is the env-normalised theta_margin
    (physical: (v+1)·3π/4 − π/2 rad, negative = inside the KOZ).
    """
    x_mu7  = float(x_mu.flatten()[7])
    x_std7 = float(x_std.flatten()[7])

    def margin_fn(state_norm: torch.Tensor) -> torch.Tensor:
        obs7 = state_norm[:, 7] * x_std7 + x_mu7
        margin_rad = (obs7 + 1.0) * (3.0 * math.pi / 4.0) - math.pi / 2.0
        return margin_rad * (180.0 / math.pi)

    return margin_fn


def make_space_boundary_sampler(
    x_mu: torch.Tensor,
    x_std: torch.Tensor,
    margin_low_deg: float = -10.0,
    margin_high_deg: float = 15.0,
) -> Callable:
    """Resample a real spaceEnv batch through the KOZ approach corridor.

    Takes collector-normalised states, denormalises (exact under the frozen
    normaliser), then per sample:
      * redraws the θ-margin uniformly in [margin_low, margin_high] degrees
        while keeping the state's own KOZ half-angle (θ − margin) consistent,
      * randomly rescales and sign-flips ω (0.5–2×) so both approach and
        retreat rates are covered,
    and renormalises.  Attitude (q_e) and avoid-direction stay real.  θ is
    clamped away from the sin-θ pole guard.  Used to evaluate the CBF penalty
    on states the (all-safe) training data never contains.
    """
    x_mu_f  = x_mu.flatten()
    x_std_f = x_std.flatten()
    lo = math.radians(margin_low_deg)
    hi = math.radians(margin_high_deg)

    def sampler(x_norm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dev = x_norm.device
            mu, std = x_mu_f.to(dev), x_std_f.to(dev)
            obs = x_norm * std + mu
            B = obs.shape[0]

            margin = (obs[:, 7] + 1.0) * (3.0 * math.pi / 4.0) - math.pi / 2.0
            theta  = (obs[:, 8] + 1.0) * (math.pi / 2.0)
            half   = (theta - margin).clamp(math.radians(5.0), math.radians(45.0))

            m_new  = torch.rand(B, device=dev) * (hi - lo) + lo
            th_new = (half + m_new).clamp(0.06, math.pi - 0.06)  # sin-θ guard
            m_new  = th_new - half

            obs = obs.clone()
            obs[:, 7] = -1.0 + (m_new + math.pi / 2.0) * 4.0 / (3.0 * math.pi)
            obs[:, 8] = -1.0 + th_new * 2.0 / math.pi
            sign  = torch.where(torch.rand(B, 1, device=dev) < 0.5, -1.0, 1.0)
            scale = (torch.rand(B, 1, device=dev) * 1.5 + 0.5) * sign
            obs[:, 4:7] = obs[:, 4:7] * scale

            return (obs - mu) / std

    return sampler


def make_space_clf_fn(
    x_mu: torch.Tensor,
    x_std: torch.Tensor,
    a_mu: torch.Tensor,
    a_std: torch.Tensor,
    inertia,               # (3,3) array-like
    c_q: float = 1.0,
    c_w: float = 0.1,
    zeta_min: float = 0.001,
    zeta_max: float = 0.06,
    j: float = 5.0,        # keep in sync with SpaceAttitudeCLF defaults
    c: float = 0.6,        # midpoint [rad]: ζ≈ζ_min beyond ~60° attitude error
    scale_omega: float = 5.0,
) -> Callable:
    """Differentiable V_dot(x,u) + ζ(x)V(x) for SatDynEnv (paper Eq. 11-13).

    Returns the CLF condition value per sample:
        positive ⇒ stability constraint violated (loss penalises this squared)
        ≤ 0       ⇒ constraint satisfied

    The sigmoid decay rate ζ(x) (Eq. 13) is reproduced exactly.
    """
    import numpy as _np
    _SCALE_TORQUE = 2.0

    I_np     = _np.array(inertia, dtype=_np.float64)
    I_inv_np = _np.linalg.inv(I_np)
    I_t      = torch.tensor(I_np,    dtype=torch.float32)
    I_inv_t  = torch.tensor(I_inv_np, dtype=torch.float32)

    x_mu_f = x_mu.flatten()
    x_std_f = x_std.flatten()
    a_mu_f = a_mu.flatten()
    a_std_f = a_std.flatten()

    def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1],
            a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2],
            a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0],
        ], dim=1)

    def clf_fn(state_norm: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
        dev = state_norm.device
        I   = I_t.to(dev)
        Ii  = I_inv_t.to(dev)

        # Undo collector normalisation → env-normalised obs
        obs = state_norm * x_std_f.to(dev) + x_mu_f.to(dev)

        q_e     = obs[:, 0:4]                                           # (B, 4)
        omega   = obs[:, 4:7] * scale_omega                             # (B, 3) rad/s
        q_e0    = q_e[:, 0]                                             # (B,)
        q_e_vec = q_e[:, 1:4]                                           # (B, 3)

        # V(x) = c_q·‖q_e_vec‖² + c_w·‖ω‖²
        V_val = c_q * q_e_vec.pow(2).sum(dim=1) + c_w * omega.pow(2).sum(dim=1)  # (B,)

        # Quaternion kinematics drift: dq_e_vec/dt = 0.5·(q_e0·ω + q_e_vec × ω)
        dqv_dt = 0.5 * (q_e0.unsqueeze(1) * omega + _cross(q_e_vec, omega))      # (B, 3)

        # Euler drift: ω_dot_f = I⁻¹(−ω × Iω)
        Iw          = omega @ I.T                                        # (B, 3)
        omega_dot_f = (-_cross(omega, Iw)) @ Ii.T                       # (B, 3)

        # LfV = c_q·2·q_e_vec·dqv_dt + c_w·2·ω·ω_dot_f
        LfV = (
            c_q * 2.0 * (q_e_vec * dqv_dt).sum(dim=1)
            + c_w * 2.0 * (omega * omega_dot_f).sum(dim=1)
        )                                                                # (B,)

        # LgV = c_w·2·τ_scale·(I⁻¹ᵀ ω)  — coefficient for u_raw ∈ [-1,1]³
        LgV = c_w * 2.0 * _SCALE_TORQUE * (omega @ Ii)                 # (B, 3)

        # ζ(x) from Eq. 13
        att_err = 2.0 * torch.acos(q_e0.clamp(-1.0 + 1e-7, 1.0 - 1e-7))  # (B,)
        zeta = zeta_min + (zeta_max - zeta_min) / (
            1.0 + torch.exp(torch.tensor(j, device=dev) * (att_err - c))
        )                                                                # (B,)

        # Undo collector norm → raw action in [-1,1]³
        u_raw = action_norm * a_std_f.to(dev) + a_mu_f.to(dev)         # (B, 3)

        # V_dot(x,u) = LfV + LgV·u + ζ·V  (≤ 0 ⇒ stability constraint met)
        V_dot = LfV + (LgV * u_raw).sum(dim=1) + zeta * V_val          # (B,)

        return V_dot   # positive ⇒ CLF constraint violated

    return clf_fn

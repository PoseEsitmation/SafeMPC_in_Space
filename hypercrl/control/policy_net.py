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
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """4-layer MLP with LayerNorm + ReLU + Dropout, tanh-bounded output.

    Architecture matches Section IV-A of the paper (256 neurons, 4 layers).
    Input:  preprocessed + normalised state (same as dynamics model input).
    Output: normalised action clipped by tanh, then scaled to [-u_max, u_max].
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
        layers.append(nn.Tanh())

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
        self.n_iters         = getattr(hparams, "policy_train_iters",     1000)
        self.bs              = getattr(hparams, "policy_bs",              128)
        self._dagger_iter    = 0   # counts DAGGER refinement rounds for curriculum

        self.cbf_fn = cbf_fn
        self.clf_fn = clf_fn

        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=getattr(hparams, "policy_lr", 1e-4),
            weight_decay=1e-4,
        )

        self._step = 0  # global step counter for TensorBoard

    # ------------------------------------------------------------------

    def train(self, dataset, writer=None) -> float:
        """Run one training phase; return mean total loss."""
        loader = DataLoader(dataset, batch_size=self.bs, shuffle=True, drop_last=True)
        it = iter(loader)

        self.policy.train()
        total = 0.0

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

            # --- imitation loss ---
            loss_imit = F.mse_loss(u_pred, u_expert)
            loss = self.lambda_imit * loss_imit

            # --- CBF loss (Eq. 16) ---
            cbf_viol_frac = torch.zeros(1)
            cbf_mean_margin = torch.zeros(1)
            if self.cbf_fn is not None and self.lambda_cbf > 0.0:
                h_dot = self.cbf_fn(x, u_pred)
                loss_cbf = torch.mean(torch.clamp(-h_dot, min=0.0) ** 2)
                loss = loss + self.lambda_cbf * loss_cbf
                with torch.no_grad():
                    cbf_viol_frac   = (h_dot < 0).float().mean()
                    cbf_mean_margin = h_dot.mean()
            else:
                loss_cbf = torch.zeros(1)

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
                writer.add_scalar("policy/loss_clf",          loss_clf.item(),         self._step)
                writer.add_scalar("policy/loss_total",        loss.item(),             self._step)
                writer.add_scalar("policy/cbf_viol_frac",     cbf_viol_frac.item(),    self._step)
                writer.add_scalar("policy/cbf_mean_margin",   cbf_mean_margin.item(),  self._step)
                writer.add_scalar("policy/clf_viol_frac",     clf_viol_frac.item(),    self._step)

        mean_loss = total / self.n_iters
        logger.info("policy training — mean loss %.5f over %d iters", mean_loss, self.n_iters)
        print(f"  [policy] mean loss: {mean_loss:.5f}")
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
    ) -> None:
        """One DAGGER refinement iteration (Algorithm 1, lines 7-20).

        Rolls out the mixed policy κ·π* + (1-κ)·π_NN in the environment,
        labels visited states with the MPC expert, adds them to the collector,
        retrains the policy on the augmented dataset with the CBF-CLF loss, and
        doubles λ_CBF / λ_CLF for the next iteration (curriculum).

        Parameters
        ----------
        env : gymnasium.Env
            Live environment (reset() / step() interface).
        mpc_agent :
            Expert with .act(obs, task_id) → tensor in physical action space.
        collector :
            DataCollector — new (obs, u_mpc, obs_next) triples are appended.
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
        """
        self._dagger_iter += 1
        kappa = max(0.0, 1.0 - self._dagger_iter / max(1, getattr(self, "_dagger_n_iter", 5)))
        print(f"  [dagger iter {self._dagger_iter}] κ={kappa:.2f}  "
              f"λ_cbf={self.lambda_cbf:.2e}  λ_clf={self.lambda_clf:.2e}")

        self.policy.eval()
        new_pairs = 0

        for _ in range(n_rollout):
            obs, _ = env.reset()
            mpc_agent.reset()
            steps = 0

            while steps < max_ep_steps:
                # Expert action (normalised, in action space)
                with torch.no_grad():
                    u_mpc_t = mpc_agent.act(obs, task_id=task_id)
                u_mpc = u_mpc_t.detach().cpu().numpy().flatten()

                # NN action: preprocess+normalised state → normalised tanh output
                with torch.no_grad():
                    x_proc = preprocess_fn(obs)                    # (1, proc_dim)
                    u_nn_t = self.policy(x_proc)                   # (1, action_dim)
                u_nn = u_nn_t.cpu().numpy().flatten()

                # Denormalise NN output to physical space so it can be mixed with
                # u_mpc (which is already in physical action space).
                if a_mu is not None and a_std is not None:
                    u_nn = u_nn * a_std + a_mu

                # Mixed policy in physical space, clipped to env action bounds.
                u_mix = kappa * u_mpc + (1.0 - kappa) * u_nn
                u_mix = np.clip(u_mix, env.action_space.low, env.action_space.high)

                obs_next, _, terminated, truncated, _ = env.step(
                    u_mix.reshape(env.action_space.shape)
                )

                # Label with expert action and aggregate into dataset
                collector.add(obs, u_mpc, obs_next, task_id)
                new_pairs += 1

                obs = obs_next
                steps += 1
                if terminated or truncated:
                    break

        print(f"  [dagger] aggregated {new_pairs} new (s, u_expert) pairs")

        # Retrain on the augmented dataset with CBF-CLF-informed loss
        train_set, _ = collector.get_dataset(task_id)
        self.train(train_set, writer=writer)

        # Curriculum: double λ_CBF and λ_CLF (Algorithm 1, line 19)
        self.lambda_cbf = min(self.lambda_cbf * 2.0, 1.0)
        self.lambda_clf = min(self.lambda_clf * 2.0, 1.0)

        if writer is not None:
            writer.add_scalar("policy/dagger_iter",  self._dagger_iter, self._step)
            writer.add_scalar("policy/kappa",         kappa,            self._step)
            writer.add_scalar("policy/lambda_cbf",    self.lambda_cbf,  self._step)
            writer.add_scalar("policy/lambda_clf",    self.lambda_clf,  self._step)


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
    a_mu_f = a_mu.flatten()
    a_std_f = a_std.flatten()

    def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1],
            a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2],
            a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0],
        ], dim=1)

    def cbf_fn(state_norm: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
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

        sin_t   = theta.sin().squeeze(1).clamp(min=1e-6)               # (B,)
        cos_t   = theta.cos().squeeze(1)                                # (B,)
        bore_b  = bore.expand_as(av_b)

        # c_perp = boresight × avoid_in_b,  |c_perp| = sin(theta)
        c_perp = _cross(bore_b, av_b)                                   # (B, 3)

        # h_dot = -ω · c_perp / sin(theta)
        h_dot = -(omega * c_perp).sum(dim=1) / sin_t                   # (B,)
        h     = h.squeeze(1)                                            # (B,)

        # H(x) = h + |ḣ|·ḣ / (2·U_MAX)
        H_val = h + h_dot.abs() * h_dot / (2.0 * _U_MAX)              # (B,)

        # Drift: ω_dot_f = I⁻¹(−ω × Iω)
        Iw           = omega @ I.T                                      # (B, 3)
        omega_dot_f  = (-_cross(omega, Iw)) @ Ii.T                     # (B, 3)

        # dc_perp/dt = boresight × (−ω × avoid_in_b)
        dc_perp_dt = _cross(bore_b, -_cross(omega, av_b))              # (B, 3)

        # ḧ drift = −(ω_dot_f·c_perp + ω·dc_perp_dt)/sinθ + ḣ·cosθ·ḣ/sin²θ
        num       = (omega_dot_f * c_perp).sum(dim=1) + (omega * dc_perp_dt).sum(dim=1)
        hdd_drift = -num / sin_t + h_dot * cos_t * h_dot / sin_t.pow(2)  # (B,)

        # ḧ linear-in-u coefficient: g = −(I⁻¹ c_perp)·τ_scale / sinθ
        # Physical torque = u_raw * _SCALE_TORQUE, so g already absorbs τ_scale.
        g_hdd = -(c_perp @ Ii.T) * _SCALE_TORQUE / sin_t.unsqueeze(1)  # (B, 3)

        # Undo collector norm → raw action in [-1,1]³
        u_raw = action_norm * a_std_f.to(dev) + a_mu_f.to(dev)         # (B, 3)

        # Ḣ(x,u) = ḣ + |ḣ|/U_MAX · (ḧ_drift + g·u_raw)
        H_dot = (
            h_dot
            + h_dot.abs() / _U_MAX * hdd_drift
            + h_dot.abs() / _U_MAX * (g_hdd * u_raw).sum(dim=1)
        )                                                                # (B,)

        return H_dot + gamma * H_val   # ≥ 0 ⇒ safe

    return cbf_fn


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
    j: float = 1.0,
    c: float = 2.0,
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

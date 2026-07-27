"""CBF and CLF for the SatDynEnv attitude-control environment.

Implements the paper's safety certificates (Section III.A–B) adapted to
attitude dynamics (Euler equations + quaternion kinematics) instead of the
translational CWH dynamics used in the original spacecraft rendezvous paper.

State layout received by the filter (13-dim, normalised by SatDynEnv._normalise):
  obs[0:4]  — q_e  (error quaternion, scalar-first, already unit-norm)
  obs[4:7]  — omega_e / scale_omega   (scale_omega = 5 rad/s)
  obs[7]    — theta_margin_norm        → physical: (obs+1)*3π/4 - π/2  [rad]
  obs[8]    — theta_norm               → physical: (obs+1)*π/2          [rad]
  obs[9:12] — rel_avoid_b  (unit vector: (avoid_in_b - boresight) / |…|)
  obs[12]   — qe_0_prev

Physical reconstruction:
  omega       = obs[4:7] * 5
  theta       = (obs[8] + 1) * π/2
  avoid_in_b  = boresight_b + 2*sin(theta/2) * rel_avoid_b   (unit vector)
  (proof: |avoid_in_b - boresight_b| = 2*sin(theta/2) for two unit vecs at angle theta)

CBF — relative-degree-2 barrier for KOZ (paper Eq. 6-7):
  h(x)  = theta_margin = theta - half_angle
  ḣ(x)  = -omega · (boresight_b × avoid_in_b) / sin(theta)
  H(x)  = h + |ḣ|·ḣ / (2·u_max)
  Ḣ(x,u) = ḣ + |ḣ|·ḧ(x,u) / u_max  [affine in u]

CLF — quadratic Lyapunov function (paper Eq. 11-13):
  V(x)  = c_q·‖q_e_vec‖² + c_w·‖omega‖²
  ζ(x)  = ζ_min + (ζ_max - ζ_min) / (1 + exp(j·(att_err - c)))  [Eq. 13]
  V(x,u) = LfV + LgV·u + ζ(x)·V(x)   [must be ≤ δ]

Two forms of the same certificates live here:
  * SpaceAttitudeCBF / SpaceAttitudeCLF — numpy + cvxpy, used by the runtime
    QP SafetyFilter (input: env-normalised obs).
  * make_space_* factories — batched, torch-differentiable closures used by
    the CBF/CLF terms of the imitation loss (input: collector-normalised
    states, hence the extra (x_mu, x_std) / (a_mu, a_std) arguments).
Keeping both in one module is deliberate: the training penalty must match the
condition the runtime filter enforces, so shared constants (γ default, the
sin-θ guard, torque/rate scales) can only drift if edited here.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import cvxpy as cp
import numpy as np
import torch

from hypercrl.control.safety_filter import CBF, CLF

_PI = math.pi

# sin θ below this ⇒ within ~3° of a pole (θ≈0: deep inside the KOZ; θ≈π:
# pointing directly away — the safest attitude).  The ḧ terms divide by sin θ
# with numerators ~|ω|² that do NOT vanish at the poles, so a tighter guard
# (the old 1e-6) let the quotient explode as |ω|²/sinθ.  Zeroing the rate
# terms there is exact in the θ→π limit and conservative for θ→0 (γH keeps
# the penalty/constraint active via h < 0).
_SIN_GUARD = 0.05

# Physical constants from space_KOZ.py
_SCALE_TORQUE = 2.0        # [Nm] max torque per axis
_SCALE_OMEGA  = 5.0        # [rad/s] obs normalisation factor
_U_MAX        = _SCALE_TORQUE * math.sqrt(3.0)   # max torque L2-norm
_BORESIGHT_B  = np.array([1.0, 0.0, 0.0])        # fixed instrument axis in body

# Class-K multiplier α(H) = γ·H, shared by the QP filter and the training loss
# so the penalty can never drift from the constraint that is enforced.
# gamma = 0.5 (paper value).  gamma=0.2 was evaluated in paper_final2: it binds
# ~2.5x earlier (gentle corrections, perfect safety even for the untrained
# policy) but floors the intervention RATE at ~2% — the reliance-declines-to-
# zero result needs 0.5 (paper_final: 8.6%→0.00%).
_GAMMA = 0.5


def _denorm_obs(obs: np.ndarray):
    """Extract physical quantities from the normalised 13-dim observation."""
    q_e      = obs[0:4].astype(float)
    omega    = obs[4:7].astype(float) * _SCALE_OMEGA
    th_marg  = float((obs[7] + 1.0) * (3.0 * _PI / 4.0) - _PI / 2.0)
    theta    = float((obs[8] + 1.0) * (_PI / 2.0))
    rel_av_b = obs[9:12].astype(float)
    return q_e, omega, th_marg, theta, rel_av_b


def _avoid_in_b(theta: float, rel_av_b: np.ndarray) -> np.ndarray:
    """Recover the avoid-vector in body frame from the stored normalised obs.

    Derivation: |a - b| = 2·sin(θ/2) for two unit vectors a, b at angle θ,
    so a = b + 2·sin(θ/2)·(a-b)/|a-b| = boresight + 2·sin(θ/2)·rel_avoid_b.
    """
    half = math.sin(theta / 2.0)
    av_b = _BORESIGHT_B + 2.0 * half * rel_av_b
    n = np.linalg.norm(av_b)
    return av_b / n if n > 1e-9 else _BORESIGHT_B.copy()


# ---------------------------------------------------------------------------
# CBF
# ---------------------------------------------------------------------------

class SpaceAttitudeCBF(CBF):
    """Relative-degree-2 attitude KOZ barrier (paper Eq. 6-7).

    The env stores the KOZ half-angle inside f_zone.half_angle.  Pass the env
    directly; the CBF reads the current half_angle on every call.

    Parameters
    ----------
    env : SatDynEnv
        Live environment — used for inertia tensor and current f_zone.
    gamma : float
        Class-K multiplier α(H) = γ·H in the CBF condition (paper Sec. III.A).
    """

    def __init__(self, env, gamma: float = _GAMMA) -> None:
        self._env   = env
        self.gamma  = gamma
        self._I     = env.inertia.astype(float)
        self._I_inv = np.linalg.inv(self._I)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _kinematics(self, obs: np.ndarray):
        """Return (h, h_dot, c_perp, sin_theta, omega, avoid_in_b).

        c_perp = boresight × avoid_in_b   (not normalised, |c_perp| = sin θ)
        """
        q_e, omega, th_marg, theta, rel_av_b = _denorm_obs(obs)

        sin_t = math.sin(theta)
        av_b  = _avoid_in_b(theta, rel_av_b)
        c_perp = np.cross(_BORESIGHT_B, av_b)          # |c_perp| = sin θ

        # ḣ = −ω · c_perp / sin(θ)
        if sin_t > _SIN_GUARD:
            h_dot = -float(np.dot(omega, c_perp)) / sin_t
        else:
            h_dot = 0.0

        return th_marg, h_dot, c_perp, sin_t, omega, av_b

    def _h_dot_dot_f(self, omega, c_perp, sin_t, theta, h_dot, av_b) -> float:
        """Nonlinear (drift) part of ḧ — no u dependence.

        ḧ = −(ω_dot_f·c_perp + ω·dc_perp_dt) / sin θ − ḣ²·cos θ / sin θ
        where ω_dot_f = I⁻¹(−ω×Iω)  and  dc_perp_dt = b × (−ω×a)

        Derivation: ḣ = −N/S where N=ω·c_perp, S=sinθ
          ḧ = −(Ṅ·S − N·Ṡ)/S² = −Ṅ/S + (N/S)·(cosθ·θ̇)/S
            = −Ṅ/S + (−ḣ)·cosθ·ḣ / S = −Ṅ/S − ḣ²·cosθ / sinθ
        """
        omega_dot_f = self._I_inv @ (-np.cross(omega, self._I @ omega))
        # dc_perp/dt = boresight × d(avoid_in_b)/dt = boresight × (−ω × avoid_in_b)
        dc_perp_dt = np.cross(_BORESIGHT_B, -np.cross(omega, av_b))

        if sin_t > _SIN_GUARD:
            drift = -(np.dot(omega_dot_f, c_perp) + np.dot(omega, dc_perp_dt)) / sin_t
            drift -= h_dot * math.cos(theta) * h_dot / sin_t
        else:
            drift = 0.0
        return drift

    # ------------------------------------------------------------------
    # CBF interface
    # ------------------------------------------------------------------

    def H(self, obs: np.ndarray) -> float:
        """Extended barrier value H(x) = h + |ḣ|·ḣ / (2·u_max)  [Eq. 6]."""
        th_marg, h_dot, _, _, _, _ = self._kinematics(obs)
        return th_marg + abs(h_dot) * h_dot / (2.0 * _U_MAX)

    def H_dot_expr(self, obs: np.ndarray, u_var: cp.Variable) -> cp.Expression:
        """CVXPY expression for Ḣ(x,u) + γ·H(x)  [Eq. 7 + class-K condition].

        Constraint in SafetyFilter QP:  H_dot_expr ≥ cbf_epsilon.
        """
        th_marg, h_dot, c_perp, sin_t, omega, av_b = self._kinematics(obs)

        H_val  = th_marg + abs(h_dot) * h_dot / (2.0 * _U_MAX)

        # Linear-in-u part of ḧ: −(I⁻¹·c_perp)·tau/sin θ,  tau = u·scale_torque
        if sin_t > _SIN_GUARD:
            g_h_dot_dot = -(self._I_inv @ c_perp) * _SCALE_TORQUE / sin_t
        else:
            g_h_dot_dot = np.zeros(3)

        # Drift part of ḧ
        theta = (obs[8] + 1.0) * (_PI / 2.0)
        f_h_dot_dot = self._h_dot_dot_f(omega, c_perp, sin_t, theta, h_dot, av_b)

        # Ḣ(x,u) = h_dot + |h_dot| * (f_hdd + g_hdd·u) / u_max
        # Ḣ + γ·H = const + linear_in_u
        abs_hd = abs(h_dot)
        const_part  = h_dot + abs_hd * f_h_dot_dot / _U_MAX + self.gamma * H_val
        linear_part = abs_hd / _U_MAX * g_h_dot_dot   # (3,) coefficient vector

        return const_part + linear_part @ u_var


# ---------------------------------------------------------------------------
# CLF
# ---------------------------------------------------------------------------

class SpaceAttitudeCLF(CLF):
    """Quadratic CLF for attitude stabilisation (paper Eq. 11-13).

    V(x) = c_q·‖q_e_vec‖² + c_w·‖omega‖²

    The state-dependent decay rate ζ(x) uses a sigmoid (Eq. 13) that ramps
    from ζ_min (far from goal) to ζ_max (near goal), encouraging aggressive
    convergence close to the equilibrium without fighting the CBF at distance.
    """

    def __init__(
        self,
        env,
        c_q: float = 1.0,      # weight on attitude error
        c_w: float = 0.1,      # weight on angular rate
        zeta_min: float = 0.001,
        zeta_max: float = 0.06,
        j: float = 5.0,        # sigmoid steepness (in 1/rad of attitude error)
        c: float = 0.6,        # sigmoid midpoint [rad] ≈ 34° attitude error
    ) -> None:
        self._env     = env
        self.c_q      = c_q
        self.c_w      = c_w
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.j        = j
        self.c        = c
        self._I     = env.inertia.astype(float)
        self._I_inv = np.linalg.inv(self._I)

    def _unpack(self, obs: np.ndarray):
        q_e, omega, _, _, _ = _denorm_obs(obs)
        q_e0    = float(q_e[0])
        q_e_vec = q_e[1:4].astype(float)
        return q_e0, q_e_vec, omega

    def _zeta(self, q_e0: float, q_e_vec: np.ndarray, omega: np.ndarray) -> float:
        """State-dependent decay rate ζ(x) from Eq. 13."""
        att_err = 2.0 * math.acos(float(np.clip(q_e0, -1.0, 1.0)))
        return self.zeta_min + (self.zeta_max - self.zeta_min) / (
            1.0 + math.exp(self.j * (att_err - self.c))
        )

    # ------------------------------------------------------------------

    def V(self, obs: np.ndarray) -> float:
        """Lyapunov value V = c_q·‖q_e_vec‖² + c_w·‖omega‖²."""
        q_e0, q_e_vec, omega = self._unpack(obs)
        return self.c_q * float(np.dot(q_e_vec, q_e_vec)) + self.c_w * float(np.dot(omega, omega))

    def V_dot_expr(self, obs: np.ndarray, u_var: cp.Variable) -> cp.Expression:
        """CVXPY expression for V(x,u) = LfV + LgV·u + ζ(x)·V(x)  [Eq. 12].

        Constraint in SafetyFilter QP:  V_dot_expr ≤ δ  (slack).
        """
        q_e0, q_e_vec, omega = self._unpack(obs)

        # Quaternion kinematics: dq_e_vec/dt = 0.5*(q_e0·ω + q_e_vec×ω)
        dqevec_dt = 0.5 * (q_e0 * omega + np.cross(q_e_vec, omega))

        # Drift: LfV = c_q·2·q_e_vec·dqevec_dt + c_w·2·ω·I⁻¹(−ω×Iω)
        omega_dot_f = self._I_inv @ (-np.cross(omega, self._I @ omega))
        Lf_V = (
            self.c_q * 2.0 * float(np.dot(q_e_vec, dqevec_dt))
            + self.c_w * 2.0 * float(np.dot(omega, omega_dot_f))
        )

        # Control gain: LgV·u = c_w·2·ω^T·I⁻¹·u·scale_torque
        g_clf = self.c_w * 2.0 * _SCALE_TORQUE * (self._I_inv.T @ omega)  # (3,)

        V_val = self.c_q * float(np.dot(q_e_vec, q_e_vec)) + self.c_w * float(np.dot(omega, omega))
        zeta  = self._zeta(q_e0, q_e_vec, omega)

        return Lf_V + g_clf @ u_var + zeta * V_val


# ---------------------------------------------------------------------------
# Torch-differentiable factories (training-loss side of the same certificates)
# ---------------------------------------------------------------------------
#
# These take collector-normalised batches: the PolicyTrainer stores states and
# actions normalised by (x_mu, x_std) / (a_mu, a_std), one step further than
# the env-normalised obs the QP filter sees, so every factory captures those
# statistics and denormalises on the fly.

def _cross_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched 3-vector cross product, (B,3) × (B,3) → (B,3)."""
    return torch.stack([
        a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1],
        a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2],
        a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0],
    ], dim=1)


def make_space_cbf_components(
    x_mu: torch.Tensor,
    x_std: torch.Tensor,
    inertia,               # (3,3) array-like
    gamma: float = _GAMMA,
) -> Callable:
    """Affine decomposition of the CBF condition for SatDynEnv (Eq. 5-7).

    Returns fn(state_norm) → (c0, b) with

        Ḣ(x,u) + γH(x) = c0(x) + b(x)·u_raw ,   u_raw ∈ [-1, 1]³.

    The affine-in-u structure is what makes the feasibility test possible
    (best case over the box = c0 + Σ|bᵢ|).

    State layout (collector-normalised, 13-dim):
        [0:4]  q_e (error quaternion, scalar-first)
        [4:7]  omega_norm  (physical: * 5 rad/s)
        [7]    theta_margin_norm  (physical: (v+1)*3π/4 - π/2)
        [8]    theta_norm         (physical: (v+1)*π/2)
        [9:12] rel_avoid_b  (unit direction: (avoid_b - boresight) / |…|)
    """
    I_np     = np.array(inertia, dtype=np.float64)
    I_t      = torch.tensor(I_np,                  dtype=torch.float32)
    I_inv_t  = torch.tensor(np.linalg.inv(I_np),   dtype=torch.float32)
    _bore    = torch.tensor(_BORESIGHT_B,          dtype=torch.float32)

    x_mu_f = x_mu.flatten()
    x_std_f = x_std.flatten()

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

        # Singularity guard (_SIN_GUARD): the ḧ terms divide by sin θ but their
        # numerators (e.g. ω·dc_perp_dt ~ |ω|²) do NOT vanish as θ → 0 or π, so
        # near the poles the quotient explodes as |ω|²/sin θ.  With the loss
        # squaring it, a single near-pole sample in a batch reached ~1e22 in run
        # baseline_17 and destroyed the policy weights.
        sin_t_raw = theta.sin().squeeze(1)                              # (B,)
        valid   = sin_t_raw > _SIN_GUARD                                # ~3° from poles
        sin_t   = sin_t_raw.clamp(min=_SIN_GUARD)                       # safe denominator
        cos_t   = theta.cos().squeeze(1)                                # (B,)
        bore_b  = bore.expand_as(av_b)

        # c_perp = boresight × avoid_in_b,  |c_perp| = sin(theta)
        c_perp = _cross_t(bore_b, av_b)                                 # (B, 3)

        # h_dot = -ω · c_perp / sin(theta)
        h_dot = -(omega * c_perp).sum(dim=1) / sin_t                   # (B,)
        h_dot = torch.where(valid, h_dot, torch.zeros_like(h_dot))
        h     = h.squeeze(1)                                            # (B,)

        # H(x) = h + |ḣ|·ḣ / (2·U_MAX)
        H_val = h + h_dot.abs() * h_dot / (2.0 * _U_MAX)              # (B,)

        # Drift: ω_dot_f = I⁻¹(−ω × Iω)
        Iw           = omega @ I.T                                      # (B, 3)
        omega_dot_f  = (-_cross_t(omega, Iw)) @ Ii.T                    # (B, 3)

        # dc_perp/dt = boresight × (−ω × avoid_in_b)
        dc_perp_dt = _cross_t(bore_b, -_cross_t(omega, av_b))           # (B, 3)

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
    gamma: float = _GAMMA,
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
    gamma: float = _GAMMA,
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
        margin_rad = (obs7 + 1.0) * (3.0 * _PI / 4.0) - _PI / 2.0
        return margin_rad * (180.0 / _PI)

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

            margin = (obs[:, 7] + 1.0) * (3.0 * _PI / 4.0) - _PI / 2.0
            theta  = (obs[:, 8] + 1.0) * (_PI / 2.0)
            half   = (theta - margin).clamp(math.radians(5.0), math.radians(45.0))

            m_new  = torch.rand(B, device=dev) * (hi - lo) + lo
            th_new = (half + m_new).clamp(0.06, _PI - 0.06)  # sin-θ guard
            m_new  = th_new - half

            obs = obs.clone()
            obs[:, 7] = -1.0 + (m_new + _PI / 2.0) * 4.0 / (3.0 * _PI)
            obs[:, 8] = -1.0 + th_new * 2.0 / _PI
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
    scale_omega: float = _SCALE_OMEGA,
) -> Callable:
    """Differentiable V_dot(x,u) + ζ(x)V(x) for SatDynEnv (paper Eq. 11-13).

    Returns the CLF condition value per sample:
        positive ⇒ stability constraint violated (loss penalises this squared)
        ≤ 0       ⇒ constraint satisfied

    The sigmoid decay rate ζ(x) (Eq. 13) is reproduced exactly.
    """
    I_np    = np.array(inertia, dtype=np.float64)
    I_t     = torch.tensor(I_np,                dtype=torch.float32)
    I_inv_t = torch.tensor(np.linalg.inv(I_np), dtype=torch.float32)

    x_mu_f = x_mu.flatten()
    x_std_f = x_std.flatten()
    a_mu_f = a_mu.flatten()
    a_std_f = a_std.flatten()

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
        dqv_dt = 0.5 * (q_e0.unsqueeze(1) * omega + _cross_t(q_e_vec, omega))    # (B, 3)

        # Euler drift: ω_dot_f = I⁻¹(−ω × Iω)
        Iw          = omega @ I.T                                        # (B, 3)
        omega_dot_f = (-_cross_t(omega, Iw)) @ Ii.T                     # (B, 3)

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

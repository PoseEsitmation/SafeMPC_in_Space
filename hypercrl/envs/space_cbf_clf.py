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
"""

from __future__ import annotations

import math
from typing import Optional

import cvxpy as cp
import numpy as np

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

    # gamma = 0.5 (paper value).  gamma=0.2 was evaluated in paper_final2:
    # it binds ~2.5x earlier (gentle corrections, perfect safety even for the
    # untrained policy) but floors the intervention RATE at ~2% — the
    # reliance-declines-to-zero result needs 0.5 (paper_final: 8.6%→0.00%).
    # KEEP IN SYNC with the torch factories in control/policy_net.py
    # (training loss must match the runtime filter).
    def __init__(self, env, gamma: float = 0.5) -> None:
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

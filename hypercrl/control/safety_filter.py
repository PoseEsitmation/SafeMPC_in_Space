from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


class CBF(ABC):
    """Abstract Control Barrier Function.

    Concrete subclasses live in the environment; the filter only calls
    H() and H_dot_expr() — it has zero knowledge of geometry.
    """

    @abstractmethod
    def H(self, state: np.ndarray) -> float:
        """Relative-degree-2 extended barrier value.

        Positive means the current state is in the safe set.
        """
        ...

    @abstractmethod
    def H_dot_expr(self, state: np.ndarray, u_var: cp.Variable) -> cp.Expression:
        """CVXPY expression for dH/dt, affine in u_var.

        The QP enforces ``H_dot_expr(state, u) >= cbf_epsilon`` as a hard
        constraint.  Must be affine in u_var so the problem stays a QP.
        """
        ...


class CLF(ABC):
    """Abstract Control Lyapunov Function.

    Concrete subclasses live in the environment.
    """

    @abstractmethod
    def V(self, state: np.ndarray) -> float:
        """Lyapunov value.  Zero only at the goal state."""
        ...

    @abstractmethod
    def V_dot_expr(self, state: np.ndarray, u_var: cp.Variable) -> cp.Expression:
        """CVXPY expression for dV/dt, affine in u_var.

        The QP enforces ``V_dot_expr(state, u) <= delta`` as a *soft*
        constraint (delta is a non-negative slack variable).
        """
        ...


class SafetyFilter:
    """QP safety filter: projects u_proposed onto the CBF-feasible set.

    Solves at every timestep::

        min  ||u - u_proposed||^2 + clf_rho * delta^2
        s.t. H_dot_expr(s, u) >= cbf_epsilon       (CBF — hard)
             V_dot_expr(s, u) <= delta              (CLF — soft, delta >= 0)
             -u_max <= u <= u_max                   (box constraint)

    The CLF constraint is omitted when ``clf=None``.
    On any solver failure ``u_proposed`` is returned unchanged and a warning
    is logged.

    Usage::

        sf = SafetyFilter(cbf=env.get_cbf(), clf=env.get_clf(),
                          u_max=1.0, control_dim=6)
        u_safe = sf.filter(state, u_proposed)
    """

    def __init__(
        self,
        cbf: CBF,
        clf: Optional[CLF] = None,
        u_max: float = 1.0,
        control_dim: int = 3,
        cbf_epsilon: float = 0.0,
        clf_rho: float = 1e3,
        cbf_rho: float = 1e6,
    ) -> None:
        self.cbf = cbf
        self.clf = clf
        self.u_max = u_max
        self.control_dim = control_dim
        self.cbf_epsilon = cbf_epsilon
        self.clf_rho = clf_rho
        self.cbf_rho = cbf_rho   # penalty for CBF slack — high = near-hard constraint

        # Populated after every filter() call; read by the monitor for TensorBoard.
        self.last_H: float = float("nan")
        self.last_V: float = float("nan")
        self.last_was_active: bool = False
        self.last_cbf_slack: float = 0.0
        self.last_du_norm: float = 0.0   # ‖u_safe − u_proposed‖ — correction magnitude

    def filter(self, state: np.ndarray, u_proposed: np.ndarray) -> np.ndarray:
        """Return the safe action closest (in L2) to u_proposed.

        Both CBF and CLF constraints use slack variables so the QP is always
        feasible.  CBF slack is penalised at ``cbf_rho`` (default 1e6, near-hard);
        CLF slack at ``clf_rho`` (default 1e3, soft — stability is secondary).

        Parameters
        ----------
        state:
            Current environment observation, shape ``(state_dim,)``.
        u_proposed:
            Action from the MPC optimizer, shape ``(control_dim,)``.

        Returns
        -------
        np.ndarray
            Safe action, shape ``(control_dim,)``.  Falls back to
            ``u_proposed`` if the QP cannot be solved even with slacks.
        """
        u_ref = u_proposed.flatten()[: self.control_dim]
        u         = cp.Variable(self.control_dim)
        delta_cbf = cp.Variable(nonneg=True)   # CBF slack — penalised heavily

        objective_terms = [cp.sum_squares(u - u_ref),
                           self.cbf_rho * cp.square(delta_cbf)]

        constraints: list = [
            u >= -self.u_max,
            u <= self.u_max,
            # CBF: soft with high penalty so QP is always feasible
            self.cbf.H_dot_expr(state, u) + delta_cbf >= self.cbf_epsilon,
        ]

        if self.clf is not None:
            delta_clf = cp.Variable(nonneg=True)
            constraints.append(self.clf.V_dot_expr(state, u) <= delta_clf)
            objective_terms.append(self.clf_rho * cp.square(delta_clf))

        self.last_H = float(self.cbf.H(state))
        self.last_V = float(self.clf.V(state)) if self.clf is not None else float("nan")

        prob = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
        try:
            # CLARABEL handles large coefficient ratios (cbf_rho=1e6 vs O(1) terms)
            # much better than OSQP's ADMM which hits user_limit on ill-conditioned problems.
            prob.solve(solver=cp.CLARABEL)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
                u_safe = u.value.astype(np.float64)
                self.last_du_norm     = float(np.linalg.norm(u_safe - u_ref))
                self.last_was_active  = self.last_du_norm > 1e-4
                self.last_cbf_slack   = float(delta_cbf.value) if delta_cbf.value is not None else 0.0
                if self.last_cbf_slack > 1e-4:
                    logger.debug("SafetyFilter CBF slack=%.4f (H=%.4f)", self.last_cbf_slack, self.last_H)
                return u_safe
        except Exception as exc:
            logger.warning("SafetyFilter QP exception: %s", exc)

        logger.warning(
            "SafetyFilter: solver status=%s — returning u_proposed unchanged",
            getattr(prob, "status", "unknown"),
        )
        self.last_was_active = False
        self.last_cbf_slack  = float("nan")
        return u_proposed.copy()

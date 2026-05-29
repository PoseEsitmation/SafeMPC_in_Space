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
    ) -> None:
        self.cbf = cbf
        self.clf = clf
        self.u_max = u_max
        self.control_dim = control_dim
        self.cbf_epsilon = cbf_epsilon
        self.clf_rho = clf_rho

    def filter(self, state: np.ndarray, u_proposed: np.ndarray) -> np.ndarray:
        """Return the safe action closest (in L2) to u_proposed.

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
            ``u_proposed`` if the QP cannot be solved.
        """
        u_ref = u_proposed.flatten()[: self.control_dim]
        u = cp.Variable(self.control_dim)

        constraints: list = [
            u >= -self.u_max,
            u <= self.u_max,
            self.cbf.H_dot_expr(state, u) >= self.cbf_epsilon,
        ]

        if self.clf is not None:
            delta = cp.Variable(nonneg=True)
            constraints.append(self.clf.V_dot_expr(state, u) <= delta)
            objective = cp.Minimize(
                cp.sum_squares(u - u_ref) + self.clf_rho * cp.square(delta)
            )
        else:
            objective = cp.Minimize(cp.sum_squares(u - u_ref))

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
                return u.value.astype(np.float64)
        except Exception as exc:
            logger.warning("SafetyFilter QP exception: %s", exc)

        logger.warning(
            "SafetyFilter: solver status=%s — returning u_proposed unchanged",
            getattr(prob, "status", "unknown"),
        )
        return u_proposed.copy()

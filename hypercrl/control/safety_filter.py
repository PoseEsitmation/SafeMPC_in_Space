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

    Primary QP (paper Eq. 19): hard CBF constraint, soft CLF.
    Fallback QP: soft CBF with large slack penalty — activated only when the
    hard CBF constraint has no feasible solution (drift exceeds control
    authority or state has crossed H=0).  Returns the least-unsafe action
    instead of abandoning correction.

    Usage::

        sf = SafetyFilter(cbf=env.get_cbf(), clf=env.get_clf(),
                          u_max=1.0, control_dim=3)
        u_safe = sf.filter(state, u_proposed)
    """

    # Penalty on CBF slack in the fallback QP.  Large enough to strongly
    # prefer hard feasibility; bounded so the fallback always has an optimum.
    _CBF_SLACK_RHO: float = 1e6

    # Activation-type codes for ``last_type`` (logged per step to TensorBoard).
    # Ordered by severity: anything >= TYPE_FALLBACK means the hard CBF
    # constraint could not be satisfied within the actuator box.
    # QP feasible, u_proposed already safe (no correction)
    TYPE_INACTIVE:  int = 0
    TYPE_CORRECTED: int = 1   # QP feasible, action projected onto the CBF-safe set
    TYPE_FALLBACK:  int = 2   # hard CBF infeasible -> soft-CBF least-unsafe action
    TYPE_FAILED:    int = 3   # both QPs errored -> u_proposed passed through unfiltered

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

        # Populated after every filter() call; read by the monitor for TensorBoard.
        self.last_H: float = float("nan")
        self.last_V: float = float("nan")
        self.last_was_active: bool = False
        self.last_cbf_slack: float = 0.0
        self.last_du_norm: float = 0.0   # ‖u_safe − u_proposed‖
        # Outcome of the last filter() call (TYPE_* code) — every consumer
        # (monitor episode counters, safety/filter_type scalar, eval fallback
        # counting) reads this; it MUST be assigned in every filter() branch.
        self.last_type: int = self.TYPE_INACTIVE

        # Consecutive steps using the soft-CBF fallback (0 = primary is feasible).
        self._fallback_count: int = 0

    def filter(self, state: np.ndarray, u_proposed: np.ndarray) -> np.ndarray:
        """Return the safe action closest (in L2) to u_proposed.

        Parameters
        ----------
        state:
            Current environment observation, shape ``(state_dim,)``.
        u_proposed:
            Proposed action, shape ``(control_dim,)``.

        Returns
        -------
        np.ndarray
            Safe action, shape ``(control_dim,)``.  Falls back to
            ``u_proposed`` only if both QP formulations fail with a solver error.
        """
        u_ref = u_proposed.flatten()[: self.control_dim]

        self.last_H = float(self.cbf.H(state))
        self.last_V = float(self.clf.V(
            state)) if self.clf is not None else float("nan")

        # ── Primary QP: hard CBF (paper Eq. 19) ─────────────────────────────
        # Constraint construction stays inside the try: H_dot_expr/V_dot_expr
        # evaluate the state (e.g. NaN checks, divisions) and an error there
        # must degrade to the fallback/pass-through path, not crash training.
        try:
            u = cp.Variable(self.control_dim)
            obj_terms: list = [cp.sum_squares(u - u_ref)]
            csts: list = [
                u >= -self.u_max,
                u <= self.u_max,
                self.cbf.H_dot_expr(
                    state, u) >= self.cbf_epsilon,   # hard (Eq. 19b)
            ]
            if self.clf is not None:
                delta_clf = cp.Variable(nonneg=True)
                csts.append(self.clf.V_dot_expr(state, u) <= delta_clf)
                obj_terms.append(self.clf_rho * cp.square(delta_clf))

            prob = cp.Problem(cp.Minimize(sum(obj_terms)), csts)
            # CLARABEL handles mixed constraint types better than OSQP's ADMM.
            prob.solve(solver=cp.CLARABEL)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
                if self._fallback_count > 0:
                    logger.warning(
                        "SafetyFilter: hard CBF feasible again after %d fallback step(s)",
                        self._fallback_count,
                    )
                    self._fallback_count = 0
                u_safe = u.value.astype(np.float64)
                self.last_du_norm = float(np.linalg.norm(u_safe - u_ref))
                self.last_was_active = self.last_du_norm > 1e-4
                self.last_cbf_slack = 0.0
                self.last_type = (self.TYPE_CORRECTED if self.last_was_active
                                  else self.TYPE_INACTIVE)
                return u_safe
        except Exception as exc:
            logger.warning("SafetyFilter primary QP exception: %s", exc)

        # ── Fallback QP: soft CBF ─────────────────────────────────────────────
        # Hard CBF was infeasible (no u in the box can satisfy Ḣ ≥ ε).
        # Return the least-unsafe action rather than passing u_proposed unchanged.
        self._fallback_count += 1
        if self._fallback_count == 1:
            logger.warning(
                "SafetyFilter: hard CBF infeasible (H=%.4f), switching to soft-CBF fallback",
                self.last_H,
            )
        elif self._fallback_count % 500 == 0:
            logger.warning(
                "SafetyFilter: still in soft-CBF fallback for %d steps (H=%.4f)",
                self._fallback_count, self.last_H,
            )

        try:
            u2 = cp.Variable(self.control_dim)
            delta_cbf2 = cp.Variable(nonneg=True)
            obj2: list = [cp.sum_squares(
                u2 - u_ref), self._CBF_SLACK_RHO * cp.square(delta_cbf2)]
            csts2: list = [
                u2 >= -self.u_max,
                u2 <= self.u_max,
                self.cbf.H_dot_expr(state, u2) + delta_cbf2 >= self.cbf_epsilon,
            ]
            if self.clf is not None:
                delta_clf2 = cp.Variable(nonneg=True)
                csts2.append(self.clf.V_dot_expr(state, u2) <= delta_clf2)
                obj2.append(self.clf_rho * cp.square(delta_clf2))

            prob2 = cp.Problem(cp.Minimize(sum(obj2)), csts2)
            prob2.solve(solver=cp.CLARABEL)
            if prob2.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u2.value is not None:
                u_safe = u2.value.astype(np.float64)
                self.last_du_norm = float(np.linalg.norm(u_safe - u_ref))
                self.last_was_active = self.last_du_norm > 1e-4
                self.last_cbf_slack = float(
                    delta_cbf2.value) if delta_cbf2.value is not None else 0.0
                self.last_type = self.TYPE_FALLBACK
                return u_safe
        except Exception as exc2:
            logger.warning("SafetyFilter fallback QP exception: %s", exc2)

        logger.warning(
            "SafetyFilter: all QP formulations failed — returning u_proposed unchanged")
        self.last_was_active = False
        self.last_cbf_slack = float("nan")
        self.last_type = self.TYPE_FAILED
        return u_proposed.copy()

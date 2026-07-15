"""SafetyFilter.last_type must reflect the QP outcome of every filter() call.

Regression test: baseline_33–35 ran with last_type never assigned, so the
"Filter Activation Breakdown" chart, safety/filter_type scalar and the eval
fallback counters all read 0 (inactive) regardless of what the filter did —
32 genuine interventions in baseline_35 task 0 were invisible.
"""

import numpy as np
import pytest

from hypercrl.control.safety_filter import SafetyFilter


class _StubCBF:
    """Affine CBF stub: condition = u[0] - need >= epsilon."""

    def __init__(self, need: float, raise_on_expr: bool = False):
        self.need = need
        self.raise_on_expr = raise_on_expr

    def H(self, state):
        return 1.0

    def H_dot_expr(self, state, u):
        if self.raise_on_expr:
            raise RuntimeError("boom")
        return u[0] - self.need


def _make(need, **kw):
    return SafetyFilter(cbf=_StubCBF(need, **kw), clf=None,
                        u_max=1.0, control_dim=3, cbf_epsilon=0.0)


STATE = np.zeros(13, dtype=np.float64)


class TestLastType:
    def test_inactive_when_action_already_safe(self):
        sf = _make(need=0.0)
        u = sf.filter(STATE, np.array([0.5, 0.0, 0.0]))
        assert sf.last_type == SafetyFilter.TYPE_INACTIVE
        assert not sf.last_was_active
        assert np.allclose(u, [0.5, 0.0, 0.0], atol=1e-5)

    def test_corrected_when_projection_needed(self):
        sf = _make(need=0.0)
        sf.filter(STATE, np.array([-0.5, 0.0, 0.0]))   # u[0] >= 0 forces change
        assert sf.last_type == SafetyFilter.TYPE_CORRECTED
        assert sf.last_was_active and sf.last_du_norm > 0.1

    def test_fallback_when_hard_cbf_infeasible(self):
        sf = _make(need=5.0)                            # u[0] >= 5 impossible in box
        u = sf.filter(STATE, np.array([0.0, 0.0, 0.0]))
        assert sf.last_type == SafetyFilter.TYPE_FALLBACK
        # least-unsafe action = max u[0]
        assert u[0] == pytest.approx(1.0, abs=1e-3)

    def test_failed_when_both_qps_error(self):
        sf = _make(need=0.0, raise_on_expr=True)
        u = sf.filter(STATE, np.array([0.3, 0.0, 0.0]))
        assert sf.last_type == SafetyFilter.TYPE_FAILED
        assert np.allclose(u, [0.3, 0.0, 0.0])

    def test_active_implies_typed(self):
        """The baseline_35 artifact: active step with type INACTIVE is impossible."""
        for need, u0 in [(0.0, -0.5), (5.0, 0.0)]:
            sf = _make(need=need)
            sf.filter(STATE, np.array([u0, 0.0, 0.0]))
            if sf.last_was_active:
                assert sf.last_type != SafetyFilter.TYPE_INACTIVE

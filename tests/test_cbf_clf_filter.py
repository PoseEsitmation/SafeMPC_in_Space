"""
Tests for the CBF/CLF QP safety filter (CBFCLFFilter) in space_KOZ.py.

Goal: construct actions that violate the CBF / CLF conditions on purpose,
run them through CBFCLFFilter.filter_action, and verify the corrected
action is actually safe.

Two layers of tests:

1. "Mocked gradient" tests patch out _dh_du / _dV_du with clean, known
   vectors. This isolates the QP projection itself (the part we actually
   care about for "is the filter correct") from the physical Jacobian
   derivation, whose real-world magnitudes are tiny (dt=0.1, inertia ~50,
   scale_torque=2 all shrink the gradients) and would make violation
   thresholds numerically fragile to hand-pick.

2. A "smoke test" runs the full pipeline with the real (non-mocked)
   _dh_du / _dV_du, using realistic state/KOZ geometry, just to confirm
   nothing crashes and the output stays finite and within actuator limits.
   This is the path actually exercised by SatDynEnv.step().
"""

import numpy as np
import pytest

from hypercrl.envs.space_KOZ import CBFCLFFilter
from hypercrl.envs.subfunctions_att_constraints import KeepOutZone

# Solver uses eps_abs = eps_rel = 1e-4 internally; give a little headroom
# when checking that a constraint is satisfied post-filter.
TOL = 1e-3

INERTIA = np.diag([50.0, 50.0, 50.0])
DT = 0.1
SCALE_TORQUE = 2.0


def make_filter(alpha_cbf=1.0, gamma_clf=0.5, beta_cbf=10.0, beta_clf=10.0):
    """Build a CBFCLFFilter with fixed, simple parameters for testing."""
    return CBFCLFFilter(
        inertia=INERTIA,
        dt=DT,
        scale_torque=SCALE_TORQUE,
        alpha_cbf=alpha_cbf,
        gamma_clf=gamma_clf,
        beta_cbf=beta_cbf,
        beta_clf=beta_clf,
    )


def make_fzone():
    """
    Fixed Keep-Out Zone geometry shared by tests:
      - boresight points along body +X
      - avoid direction sits 90 deg away in the inertial frame
      - half_angle = 20 deg, comfortably clear of the theta~0/pi
        singularity handled in _dh_du
    """
    boresight_b = np.array([1.0, 0.0, 0.0])
    avoid_i = np.array([0.0, 1.0, 0.0])
    half_angle = np.radians(20.0)
    return KeepOutZone(boresight_b, avoid_i, half_angle)


def make_state(theta_margin, theta, qe_deg=30.0, qe_axis=(0, 0, 1)):
    """
    Build a 13-element state vector matching the layout documented in
    space_KOZ.py:
        [qe(4), omega_e(3), theta_margin(1), theta(1), rel_avoid_in_b(3), qe_0_prev(1)]

    Only state[0:4] (qe), state[7] (theta_margin) and state[8] (theta) are
    actually read by the filter's _dh_du / _dV_du -- the remaining entries
    are physically-irrelevant placeholders for the rest of the layout.
    """
    axis = np.array(qe_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = np.radians(qe_deg) / 2.0
    qe = np.array([np.cos(half), *(axis * np.sin(half))])

    omega = np.zeros(3)
    rel_avoid_b = np.array([0.0, 1.0, 0.0])  # unused by the filter
    qe_0_prev = qe[0]

    return np.concatenate((qe, omega, [theta_margin], [theta], rel_avoid_b, [qe_0_prev]))



# 1a. CBF: an action constructed to violate the hard constraint must come
#     back out of the filter satisfying it.


def test_qp_filter_corrects_cbf_violation(monkeypatch):
    f = make_filter(alpha_cbf=1.0)
    fzone = make_fzone()

    dh_fixed = np.array([1.0, 0.0, 0.0])
    dV_fixed = np.zeros(3)  # isolate CBF: zero CLF gradient means CLF never binds

    monkeypatch.setattr(f, "_dh_du", lambda state, fzone: dh_fixed)
    monkeypatch.setattr(f, "_dV_du", lambda state: dV_fixed)

    h_val = 0.2  # state[7] -> theta_margin
    state = make_state(theta_margin=h_val, theta=fzone.half_angle + h_val)

    cbf_rhs = -f.alpha_cbf * h_val  # = -0.2

    # Deliberately unsafe: dh . u = -1.0, well below cbf_rhs = -0.2
    u_unsafe = np.array([-1.0, 0.3, -0.5])
    cbf_residual_before = dh_fixed @ u_unsafe - cbf_rhs
    assert cbf_residual_before < 0, "constructed action is not actually unsafe; check test setup"

    u_safe, cbf_viol, clf_viol = f.filter_action(u_unsafe, state, fzone)

    # The pre-filter violation must have been flagged (drives the reward penalty)
    assert cbf_viol > 0.0

    # The corrected action must satisfy the CBF constraint, within solver tolerance
    assert dh_fixed @ u_safe >= cbf_rhs - TOL

    # Actuator limits respected
    assert np.all(u_safe >= -1.0 - 1e-6)
    assert np.all(u_safe <= 1.0 + 1e-6)



# 1b. CBF: an action that's already safe shouldn't be needlessly distorted.


def test_qp_filter_leaves_already_safe_action_alone(monkeypatch):
    f = make_filter(alpha_cbf=1.0, gamma_clf=0.5)
    fzone = make_fzone()

    dh_fixed = np.array([1.0, 0.0, 0.0])
    dV_fixed = np.array([0.0, 0.0, 1.0])

    monkeypatch.setattr(f, "_dh_du", lambda state, fzone: dh_fixed)
    monkeypatch.setattr(f, "_dV_du", lambda state: dV_fixed)

    h_val = 0.2
    state = make_state(theta_margin=h_val, theta=fzone.half_angle + h_val, qe_deg=0.0)
    # qe_deg=0 -> qe_0=1 -> V_val=0 -> clf_rhs=0, so u=0 satisfies CLF too

    cbf_rhs = -f.alpha_cbf * h_val
    u_already_safe = np.zeros(3)
    assert dh_fixed @ u_already_safe >= cbf_rhs  # trivially safe by construction

    u_safe, cbf_viol, clf_viol = f.filter_action(u_already_safe, state, fzone)

    assert cbf_viol == 0.0
    assert clf_viol == 0.0
    np.testing.assert_allclose(u_safe, u_already_safe, atol=1e-3)



# 2. CLF: violations are soft (allowed via slack) but still flagged for the
#    reward penalty, and must not break the QP or violate actuator limits.


def test_qp_filter_allows_clf_violation_via_slack(monkeypatch):
    f = make_filter(alpha_cbf=1.0, gamma_clf=0.5, beta_clf=10.0)
    fzone = make_fzone()

    dh_fixed = np.zeros(3)  # CBF constraint trivially satisfied for any u
    dV_fixed = np.array([0.0, 0.0, 1.0])

    monkeypatch.setattr(f, "_dh_du", lambda state, fzone: dh_fixed)
    monkeypatch.setattr(f, "_dV_du", lambda state: dV_fixed)

    state = make_state(theta_margin=0.2, theta=fzone.half_angle + 0.2, qe_deg=60.0)
    V_val = 1.0 - state[0] ** 2  # qe_0 = cos(30deg) -> V_val = 0.25
    clf_rhs = f.gamma_clf * V_val  # = 0.125

    u_proposed = np.array([0.0, 0.0, 1.0])  # pushes straight along +dV
    clf_residual_before = dV_fixed @ u_proposed - clf_rhs
    assert clf_residual_before > 0, "constructed action does not actually violate CLF; check test setup"

    u_safe, cbf_viol, clf_viol = f.filter_action(u_proposed, state, fzone)

    assert clf_viol > 0.0  # flagged for the reward penalty
    assert cbf_viol == 0.0  # CBF never binds here (dh_fixed = 0)
    assert np.all(np.isfinite(u_safe))
    assert np.all(u_safe >= -1.0 - 1e-6)
    assert np.all(u_safe <= 1.0 + 1e-6)



# 3. Smoke test with the REAL (non-mocked) gradients, end-to-end.


def test_full_pipeline_smoke_with_real_gradients():
    """
    Exercises _dh_du / _dV_du as actually wired up via state + f_zone
    (the same path SatDynEnv.step() uses), to catch integration bugs that
    the mocked tests above can't see.
    """
    f = make_filter()
    fzone = make_fzone()

    theta_margin = np.radians(5.0)
    theta = fzone.half_angle + theta_margin
    state = make_state(theta_margin, theta, qe_deg=30.0)

    u_proposed = np.array([0.9, -0.9, 0.9])  # near actuator limits, arbitrary direction

    u_safe, cbf_viol, clf_viol = f.filter_action(u_proposed, state, fzone)

    assert np.all(np.isfinite(u_safe))
    assert np.all(u_safe >= -1.0 - 1e-6)
    assert np.all(u_safe <= 1.0 + 1e-6)
    assert cbf_viol >= 0.0
    assert clf_viol >= 0.0
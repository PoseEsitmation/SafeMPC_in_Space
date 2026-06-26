"""
Tests for the CBF/CLF QP safety filter (CBFCLFFilter) in space_KOZ.py.

All states/KeepOutZones here come from a REAL SatDynEnv (or
SatDynEnvValidation), not hand-invented numbers:
  - test 1/2 use the real post-reset() state.
  - test 3 rolls the real env forward with an adversarial control intent
    to find a real near/past-boundary state (the case that actually
    matters for safety), then tests the filter there.

KNOWN LIMITATION DISCOVERED WHILE WRITING THESE TESTS (see test 3 and the
comment in CBFCLFFilter.filter_action):
  When theta_margin is already deeply negative, dh (the linearized CBF
  gradient) is too small to satisfy `dh @ u >= cbf_rhs` for any u in
  [-1, 1]. The QP then reports infeasible (u_var.value is None), and
  filter_action's fallback returns the ORIGINAL unfiltered action --
  i.e. no safety correction is applied exactly when the satellite is
  already past the Keep-Out Zone boundary. test_filter_infeasible_case_*
  pins down this current behavior so a future fix is a deliberate,
  visible change rather than a silent one.
"""

import numpy as np
import pytest

from hypercrl.envs.space_KOZ import SatDynEnv, SatDynEnvValidation

TOL = 1e-3  # solver uses eps_abs = eps_rel = 1e-4; allow a little headroom


def get_real_initial_state(env_cls=SatDynEnv, seed=0, **env_kwargs):
    """
    Real env, real reset() state and KeepOutZone.

    NOTE: space_KOZ.py's sampling helpers (random_unit_quat_with_angle_bound,
    generate_avoid_vector_in_i_for_1Fzone_phase1_v2, etc.) call np.random
    directly rather than the seeded self.np_random gymnasium sets up, so
    reset(seed=...) alone does NOT make this reproducible. We seed the
    global RNG explicitly so these tests are deterministic.
    """
    np.random.seed(seed)
    env = env_cls(**env_kwargs)
    env.reset(seed=seed)
    return env, env.state.copy(), env.f_zone


def rollout_to_worst_margin(env, max_steps=None):
    """
    Step a real env forward using an adversarial *raw* control intent
    (push toward the KOZ every step, computed from the real dh at each
    real state), and return the state/f_zone at the point where
    theta_margin was smallest during the episode.

    This is how we find a genuinely unsafe state to test against,
    instead of inventing one.
    """
    f = env.cbf_clf_filter
    max_steps = max_steps or env.max_steps

    min_margin, min_state = None, None
    for _ in range(max_steps):
        dh = f._dh_du(env.state, env.f_zone)
        action = -np.sign(dh) if np.linalg.norm(dh) > 1e-8 else np.zeros(3)
        _, _, done, _, _ = env.step(action)
        h = float(env.state[7])
        if min_margin is None or h < min_margin:
            min_margin, min_state = h, env.state.copy()
        if done:
            break
    return min_margin, min_state, env.f_zone



# 1. Real reset() state, feasible region: construct the worst-case action
#    available within actuator limits and confirm it's still within the
#    filter's reach (it usually is near reset, since theta_margin starts
#    comfortably positive).


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_filter_safe_at_real_reset_state(seed):
    env, state, f_zone = get_real_initial_state(seed=seed)
    f = env.cbf_clf_filter

    h_val = float(state[7])
    dh = f._dh_du(state, f_zone)
    cbf_rhs = -f.alpha_cbf * h_val

    if np.linalg.norm(dh) < 1e-8:
        pytest.skip("degenerate gradient at this seed (theta near 0/pi)")

    u_worst_case = -np.sign(dh)  # most adversarial action available in [-1, 1]
    u_safe, cbf_viol, clf_viol = f.filter_action(u_worst_case, state, f_zone)

    assert dh @ u_safe >= cbf_rhs - TOL
    assert np.all(u_safe >= -1.0 - 1e-6)
    assert np.all(u_safe <= 1.0 + 1e-6)


def test_env_step_runs_with_adversarial_action_from_real_reset():
    """step() (the real training code path) must run end-to-end without errors."""
    env, state, f_zone = get_real_initial_state(seed=0)
    obs, reward, done, truncated, info = env.step(np.array([1.0, -1.0, 1.0], dtype=np.float32))
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)
    assert obs.shape == (13,)


def test_filter_safe_at_real_reset_state_on_validation_env():
    env, state, f_zone = get_real_initial_state(env_cls=SatDynEnvValidation, seed=0)
    f = env.cbf_clf_filter

    h_val = float(state[7])
    dh = f._dh_du(state, f_zone)
    if np.linalg.norm(dh) < 1e-8:
        pytest.skip("degenerate gradient at this seed (theta near 0/pi)")

    cbf_rhs = -f.alpha_cbf * h_val
    u_worst_case = -np.sign(dh)
    u_safe, cbf_viol, clf_viol = f.filter_action(u_worst_case, state, f_zone)

    assert dh @ u_safe >= cbf_rhs - TOL
    assert np.all(u_safe >= -1.0 - 1e-6)
    assert np.all(u_safe <= 1.0 + 1e-6)



# 2. Real rollout, deep-violation region: documents the actual current
#    behavior of filter_action when the CBF constraint is infeasible
#    within actuator limits.


def test_filter_infeasible_case_falls_back_to_unfiltered_action():
    """
    KNOWN LIMITATION (see module docstring). At a real, deeply-past-boundary
    state reached via rollout, dh is too small relative to cbf_rhs for the
    QP to be feasible within u in [-1, 1]. filter_action currently falls
    back to returning the ORIGINAL action -- i.e. it does NOT correct an
    unsafe action in this regime.

    NOTE ON RANDOMNESS: random_unit_quat_with_angle_bound (and friends) are
    @njit-compiled, so they draw from Numba's own internal RNG -- NOT the
    plain np.random.seed() set below. That means a single fixed seed isn't
    guaranteed to reproduce a boundary violation across machines/numba
    versions. To stay robust, we try several seeds and use whichever one
    actually produces a violation; only skip if NONE of them do.
    """
    min_margin, state, f_zone, env = None, None, None, None
    for seed in range(20):
        candidate_env = SatDynEnv()
        np.random.seed(seed)
        candidate_env.reset(seed=seed)
        candidate_margin, candidate_state, candidate_fzone = rollout_to_worst_margin(candidate_env)
        if candidate_margin < 0:
            min_margin, state, f_zone, env = candidate_margin, candidate_state, candidate_fzone, candidate_env
            break

    if min_margin is None:
        pytest.skip("no seed in range(20) produced a real boundary violation; widen the seed search")

    f = env.cbf_clf_filter
    dh = f._dh_du(state, f_zone)
    h_val = float(state[7])
    cbf_rhs = -f.alpha_cbf * h_val

    u_bad = np.sign(dh)  # actively pushes further into violation
    u_safe, cbf_viol, clf_viol = f.filter_action(u_bad, state, f_zone)

    # Current behavior: unfiltered action passed straight through, still unsafe.
    np.testing.assert_allclose(u_safe, u_bad.astype(np.float32), atol=1e-6)
    assert dh @ u_safe < cbf_rhs  # still unsafe -- this IS the bug, not a false negative
    assert cbf_viol > 0.0  # at least correctly flagged for the reward penalty
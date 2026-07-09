import numpy as np
import pytest

from hypercrl.envs.space_KOZ import SatDynEnv


@pytest.fixture
def env():
    e = SatDynEnv()
    yield e
    e.close()


def _snapshot(env):
    """Save what step() mutates: the state vector and step counter."""
    return env.state.copy(), env.steps


def _restore(env, snap):
    """Undo a trial step by restoring the saved state and step counter."""
    env.state, env.steps = snap[0].copy(), snap[1]


def test_filter_improves_koz_margin(env):
    state, _ = env.reset()
    unsafe_action = env.action_space.sample()

    # Search for an action whose next step would enter the KOZ.
    while True:
        snap = _snapshot(env)
        obs, _, _, done, info = env.step(unsafe_action)
        if info["keep_out_violation"]:
            _restore(env, snap)   # dangerous action found; rewind to the state before it
            break
        if done:
            pytest.skip("episode ended before any action reached the KOZ")
        state = obs               # keep the advanced state; obs is the normalised observation
        unsafe_action = env.action_space.sample()

    # Margin after the RAW action (theta_margin = state[7]; >0 safe, <0 inside KOZ).
    snap = _snapshot(env)
    env.step(unsafe_action)
    margin_unfiltered = env.state[7]
    _restore(env, snap)           # rewind so the filtered action starts from the same state

    # Margin after the FILTERED action, applied from the identical state.
    sf = env.get_safety_filter()
    safe_action = sf.filter(state, unsafe_action)
    env.step(safe_action)
    margin_filtered = env.state[7]

    print(f"\nunsafe action   = {unsafe_action}")
    print(f"safe action     = {safe_action}")
    print(f"margin unfilter = {margin_unfiltered:.5f}")
    print(f"margin filtered = {margin_filtered:.5f}")

    # The filter must return one value per control axis.
    assert safe_action.shape == (sf.control_dim,)
    # The filtered action must respect the torque box constraint.
    assert np.all(np.abs(safe_action) <= sf.u_max + 1e-6)
    # Core guarantee: the filtered action leaves the boresight no closer to the KOZ.
    assert margin_filtered >= margin_unfiltered - 1e-6, (
        "safety filter made the KOZ margin worse than the unfiltered action"
    )
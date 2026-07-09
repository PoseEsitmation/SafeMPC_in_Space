"""
Safety filter test for the SatDynEnv keep-out-zone (KOZ) environment.

This test mirrors the real control flow used by SafeAgent.act() in agent.py:
    u_proposed = mpc.act(state)                            # controller proposes an action
    u_safe     = safety_filter.filter(state, u_proposed)  # filter corrects it
so we validate the safety filter exactly the way the agent uses it.

Design note (why we check "no worse", not "no violation"):
In safety_filter.py the CBF constraint is SOFT -- it is enforced with a slack
variable (delta_cbf) penalised by cbf_rho, not as a hard constraint. The filter
therefore promises a MINIMAL safe correction, not a guaranteed zero-violation
result. With bounded torque (2 Nm/axis) a single 0.1 s step also cannot always
pull the boresight out once it is right on the boundary -- a physical limit, not
a filter failure. So we assert the filtered action leaves the KOZ margin NO
WORSE than the raw action, which is the guarantee the filter actually makes.

Run with: python -m pytest -s tests/test_cbf_clf_filter.py -v
"""

import numpy as np
import pytest

from hypercrl.envs.space_KOZ import SatDynEnv


@pytest.fixture
def env(tmp_path, monkeypatch):
    # Run each test in a throwaway temp dir so render/frame files don't pollute the repo.
    monkeypatch.chdir(tmp_path)
    # Build a fresh environment for the test.
    e = SatDynEnv()
    # Hand the environment to the test.
    yield e
    # Tear down (close plotter / free resources) after the test finishes.
    e.close()


def _snapshot(env):
    # Save the only things step() mutates: the 13-dim state and the step counter.
    # f_zone (the KOZ) is fixed for the whole episode, so it needs no saving.
    return env.state.copy(), env.steps


def _restore(env, snap):
    # Put the saved state and step counter back, undoing a trial step().
    env.state, env.steps = snap[0].copy(), snap[1]


def test_filter_improves_koz_margin(env):
    # Start a new episode; `state` is the normalised observation SafeAgent feeds to filter().
    state, _ = env.reset()
    # Draw a random candidate command, standing in for the controller's proposed action.
    unsafe_action = env.action_space.sample()

    # Search forward until a single step with this action would enter the KOZ.
    while True:
        # Remember the current state so a trial step can be undone.
        snap = _snapshot(env)
        # Trial-step the real env with the candidate action and read its violation flag.
        _, _, _, done, info = env.step(unsafe_action)
        # Dangerous action found -> undo the trial step and stop searching.
        if info["keep_out_violation"]:
            _restore(env, snap)
            break
        # Not dangerous: keep this advanced state, but stop if the episode ended.
        if done:
            pytest.skip("episode ended before any action reached the KOZ")
        # `state` now holds the current normalised observation for the next candidate.
        state = env._normalise()
        # Draw a new candidate and try again.
        unsafe_action = env.action_space.sample()

    # --- Outcome of the RAW (unfiltered) action ---
    snap = _snapshot(env)
    # Apply the raw dangerous action.
    env.step(unsafe_action)
    # theta_margin (state index 7): >0 safe, <0 inside the KOZ. This is the raw baseline.
    margin_unfiltered = env.state[7]
    # Undo it so the filtered action starts from the same state.
    _restore(env, snap)

    # --- Outcome of the FILTERED action ---
    # Build the safety filter, the same object agent.py uses (env-owned CBF+CLF QP filter).
    sf = env.get_safety_filter()
    # Correct the dangerous action; pass the normalised obs, exactly like SafeAgent.act().
    safe_action = sf.filter(state, unsafe_action)
    # Apply the filtered action from the identical starting state.
    env.step(safe_action)
    # Margin after the filtered action -- what we compare against the raw baseline.
    margin_filtered = env.state[7]

    # Diagnostics (visible with -s): the two actions, the two margins, and filter internals.
    print(f"\nunsafe action   = {unsafe_action}")
    print(f"safe action     = {safe_action}")
    print(f"margin unfilter = {margin_unfiltered:.5f}")
    print(f"margin filtered = {margin_filtered:.5f}")
    print(f"filter correction |du| = {sf.last_du_norm:.5f}")
    print(f"barrier value H = {sf.last_H:.5f}")

    # we have to check that margin filtered >= margin unfilter 
    # Contract check: filter() must return one action per control dimension.
    assert safe_action.shape == (sf.control_dim,)
    # The filtered action must stay inside the torque box [-u_max, u_max].
    assert np.all(np.abs(safe_action) <= sf.u_max + 1e-6)
    # Core guarantee: the filtered action leaves the boresight NO CLOSER to the KOZ than the raw action.
    assert margin_filtered >= margin_unfiltered - 1e-6, (
        "safety filter made the KOZ margin worse than the unfiltered action"
    )
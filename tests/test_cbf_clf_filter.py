"""
Run with: python -m pytest -s tests/test_cbf_clf_filter.py -v
"""

import copy

import numpy as np
import pytest
from numba import njit

from hypercrl.envs.space_KOZ import SatDynEnv


@njit
def _seed_numba(seed):
    np.random.seed(seed)


@pytest.fixture
def envs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # SatDynEnv has no use_safety_filter flag: the filter is obtained via
    # env.get_safety_filter() and applied to actions manually ("filter ON").
    env_off = SatDynEnv()
    env_on = SatDynEnv()

    yield env_off, env_on

    env_off.close()
    env_on.close()


def _reset_same(env, seed):
    np.random.seed(seed)
    _seed_numba(seed)
    return env.reset()


def test_same_start_state(envs):
    env_off, env_on = envs

    obs_a, _ = _reset_same(env_off, seed=42)
    obs_b, _ = _reset_same(env_on, seed=42)

    print(f"\ntheta_margin OFF={env_off.state[7]:.4f}  ON={env_on.state[7]:.4f}")
    assert np.allclose(obs_a, obs_b)


def test_filter_prevents_violation_at_boundary(envs):
    env, _ = envs
    _reset_same(env, seed=42)
    env.action_space.seed(42)

    unsafe_action = env.action_space.sample()

    found = False
    for _ in range(env.max_steps):
        env_cp = copy.deepcopy(env)
        _, _, _, _, info = env_cp.step(unsafe_action)
        if info["keep_out_violation"]:
            found = True
            break

        env.step(unsafe_action)
        unsafe_action = env.action_space.sample()

    if not found:
        pytest.skip("no violating action found in one episode")

    print(f"\nunsafe state: theta_margin={env.state[7]:.4f}")
    print(f"unsafe action = {unsafe_action}")

    sf = env.get_safety_filter()
    safe_action = sf.filter(env._normalise(), np.asarray(unsafe_action, dtype=np.float64))

    print(f"safe action  = {safe_action}")
    print(f"H = {sf.last_H:.4f}  filter_active = {sf.last_was_active}  "
          f"|du| = {sf.last_du_norm:.4f}")

    # filtered action must respect the box constraint
    assert np.all(np.abs(safe_action) <= sf.u_max + 1e-6)

    _, _, _, _, info = env.step(safe_action)
    print(f"after filtered action -> violation = {info['keep_out_violation']}")

    assert not info["keep_out_violation"], "safety filter failed to prevent KOZ violation"


def test_filter_effect_on_koz_violation(envs):
    """Find an action that would cause a KOZ violation, then confirm the
    safety filter's corrected action avoids it. Same start, filter OFF vs ON."""
    env_off, env_on = envs
    _reset_same(env_off, seed=42)
    _reset_same(env_on, seed=42)

    env_off.action_space.seed(42)
    unsafe_action = env_off.action_space.sample()

    # Walk forward until a single step would land inside the KOZ.
    found = False
    for _ in range(env_off.max_steps):
        env_CP = copy.deepcopy(env_off)
        _, _, _, _, info = env_CP.step(unsafe_action)
        if info["keep_out_violation"]:
            found = True
            break

        env_off.step(unsafe_action)
        env_on.step(unsafe_action)
        unsafe_action = env_off.action_space.sample()

    if not found:
        pytest.skip("no violating action found in one episode")

    print(f"\ntheta_margin OFF={env_off.state[7]:.4f}  ON={env_on.state[7]:.4f}")
    print(f"unsafe action = {unsafe_action}")

    # Filter the unsafe action on the ON env (CBF expects normalised obs).
    sf = env_on.get_safety_filter()
    safe_action = sf.filter(env_on._normalise(), np.asarray(unsafe_action, dtype=np.float64))

    print(f"safe action   = {safe_action}")
    print(f"H = {sf.last_H:.4f}  active = {sf.last_was_active}  |du| = {sf.last_du_norm:.4f}")

    # OFF env takes the unsafe action, ON env takes the filtered one.
    _, _, _, _, info_off = env_off.step(unsafe_action)
    _, _, _, _, info_on = env_on.step(safe_action)

    print(f"filter OFF -> violation={info_off['keep_out_violation']}  "
          f"theta_margin={env_off.state[7]:.4f}")
    print(f"filter ON  -> violation={info_on['keep_out_violation']}  "
          f"theta_margin={env_on.state[7]:.4f}")

    assert np.all(np.abs(safe_action) <= sf.u_max + 1e-6)
    assert not info_on["keep_out_violation"], "safety filter failed to prevent KOZ violation"


def test_filter_action_directly(envs):
    env_off, _ = envs
    _reset_same(env_off, seed=42)

    env_off.action_space.seed(42)
    action = env_off.action_space.sample()

    sf = env_off.get_safety_filter()
    u_safe = sf.filter(env_off._normalise(), np.asarray(action, dtype=np.float64))

    print(f"\ntheta_margin = {env_off.state[7]:.4f}")
    print(f"proposed = {action}")
    print(f"filtered = {u_safe}")
    print(f"H = {sf.last_H:.4f}  cbf_slack = {sf.last_cbf_slack:.4f}  "
          f"active = {sf.last_was_active}")

    assert u_safe.shape == (sf.control_dim,)
    assert np.all(np.isfinite(u_safe))
    assert np.all(np.abs(u_safe) <= sf.u_max + 1e-6)
    # state is safe at reset (boresight starts outside the KOZ) -> H must be positive
    assert sf.last_H > 0.0


def test_filter_effect_over_full_episode(envs):
    env_off, env_on = envs
    _reset_same(env_off, seed=42)
    _reset_same(env_on, seed=42)

    sf = env_on.get_safety_filter()

    env_off.action_space.seed(42)
    action = env_off.action_space.sample().astype(np.float64)

    violations_off = 0
    violations_on = 0

    for _ in range(env_off.max_steps):
        _, _, _, _, info_off = env_off.step(action)

        # CBF/CLF expect the NORMALISED observation (see space_cbf_clf docstring),
        # not the raw physical state.
        safe_action = sf.filter(env_on._normalise(), action)
        _, _, _, _, info_on = env_on.step(safe_action)

        violations_off += int(info_off["keep_out_violation"])
        violations_on += int(info_on["keep_out_violation"])

    print(f"\nsteps run = {env_off.max_steps}")
    print(f"filter OFF -> KOZ violated in {violations_off} steps")
    print(f"filter ON  -> KOZ violated in {violations_on} steps")

    if violations_off == 0:
        pytest.skip("action never reached the KOZ, try another seed")

    assert violations_on <= violations_off
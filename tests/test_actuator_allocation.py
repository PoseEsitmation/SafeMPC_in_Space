"""tau = B @ u: nominal B must reproduce the old scale_torque=2 behaviour, and a
faulted B must change the plant and the certificates together."""
import cvxpy as cp
import numpy as np
import pytest
import torch

from hypercrl.envs.space_KOZ import SatDynEnv
from hypercrl.envs.space_cbf_clf import (_SCALE_TORQUE, _U_MAX, _B_NOMINAL,
                                         make_space_cbf_fn, make_space_clf_fn)
from hypercrl.envs.space_tasks import get_task_spec


def _observations(env, n=24):
    obs, (o, _) = [], env.reset(seed=11)
    for i in range(n):
        obs.append(np.asarray(o, dtype=np.float64))
        o, _, done, _, _ = env.step(env.action_space.sample())
        if done:
            o, _ = env.reset(seed=12 + i)
    return obs


def _control_coeffs(expr_fn, obs):
    """Linear-in-u coefficients of an affine cvxpy expression."""
    u = cp.Variable(3)
    expr = expr_fn(obs, u)
    u.value = np.zeros(3)
    c0 = float(expr.value)
    out = []
    for e in np.eye(3):
        u.value = e
        out.append(float(expr.value) - c0)
    return np.array(out)


def test_nominal_cbf_matches_legacy():
    env = SatDynEnv()
    cbf, I_inv = env.get_cbf(), np.linalg.inv(env.inertia.astype(float))
    for obs in _observations(env):
        _, h_dot, c_perp, sin_t, _, _ = cbf._kinematics(obs)
        if sin_t <= 0.05:
            continue
        legacy = abs(h_dot) / _U_MAX * (-(I_inv @ c_perp) * _SCALE_TORQUE / sin_t)
        assert np.allclose(_control_coeffs(cbf.H_dot_expr, obs), legacy, atol=1e-9)


def test_nominal_clf_matches_legacy():
    env = SatDynEnv()
    clf, I_inv = env.get_clf(), np.linalg.inv(env.inertia.astype(float))
    for obs in _observations(env):
        legacy = clf.c_w * 2.0 * _SCALE_TORQUE * (I_inv.T @ (obs[4:7] * 5.0))
        assert np.allclose(_control_coeffs(clf.V_dot_expr, obs), legacy, atol=1e-9)


def test_nominal_env_actuator():
    env = SatDynEnv()
    assert np.allclose(env.B, _B_NOMINAL) and env.u_max_torque == pytest.approx(_U_MAX)


def test_torch_closures_take_the_actuator():
    env = SatDynEnv()
    x = torch.tensor(np.stack(_observations(env)), dtype=torch.float32)
    u = torch.empty(x.shape[0], 3).uniform_(-1.0, 1.0)
    ident = (torch.zeros(1, 13), torch.ones(1, 13), torch.zeros(1, 3), torch.ones(1, 3))
    faulted = get_task_spec("spaceEnv_thruster", 2).B()
    for make in (make_space_cbf_fn, make_space_clf_fn):
        default = make(*ident, env.inertia)(x, u)
        assert torch.allclose(default, make(*ident, env.inertia, B=_B_NOMINAL)(x, u), atol=1e-6)
        assert not torch.allclose(default, make(*ident, env.inertia, B=faulted)(x, u))


@pytest.mark.parametrize("task_id", [1, 2, 3, 4])
def test_fault_changes_plant_and_cbf(task_id):
    nominal = SatDynEnv()
    faulted = SatDynEnv(**get_task_spec("spaceEnv_thruster", task_id).env_kwargs())
    nominal.reset(seed=7)
    faulted.state, faulted.f_zone = nominal.state.copy(), nominal.f_zone

    u = np.ones(3, dtype=np.float32)
    o_n, *_ = nominal.step(u)
    o_f, *_ = faulted.step(u)
    assert not np.allclose(o_n[4:7], o_f[4:7], atol=1e-9)

    obs = np.asarray(o_n, dtype=np.float64)
    c_n = _control_coeffs(nominal.get_cbf().H_dot_expr, obs)
    c_f = _control_coeffs(faulted.get_cbf().H_dot_expr, obs)
    if np.linalg.norm(c_n) > 1e-9:
        assert not np.allclose(c_n, c_f, atol=1e-9)


def test_filter_stays_in_box_under_fault():
    env = SatDynEnv(**get_task_spec("spaceEnv_thruster", 3).env_kwargs())
    sf = env.get_safety_filter()
    obs, _ = env.reset(seed=6)
    for _ in range(20):
        u = sf.filter(np.asarray(obs, dtype=np.float64), env.action_space.sample().astype(np.float64))
        assert np.all(np.abs(u) <= sf.u_max + 1e-6)
        obs, *_ = env.step(u.astype(np.float32))

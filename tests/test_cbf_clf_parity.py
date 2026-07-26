"""Numeric parity between the runtime (numpy/cvxpy) and training (torch)
CBF/CLF implementations for SatDynEnv.

space_cbf_clf.py (SpaceAttitudeCBF/CLF) backs the deployed QP safety filter;
policy_net.py's make_space_cbf_fn/make_space_clf_fn back the differentiable
training loss. Both independently reimplement the same rigid-body math — a
torch autograd graph can't share code with a cvxpy expression — and are kept
in sync only by "KEEP IN SYNC" comments in both files. This test compares
their outputs on identical (state, action) pairs so drift between the two
fails loudly instead of silently desyncing the training signal from the
deployed filter.

Two state distributions are used:

* ``rollout`` — real env transitions. Realistic, but SatDynEnv resets with a
  near-zero angular rate, so |h_dot| stays ~1e-2 and the relative-degree-2
  extension |h_dot|*h_dot/(2*u_max) contributes only ~1e-5. These states
  barely exercise the h_ddot drift/control terms.
* ``boundary`` — synthetic states with substantial angular rates near the KOZ
  edge, built to be exactly self-consistent with the observation layout. These
  drive |h_dot| high enough that the h_ddot terms (inertia coupling,
  dc_perp/dt, the sin(theta) divisions) dominate the condition value, which is
  where an edit to one implementation is most likely to silently diverge.

``test_boundary_states_exercise_rate_terms`` guards the second distribution:
if a future change makes those states tame again, the parity checks would
degrade to validating only the trivial ``h + gamma*h`` part, and that test
fails rather than letting the suite quietly lose its teeth.
"""

import math

import cvxpy as cp
import numpy as np
import pytest
import torch

from hypercrl.control.policy_net import make_space_cbf_fn, make_space_clf_fn
from hypercrl.envs.space_KOZ import SatDynEnv

GAMMA = 0.5

# Both implementations zero the rate terms below sin(theta) = 0.05; stay well
# clear so the comparison exercises the shared non-degenerate branch.
SIN_THETA_CLEARANCE = 0.15

# Observed agreement is ~1e-9, so these are loose by three orders of magnitude
# while still far tighter than the CLF condition's own scale (~1e-3) — an
# implementation returning zeros must not be able to pass.
TOL = dict(rel=1e-4, abs=1e-7)

_U_MAX = 2.0 * math.sqrt(3.0)


# ---------------------------------------------------------------------------
# Fixtures / state generation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    e = SatDynEnv()
    yield e
    e.close()


def _rollout_states(env, n=40, seed=0):
    """Real (obs, action) pairs from env dynamics, clear of the pole guard."""
    rng = np.random.default_rng(seed)
    states, actions = [], []
    obs, _ = env.reset()
    for _ in range(5000):
        if len(states) >= n:
            break
        theta = (obs[8] + 1.0) * (np.pi / 2.0)
        u = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        if np.sin(theta) > SIN_THETA_CLEARANCE:
            states.append(obs.copy())
            actions.append(u.copy())
        obs, _, terminated, truncated, _ = env.step(u)
        if terminated or truncated:
            obs, _ = env.reset()
    assert len(states) == n, f"only collected {len(states)}/{n} clear-of-pole states"
    return np.stack(states), np.stack(actions)


def _boundary_states(n=40, seed=1):
    """Synthetic high-angular-rate states near the KOZ boundary.

    Built directly in the observation layout so theta and rel_avoid_b stay
    exactly consistent: for a unit avoid-vector at angle theta from the
    boresight (+X) with azimuth phi,

        avoid_in_b  = [cos(theta), sin(theta)cos(phi), sin(theta)sin(phi)]
        rel_avoid_b = (avoid_in_b - boresight) / |avoid_in_b - boresight|

    which is precisely what SatDynEnv stores and what both implementations
    invert. Angular rates run up to 0.5 rad/s (vs ~0.03 in a fresh rollout) so
    the |h_dot|-weighted h_ddot terms actually carry the condition value.
    """
    rng = np.random.default_rng(seed)
    bore = np.array([1.0, 0.0, 0.0])

    obs = np.zeros((n, 13), dtype=np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)

    # Random unit quaternions (scalar-first), real attitude errors.
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0

    theta = rng.uniform(math.radians(12.0), math.radians(60.0), size=n)
    half_angle = rng.uniform(math.radians(10.0), math.radians(30.0), size=n)
    margin = theta - half_angle           # spans both sides of the boundary
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    omega = rng.uniform(-0.5, 0.5, size=(n, 3))   # rad/s — the point of this set

    avoid_b = np.stack([
        np.cos(theta),
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
    ], axis=1)
    rel = avoid_b - bore
    rel /= np.linalg.norm(rel, axis=1, keepdims=True)

    obs[:, 0:4] = q
    obs[:, 4:7] = omega / 5.0                                    # scale_omega
    obs[:, 7] = -1.0 + (margin + np.pi / 2) * 4.0 / (3.0 * np.pi)
    obs[:, 8] = -1.0 + theta * 2.0 / np.pi
    obs[:, 9:12] = rel
    obs[:, 12] = q[:, 0]
    return obs, actions


@pytest.fixture(scope="module")
def state_sets(env):
    return {
        "rollout": _rollout_states(env),
        "boundary": _boundary_states(),
    }


# ---------------------------------------------------------------------------
# Runtime (numpy/cvxpy) reference values
# ---------------------------------------------------------------------------

def _numpy_cbf_condition(env, obs, u):
    """H_dot(x,u) + gamma*H(x), evaluated through the same cvxpy expression the
    QP filter builds as its hard constraint."""
    u_var = cp.Variable(3)
    expr = env.get_cbf(gamma=GAMMA).H_dot_expr(obs, u_var)
    u_var.value = u
    return float(expr.value)


def _numpy_clf_condition(env, obs, u):
    """V_dot(x,u) + zeta(x)*V(x), via the QP filter's soft-constraint expression."""
    u_var = cp.Variable(3)
    expr = env.get_clf().V_dot_expr(obs, u_var)
    u_var.value = u
    return float(expr.value)


def _torch_conditions(env, states, actions, maker, **kw):
    x_mu, x_std = torch.zeros(1, 13), torch.ones(1, 13)
    a_mu, a_std = torch.zeros(1, 3), torch.ones(1, 3)
    fn = maker(x_mu, x_std, a_mu, a_std, env.inertia, **kw)
    return fn(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    ).detach().numpy()


def _assert_parity(env, states, actions, torch_vals, numpy_fn, label):
    for i in range(len(states)):
        np_val = numpy_fn(env, states[i], actions[i])
        assert torch_vals[i] == pytest.approx(np_val, **TOL), (
            f"{label} mismatch at sample {i}: "
            f"numpy/cvxpy={np_val:.9f} torch={torch_vals[i]:.9f} "
            f"(diff={abs(torch_vals[i] - np_val):.3e})"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist", ["rollout", "boundary"])
class TestCbfParity:
    def test_condition_matches_runtime_filter(self, env, state_sets, dist):
        states, actions = state_sets[dist]
        torch_vals = _torch_conditions(
            env, states, actions, make_space_cbf_fn, gamma=GAMMA)
        _assert_parity(env, states, actions, torch_vals,
                       _numpy_cbf_condition, "CBF condition (H_dot + gamma*H)")

    def test_gamma_drift_is_detected(self, env, state_sets, dist):
        """The parity check must fail on real drift — e.g. the stale gamma=0.2
        the torch docstring used to claim against the runtime's 0.5."""
        states, actions = state_sets[dist]
        torch_vals = _torch_conditions(
            env, states, actions, make_space_cbf_fn, gamma=0.2)
        mismatches = [
            i for i in range(len(states))
            if torch_vals[i] != pytest.approx(
                _numpy_cbf_condition(env, states[i], actions[i]), **TOL)
        ]
        assert mismatches, (
            "a gamma mismatch (0.2 vs runtime 0.5) went undetected — "
            "the parity tolerance is too loose to catch real drift"
        )


@pytest.mark.parametrize("dist", ["rollout", "boundary"])
class TestClfParity:
    def test_condition_matches_runtime_filter(self, env, state_sets, dist):
        states, actions = state_sets[dist]
        torch_vals = _torch_conditions(
            env, states, actions, make_space_clf_fn)
        _assert_parity(env, states, actions, torch_vals,
                       _numpy_clf_condition, "CLF condition (V_dot + zeta*V)")

    def test_tolerance_rejects_a_null_implementation(self, env, state_sets, dist):
        """CLF condition values are small (~1e-3); guard that TOL is tight
        enough that returning zeros cannot pass as agreement."""
        states, actions = state_sets[dist]
        zeros = np.zeros(len(states))
        mismatches = [
            i for i in range(len(states))
            if zeros[i] != pytest.approx(
                _numpy_clf_condition(env, states[i], actions[i]), **TOL)
        ]
        assert len(mismatches) == len(states), (
            "an all-zeros CLF would pass the parity tolerance on "
            f"{len(states) - len(mismatches)}/{len(states)} samples"
        )


def test_boundary_states_exercise_rate_terms(env, state_sets):
    """The boundary set must actually stress the relative-degree-2 machinery.

    Without this, a change that tamed those states would silently reduce both
    parity tests to checking the trivial h + gamma*h path, since the h_ddot
    drift and control terms are all weighted by |h_dot|/u_max.
    """
    states, _ = state_sets["boundary"]
    cbf = env.get_cbf(gamma=GAMMA)
    h_dots = np.array([abs(cbf._kinematics(s)[1]) for s in states])

    # Rate extension |h_dot|*h_dot/(2*u_max) vs the barrier value it corrects.
    extension = h_dots.max() ** 2 / (2.0 * _U_MAX)
    assert h_dots.max() > 0.3, (
        f"boundary states have max |h_dot|={h_dots.max():.4f} — too tame to "
        "exercise the h_ddot terms"
    )
    assert extension > 1e-2, (
        f"relative-degree-2 extension term is only {extension:.2e}; the "
        "parity tests would be validating the trivial path only"
    )

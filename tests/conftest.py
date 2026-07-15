"""Shared fixtures and lightweight mocks for SafeStuff unit tests."""

import types
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset


# ---------------------------------------------------------------------------
# Shared hyperparameter fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def hparams():
    return types.SimpleNamespace(
        device="cpu",
        policy_lr=1e-4,
        policy_bs=32,
        policy_train_iters=2,       # keep tests fast
        policy_lambda_imit=1.0,
        policy_lambda_cbf=0.0,
        policy_lambda_clf=0.0,
        dagger_n_iter=5,
        normalize_xu=True,
        state_dim=13,
        control_dim=3,
        env="spaceEnv",
        # Fields required by Agent base class
        model="hnet",
        dnn_out="diff",
        reward_discount=0.99,
    )


# ---------------------------------------------------------------------------
# Mock DataCollector
# ---------------------------------------------------------------------------

class MockCollector:
    """Minimal stand-in for DataCollector.

    Tracks every call to add() so tests can inspect what was stored.
    norm() returns fixed tensors so _make_policy_train_set can normalise.
    get_dataset() returns a small TensorDataset with the recorded transitions.
    finalize() updates the cached norms (no-op here since they're fixed).
    """

    STATE_DIM  = 13
    ACTION_DIM = 3

    def __init__(self, task_id: int = 0, n_existing: int = 64):
        self.task_id = task_id
        self._calls: list[tuple] = []          # list of (x, u, x_next) numpy arrays

        # Populate with fake pre-existing expert transitions
        rng = np.random.default_rng(0)
        self._base_states  = rng.standard_normal((n_existing, self.STATE_DIM)).astype(np.float32)
        self._base_actions = rng.standard_normal((n_existing, self.ACTION_DIM)).astype(np.float32) * 0.62
        self._base_nexts   = rng.standard_normal((n_existing, self.STATE_DIM)).astype(np.float32)

        # Fixed action norms (a_std ≈ 0.62 as in real spaceEnv runs)
        self._a_mu  = torch.zeros(self.ACTION_DIM)
        self._a_std = torch.full((self.ACTION_DIM,), 0.62)
        self._x_mu  = torch.zeros(self.STATE_DIM)
        self._x_std = torch.ones(self.STATE_DIM)

    # ------------------------------------------------------------------

    def add(self, x, u, x_next, task_id):
        self._calls.append((
            np.asarray(x).flatten().copy(),
            np.asarray(u).flatten().copy(),
            np.asarray(x_next).flatten().copy(),
        ))

    def norm(self, task_id):
        return self._x_mu, self._x_std, self._a_mu, self._a_std

    def finalize(self, task_id):
        return self._x_mu, self._x_std, self._a_mu, self._a_std

    def get_dataset(self, task_id, ds_range=None, skip_first_n=0):
        """Return a TensorDataset of (state_norm, action_norm, delta_state_norm)."""
        x   = torch.tensor(self._base_states[skip_first_n:])
        u   = torch.tensor(self._base_actions[skip_first_n:]) / self._a_std   # normalise
        dx  = torch.zeros_like(x)
        return TensorDataset(x, u, dx), None

    # helper
    @property
    def add_calls(self):
        return self._calls


@pytest.fixture
def mock_collector():
    return MockCollector()


# ---------------------------------------------------------------------------
# Mock MPC / SafeAgent
# ---------------------------------------------------------------------------

class MockMPCAgent:
    """Returns a fixed expert action for every state."""

    def __init__(self, action_dim: int = 3, u_val: float = 0.8):
        self._u = torch.full((action_dim,), u_val)

    def act(self, obs, task_id=None, **kw):
        return self._u.clone()

    def reset(self):
        pass

    def cache_hnet(self, *a, **kw):
        pass


@pytest.fixture
def mock_mpc_agent():
    return MockMPCAgent()


# ---------------------------------------------------------------------------
# Mock Gym environment
# ---------------------------------------------------------------------------

class MockActionSpace:
    low   = np.full((3,), -1.0, dtype=np.float32)
    high  = np.full((3,),  1.0, dtype=np.float32)
    shape = (3,)


class MockEnv:
    action_space = MockActionSpace()
    _step_count  = 0
    _max_steps   = 10   # short episodes to keep tests fast

    def reset(self):
        self._step_count = 0
        return np.zeros(13, dtype=np.float32), {}

    def step(self, u):
        self._step_count += 1
        obs_next  = np.random.default_rng().standard_normal(13).astype(np.float32)
        terminated = self._step_count >= self._max_steps
        return obs_next, 0.0, terminated, False, {}


@pytest.fixture
def mock_env():
    return MockEnv()

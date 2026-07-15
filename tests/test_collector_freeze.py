"""Tests for DataCollector frozen-normalisation mode.

freeze_norms() must fix the affine transform for the rest of the run:
statistics stop updating, actions switch to the identity transform, later
tasks inherit the same stats, and load_frozen_norms() reproduces a previous
run's coordinate system exactly.
"""

import types

import numpy as np
import pytest
import torch

from hypercrl.dataset.datautil import DataCollector


STATE_DIM = 13
ACTION_DIM = 3


def make_collector(**overrides):
    hp = types.SimpleNamespace(
        dnn_out="diff",
        normalize_xu=True,
        normalize_diff=True,
        env="spaceEnv",
    )
    for k, v in overrides.items():
        setattr(hp, k, v)
    return DataCollector(hp)


def fill(collector, task_id, n, scale=1.0, seed=0):
    """Add n random transitions for task_id, states scaled by `scale`."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = (rng.standard_normal(STATE_DIM) * scale).astype(np.float32)
        u = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        x_next = x + (rng.standard_normal(STATE_DIM) * 0.01 * scale).astype(np.float32)
        collector.add(x, u, x_next, task_id)


class TestFreezeNorms:
    def test_stats_stop_updating_after_freeze(self):
        col = make_collector()
        fill(col, 0, 200, scale=1.0, seed=0)
        x_mu, x_std, _, _ = col.freeze_norms(0)

        # Wildly different data afterwards must not move the stats.
        fill(col, 0, 200, scale=100.0, seed=1)
        col.get_dataset(0)   # triggers finalize()
        x_mu2, x_std2, _, _ = col.norm(0)

        assert torch.equal(x_mu, x_mu2)
        assert torch.equal(x_std, x_std2)

    def test_actions_use_identity_transform(self):
        col = make_collector()
        fill(col, 0, 200)
        _, _, a_mu, a_std = col.freeze_norms(0)

        assert torch.all(a_mu == 0.0)
        assert torch.all(a_std == 1.0)

    def test_actions_pass_through_unchanged(self):
        col = make_collector()
        # Constant action so we can recognise it after the train/val split.
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = rng.standard_normal(STATE_DIM).astype(np.float32)
            col.add(x, np.full(ACTION_DIM, 0.5, dtype=np.float32), x, 0)
        col.freeze_norms(0)

        train_set, _ = col.get_dataset(0)
        _, actions, _ = train_set.tensors
        assert torch.allclose(actions, torch.full_like(actions, 0.5))

    def test_new_task_inherits_frozen_norms(self):
        col = make_collector()
        fill(col, 0, 200, scale=1.0, seed=0)
        frozen = col.freeze_norms(0)

        # Task 1 data has a different scale; stats must not be recomputed.
        fill(col, 1, 200, scale=10.0, seed=1)
        col.freeze_norms(1)
        col.get_dataset(1)

        for a, b in zip(frozen, col.norm(1)):
            assert torch.equal(a, b)

    def test_diff_norms_frozen_and_shared(self):
        col = make_collector()
        fill(col, 0, 200, scale=1.0, seed=0)
        col.freeze_norms(0)
        dx_mu0, dx_std0 = col.norm_diff(0)

        fill(col, 1, 200, scale=10.0, seed=1)
        col.freeze_norms(1)
        col.get_dataset(1)
        dx_mu1, dx_std1 = col.norm_diff(1)

        assert torch.equal(dx_mu0, dx_mu1)
        assert torch.equal(dx_std0, dx_std1)

    def test_freeze_is_idempotent(self):
        col = make_collector()
        fill(col, 0, 200)
        first = col.freeze_norms(0)
        fill(col, 0, 200, scale=50.0, seed=2)
        second = col.freeze_norms(0)

        for a, b in zip(first, second):
            assert torch.equal(a, b)


class TestLoadFrozenNorms:
    def _synthetic_norms(self):
        x_mu = torch.zeros(1, STATE_DIM)
        x_std = torch.full((1, STATE_DIM), 2.0)
        a_mu = torch.zeros(1, ACTION_DIM)
        a_std = torch.ones(1, ACTION_DIM)
        dx_mu = torch.zeros(1, STATE_DIM)
        dx_std = torch.full((1, STATE_DIM), 0.25)
        return (x_mu, x_std, a_mu, a_std), (dx_mu, dx_std)

    def test_loaded_norms_used_from_first_transition(self):
        norms, diff_norms = self._synthetic_norms()
        col = make_collector()
        col.load_frozen_norms(norms, diff_norms)

        fill(col, 0, 100, scale=7.0, seed=3)
        col.get_dataset(0)

        for a, b in zip(norms, col.norm(0)):
            assert torch.equal(a, b)
        for a, b in zip(diff_norms, col.norm_diff(0)):
            assert torch.equal(a, b)

    def test_save_load_roundtrip(self, tmp_path):
        """Mirror the norms.pt payload written by hnet_exp.run()."""
        col = make_collector()
        fill(col, 0, 200)
        col.freeze_norms(0)

        path = tmp_path / "norms.pt"
        torch.save({"norms": col._frozen_norms,
                    "diff_norms": col._frozen_diff_norms}, path)

        payload = torch.load(path, weights_only=True)
        col2 = make_collector()
        col2.load_frozen_norms(payload["norms"], payload.get("diff_norms"))

        fill(col2, 0, 100, scale=42.0, seed=4)
        col2.get_dataset(0)

        for a, b in zip(col.norm(0), col2.norm(0)):
            assert torch.equal(a, b)
        for a, b in zip(col.norm_diff(0), col2.norm_diff(0)):
            assert torch.equal(a, b)

    def test_states_normalised_with_loaded_stats(self):
        norms, diff_norms = self._synthetic_norms()
        col = make_collector()
        col.load_frozen_norms(norms, diff_norms)

        rng = np.random.default_rng(5)
        raw_states = []
        for _ in range(100):
            x = rng.standard_normal(STATE_DIM).astype(np.float32)
            raw_states.append(x)
            col.add(x, rng.uniform(-1, 1, ACTION_DIM).astype(np.float32), x, 0)

        train_set, _ = col.get_dataset(0)
        states, _, _ = train_set.tensors
        raw = torch.tensor(np.stack(raw_states))[col.train_inds[0]]
        # x_mu = 0, x_std = 2 → normalised states are exactly raw / 2
        assert torch.allclose(states, raw / 2.0, atol=1e-6)

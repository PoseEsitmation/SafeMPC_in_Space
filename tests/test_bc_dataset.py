"""Tests for excluding the random-exploration phase from the policy BC dataset.

The policy's imitation targets must be expert (MPC) actions only.  The first
init_rand_steps transitions per task carry RandomAgent noise as "labels"
(baseline_27: 27% of the BC base, physical torques up to 4x the actuator box,
u_target_norm_max pinned at 3.98) — get_dataset(skip_first_n=N) slices them
off while the dynamics model keeps the full set.
"""

import types

import numpy as np
import pytest
import torch

from hypercrl.dataset.datautil import DataCollector
from hypercrl.control.policy_net import PolicyNet, PolicyTrainer

STATE_DIM = 13
ACTION_DIM = 3
RANDOM_MARK = 9.9   # sentinel action value for random-phase rows
EXPERT_MARK = 0.5   # sentinel action value for MPC-phase rows


def make_collector():
    hp = types.SimpleNamespace(
        dnn_out="diff",
        normalize_xu=True,
        normalize_diff=True,
        env="spaceEnv",
    )
    return DataCollector(hp)


def fill_marked(collector, n_random=50, n_expert=150, task_id=0, seed=0):
    """n_random rows with RANDOM_MARK actions, then n_expert with EXPERT_MARK."""
    rng = np.random.default_rng(seed)
    for i in range(n_random + n_expert):
        x = rng.standard_normal(STATE_DIM).astype(np.float32)
        mark = RANDOM_MARK if i < n_random else EXPERT_MARK
        u = np.full(ACTION_DIM, mark, dtype=np.float32)
        collector.add(x, u, x, task_id)
    return n_random


class TestSkipFirstN:
    def test_random_rows_excluded_from_bc_set(self):
        col = make_collector()
        n_random = fill_marked(col)
        col.freeze_norms(0)   # identity actions → dataset actions stay physical

        bc_set, bc_val = col.get_dataset(0, skip_first_n=n_random)
        for ds in (bc_set, bc_val):
            if len(ds) == 0:
                continue
            actions = ds.tensors[1]
            assert torch.allclose(actions, torch.full_like(actions, EXPERT_MARK)), (
                "BC dataset contains random-phase actions — the policy would "
                "imitate RandomAgent noise."
            )

    def test_dynamics_set_keeps_all_rows(self):
        col = make_collector()
        n_random = fill_marked(col)
        col.freeze_norms(0)

        full_set, full_val = col.get_dataset(0)
        bc_set, bc_val = col.get_dataset(0, skip_first_n=n_random)
        assert len(full_set) + len(full_val) == 200
        assert len(bc_set) + len(bc_val) == 150

    def test_skip_zero_is_default_behaviour(self):
        col = make_collector()
        fill_marked(col)
        col.freeze_norms(0)

        a, av = col.get_dataset(0)
        b, bv = col.get_dataset(0, skip_first_n=0)
        assert len(a) == len(b) and len(av) == len(bv)

    def test_skip_is_per_task(self):
        """Each task's random phase is its own first N transitions."""
        col = make_collector()
        fill_marked(col, task_id=0)
        col.freeze_norms(0)
        fill_marked(col, task_id=1, seed=1)
        col.freeze_norms(1)

        bc_set, bc_val = col.get_dataset(1, skip_first_n=50)
        for ds in (bc_set, bc_val):
            if len(ds) == 0:
                continue
            actions = ds.tensors[1]
            assert torch.allclose(actions, torch.full_like(actions, EXPERT_MARK))


class TestEmptyBCDataset:
    def test_train_skips_gracefully_below_batch_size(self, hparams):
        """Right after policy_train_start the BC set can be smaller than one
        batch (random phase excluded, few expert rows yet) — train() must
        skip, not raise StopIteration."""
        policy = PolicyNet(STATE_DIM, ACTION_DIM, hidden_dims=(16,), dropout=0.0)
        trainer = PolicyTrainer(policy, hparams)

        from torch.utils.data import TensorDataset
        tiny = TensorDataset(torch.zeros(3, STATE_DIM), torch.zeros(3, ACTION_DIM),
                             torch.zeros(3, STATE_DIM))
        loss = trainer.train(tiny, writer=None)
        assert loss == 0.0

        # 0 rows would crash inside the DataLoader constructor itself
        # (RandomSampler rejects num_samples=0) — must be guarded before it.
        empty = TensorDataset(torch.zeros(0, STATE_DIM), torch.zeros(0, ACTION_DIM),
                              torch.zeros(0, STATE_DIM))
        loss = trainer.train(empty, writer=None)
        assert loss == 0.0

from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from hypercrl.control.policy_net import PolicyNet, PolicyTrainer

STATE_DIM, ACT_DIM = 6, 3


def _trainer(n_tasks, **hp):
    hparams = SimpleNamespace(device="cpu", policy_train_iters=2, policy_bs=8, **hp)
    return PolicyTrainer(PolicyNet(STATE_DIM, ACT_DIM, n_tasks=n_tasks), hparams)


def _base(n):
    return TensorDataset(torch.randn(n, STATE_DIM), torch.randn(n, ACT_DIM),
                         torch.randn(n, STATE_DIM))


def test_no_replay_trains_past_task_0():
    # cl_s4: the no-replay arms crashed at task 1 with an IndexError in _onehot.
    t = _trainer(n_tasks=0)
    ds, _ = t._make_policy_train_set(_base(64), collector=None, task_id=1)
    assert ds.tensors[2].shape == (64, 1)
    t.train(ds, task_id=1)


def test_replay_is_balanced_across_tasks():
    t = _trainer(n_tasks=3)
    t._replay = {0: (torch.randn(20, STATE_DIM), torch.randn(20, ACT_DIM)),
                 1: (torch.randn(20, STATE_DIM), torch.randn(20, ACT_DIM))}
    ds, w = t._make_policy_train_set(_base(1000), collector=None, task_id=2)
    task = ds.tensors[2].argmax(dim=1)
    for j in range(3):
        assert torch.isclose(w[task == j].sum() / w.sum(), torch.tensor(1 / 3))


def test_replay_frac_override():
    t = _trainer(n_tasks=2, policy_replay_frac=0.2)
    t._replay = {0: (torch.randn(20, STATE_DIM), torch.randn(20, ACT_DIM))}
    ds, w = t._make_policy_train_set(_base(1000), collector=None, task_id=1)
    old = ds.tensors[2].argmax(dim=1) == 0
    assert torch.isclose(w[old].sum() / w.sum(), torch.tensor(0.2))


class _Ctrl:
    def __init__(self):
        self.mean = torch.full((4,), 7.0)

    def reset(self):
        self.mean = torch.zeros(4)


class _Agent:
    """Records the planner's warm start seen by every act() call."""

    def __init__(self):
        self.control = _Ctrl()
        self.seen = []

    def reset(self):
        self.control.reset()

    def act(self, state, task_id=None):
        self.seen.append(self.control.mean.clone())
        self.control.mean = torch.full((4,), float(state[0]))   # plan depends on state
        return torch.zeros(ACT_DIM)

    def cache_hnet(self, task_id):
        pass

    def set_safety_filter(self, f):
        pass


class _Collector:
    def get_dataset(self, j, skip_first_n=0):
        return _base(10), None

    def norm(self, j):
        return (torch.zeros(STATE_DIM), torch.ones(STATE_DIM),
                torch.zeros(ACT_DIM), torch.ones(ACT_DIM))


def test_refresh_resets_planner_and_restores_current_plan():
    t = _trainer(n_tasks=2, policy_replay_n=5)
    agent = _Agent()
    t.refresh_replay(agent, _Collector(), task_id=1, env_for_task=lambda j: object())
    assert len(agent.seen) == 10                               # labels + noise re-solve
    assert all(torch.all(m == 0) for m in agent.seen)          # cold start each state
    assert torch.all(agent.control.mean == 7.0)                # current plan restored


class _StoredCollector(_Collector):
    def get_dataset(self, j, skip_first_n=0):
        self.skip = skip_first_n
        x = torch.arange(30.0).repeat(STATE_DIM, 1).T
        u = torch.arange(30.0).repeat(ACT_DIM, 1).T
        return TensorDataset(x, u, x), None


def test_stored_labels_skip_planner_and_random_phase():
    t = _trainer(n_tasks=2, policy_replay_n=8, policy_replay_labels="stored",
                 init_rand_steps=5)
    agent, coll = _Agent(), _StoredCollector()
    t.refresh_replay(agent, coll, task_id=1, env_for_task=lambda j: object())
    x, u = t._replay[0]
    assert coll.skip == 5                      # random phase excluded
    assert agent.seen == []                    # no planner calls
    assert torch.equal(x[:, 0], u[:, 0])       # labels stay paired with their states


def test_expert_labels_report_gap_to_stored():
    t = _trainer(n_tasks=2, policy_replay_n=5)
    t.refresh_replay(_Agent(), _Collector(), task_id=1, env_for_task=lambda j: object())
    assert {"gap_stored", "cos_stored", "gap_self"} <= set(t.replay_stats[0])

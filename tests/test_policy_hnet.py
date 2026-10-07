from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from hypercrl.control.policy_hnet import HnetPolicy, HnetPolicyTrainer

STATE_DIM, ACT_DIM = 6, 3


def _trainer(reg_share=0.8, iters=60):
    torch.manual_seed(0)
    policy = HnetPolicy(STATE_DIM, ACT_DIM, n_tasks=2, hidden_dims=(16, 16), hnet_arch=(8, 8))
    hp = SimpleNamespace(device="cpu", policy_train_iters=iters, policy_bs=16,
                         policy_lr=1e-2, policy_reg_share=reg_share)
    return HnetPolicyTrainer(policy, hp)


def _task_data(n, sign):
    x = torch.randn(n, STATE_DIM)
    return TensorDataset(x, sign * x[:, :ACT_DIM], x)   # task 1 wants the opposite action


def _drift_after_task_1(reg_share):
    t = _trainer(reg_share)
    t.begin_task(0)
    ds, w = t._make_policy_train_set(_task_data(256, +1.0), None, 0)
    t.train(ds, sample_weights=w, task_id=0)
    t.begin_task(1)
    target = [W.clone() for W in t._targets[0]]
    ds, w = t._make_policy_train_set(_task_data(256, -1.0), None, 1)
    t.train(ds, sample_weights=w, task_id=1)
    now = t.policy.hnet.forward(task_id=0)
    return sum(((a - b) ** 2).sum().item() for a, b in zip(now, target))


def test_forward_routes_rows_to_their_task():
    t = _trainer()
    t.begin_task(0)
    t.begin_task(1)
    with torch.no_grad():                       # copied embedding: make the tasks differ
        t.policy.hnet.get_task_emb(1).add_(1.0)
    x = torch.randn(5, STATE_DIM)
    assert not torch.allclose(t.policy(x, task_id=0), t.policy(x, task_id=1))
    onehot = torch.eye(2)[torch.tensor([0, 1, 0, 1, 1])]
    mixed = t.policy(x, task_id=onehot)
    assert torch.allclose(mixed[[0, 2]], t.policy(x[[0, 2]], task_id=0), atol=1e-6)
    assert torch.allclose(mixed[[1, 3, 4]], t.policy(x[[1, 3, 4]], task_id=1), atol=1e-6)


def test_new_task_starts_from_previous_embedding_and_old_ones_are_not_optimised():
    t = _trainer()
    t.begin_task(0)
    t.begin_task(1)
    hnet = t.policy.hnet
    assert torch.equal(hnet.get_task_emb(1), hnet.get_task_emb(0))
    optimised = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
    assert id(hnet.get_task_emb(1)) in optimised
    assert id(hnet.get_task_emb(0)) not in optimised


def test_regulariser_protects_old_task_weights():
    protected = _drift_after_task_1(0.8)
    unprotected = _drift_after_task_1(None)
    assert protected < 0.5 * unprotected


def test_plain_trainer_is_untouched_by_hnet_module():
    # Importing the hypernetwork student must not change PolicyNet/PolicyTrainer.
    from hypercrl.control.policy_net import PolicyNet, PolicyTrainer
    hp = SimpleNamespace(device="cpu", policy_train_iters=2, policy_bs=8)
    t = PolicyTrainer(PolicyNet(STATE_DIM, ACT_DIM), hp)
    assert isinstance(t.optimizer, torch.optim.AdamW)
    assert not hasattr(t, "begin_task")

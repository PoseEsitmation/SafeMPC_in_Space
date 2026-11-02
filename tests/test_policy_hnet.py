from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from hypercrl.control.policy_hnet import HnetPolicy, HnetPolicyTrainer, _FixedBeta
from hypercrl.tools.reg_share import RegShareBeta

STATE_DIM, ACT_DIM = 6, 3


def _trainer(reg_share=0.8, iters=60, mode=None, beta=0.05):
    torch.manual_seed(0)
    policy = HnetPolicy(STATE_DIM, ACT_DIM, n_tasks=2, hidden_dims=(16, 16), hnet_arch=(8, 8))
    hp = SimpleNamespace(device="cpu", policy_train_iters=iters, policy_bs=16,
                         policy_lr=1e-2, policy_reg_share=reg_share,
                         policy_reg_mode=mode, policy_hnet_beta=beta)
    return HnetPolicyTrainer(policy, hp)


def _task_data(n, sign):
    x = torch.randn(n, STATE_DIM)
    return TensorDataset(x, sign * x[:, :ACT_DIM], x)   # task 1 wants the opposite action


def _drift_after_task_1(reg_share, mode=None, beta=0.05):
    t = _trainer(reg_share, mode=mode, beta=beta)
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
    unprotected = _drift_after_task_1(None)
    for mode in ("fixed", "share"):
        protected = _drift_after_task_1(0.8, mode=mode)
        assert protected < 0.5 * unprotected, f"{mode} mode did not protect task 0"


def _fit_after_task_1(mode, beta, iters=400):
    """Train task 0, then the sign-flipped task 1; return (task1 error, task0 error)."""
    t = _trainer(iters=iters, mode=mode, beta=beta)
    t.begin_task(0)
    ds, w = t._make_policy_train_set(_task_data(256, +1.0), None, 0)
    t.train(ds, sample_weights=w, task_id=0)
    t.begin_task(1)
    ds, w = t._make_policy_train_set(_task_data(256, -1.0), None, 1)
    t.train(ds, sample_weights=w, task_id=1)
    x = torch.randn(512, STATE_DIM)
    with torch.no_grad():
        return ((t.policy(x, task_id=1) + x[:, :ACT_DIM]).pow(2).mean().item(),
                (t.policy(x, task_id=0) - x[:, :ACT_DIM]).pow(2).mean().item())


def test_fixed_beta_learns_a_conflicting_task_while_keeping_the_old_one():
    # cl_c9 in miniature.  Beta there was pinned at the beta_max clamp, which keeps
    # task 0 intact and leaves the sign-flipped task unlearnable; a sane fixed beta
    # has to do both.  Guards against anyone restoring a stiff default.
    learn, keep = _fit_after_task_1("fixed", 0.05)
    assert learn < 0.1, f"the flipped task was not learned (err {learn:.3f})"
    assert keep < 0.1, f"task 0 was not kept (err {keep:.3f})"
    stiff_learn, stiff_keep = _fit_after_task_1("fixed", 100.0)
    assert stiff_keep < 0.1                       # still remembers
    assert stiff_learn > 10 * learn               # but can no longer adapt


def test_fixed_is_the_default_mode_and_beta_is_constant():
    t = _trainer(beta=0.3)
    assert t.reg_mode == "fixed"
    t.begin_task(0)
    assert t.reg_ctrl is None                       # nothing to protect on task 0
    t.begin_task(1)
    assert isinstance(t.reg_ctrl, _FixedBeta)
    assert t.reg_ctrl.beta == 0.3
    assert t.reg_ctrl.update(1, 123.0, 1e-9) == 0.3   # no adaptation, whatever the loss


def test_share_mode_must_be_asked_for():
    t = _trainer(mode="share")
    t.begin_task(0)
    t.begin_task(1)
    assert isinstance(t.reg_ctrl, RegShareBeta)


def test_reg_share_zero_still_means_no_protection():
    # The phnet_unprot arm passes --policy-reg-share 0; it must win over any mode.
    for mode in (None, "fixed", "share"):
        t = _trainer(reg_share=0, mode=mode)
        assert t.reg_mode == "off"
        t.begin_task(0)
        t.begin_task(1)
        assert t.reg_ctrl is None


def test_unknown_mode_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        _trainer(mode="adaptive")


def test_share_controller_warns_once_when_beta_saturates(caplog):
    # What every cl_c7/c8/c9 phnet run did: theta sits on the targets, so the raw
    # regulariser is ~1e-5, the controller asks for beta ~1e5 and pins to beta_max.
    ctrl = RegShareBeta(0.8, 0.05, name="policy hnet", warn_after=5)
    with caplog.at_level("WARNING"):
        for _ in range(50):
            beta = ctrl.update(1, 0.645, 1.4e-5)
    assert beta == ctrl.beta_max
    saturation = [r for r in caplog.records if "clamped at beta_max" in r.message]
    assert len(saturation) == 1, "expected exactly one saturation warning"


def test_no_warning_when_the_share_target_is_reachable():
    import pytest
    ctrl = RegShareBeta(0.8, 0.05, warn_after=5)
    for _ in range(50):
        ctrl.update(1, 1.0, 1.0)          # beta wants 4.0, well under the clamp
    assert not ctrl._warned
    assert ctrl.beta == pytest.approx(4.0)


def test_policy_hnet_lr_overrides_policy_lr():
    hp = SimpleNamespace(device="cpu", policy_train_iters=2, policy_bs=8,
                         policy_lr=1e-4, policy_hnet_lr=1e-3)
    policy = HnetPolicy(STATE_DIM, ACT_DIM, n_tasks=1, hidden_dims=(16,), hnet_arch=(8,))
    t = HnetPolicyTrainer(policy, hp)
    t.begin_task(0)
    assert t.optimizer.param_groups[0]["lr"] == 1e-3


def test_plain_trainer_is_untouched_by_hnet_module():
    # Importing the hypernetwork student must not change PolicyNet/PolicyTrainer.
    from hypercrl.control.policy_net import PolicyNet, PolicyTrainer
    hp = SimpleNamespace(device="cpu", policy_train_iters=2, policy_bs=8)
    t = PolicyTrainer(PolicyNet(STATE_DIM, ACT_DIM), hp)
    assert isinstance(t.optimizer, torch.optim.AdamW)
    assert not hasattr(t, "begin_task")

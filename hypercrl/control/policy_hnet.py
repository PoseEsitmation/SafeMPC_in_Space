"""Hypernetwork student: the policy's weights are generated per task.

PolicyNet is a plain network, so training it on a new task overwrites the old
ones.  In cl_c6 every no-replay run had 0 s on target on task 0 after task 1,
while the dynamics hypernetwork kept the teacher's memory intact.  Here the
student gets the same protection (von Oswald et al. 2020): a task embedding
goes into a hypernetwork, a whole MLP's weights come out, and an output
regulariser keeps the weights generated for old embeddings fixed.

Separate from policy_net.py on purpose.  PolicyNet and PolicyTrainer are used
unchanged; HnetPolicyTrainer only subclasses PolicyTrainer:
  * the parent constructor is given a 1-unit placeholder network (it builds an
    AdamW over policy.net), then the hypernetwork student replaces it;
  * the regulariser enters through an optimiser wrapper: the parent calls
    optimizer.step() right after loss.backward(), and the wrapper adds
    beta * reg there;
  * the wrapper never sees the per-step task loss, so beta is set from the mean
    task loss of the previous training phase (the value train() returns).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypercrl.control.policy_net import PolicyNet, PolicyTrainer
from hypercrl.hypercl import HyperNetwork
from hypercrl.hypercl.utils import hnet_regularizer as hreg


class HnetPolicy(nn.Module):
    """MLP student whose weights its own hypernetwork generates per task.

    Its own hypernetwork, not the dynamics one: the two are trained at
    different times on different losses.  The generated MLP has PolicyNet's
    shape (4 × 256, ReLU, linear head) with parameter-free LayerNorm and no
    Dropout.  ``n_tasks`` > 0 makes PolicyTrainer tag every row with its task,
    which forward() uses to pick that task's weights.
    """

    def __init__(self, state_dim: int, action_dim: int, n_tasks: int,
                 hidden_dims: tuple = (256, 256, 256, 256),
                 hnet_arch: tuple = (128, 128), te_dim: int = 10,
                 temb_std: float = 1.0) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.action_dim = action_dim
        self.temb_std = temb_std
        dims = [state_dim, *hidden_dims, action_dim]
        shapes = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            shapes += [[d_out, d_in], [d_out]]
        self.hnet = HyperNetwork(shapes, layers=list(hnet_arch), te_dim=te_dim,
                                 activation_fn=nn.ReLU(), verbose=False)
        for W in self.hnet.theta:          # same init as the dynamics hnet
            if W.ndimension() == 1:
                nn.init.constant_(W, 0)
            else:
                nn.init.xavier_uniform_(W)
        # Output heads scaled so the generated MLP starts Kaiming-like.
        self.hnet.apply_hyperfan_init(temb_var=temb_std ** 2)

    @property
    def num_tasks_seen(self) -> int:
        return self.hnet._num_tasks

    def add_task(self, task_id: int) -> None:
        """New embedding for task_id, started from the previous task's.

        Copying makes the new task start from the last skill instead of a
        random network; the old embedding itself stays untouched.
        """
        if task_id < self.hnet._num_tasks:
            return
        self.hnet.add_task(task_id, self.temb_std)
        emb = self.hnet.get_task_emb(task_id)
        emb.data = emb.data.to(next(iter(self.hnet.theta)).device)
        if task_id > 0:
            with torch.no_grad():
                emb.copy_(self.hnet.get_task_emb(task_id - 1))

    def _mlp(self, x: torch.Tensor, W) -> torch.Tensor:
        for i in range(0, len(W) - 2, 2):
            x = F.relu(F.layer_norm(F.linear(x, W[i], W[i + 1]), (W[i].shape[0],)))
        return F.linear(x, W[-2], W[-1])

    def forward(self, x: torch.Tensor, task_id=None) -> torch.Tensor:
        if task_id is None:
            raise ValueError("HnetPolicy requires task_id")
        if torch.is_tensor(task_id) and task_id.dim() == 2:   # one-hot rows
            ids = task_id.argmax(dim=1)
            tasks = ids.unique().tolist()
            if len(tasks) == 1:
                return self._mlp(x, self.hnet.forward(task_id=tasks[0]))
            out = x.new_zeros(x.shape[0], self.action_dim)
            for j in tasks:
                idx = (ids == j).nonzero(as_tuple=True)[0]
                out = out.index_copy(0, idx, self._mlp(x[idx], self.hnet.forward(task_id=j)))
            return out
        return self._mlp(x, self.hnet.forward(task_id=int(task_id)))


class _RegularisedStep:
    """Optimiser wrapper: adds the hypernetwork output regulariser at step().

    PolicyTrainer.train() runs zero_grad → loss.backward → clip → step.  At
    step() the task gradients are in place; the regulariser's are added on top,
    the total is clipped again, then the real optimiser steps.
    """

    def __init__(self, trainer: "HnetPolicyTrainer", optimizer) -> None:
        self.trainer = trainer
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        t = self.trainer
        if t._targets is not None and t.reg_ctrl is not None:
            reg = hreg.calc_fix_target_reg(t.policy.hnet, t._task_id, targets=t._targets)
            beta = (t.reg_ctrl.update(t._task_id, t._phase_loss, reg.item())
                    if t._phase_loss else t.reg_ctrl.beta)
            (beta * reg).backward()
            torch.nn.utils.clip_grad_norm_(t.policy.parameters(), 0.5)
            t._log_reg(reg.item(), beta)
        self.optimizer.step()


class HnetPolicyTrainer(PolicyTrainer):
    """PolicyTrainer for an HnetPolicy, adding the forgetting protection.

    Call begin_task(task_id) at the start of every task.  reg_share is the
    regulariser's target share of the student's loss (default 0.8, like the
    dynamics model); None turns the protection off (ablation).
    """

    def __init__(self, policy: HnetPolicy, hparams, cbf_fn=None, clf_fn=None) -> None:
        # The parent builds its AdamW over policy.net; give it a placeholder.
        super().__init__(PolicyNet(1, 1, hidden_dims=(), dropout=0.0), hparams,
                         cbf_fn=cbf_fn, clf_fn=clf_fn)
        self.policy = policy
        self.n_tasks = policy.n_tasks
        self.optimizer = None                     # per task, see begin_task
        self.lr = getattr(hparams, "policy_lr", 1e-4)
        self.reg_share = getattr(hparams, "policy_reg_share", 0.8)
        self.reg_beta_init = getattr(hparams, "policy_hnet_beta", 0.05)
        self.reg_ctrl = None
        self._targets = None
        self._task_id = 0
        self._phase_loss = None   # mean task loss of the last train() phase
        self._writer = None

    def begin_task(self, task_id: int) -> None:
        """Freeze the old tasks' generated weights as targets; add task_id.

        Only theta and the new task's embedding are optimised, so the old
        embeddings never move.
        """
        # Local: hypercrl.tools imports hypercrl.control (import cycle).
        from hypercrl.tools.reg_share import RegShareBeta

        hnet = self.policy.hnet
        self._task_id = task_id
        self._targets = hreg.get_current_targets(task_id, hnet) if task_id > 0 else None
        self.policy.add_task(task_id)
        self.optimizer = _RegularisedStep(
            self, torch.optim.Adam(list(hnet.theta) + [hnet.get_task_emb(task_id)], lr=self.lr))
        self.reg_ctrl = (RegShareBeta(self.reg_share, self.reg_beta_init)
                         if task_id > 0 and self.reg_share else None)

    def train(self, dataset, writer=None, sample_weights=None, task_id=0) -> float:
        if self.optimizer is None:
            raise RuntimeError("HnetPolicyTrainer: call begin_task(task_id) first")
        self._writer = writer
        mean_loss = super().train(dataset, writer=writer, sample_weights=sample_weights,
                                  task_id=task_id)
        if mean_loss:
            self._phase_loss = mean_loss    # carries into the next task's first phase
        return mean_loss

    def _log_reg(self, reg: float, beta: float) -> None:
        if self._writer is None or (self._step + 1) % 200:
            return
        reg_w = beta * reg
        L = self._phase_loss or 0.0
        self._writer.add_scalar("policy_hnet/loss_reg", reg_w, self._step + 1)
        self._writer.add_scalar("policy_hnet/loss_task_phase", L, self._step + 1)
        self._writer.add_scalar("policy_hnet/reg_share", reg_w / max(L + reg_w, 1e-12),
                                self._step + 1)
        self._writer.add_scalar("policy_hnet/beta", beta, self._step + 1)

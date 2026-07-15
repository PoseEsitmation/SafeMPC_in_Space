"""Regression test for TaskLoss.regularize scaling.

History: the regularizer originally overwrote `loss` in the loop (keeping only
the last tensor's norm ≈ no regularization — dynamics model fit to ~0.02 L1).
A later fix accumulated the norms but left the sum UNWEIGHTED, adding an O(100)
L1 penalty that lasso-crushed the hnet-generated weights: every run from
baseline_11 to baseline_19 had train/loss pinned at ~103 and a dynamics model
stuck at ~0.55 normalized L1 on data a plain MLP fits to 0.04.  The correct
form is reg_lambda * sum of norms.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import types
import pytest
import torch

from hypercrl.hnet_exp import TaskLoss


def _make_loss(reg_lambda):
    hp = types.SimpleNamespace(
        out_var=False, out_dim=13, mlp_var_minmax=False, reg_lambda=reg_lambda)
    return TaskLoss(hp, mnet=None)


class TestTaskLossRegularizer:

    def test_regularize_sums_all_tensors(self):
        """Every generated tensor must contribute (not just the last one)."""
        tl = _make_loss(reg_lambda=1.0)
        weights = [torch.ones(4), torch.ones(4), torch.ones(4)]  # L1 = 4 each
        assert float(tl.regularize(weights)) == pytest.approx(12.0)

    def test_regularize_scaled_by_reg_lambda(self):
        """The sum of norms must be multiplied by reg_lambda — unweighted it
        is O(100) and dominates a task MSE of O(0.1)."""
        tl = _make_loss(reg_lambda=1e-4)
        weights = [torch.ones(100), torch.ones(100)]  # unweighted sum = 200
        assert float(tl.regularize(weights)) == pytest.approx(0.02)

    def test_regularize_off_when_lambda_zero(self):
        tl = _make_loss(reg_lambda=0.0)
        assert tl.regularize([torch.ones(10)]) == 0

    def test_forward_task_term_dominates(self):
        """With realistic magnitudes the task MSE must dominate the loss:
        a batch of 100 with per-element error 0.1 vs 6 weight tensors of
        256x256 entries at |w|=0.01 (unweighted L1 sum ≈ 3932)."""
        tl = _make_loss(reg_lambda=1e-4)
        pred = torch.zeros(100, 13)
        gt   = torch.full((100, 13), 0.1)
        weights = [torch.full((256, 256), 0.01) for _ in range(6)]
        loss = tl.forward(pred, gt, weights)
        task_term = torch.nn.functional.mse_loss(pred, gt, reduction="sum") / 13
        reg_term  = float(loss) - float(task_term)
        assert reg_term < float(task_term), (
            f"regularizer ({reg_term:.3f}) dominates task loss ({float(task_term):.3f})"
        )

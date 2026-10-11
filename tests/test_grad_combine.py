import torch

from hypercrl.tools import grad_combine


def test_norm_rescales_reg_gradient():
    g_task, g_reg = torch.tensor([3.0, 4.0]), torch.tensor([0.0, 0.01])
    g, cos, scale = grad_combine.combine(g_task, g_reg, "norm", beta=2.0)
    assert torch.isclose((g - g_task).norm(), torch.tensor(10.0))
    assert abs(scale - 1000.0) < 1e-3
    assert abs(cos - 0.8) < 1e-6


def test_project_removes_conflict_only():
    g_task, g_reg = torch.tensor([1.0, 1.0]), torch.tensor([-1.0, 0.0])
    g, cos, _ = grad_combine.combine(g_task, g_reg, "project")
    assert cos < 0
    assert torch.allclose(g, torch.tensor([-1.0, 1.0]))   # (0, 1) + g_reg

    g_reg = torch.tensor([1.0, 0.0])                       # no conflict: plain sum
    g, _, _ = grad_combine.combine(g_task, g_reg, "project")
    assert torch.allclose(g, g_task + g_reg)


def test_zero_reg_gradient_is_safe():
    g_task = torch.tensor([1.0, 2.0])
    for mode in ("norm", "project"):
        g, cos, _ = grad_combine.combine(g_task, torch.zeros(2), mode)
        assert torch.allclose(g, g_task) and cos == 0.0


def test_flat_roundtrip():
    ps = [torch.nn.Parameter(torch.zeros(2, 3)), torch.nn.Parameter(torch.zeros(4))]
    flat = torch.arange(10, dtype=torch.float32)
    grad_combine.write_grads(ps, flat)
    assert torch.equal(grad_combine.flat_grads(ps), flat)

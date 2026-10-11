"""Combine the hnet's task and regulariser gradients by hand (hparams.reg_grad_mode).

The default (None) keeps the summed backward passes, so earlier runs stay
reproducible.  The other modes act on the flattened gradients:

  "norm":    rescale g_reg so that ||g_reg|| = beta * ||g_task||; the
             regulariser's loss weight then no longer matters.
  "project": if cos(g_task, g_reg) < 0, remove from g_task its component
             along g_reg (PCGrad / A-GEM style), then add g_reg.
"""
import torch

MODES = (None, "norm", "project")


def combine(g_task, g_reg, mode, beta=1.0, eps=1e-12):
    """Flat gradients in, (combined gradient, cos(g_task, g_reg), reg scale) out.

    reg scale is the factor g_reg was multiplied by (1 for "project").
    """
    task_norm, reg_norm = g_task.norm(), g_reg.norm()
    dot = torch.dot(g_task, g_reg)
    cos = (dot / (task_norm * reg_norm)).item() if task_norm > eps and reg_norm > eps else 0.0

    if mode == "norm":
        scale = (beta * task_norm / reg_norm).item() if reg_norm > eps else 0.0
        return g_task + scale * g_reg, cos, scale
    if mode == "project":
        if cos < 0:
            g_task = g_task - dot / (reg_norm ** 2) * g_reg
        return g_task + g_reg, cos, 1.0
    raise ValueError(f"unknown reg_grad_mode {mode!r}, expected one of {MODES}")


def flat_grads(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in params])


def write_grads(params, flat):
    """Write a flat gradient back into p.grad for every parameter."""
    i = 0
    for p in params:
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n

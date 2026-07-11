"""Unit tests for NNPolicyAgent — output clipping and inference pipeline.

Run with:
    cd /path/to/SafeStuff && python -m pytest tests/test_nn_agent.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import types
import numpy as np
import pytest
import torch
import torch.nn as nn

from hypercrl.control.policy_net import PolicyNet
from hypercrl.control.agent import NNPolicyAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(hparams, policy=None, collector=None, safety_filter=None):
    if policy is None:
        policy = PolicyNet(hparams.state_dim, hparams.control_dim,
                           hidden_dims=(32, 32), dropout=0.0)
    return NNPolicyAgent(hparams, policy,
                         collector=collector,
                         safety_filter=safety_filter)


def _large_output_policy(action_dim: int = 3, value: float = 10.0):
    """A policy whose linear layer is initialised to output `value` for any input."""
    net = PolicyNet(13, action_dim, hidden_dims=(32, 32), dropout=0.0)
    with torch.no_grad():
        for p in net.net[-1].parameters():
            p.fill_(value)
    return net


# ===========================================================================
# NNPolicyAgent — clipping
# ===========================================================================

class TestNNPolicyAgentClip:

    def test_output_clipped_to_minus_one(self, hparams):
        """act() must never return a component < −1."""
        policy = _large_output_policy(value=-10.0)
        agent  = _make_agent(hparams, policy=policy)
        obs    = np.zeros(hparams.state_dim, dtype=np.float32)
        u = agent.act(obs)
        assert (u.cpu() >= -1.0).all(), (
            f"act() returned {u.tolist()} — component < −1 found. "
            "An unbounded linear head must be clipped before reaching the QP solver."
        )

    def test_output_clipped_to_plus_one(self, hparams):
        """act() must never return a component > +1."""
        policy = _large_output_policy(value=10.0)
        agent  = _make_agent(hparams, policy=policy)
        obs    = np.zeros(hparams.state_dim, dtype=np.float32)
        u = agent.act(obs)
        assert (u.cpu() <= 1.0).all(), (
            f"act() returned {u.tolist()} — component > +1 found."
        )

    def test_clip_applied_without_normalisation(self, hparams):
        """Clip must still work when normalize_xu=False (no a_mu/a_std scaling)."""
        hparams.normalize_xu = False
        policy = _large_output_policy(value=5.0)
        agent  = _make_agent(hparams, policy=policy)
        obs    = np.zeros(hparams.state_dim, dtype=np.float32)
        u = agent.act(obs)
        assert (u.cpu().abs() <= 1.0).all()

    def test_clip_applied_with_denormalisation(self, hparams, mock_collector):
        """After u * a_std + a_mu, output may exceed ±1 — clip must still enforce bounds."""
        # With a_std = 0.62 and policy output ≈ 10.0,
        # denorm gives 10.0 * 0.62 = 6.2 → must be clipped to 1.0
        policy = _large_output_policy(value=10.0)
        agent  = _make_agent(hparams, policy=policy, collector=mock_collector)
        agent.cache_state_norm(0)   # loads a_mu=0, a_std=0.62 from mock_collector
        obs    = np.zeros(hparams.state_dim, dtype=np.float32)
        u = agent.act(obs)
        assert (u.cpu().abs() <= 1.0 + 1e-5).all(), (
            f"After denormalisation (×0.62) the output was not clipped: {u.tolist()}"
        )

    def test_clip_before_safety_filter(self, hparams):
        """The clip happens before the safety filter, so the QP never sees extreme values."""
        class RecordingFilter:
            received = None
            def filter(self, state, u_proposed):
                RecordingFilter.received = u_proposed.copy()
                return u_proposed   # pass-through

        policy = _large_output_policy(value=10.0)
        agent  = _make_agent(hparams, policy=policy, safety_filter=RecordingFilter())
        obs    = np.zeros(hparams.state_dim, dtype=np.float32)
        agent.act(obs)

        assert RecordingFilter.received is not None
        assert (np.abs(RecordingFilter.received) <= 1.0 + 1e-5).all(), (
            f"Safety filter received un-clipped input {RecordingFilter.received}. "
            "Extreme values can destabilise the CLARABEL QP solver."
        )

    # -----------------------------------------------------------------------
    # cache_state_norm

    def test_cache_state_norm_populates_stats(self, hparams, mock_collector):
        agent = _make_agent(hparams, collector=mock_collector)
        assert agent.x_mu is None
        agent.cache_state_norm(0)
        assert agent.x_mu is not None
        assert agent.a_std is not None

    def test_cache_state_norm_device(self, hparams, mock_collector):
        agent = _make_agent(hparams, collector=mock_collector)
        agent.cache_state_norm(0)
        assert str(agent.x_mu.device) == hparams.device

    # -----------------------------------------------------------------------
    # Inference pipeline order

    def test_act_returns_tensor(self, hparams):
        agent = _make_agent(hparams)
        obs   = np.zeros(hparams.state_dim, dtype=np.float32)
        u = agent.act(obs)
        assert isinstance(u, torch.Tensor)

    def test_act_accepts_numpy_obs(self, hparams):
        agent = _make_agent(hparams)
        obs   = np.random.default_rng(0).standard_normal(hparams.state_dim).astype(np.float32)
        u = agent.act(obs)
        assert u.shape == (hparams.control_dim,)

    def test_act_accepts_tensor_obs(self, hparams):
        agent = _make_agent(hparams)
        obs   = torch.randn(hparams.state_dim)
        u = agent.act(obs)
        assert u.shape == (hparams.control_dim,)


# ===========================================================================
# Regression: tanh saturation is gone
# ===========================================================================

class TestTanhSaturationRegression:
    """Confirm the full path from expert target → gradient → weight update works
    for near-boundary actions that the old Tanh could not represent."""

    def test_policy_can_represent_full_torque_target(self, hparams):
        """The policy output must be able to reach 1.0 in physical space.

        With the old final Tanh and a_std=0.62:
            max physical = tanh(∞) × 0.62 = 0.62  <  1.0   ← saturated
        With the linear head:
            physical = u_norm × 0.62, u_norm can be 1.0/0.62 = 1.61 → physical = 1.0 ✓
        """
        a_std = 0.62
        u_max = 1.0
        # In normalised space the policy must be able to output u_max / a_std
        required_norm = u_max / a_std   # ≈ 1.61

        net = PolicyNet(13, 3, hidden_dims=(64, 64), dropout=0.0)

        # Train for a few steps on the near-max target
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        target = torch.full((32, 3), required_norm)
        x      = torch.randn(32, 13)

        for _ in range(200):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x), target)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = net(x).mean(0)

        # After training the policy should get close to the target
        assert (pred > 1.0).all(), (
            f"Policy cannot represent full-torque target (got {pred.tolist()}). "
            "This indicates Tanh is still present and saturating for safety-critical states."
        )

    def test_gradient_norm_non_zero_at_large_target(self, hparams):
        """The gradient L2 norm must be >0 for a batch where all targets are 1.6."""
        net = PolicyNet(13, 3, hidden_dims=(32, 32), dropout=0.0)
        x      = torch.randn(16, 13)
        target = torch.full((16, 3), 1.6)
        loss = torch.nn.functional.mse_loss(net(x), target)
        loss.backward()

        total_grad = sum(
            p.grad.norm().item()
            for p in net.parameters()
            if p.grad is not None
        )
        assert total_grad > 1e-6, (
            "Gradient norm is essentially zero for target=1.6. "
            "Tanh saturation is still blocking gradient flow."
        )

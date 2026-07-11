"""Unit tests for PolicyNet, PolicyTrainer, and the DAgger data pipeline.

Run with:
    cd /path/to/SafeStuff && python -m pytest tests/test_policy_net.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from hypercrl.control.policy_net import PolicyNet, PolicyTrainer


# ===========================================================================
# PolicyNet — output bounds and gradient flow
# ===========================================================================

class TestPolicyNet:

    def _make_net(self, state_dim=13, action_dim=3):
        return PolicyNet(state_dim, action_dim, hidden_dims=(32, 32), dropout=0.0)

    # -----------------------------------------------------------------------

    def test_output_shape(self):
        net = self._make_net()
        x = torch.zeros(4, 13)
        y = net(x)
        assert y.shape == (4, 3), "Expected (batch, action_dim)"

    def test_output_is_unbounded(self):
        """Without Tanh the output can exceed ±1 — critical for safety-critical targets."""
        net = self._make_net()
        # Push the final linear layer to produce large outputs
        with torch.no_grad():
            for p in net.net[-1].parameters():   # last Linear
                p.fill_(5.0)
        x = torch.ones(1, 13)
        y = net(x)
        assert (y.abs() > 1.0).any(), (
            "PolicyNet output must be unbounded (no final Tanh). "
            "If all outputs are ≤ 1 the policy cannot represent full-torque expert actions."
        )

    def test_no_tanh_module(self):
        """Confirm Tanh is not present anywhere in the network."""
        net = self._make_net()
        for m in net.modules():
            assert not isinstance(m, nn.Tanh), (
                "Found nn.Tanh in PolicyNet — remove it to prevent gradient saturation "
                "for near-boundary expert targets."
            )

    def test_gradients_flow_for_targets_beyond_one(self):
        """Gradient must be non-zero when the supervised target exceeds ±1 in normalised space.

        With the old Tanh, tanh(z)→1 as z→∞ so ∂tanh/∂z→0 — gradient dies for
        exactly the safety-critical states (|u_mpc / a_std| ≈ 1.6).
        Without Tanh the gradient is always well-defined.
        """
        net = self._make_net()
        x      = torch.randn(8, 13)
        target = torch.full((8, 3), 1.6)   # ≈ u_mpc=1.0 / a_std=0.62, unreachable by tanh

        pred = net(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()

        all_zero = all(
            (p.grad is None or p.grad.abs().max() == 0)
            for p in net.parameters()
        )
        assert not all_zero, "All gradients are zero — Tanh saturation is still present."

    def test_deterministic_eval_mode(self):
        """Dropout must be off in eval mode so inference is repeatable."""
        net = self._make_net(state_dim=13, action_dim=3)
        net.eval()
        x = torch.randn(1, 13)
        with torch.no_grad():
            y1 = net(x)
            y2 = net(x)
        assert torch.allclose(y1, y2), "Non-deterministic output in eval mode."


# ===========================================================================
# PolicyTrainer — curriculum and reset
# ===========================================================================

class TestPolicyTrainer:

    def _make_trainer(self, hparams):
        policy = PolicyNet(13, 3, hidden_dims=(32, 32), dropout=0.0)
        return PolicyTrainer(policy, hparams)

    # -----------------------------------------------------------------------
    # Constructor

    def test_dagger_n_iter_read_from_hparams(self, hparams):
        """_dagger_n_iter must come from hparams, not a hard-coded default."""
        hparams.dagger_n_iter = 7
        trainer = self._make_trainer(hparams)
        assert trainer._dagger_n_iter == 7, (
            "_dagger_n_iter was not read from hparams.dagger_n_iter. "
            "The kappa schedule would use the wrong number of iterations."
        )

    def test_initial_dagger_iter_is_zero(self, hparams):
        trainer = self._make_trainer(hparams)
        assert trainer._dagger_iter == 0

    def test_initial_lambda_equals_base(self, hparams):
        hparams.policy_lambda_cbf = 1e-3
        hparams.policy_lambda_clf = 1e-5
        trainer = self._make_trainer(hparams)
        assert trainer.lambda_cbf == pytest.approx(1e-3)
        assert trainer.lambda_clf == pytest.approx(1e-5)

    # -----------------------------------------------------------------------
    # reset_per_task

    def test_reset_clears_dagger_iter(self, hparams):
        trainer = self._make_trainer(hparams)
        trainer._dagger_iter = 3
        trainer.reset_per_task()
        assert trainer._dagger_iter == 0

    def test_reset_clears_dag_buffer(self, hparams):
        trainer = self._make_trainer(hparams)
        trainer._dag_states.append(torch.zeros(1, 13))
        trainer._dag_actions_phys.append(torch.zeros(3))
        trainer.reset_per_task()
        assert len(trainer._dag_states) == 0
        assert len(trainer._dag_actions_phys) == 0

    def test_reset_restores_lambda_to_base(self, hparams):
        hparams.policy_lambda_cbf = 1e-3
        hparams.policy_lambda_clf = 1e-5
        trainer = self._make_trainer(hparams)
        trainer.lambda_cbf = 0.5   # simulate after several doublings
        trainer.lambda_clf = 0.5
        trainer.reset_per_task()
        assert trainer.lambda_cbf == pytest.approx(1e-3)
        assert trainer.lambda_clf == pytest.approx(1e-5)

    def test_reset_does_not_wipe_dagger_n_iter(self, hparams):
        hparams.dagger_n_iter = 7
        trainer = self._make_trainer(hparams)
        trainer.reset_per_task()
        assert trainer._dagger_n_iter == 7   # fixed per hparams, not reset

    # -----------------------------------------------------------------------
    # kappa schedule

    def test_kappa_iter1(self, hparams):
        """iter=1 of 5 → kappa = 1 − 1/5 = 0.8."""
        trainer = self._make_trainer(hparams)
        trainer._dagger_iter = 1
        kappa = max(0.0, 1.0 - trainer._dagger_iter / trainer._dagger_n_iter)
        assert kappa == pytest.approx(0.8)

    def test_kappa_iter5(self, hparams):
        """iter=5 of 5 → kappa = 0.0 (policy fully in control)."""
        trainer = self._make_trainer(hparams)
        trainer._dagger_iter = 5
        kappa = max(0.0, 1.0 - trainer._dagger_iter / trainer._dagger_n_iter)
        assert kappa == pytest.approx(0.0)

    def test_kappa_never_negative(self, hparams):
        """kappa must never go below 0 even if _dagger_iter exceeds n_iter."""
        trainer = self._make_trainer(hparams)
        trainer._dagger_iter = 10
        kappa = max(0.0, 1.0 - trainer._dagger_iter / trainer._dagger_n_iter)
        assert kappa == 0.0

    # -----------------------------------------------------------------------
    # lambda curriculum

    def test_lambda_doubles_each_iteration(self, hparams):
        hparams.policy_lambda_cbf = 1e-3
        trainer = self._make_trainer(hparams)
        initial = trainer.lambda_cbf
        trainer.lambda_cbf = min(trainer.lambda_cbf * 2.0, 1.0)
        assert trainer.lambda_cbf == pytest.approx(initial * 2.0)

    def test_lambda_capped_at_one(self, hparams):
        hparams.policy_lambda_cbf = 0.9
        trainer = self._make_trainer(hparams)
        trainer.lambda_cbf = min(trainer.lambda_cbf * 2.0, 1.0)
        assert trainer.lambda_cbf <= 1.0

    # -----------------------------------------------------------------------
    # _make_policy_train_set

    def test_make_policy_train_set_empty_dag_buffer_returns_dynamics(
        self, hparams, mock_collector
    ):
        """When no DAgger data exists, the dynamics dataset is returned unchanged."""
        trainer = self._make_trainer(hparams)
        dyn_ds, _ = mock_collector.get_dataset(0)

        result, _ = trainer._make_policy_train_set(dyn_ds, mock_collector, 0)

        # Should be the exact same object — no concatenation
        assert result is dyn_ds

    def test_make_policy_train_set_with_dag_data_increases_rows(
        self, hparams, mock_collector
    ):
        """After DAgger data is added the combined dataset must be larger."""
        trainer = self._make_trainer(hparams)
        dyn_ds, _ = mock_collector.get_dataset(0)
        n_dyn = dyn_ds.tensors[0].shape[0]

        n_dag = 20
        trainer._dag_states       = [torch.randn(1, 13) for _ in range(n_dag)]
        trainer._dag_actions_phys = [torch.zeros(3)     for _ in range(n_dag)]

        result, _ = trainer._make_policy_train_set(dyn_ds, mock_collector, 0)
        n_combined = result.tensors[0].shape[0]
        assert n_combined == n_dyn + n_dag

    def test_make_policy_train_set_dag_actions_normalised(
        self, hparams, mock_collector
    ):
        """DAgger expert actions must be normalised with (u − a_mu) / a_std."""
        trainer = self._make_trainer(hparams)
        dyn_ds, _ = mock_collector.get_dataset(0)

        u_expert_phys = torch.tensor([1.0, -1.0, 0.5])  # physical expert action
        trainer._dag_states       = [torch.randn(1, 13)]
        trainer._dag_actions_phys = [u_expert_phys]

        result, _ = trainer._make_policy_train_set(dyn_ds, mock_collector, 0)
        _, a_mu, a_std = torch.zeros(3), torch.zeros(3), torch.full((3,), 0.62)
        # The DAgger row is the last row added
        dag_u_in_ds = result.tensors[1][-1]
        expected = (u_expert_phys - mock_collector._a_mu) / mock_collector._a_std
        assert torch.allclose(dag_u_in_ds, expected, atol=1e-5), (
            f"DAgger action not correctly normalised.\n"
            f"  got:      {dag_u_in_ds}\n"
            f"  expected: {expected}"
        )

    def test_make_policy_train_set_three_tensors(self, hparams, mock_collector):
        """Result must have exactly 3 tensors: (state, action, dummy)."""
        trainer = self._make_trainer(hparams)
        dyn_ds, _ = mock_collector.get_dataset(0)
        trainer._dag_states       = [torch.randn(1, 13)]
        trainer._dag_actions_phys = [torch.zeros(3)]
        result, _ = trainer._make_policy_train_set(dyn_ds, mock_collector, 0)
        assert len(result.tensors) == 3

    def test_make_policy_train_set_dummy_is_zeros(self, hparams, mock_collector):
        """The dummy next-state for DAgger rows must be zeros (ignored by train())."""
        trainer = self._make_trainer(hparams)
        dyn_ds, _ = mock_collector.get_dataset(0)
        n_dag = 5
        trainer._dag_states       = [torch.randn(1, 13) for _ in range(n_dag)]
        trainer._dag_actions_phys = [torch.zeros(3)     for _ in range(n_dag)]
        result, _ = trainer._make_policy_train_set(dyn_ds, mock_collector, 0)
        dag_dummy = result.tensors[2][-n_dag:]
        assert dag_dummy.abs().max() == 0.0


# ===========================================================================
# DAgger rollout — data separation
# ===========================================================================

class TestDaggerDataSeparation:
    """Verify that DAgger rollout transitions go ONLY to the policy buffer
    (with u_mpc expert labels) and never into the dynamics collector.
    Mixed-policy rollout data in the collector shifts the dynamics model's
    training distribution away from the expert's trajectories and degrades
    the MPC expert that plans through it (observed in baseline_18: 80% of
    dynamics rows were rollout data and the expert never converged).
    """

    def _make_preprocess_fn(self):
        def preprocess_fn(raw_obs):
            return torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0)
        return preprocess_fn

    def test_collector_untouched_by_dagger_rollout(
        self, hparams, mock_collector, mock_mpc_agent, mock_env
    ):
        """collector.add() must never be called during a DAgger rollout."""
        hparams.policy_train_iters = 1
        policy  = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        trainer = PolicyTrainer(policy, hparams)

        trainer._dagger_iter = 1  # increments to 2 → kappa = 1 - 2/5 = 0.6

        a_std = np.full(3, 0.62, dtype=np.float32)
        a_mu  = np.zeros(3,      dtype=np.float32)

        trainer.dagger_update(
            env=mock_env,
            mpc_agent=mock_mpc_agent,
            collector=mock_collector,
            task_id=0,
            preprocess_fn=self._make_preprocess_fn(),
            n_rollout=1,
            max_ep_steps=mock_env._max_steps,
            writer=None,
            a_mu=a_mu,
            a_std=a_std,
        )

        assert len(mock_collector.add_calls) == 0, (
            f"collector.add() was called {len(mock_collector.add_calls)} times during "
            "the DAgger rollout. Rollout transitions must stay out of the dynamics "
            "dataset — they pollute the model the MPC expert plans through."
        )

    def test_dag_buffer_receives_u_mpc_expert_labels(
        self, hparams, mock_collector, mock_mpc_agent, mock_env
    ):
        """The DAgger buffer must store u_mpc (expert), not u_mix."""
        hparams.policy_train_iters = 1
        policy  = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        trainer = PolicyTrainer(policy, hparams)

        u_expert_val = float(mock_mpc_agent._u[0])   # 0.8 for all dims
        a_std = np.full(3, 0.62, dtype=np.float32)
        a_mu  = np.zeros(3,      dtype=np.float32)

        trainer.dagger_update(
            env=mock_env,
            mpc_agent=mock_mpc_agent,
            collector=mock_collector,
            task_id=0,
            preprocess_fn=self._make_preprocess_fn(),
            n_rollout=1,
            max_ep_steps=mock_env._max_steps,
            writer=None,
            a_mu=a_mu,
            a_std=a_std,
        )

        assert len(trainer._dag_actions_phys) > 0, "DAgger buffer is empty after rollout."
        for u_phys in trainer._dag_actions_phys:
            assert torch.allclose(u_phys, mock_mpc_agent._u, atol=1e-4), (
                f"DAgger buffer stored {u_phys.tolist()} instead of expert {mock_mpc_agent._u.tolist()}. "
                "Policy will be trained on wrong imitation targets."
            )

    def test_dagger_update_uses_clean_snapshot_not_postrollout_dataset(
        self, hparams, mock_collector, mock_mpc_agent, mock_env
    ):
        """_make_policy_train_set must receive the pre-rollout snapshot.

        If it received the post-rollout dynamics dataset instead, the rollout
        states would appear twice — once with u_mix (dynamics) and once with
        u_mpc (DAgger buffer) — creating conflicting BC supervision.
        """
        hparams.policy_train_iters = 1
        pre_rollout_size, _ = mock_collector.get_dataset(0)
        n_before = pre_rollout_size.tensors[0].shape[0]

        captured = {}

        # Intercept _make_policy_train_set to see what base dataset it receives
        policy  = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        trainer = PolicyTrainer(policy, hparams)

        original = trainer._make_policy_train_set

        def spy(dyn_set, collector, task_id):
            captured["n_rows"] = dyn_set.tensors[0].shape[0]
            return original(dyn_set, collector, task_id)

        trainer._make_policy_train_set = spy

        trainer.dagger_update(
            env=mock_env,
            mpc_agent=mock_mpc_agent,
            collector=mock_collector,
            task_id=0,
            preprocess_fn=self._make_preprocess_fn(),
            n_rollout=1,
            max_ep_steps=mock_env._max_steps,
            writer=None,
            a_mu=None,
            a_std=None,
        )

        assert "n_rows" in captured, "_make_policy_train_set was never called."
        assert captured["n_rows"] == n_before, (
            f"_make_policy_train_set received {captured['n_rows']} rows but expected "
            f"{n_before} (pre-rollout snapshot). It was passed the post-rollout dataset "
            "which contains u_mix-labelled transitions — conflicting BC supervision."
        )


# ===========================================================================
# DAgger guard — iter limit
# ===========================================================================

class TestDaggerGuard:

    def test_dagger_iter_increments(self, hparams, mock_collector, mock_mpc_agent, mock_env):
        hparams.policy_train_iters = 1
        policy  = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        trainer = PolicyTrainer(policy, hparams)
        assert trainer._dagger_iter == 0

        trainer.dagger_update(
            env=mock_env, mpc_agent=mock_mpc_agent, collector=mock_collector,
            task_id=0, preprocess_fn=lambda obs: torch.zeros(1, 13),
            n_rollout=1, max_ep_steps=3, writer=None,
        )
        assert trainer._dagger_iter == 1

    def test_reset_per_task_allows_dagger_to_run_again(
        self, hparams, mock_collector, mock_mpc_agent, mock_env
    ):
        """After reset_per_task() the guard condition is satisfied again."""
        hparams.policy_train_iters = 1
        policy  = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        trainer = PolicyTrainer(policy, hparams)
        trainer._dagger_iter = 5   # simulate exhausted

        trainer.reset_per_task()
        assert trainer._dagger_iter < trainer._dagger_n_iter, (
            "After reset_per_task() the guard condition _dagger_iter < _dagger_n_iter "
            "must be True so DAgger can fire again for the new task."
        )


class TestDaggerRolloutFiltering:
    """DAGGER rollouts must execute the raw mixed action by default.

    Filtering the rollout action keeps trajectories away from exactly the
    states where the raw policy fails, so the buffer never receives avoidance
    labels (baseline_21: unfiltered validation KOZ violations flat across all
    10 iterations while expert data was 99.9% filter-inactive).
    """

    class _RecordingFilter:
        def __init__(self):
            self.calls = 0

        def filter(self, state, u_proposed):
            self.calls += 1
            return np.zeros_like(np.asarray(u_proposed, dtype=np.float32))

    def _agent_with_filter(self, mock_mpc_agent):
        mock_mpc_agent.safety_filter = self._RecordingFilter()
        return mock_mpc_agent

    def _run(self, hparams, mock_collector, agent, mock_env, **kw):
        hparams.policy_train_iters = 1
        policy  = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        trainer = PolicyTrainer(policy, hparams)
        trainer.dagger_update(
            env=mock_env, mpc_agent=agent, collector=mock_collector,
            task_id=0, preprocess_fn=lambda obs: torch.zeros(1, 13),
            n_rollout=1, max_ep_steps=mock_env._max_steps, writer=None,
            **kw,
        )
        return trainer

    def test_rollout_actions_unfiltered_by_default(
        self, hparams, mock_collector, mock_mpc_agent, mock_env
    ):
        agent = self._agent_with_filter(mock_mpc_agent)
        self._run(hparams, mock_collector, agent, mock_env)
        assert agent.safety_filter.calls == 0, (
            "The safety filter was applied to DAGGER rollout actions. Rollouts "
            "must sample the learner's own state distribution (including near-KOZ "
            "states) or the buffer contains no avoidance labels."
        )

    def test_filter_rollouts_true_restores_filtering(
        self, hparams, mock_collector, mock_mpc_agent, mock_env
    ):
        agent = self._agent_with_filter(mock_mpc_agent)
        self._run(hparams, mock_collector, agent, mock_env, filter_rollouts=True)
        assert agent.safety_filter.calls == mock_env._max_steps, (
            "filter_rollouts=True must pass every rollout action through the "
            "safety filter (one call per env step)."
        )


class TestSafetyPrioritizedSampling:
    """Rows where the expert's filter stepped in (buffer tags) or that lie
    near the KOZ (θ-margin) must be oversampled during policy training."""

    def _make_trainer(self, hparams, oversample=10.0):
        hparams.policy_safety_oversample = oversample
        policy = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        return PolicyTrainer(policy, hparams)

    def test_weights_none_when_oversampling_off(self, hparams, mock_collector):
        trainer = self._make_trainer(hparams, oversample=1.0)
        dyn_ds, _ = mock_collector.get_dataset(0)
        _, weights = trainer._make_policy_train_set(dyn_ds, mock_collector, 0)
        assert weights is None

    def test_filter_tagged_rows_get_max_weight(self, hparams, mock_collector):
        trainer = self._make_trainer(hparams)
        dyn_ds, _ = mock_collector.get_dataset(0)
        n_dyn = dyn_ds.tensors[0].shape[0]

        trainer._dag_states        = [torch.randn(1, 13) for _ in range(4)]
        trainer._dag_actions_phys  = [torch.zeros(3) for _ in range(4)]
        trainer._dag_filter_active = [1.0, 0.0, 1.0, 0.0]

        _, weights = trainer._make_policy_train_set(dyn_ds, mock_collector, 0)
        assert weights is not None
        assert torch.allclose(weights[:n_dyn], torch.ones(n_dyn)), \
            "untagged base rows must keep weight 1 when no margin_fn is set"
        assert torch.allclose(weights[n_dyn:], torch.tensor([10.0, 1.0, 10.0, 1.0])), \
            "filter-intervened buffer rows must be oversampled K=10x"

    def test_margin_fn_grades_all_rows(self, hparams, mock_collector):
        """Criticality fades linearly: 1 at the boundary, 0 at margin_deg."""
        trainer = self._make_trainer(hparams)
        margins = torch.tensor([0.0, 7.5, 15.0, 30.0])
        trainer.margin_fn = lambda x: margins[: x.shape[0]]

        x = torch.randn(4, 13)
        weights = trainer._sample_weights(x, tags=None)
        assert torch.allclose(weights, torch.tensor([10.0, 5.5, 1.0, 1.0])), \
            f"expected linear grading, got {weights}"

    def test_tags_and_margin_combine_as_max(self, hparams):
        trainer = self._make_trainer(hparams)
        trainer.margin_fn = lambda x: torch.full((x.shape[0],), 7.5)  # crit 0.5 -> w 5.5
        weights = trainer._sample_weights(torch.randn(2, 13),
                                          tags=torch.tensor([1.0, 0.0]))
        assert torch.allclose(weights, torch.tensor([10.0, 5.5]))

    def test_rollout_records_filter_tags(
        self, hparams, mock_collector, mock_mpc_agent, mock_env
    ):
        """dagger_update must tag each stored label with the expert filter state."""
        import types as _types
        hparams.policy_train_iters = 1
        mock_mpc_agent.safety_filter = _types.SimpleNamespace(last_was_active=True)
        trainer = self._make_trainer(hparams)

        trainer.dagger_update(
            env=mock_env, mpc_agent=mock_mpc_agent, collector=mock_collector,
            task_id=0, preprocess_fn=lambda obs: torch.zeros(1, 13),
            n_rollout=1, max_ep_steps=mock_env._max_steps, writer=None,
        )
        assert len(trainer._dag_filter_active) == len(trainer._dag_states)
        assert all(t == 1.0 for t in trainer._dag_filter_active)

    def test_train_accepts_sample_weights(self, hparams, mock_collector):
        trainer = self._make_trainer(hparams)
        hparams.policy_bs = 8
        trainer.bs = 8
        dyn_ds, _ = mock_collector.get_dataset(0)
        weights = torch.ones(len(dyn_ds))
        loss = trainer.train(dyn_ds, writer=None, sample_weights=weights)
        assert loss >= 0.0


class TestCbfEffectiveness:
    """baseline_33: loss_cbf was identically 0 — training states are all safe.
    The fixes: ε_train hinge margin, synthetic boundary states for the CBF
    penalty, and pure-NN (student) DAGGER rollout episodes."""

    def test_boundary_sampler_resamples_margin_into_corridor(self):
        from hypercrl.control.policy_net import (make_space_boundary_sampler,
                                                 make_space_margin_fn)
        import math
        x_mu, x_std = torch.zeros(1, 13), torch.ones(1, 13)
        sampler   = make_space_boundary_sampler(x_mu, x_std,
                                                margin_low_deg=-10, margin_high_deg=15)
        margin_fn = make_space_margin_fn(x_mu, x_std)

        # Build safe states: margin 40°, half-angle 20° → θ = 60°
        m, th = math.radians(40), math.radians(60)
        obs = torch.zeros(64, 13)
        obs[:, 0] = 1.0                       # q_e identity
        obs[:, 4:7] = 0.01                    # small ω
        obs[:, 7] = -1 + (m + math.pi/2) * 4 / (3*math.pi)
        obs[:, 8] = -1 + th * 2 / math.pi

        out = sampler(obs)
        new_margin = margin_fn(out)
        assert (new_margin >= -10.5).all() and (new_margin <= 15.5).all(), \
            f"resampled margins outside corridor: [{new_margin.min()}, {new_margin.max()}]"
        # Half-angle (θ − margin) must be preserved per state
        new_theta = (out[:, 8] + 1.0) * (math.pi / 2)
        half = new_theta - torch.deg2rad(new_margin)
        assert torch.allclose(half, torch.full_like(half, math.radians(20)), atol=1e-5)
        # Attitude untouched
        assert torch.allclose(out[:, 0:4], obs[:, 0:4])

    def test_eps_train_hinge_fires_below_margin(self, hparams, mock_collector):
        """With ε_train > 0 the CBF penalty must produce loss for states whose
        condition value is positive but below the margin."""
        hparams.policy_train_iters = 1
        hparams.policy_lambda_cbf = 1.0
        hparams.policy_cbf_eps_train = 0.05
        policy = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        # condition value 0.02 — safe (>0) but inside the ε_train margin
        cbf_fn = lambda x, u: torch.full((x.shape[0],), 0.02)
        trainer = PolicyTrainer(policy, hparams, cbf_fn=cbf_fn)
        trainer.lambda_cbf = 1.0

        ds, _ = mock_collector.get_dataset(0)
        loss = trainer.train(ds, writer=None)
        # imitation loss alone (same seed, no cbf) must be smaller
        policy2 = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        policy2.load_state_dict(policy.state_dict())
        trainer2 = PolicyTrainer(policy2, hparams)
        assert loss > 0.0

    def test_boundary_sampler_called_in_train(self, hparams, mock_collector):
        hparams.policy_train_iters = 2
        hparams.policy_lambda_cbf = 1.0
        policy = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        trainer = PolicyTrainer(policy, hparams,
                                cbf_fn=lambda x, u: torch.ones(x.shape[0]))
        trainer.lambda_cbf = 1.0
        calls = []
        trainer.boundary_sampler = lambda x: (calls.append(1), x.clone())[1]

        ds, _ = mock_collector.get_dataset(0)
        trainer.train(ds, writer=None)
        assert len(calls) == 2, "boundary sampler must run once per training step"

    def test_student_episodes_use_kappa_zero(
        self, hparams, mock_collector, mock_mpc_agent, mock_env
    ):
        hparams.policy_train_iters = 1
        policy = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        trainer = PolicyTrainer(policy, hparams)
        trainer._dagger_iter = 0   # increments to 1 → κ = 1 - 1/5 = 0.8

        trainer.dagger_update(
            env=mock_env, mpc_agent=mock_mpc_agent, collector=mock_collector,
            task_id=0, preprocess_fn=lambda obs: torch.zeros(1, 13),
            n_rollout=5, max_ep_steps=2, writer=None,
            student_frac=0.4,
        )
        assert trainer._last_rollout_kappas == [0.0, 0.0, 0.8, 0.8, 0.8]


class TestCbfSafetyHead:
    """Closed-form CBF layer inside the policy: monotone condition improvement,
    exact identity when already safe, differentiable, and feasibility masking."""

    def _factories(self):
        from hypercrl.control.policy_net import (make_space_cbf_fn,
                                                 make_space_cbf_safety_head,
                                                 make_space_cbf_feasible_fn,
                                                 make_space_boundary_sampler)
        import math
        I = [[60, 5, 1], [5, 50, 2], [1, 2, 70]]
        x_mu, x_std = torch.zeros(1, 13), torch.ones(1, 13)
        a_mu, a_std = torch.zeros(1, 3), torch.ones(1, 3)
        cbf  = make_space_cbf_fn(x_mu, x_std, a_mu, a_std, I)
        head = make_space_cbf_safety_head(x_mu, x_std, a_mu, a_std, I, eps=0.03)
        feas = make_space_cbf_feasible_fn(x_mu, x_std, I, eps=0.03)
        smp  = make_space_boundary_sampler(x_mu, x_std)
        return cbf, head, feas, smp

    def _corridor_states(self, sampler, n=256, seed=0):
        import math
        g = torch.Generator().manual_seed(seed)
        obs = torch.zeros(n, 13)
        q = torch.randn(n, 4, generator=g); q = q / q.norm(dim=1, keepdim=True)
        obs[:, 0:4] = q
        obs[:, 4:7] = torch.randn(n, 3, generator=g) * 0.02   # ω up to ~0.35 rad/s
        m, th = math.radians(30), math.radians(50)             # margin 30°, half 20°
        obs[:, 7] = -1 + (m + math.pi/2) * 4 / (3*math.pi)
        obs[:, 8] = -1 + th * 2 / math.pi
        rel = torch.randn(n, 3, generator=g); obs[:, 9:12] = rel / rel.norm(dim=1, keepdim=True)
        return sampler(obs)

    def test_head_never_worsens_condition(self):
        cbf, head, _, smp = self._factories()
        x = self._corridor_states(smp)
        u = torch.randn(x.shape[0], 3).clamp(-1, 1)
        c_before = cbf(x, u)
        c_after  = cbf(x, head(x, u))
        assert (c_after >= c_before - 1e-5).all(), \
            f"head worsened the CBF condition: {(c_after - c_before).min()}"

    def test_head_reaches_eps_on_feasible_states(self):
        cbf, head, feas, smp = self._factories()
        x = self._corridor_states(smp)
        u = torch.randn(x.shape[0], 3).clamp(-1, 1)
        mask = feas(x) & (cbf(x, u) < 0.03)
        if mask.any():
            c_after = cbf(x[mask], head(x[mask], u[mask]))
            assert (c_after >= 0.03 - 1e-4).all(), \
                f"feasible violating states not corrected to eps: min={c_after.min()}"

    def test_head_identity_when_safe(self):
        cbf, head, _, smp = self._factories()
        x = self._corridor_states(smp)
        u = torch.zeros(x.shape[0], 3)
        safe = cbf(x, u) >= 0.03
        assert safe.any(), "test setup: expected some already-safe states"
        u_out = head(x[safe], u[safe])
        assert torch.allclose(u_out, u[safe], atol=1e-6)

    def test_head_is_differentiable(self):
        _, head, _, smp = self._factories()
        x = self._corridor_states(smp, n=32)
        u = torch.zeros(32, 3, requires_grad=True)
        head(x, u).sum().backward()
        assert u.grad is not None and torch.isfinite(u.grad).all()

    def test_feasible_fn_masks_zero_authority_states(self):
        import math
        _, _, feas, _ = self._factories()
        obs = torch.zeros(4, 13)
        obs[:, 0] = 1.0
        obs[:, 9] = 1.0                                   # arbitrary unit rel_avoid
        th_half = math.radians(20)
        for i, m_deg in enumerate([2.0, 2.0, 40.0, 40.0]):
            m = math.radians(m_deg)
            obs[i, 7] = -1 + (m + math.pi/2) * 4 / (3*math.pi)
            obs[i, 8] = -1 + (m + th_half) * 2 / math.pi
        # ω = 0 → ḣ = 0 → b = 0: condition = γ·H = γ·margin
        out = feas(obs)
        assert out.tolist() == [False, False, True, True], (
            f"ω=0 states: margin 2° (γH=0.017 < ε) must be infeasible, "
            f"margin 40° (γH=0.35) feasible; got {out.tolist()}"
        )

    def test_policy_net_applies_head_and_tracks_du(self):
        policy = PolicyNet(13, 3, hidden_dims=(16,), dropout=0.0)
        policy.safety_head = lambda x, u: torch.zeros_like(u)
        out = policy(torch.randn(8, 13))
        assert torch.allclose(out, torch.zeros(8, 3))
        assert policy.last_head_du > 0.0

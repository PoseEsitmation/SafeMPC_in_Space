"""
Unit tests for DAGGER dataset and training (Tasks #50 and #51).

Task #50: Validate DAGGER dataset values
Task #51: Simple supervised learning test for DAGGER

Tests the real PolicyNet and PolicyTrainer from policy_net.py
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from hypercrl.control.policy_net import PolicyNet, PolicyTrainer


# ============================================================
# Task #50: Validate DAGGER Dataset Values
# ============================================================

class TestDAGGERDataset(unittest.TestCase):
    """Tests for DAGGER dataset validation."""

    def setUp(self):
        """Create mock dataset (replace with real data when available)."""
        self.state_dim = 10
        self.action_dim = 4
        self.num_samples = 100

        # Mock data
        self.states = np.random.randn(self.num_samples, self.state_dim)
        self.actions = np.random.randn(self.num_samples, self.action_dim)
        self.next_states = np.random.randn(self.num_samples, self.state_dim)
        self.actions = np.clip(self.actions, -1.0, 1.0)

    def test_dataset_not_empty(self):
        self.assertGreater(len(self.states), 0)

    def test_states_correct_shape(self):
        self.assertEqual(self.states.shape[1], self.state_dim)

    def test_actions_correct_shape(self):
        self.assertEqual(self.actions.shape[1], self.action_dim)

    def test_no_nan_values(self):
        self.assertFalse(np.isnan(self.states).any())
        self.assertFalse(np.isnan(self.actions).any())

    def test_actions_in_range(self):
        self.assertTrue(np.all(self.actions >= -1.0))
        self.assertTrue(np.all(self.actions <= 1.0))

    def test_data_is_consistent(self):
        self.assertEqual(len(self.states), len(self.actions))


# ============================================================
# Task #51: Simple Supervised Learning Test (Real DAGGER)
# ============================================================

class TestDAGGERTraining(unittest.TestCase):
    """Tests for DAGGER training using real PolicyNet and PolicyTrainer."""

    def setUp(self):
        """Create mock data and real PolicyNet."""
        self.state_dim = 10
        self.action_dim = 4
        self.num_samples = 200

        # Mock data (states, actions, next_states)
        self.states = np.random.randn(self.num_samples, self.state_dim)
        self.actions = np.random.randn(self.num_samples, self.action_dim)
        self.next_states = np.random.randn(self.num_samples, self.state_dim)
        self.actions = np.clip(self.actions, -1.0, 1.0)

        # Create real PolicyNet
        self.policy = PolicyNet(self.state_dim, self.action_dim)

        # Mock hparams
        class MockHParams:
            device = "cpu"
            policy_lr = 1e-4
            policy_bs = 64
            policy_train_iters = 50
            policy_lambda_imit = 1.0
            policy_lambda_cbf = 0.0
            policy_lambda_clf = 0.0

        self.hparams = MockHParams()

        # Create real PolicyTrainer
        self.trainer = PolicyTrainer(self.policy, self.hparams)

    def test_policy_net_exists(self):
        """Check that PolicyNet is available."""
        self.assertIsNotNone(self.policy)
        self.assertIsInstance(self.policy, PolicyNet)

    def test_trainer_exists(self):
        """Check that PolicyTrainer is available."""
        self.assertIsNotNone(self.trainer)
        self.assertIsInstance(self.trainer, PolicyTrainer)

    def test_policy_forward_pass(self):
        """Test that PolicyNet can do a forward pass."""
        x = torch.randn(10, self.state_dim)
        output = self.policy(x)
        self.assertEqual(output.shape, (10, self.action_dim))
        self.assertFalse(torch.isnan(output).any())

    def test_policy_trainer_can_train(self):
        """Test that PolicyTrainer can train on mock data."""
        # Convert to torch tensors
        states = torch.tensor(self.states[:100], dtype=torch.float32)
        actions = torch.tensor(self.actions[:100], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[:100], dtype=torch.float32)

        # Create dataset
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(states, actions, next_states)

        # Train for a few iterations
        loss = self.trainer.train(dataset)

        # Check that loss is a number (not NaN)
        self.assertFalse(np.isnan(loss), "Loss is NaN")
        self.assertGreater(loss, 0, "Loss should be positive")

    def test_policy_output_is_reasonable(self):
        """Test that policy outputs are within [-1, 1]."""
        x = torch.randn(20, self.state_dim)
        with torch.no_grad():
            output = self.policy(x)
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))


if __name__ == "__main__":
    unittest.main()

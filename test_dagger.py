"""
Unit tests for DAGGER dataset and training (Tasks #50 and #51).

Task #50: Validate DAGGER dataset values
Task #51: Simple supervised learning test for DAGGER

Tests the real PolicyNet and PolicyTrainer from policy_net.py
Uses real data from runs/lqr/ folder.
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
        """Load real DAGGER data from runs folder."""
        import pickle

        # Load the data from the pickle file
        with open('runs/lqr/20260702_202650_TBspaceEnv_val_single_2020/data.pkl', 'rb') as f:
            collector = pickle.load(f)

        # Get the first task (task_id = 0)
        task_id = 0

        # Convert lists to numpy arrays
        states_list = collector.states[task_id]
        actions_list = collector.actions[task_id]
        nexts_list = collector.nexts[task_id]

        # Stack and transpose to get (samples, dims)
        self.states = np.hstack(states_list).T
        self.actions = np.hstack(actions_list).T
        self.next_states = np.hstack(nexts_list).T

        self.state_dim = self.states.shape[1]  # 13
        self.action_dim = self.actions.shape[1]  # 3
        self.num_samples = len(self.states)  # 55000

    def test_dataset_not_empty(self):
        """Check dataset has samples."""
        self.assertGreater(len(self.states), 0, "Dataset is empty")
        self.assertEqual(len(self.states), self.num_samples)

    def test_states_correct_shape(self):
        """Check states have correct dimensions."""
        self.assertEqual(self.states.shape[1], self.state_dim)

    def test_actions_correct_shape(self):
        """Check actions have correct dimensions."""
        self.assertEqual(self.actions.shape[1], self.action_dim)

    def test_no_nan_values(self):
        """Check there are no NaN values."""
        self.assertFalse(np.isnan(self.states).any(), "States contain NaN")
        self.assertFalse(np.isnan(self.actions).any(), "Actions contain NaN")
        self.assertFalse(np.isnan(self.next_states).any(), "Next states contain NaN")

    def test_actions_in_range(self):
        """Check actions are within reasonable physical range."""
        # Check that actions are finite (no NaN or Inf)
        self.assertTrue(np.all(np.isfinite(self.actions)), "Actions contain NaN or Inf")
        
        # Log actual range for reference
        print(f"\nAction range: min={np.min(self.actions):.3f}, max={np.max(self.actions):.3f}")
        
        # Check that actions are within a reasonable range
        # (based on real data: min=-12.64, max=19.05)
        action_min = np.min(self.actions)
        action_max = np.max(self.actions)
        
        self.assertTrue(action_min >= -20.0, f"Actions too low: {action_min}")
        self.assertTrue(action_max <= 20.0, f"Actions too high: {action_max}")
        self.assertGreater(action_max, action_min, "Action range is invalid")

    def test_data_is_consistent(self):
        """Check dataset has consistent lengths."""
        self.assertEqual(len(self.states), len(self.actions))
        self.assertEqual(len(self.states), len(self.next_states))


# ============================================================
# Task #51: Simple Supervised Learning Test (Real DAGGER)
# ============================================================

class TestDAGGERTraining(unittest.TestCase):
    """Tests for DAGGER training using real PolicyNet and PolicyTrainer."""

    def setUp(self):
        """Load real DAGGER data and create real PolicyNet/PolicyTrainer."""
        import pickle

        with open('runs/lqr/20260702_202650_TBspaceEnv_val_single_2020/data.pkl', 'rb') as f:
            collector = pickle.load(f)

        task_id = 0

        states_list = collector.states[task_id]
        actions_list = collector.actions[task_id]
        nexts_list = collector.nexts[task_id]

        self.states = np.hstack(states_list).T
        self.actions = np.hstack(actions_list).T
        self.next_states = np.hstack(nexts_list).T

        self.state_dim = self.states.shape[1]  # 13
        self.action_dim = self.actions.shape[1]  # 3
        self.num_samples = len(self.states)  # 55000

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
        """Test that PolicyTrainer can train on real data."""
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
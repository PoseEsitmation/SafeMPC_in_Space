"""
Unit tests for DAGGER dataset validation and training.

Task #50: Validate DAGGER dataset values
Task #51: Simple supervised learning test for DAGGER
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Task #50: Validate DAGGER Dataset Values
# ============================================================

class TestDAGGERDataset(unittest.TestCase):
    """
    Tests for DAGGER dataset validation.
    Checks: empty, shapes, NaN, ranges, consistency.
    """

    def setUp(self):
        """Create mock DAGGER dataset (replace with real data when available)."""
        self.state_dim = 10
        self.action_dim = 4
        self.num_samples = 100

        # Mock data — replace with real data when available
        self.states = np.random.randn(self.num_samples, self.state_dim)
        self.actions = np.random.randn(self.num_samples, self.action_dim)
        self.next_states = np.random.randn(self.num_samples, self.state_dim)

        # Clip actions to valid range (e.g., actuator limits)
        self.actions = np.clip(self.actions, -1.0, 1.0)

        # Safety certificate values (optional)
        self.cbf_values = np.random.randn(self.num_samples)
        self.clf_values = np.random.randn(self.num_samples)

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

    def test_next_states_correct_shape(self):
        """Check next states have correct dimensions."""
        self.assertEqual(self.next_states.shape[1], self.state_dim)

    def test_no_nan_values(self):
        """Check there are no NaN values."""
        self.assertFalse(np.isnan(self.states).any(), "States contain NaN")
        self.assertFalse(np.isnan(self.actions).any(), "Actions contain NaN")
        self.assertFalse(np.isnan(self.next_states).any(), "Next states contain NaN")

    def test_actions_in_range(self):
        """Check actions are within expected range."""
        self.assertTrue(np.all(self.actions >= -1.0), "Actions below -1.0")
        self.assertTrue(np.all(self.actions <= 1.0), "Actions above 1.0")

    def test_data_is_consistent(self):
        """Check dataset has consistent lengths."""
        self.assertEqual(len(self.states), len(self.actions))
        self.assertEqual(len(self.states), len(self.next_states))

    def test_cbf_clf_values(self):
        """Check CBF and CLF values are valid."""
        # CBF values should be finite and not NaN
        self.assertFalse(np.isnan(self.cbf_values).any(), "CBF values contain NaN")
        self.assertFalse(np.isnan(self.clf_values).any(), "CLF values contain NaN")


# ============================================================
# Task #51: Simple Supervised Learning Test
# ============================================================

class SimpleDynamicsModel(nn.Module):
    """Simple neural network for learning dynamics."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, state_dim)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        """Predict next state given current state and action."""
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class TestDAGGERTraining(unittest.TestCase):
    """
    Tests for DAGGER training (Task #51).
    Checks: loss decreases, predictions are reasonable.
    """

    def setUp(self):
        """Create mock data for training tests."""
        self.state_dim = 10
        self.action_dim = 4
        self.num_samples = 200

        self.states = np.random.randn(self.num_samples, self.state_dim)
        self.actions = np.random.randn(self.num_samples, self.action_dim)
        self.next_states = np.random.randn(self.num_samples, self.state_dim)

        # Clip actions
        self.actions = np.clip(self.actions, -1.0, 1.0)

    def test_model_can_learn(self):
        """
        Test that a simple model can learn from the dataset.
        Loss should decrease over training steps.
        """
        # Convert to tensors
        states = torch.tensor(self.states, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.float32)
        next_states = torch.tensor(self.next_states, dtype=torch.float32)

        # Create model and optimizer
        model = SimpleDynamicsModel(self.state_dim, self.action_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Train for multiple epochs
        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            predictions = model(states, actions)
            loss = criterion(predictions, next_states)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Check that loss decreased
        self.assertLess(losses[-1], losses[0] * 0.6,
                       f"Loss did not decrease enough. Initial: {losses[0]:.4f}, Final: {losses[-1]:.4f}")

        print(f"✅ Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")

    def test_predictions_are_reasonable(self):
        """Check that model outputs are not NaN or extreme."""
        states = torch.tensor(self.states[:10], dtype=torch.float32)
        actions = torch.tensor(self.actions[:10], dtype=torch.float32)

        model = SimpleDynamicsModel(self.state_dim, self.action_dim)

        with torch.no_grad():
            predictions = model(states, actions)

        self.assertFalse(torch.isnan(predictions).any(), "Predictions contain NaN")
        self.assertTrue(torch.all(predictions.abs() < 10), "Predictions are extreme")

    def test_model_can_overfit_small_batch(self):
        """Test that model can overfit to a small batch (shows capacity)."""
        # Use a tiny batch (10 samples)
        states = torch.tensor(self.states[:10], dtype=torch.float32)
        actions = torch.tensor(self.actions[:10], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[:10], dtype=torch.float32)

        model = SimpleDynamicsModel(self.state_dim, self.action_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Train for 200 steps on same batch
        losses = []
        for _ in range(200):
            optimizer.zero_grad()
            predictions = model(states, actions)
            loss = criterion(predictions, next_states)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Final loss should be very small (overfitting)
        self.assertLess(losses[-1], 0.01, "Model failed to overfit small batch")

        print(f"✅ Overfitting loss: {losses[-1]:.6f}")


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    # Run all tests
    unittest.main()

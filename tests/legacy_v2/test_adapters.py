"""Tests for environment adapters."""

import pytest
import torch
import numpy as np

from experiments.cyborg_mind_v2.envs import create_adapter, BrainInputs
from experiments.cyborg_mind_v2.envs.base_adapter import BaseEnvAdapter


class TestAdapterInterface:
    """Test that all adapters implement the correct interface."""

    @pytest.mark.parametrize("adapter_type,env_name", [
        ("gym", "CartPole-v1"),
        # ("minerl", "MineRLTreechop-v0"),  # Requires MineRL installation
    ])
    def test_adapter_creation(self, adapter_type, env_name):
        """Test adapter can be created."""
        adapter = create_adapter(adapter_type, env_name)
        assert adapter is not None

    @pytest.mark.parametrize("adapter_type,env_name", [
        ("gym", "CartPole-v1"),
    ])
    def test_adapter_properties(self, adapter_type, env_name):
        """Test adapter has required properties."""
        adapter = create_adapter(adapter_type, env_name)

        assert hasattr(adapter, "action_space_size")
        assert hasattr(adapter, "observation_shape")
        assert hasattr(adapter, "scalar_dim")
        assert hasattr(adapter, "goal_dim")

        assert isinstance(adapter.action_space_size, int)
        assert adapter.action_space_size > 0

        assert isinstance(adapter.observation_shape, tuple)
        assert len(adapter.observation_shape) == 3

        assert isinstance(adapter.scalar_dim, int)
        assert adapter.scalar_dim > 0

        assert isinstance(adapter.goal_dim, int)
        assert adapter.goal_dim > 0

    @pytest.mark.parametrize("adapter_type,env_name", [
        ("gym", "CartPole-v1"),
    ])
    def test_adapter_reset(self, adapter_type, env_name):
        """Test adapter reset returns valid BrainInputs."""
        adapter = create_adapter(adapter_type, env_name)

        obs = adapter.reset()

        assert isinstance(obs, BrainInputs)
        assert isinstance(obs.pixels, torch.Tensor)
        assert isinstance(obs.scalars, torch.Tensor)
        assert isinstance(obs.goal, torch.Tensor)

        # Check shapes
        assert obs.pixels.shape == adapter.observation_shape
        assert obs.scalars.shape[0] == adapter.scalar_dim
        assert obs.goal.shape[0] == adapter.goal_dim

        # Check dtypes
        assert obs.pixels.dtype == torch.float32
        assert obs.scalars.dtype == torch.float32
        assert obs.goal.dtype == torch.float32

        adapter.close()

    @pytest.mark.parametrize("adapter_type,env_name", [
        ("gym", "CartPole-v1"),
    ])
    def test_adapter_step(self, adapter_type, env_name):
        """Test adapter step returns valid outputs."""
        adapter = create_adapter(adapter_type, env_name)

        obs = adapter.reset()

        # Take a random action
        action = np.random.randint(0, adapter.action_space_size)
        next_obs, reward, done, info = adapter.step(action)

        # Check types
        assert isinstance(next_obs, BrainInputs)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        # Check obs validity
        assert next_obs.pixels.shape == adapter.observation_shape
        assert next_obs.scalars.shape[0] == adapter.scalar_dim
        assert next_obs.goal.shape[0] == adapter.goal_dim

        adapter.close()

    @pytest.mark.parametrize("adapter_type,env_name", [
        ("gym", "CartPole-v1"),
    ])
    def test_adapter_episode(self, adapter_type, env_name):
        """Test running a complete episode."""
        adapter = create_adapter(adapter_type, env_name)

        obs = adapter.reset()
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = np.random.randint(0, adapter.action_space_size)
            obs, reward, done, info = adapter.step(action)
            steps += 1

        assert steps > 0
        adapter.close()


class TestGymAdapter:
    """Gym-specific adapter tests."""

    def test_cartpole_discrete_actions(self):
        """Test CartPole has 2 discrete actions."""
        adapter = create_adapter("gym", "CartPole-v1")
        assert adapter.action_space_size == 2
        adapter.close()

    def test_state_to_scalars(self):
        """Test state is converted to scalars."""
        adapter = create_adapter("gym", "CartPole-v1", use_pixels=False)
        obs = adapter.reset()

        # CartPole has 4 state dimensions
        assert adapter.scalar_dim >= 4
        adapter.close()


class TestBaseEnvAdapter:
    """Test BaseEnvAdapter utilities."""

    def test_preprocess_pixels(self):
        """Test pixel preprocessing."""
        from experiments.cyborg_mind_v2.envs.gym_adapter import GymAdapter

        adapter = GymAdapter("CartPole-v1")

        # Create test image
        test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        # Preprocess
        processed = adapter.preprocess_pixels(test_img, normalize=True)

        # Check shape
        assert processed.shape == (3, 128, 128)

        # Check range [0, 1]
        assert processed.min() >= 0
        assert processed.max() <= 1

        # Check dtype
        assert processed.dtype == torch.float32

        adapter.close()

    def test_normalize_scalars(self):
        """Test scalar normalization."""
        from experiments.cyborg_mind_v2.envs.gym_adapter import GymAdapter

        adapter = GymAdapter("CartPole-v1")

        scalars = np.array([1.0, 2.0, 3.0, 4.0])
        mean = np.array([2.0, 2.0, 2.0, 2.0])
        std = np.array([1.0, 1.0, 1.0, 1.0])

        normalized = adapter.normalize_scalars(scalars, mean, std)

        # Check normalization
        expected = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        assert torch.allclose(normalized, expected, atol=1e-6)

        adapter.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

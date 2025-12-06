"""Unit tests for environment adapters."""

import pytest
import torch
import numpy as np

from cyborg_rl.envs import GymAdapter, BaseEnvAdapter


class TestGymAdapter:
    """Tests for GymAdapter."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def cartpole_env(self, device: torch.device) -> GymAdapter:
        """Create CartPole environment."""
        return GymAdapter(
            env_name="CartPole-v1",
            device=device,
            seed=42,
            normalize_obs=False,
        )

    def test_initialization(self, cartpole_env: GymAdapter) -> None:
        """Test environment initialization."""
        assert cartpole_env.observation_dim == 4
        assert cartpole_env.action_dim == 2
        assert cartpole_env.is_discrete is True

    def test_reset(self, cartpole_env: GymAdapter) -> None:
        """Test environment reset."""
        obs = cartpole_env.reset()
        
        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (4,)
        assert obs.device == cartpole_env.device

    def test_step(self, cartpole_env: GymAdapter) -> None:
        """Test environment step."""
        cartpole_env.reset()
        action = torch.tensor([0], device=cartpole_env.device)
        
        obs, reward, terminated, truncated, info = cartpole_env.step(action)
        
        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_sample_action(self, cartpole_env: GymAdapter) -> None:
        """Test random action sampling."""
        action = cartpole_env.sample_action()
        
        assert isinstance(action, torch.Tensor)
        assert action.device == cartpole_env.device

    def test_episode_rollout(self, cartpole_env: GymAdapter) -> None:
        """Test running a complete episode."""
        obs = cartpole_env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < 100:
            action = cartpole_env.sample_action()
            obs, reward, terminated, truncated, _ = cartpole_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        assert steps > 0
        assert isinstance(total_reward, float)

    def test_close(self, cartpole_env: GymAdapter) -> None:
        """Test environment close."""
        cartpole_env.close()


class TestContinuousEnv:
    """Tests for continuous action space environments."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def pendulum_env(self, device: torch.device) -> GymAdapter:
        """Create Pendulum environment."""
        return GymAdapter(
            env_name="Pendulum-v1",
            device=device,
            seed=42,
            normalize_obs=False,
        )

    def test_continuous_action_space(self, pendulum_env: GymAdapter) -> None:
        """Test continuous action space detection."""
        assert pendulum_env.is_discrete is False
        assert pendulum_env.action_dim == 1

    def test_continuous_step(self, pendulum_env: GymAdapter) -> None:
        """Test step with continuous action."""
        pendulum_env.reset()
        action = torch.tensor([0.5], device=pendulum_env.device)
        
        obs, reward, terminated, truncated, info = pendulum_env.step(action)
        
        assert isinstance(obs, torch.Tensor)
        assert isinstance(reward, float)

    def test_cleanup(self, pendulum_env: GymAdapter) -> None:
        """Cleanup."""
        pendulum_env.close()

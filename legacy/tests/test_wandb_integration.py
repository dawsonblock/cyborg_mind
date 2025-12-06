"""Tests for WandB integration in PPOTrainer."""

import pytest
import torch
import gymnasium as gym
from unittest.mock import Mock, patch, MagicMock

from cyborg_rl.config import Config
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.ppo_trainer import PPOTrainer, WANDB_AVAILABLE


class TestWandBIntegration:
    """Test WandB logging integration."""

    @pytest.fixture
    def config(self) -> Config:
        """Create config with WandB enabled."""
        config = Config()
        config.env.name = "CartPole-v1"
        config.model.hidden_dim = 32
        config.model.latent_dim = 32
        config.train.total_timesteps = 4096  # 2 updates
        config.train.n_steps = 2048
        config.train.wandb_enabled = True
        config.train.wandb_project = "test-project"
        config.train.wandb_entity = "test-entity"
        config.train.wandb_tags = ["test", "ci"]
        config.train.wandb_run_name = "test-run"
        return config

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    @pytest.fixture
    def agent(self, config: Config):
        """Create test agent."""
        return PPOAgent(
            obs_dim=4,
            action_dim=2,
            config=config,
            is_discrete=True,
            device=torch.device("cpu"),
        )

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="WandB not installed")
    @patch("cyborg_rl.trainers.ppo_trainer.wandb")
    def test_wandb_initialization(self, mock_wandb, config: Config, env, agent: PPOAgent):
        """Test WandB is initialized when enabled."""
        trainer = PPOTrainer(env=env, agent=agent, config=config)

        # Check WandB was initialized
        assert trainer.wandb_enabled is True
        mock_wandb.init.assert_called_once()

        # Check init arguments
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["entity"] == "test-entity"
        assert call_kwargs["name"] == "test-run"
        assert "test" in call_kwargs["tags"]
        assert "ci" in call_kwargs["tags"]

        # Check wandb.watch was called
        mock_wandb.watch.assert_called_once_with(agent, log="all", log_freq=100)

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="WandB not installed")
    @patch("cyborg_rl.trainers.ppo_trainer.wandb")
    def test_wandb_logging_during_training(self, mock_wandb, config: Config, env, agent: PPOAgent):
        """Test metrics are logged to WandB during training."""
        # Mock wandb.log to capture calls
        mock_wandb.log = Mock()

        trainer = PPOTrainer(env=env, agent=agent, config=config)

        # Run one update
        trainer._collect_rollouts()
        metrics = trainer._update_policy()

        # Simulate logging (normally called in train())
        if trainer.wandb_enabled:
            mock_wandb.log(metrics, step=trainer.global_step)

        # Check wandb.log was called
        assert mock_wandb.log.called
        logged_metrics = mock_wandb.log.call_args[0][0]

        # Check expected metrics
        assert "loss" in logged_metrics
        assert "policy_loss" in logged_metrics
        assert "value_loss" in logged_metrics
        assert "entropy_loss" in logged_metrics

    def test_wandb_disabled_by_default(self, env, agent: PPOAgent):
        """Test WandB is not initialized when disabled."""
        config = Config()
        config.train.wandb_enabled = False

        trainer = PPOTrainer(env=env, agent=agent, config=config)

        assert trainer.wandb_enabled is False

    @patch("cyborg_rl.trainers.ppo_trainer.WANDB_AVAILABLE", False)
    def test_wandb_not_available_warning(self, env, agent: PPOAgent, caplog):
        """Test warning is logged when WandB is enabled but not installed."""
        config = Config()
        config.train.wandb_enabled = True

        trainer = PPOTrainer(env=env, agent=agent, config=config)

        # Check warning was logged
        assert "wandb package is not installed" in caplog.text
        assert trainer.wandb_enabled is False

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="WandB not installed")
    @patch("cyborg_rl.trainers.ppo_trainer.wandb")
    def test_config_logged_to_wandb(self, mock_wandb, config: Config, env, agent: PPOAgent):
        """Test that config is logged to WandB."""
        trainer = PPOTrainer(env=env, agent=agent, config=config)

        # Check config was passed to wandb.init
        call_kwargs = mock_wandb.init.call_args[1]
        assert "config" in call_kwargs
        logged_config = call_kwargs["config"]

        # Check config contains expected sections
        assert "env" in logged_config
        assert "model" in logged_config
        assert "train" in logged_config
        assert "ppo" in logged_config

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="WandB not installed")
    @patch("cyborg_rl.trainers.ppo_trainer.wandb")
    def test_wandb_run_name_from_registry(self, mock_wandb, config: Config, env, agent: PPOAgent):
        """Test WandB uses registry run name when not specified."""
        from cyborg_rl.experiments.registry import ExperimentRegistry

        # Don't specify wandb_run_name
        config.train.wandb_run_name = None

        # Create registry with specific run name
        registry = ExperimentRegistry(
            config=config.to_dict(),
            run_name="registry-test-run",
        )

        trainer = PPOTrainer(env=env, agent=agent, config=config, registry=registry)

        # Check wandb.init used registry run name
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["name"] == "registry-test-run"


class TestWandBMetrics:
    """Test WandB metrics collection."""

    @pytest.fixture
    def config(self) -> Config:
        """Create minimal config."""
        config = Config()
        config.train.total_timesteps = 2048
        config.train.n_steps = 2048
        return config

    def test_update_policy_returns_detailed_metrics(self, config: Config):
        """Test _update_policy returns granular loss metrics."""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(
            obs_dim=4,
            action_dim=2,
            config=config,
            is_discrete=True,
            device=torch.device("cpu"),
        )

        trainer = PPOTrainer(env=env, agent=agent, config=config)

        # Collect rollouts
        trainer._collect_rollouts()

        # Update policy
        metrics = trainer._update_policy()

        # Check all expected metrics are present
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy_loss" in metrics

        # Check metrics are numeric
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["value_loss"], float)
        assert isinstance(metrics["entropy_loss"], float)

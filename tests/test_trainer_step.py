"""Unit tests for PPO trainer."""

import pytest
import torch
import numpy as np

from cyborg_rl.config import Config
from cyborg_rl.envs import GymAdapter
from cyborg_rl.agents import PPOAgent
from cyborg_rl.trainers import PPOTrainer, RolloutBuffer


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def buffer(self, device: torch.device) -> RolloutBuffer:
        """Create rollout buffer."""
        return RolloutBuffer(
            buffer_size=100,
            obs_dim=4,
            action_dim=2,
            device=device,
            is_discrete=True,
            gamma=0.99,
            gae_lambda=0.95,
        )

    def test_add(self, buffer: RolloutBuffer) -> None:
        """Test adding transitions."""
        for i in range(10):
            buffer.add(
                obs=np.random.randn(4).astype(np.float32),
                action=np.array(0),
                reward=1.0,
                value=0.5,
                log_prob=-0.5,
                done=False,
            )
        
        assert len(buffer) == 10
        assert buffer.ptr == 10

    def test_compute_advantages(self, buffer: RolloutBuffer) -> None:
        """Test GAE computation."""
        for i in range(10):
            buffer.add(
                obs=np.random.randn(4).astype(np.float32),
                action=np.array(0),
                reward=1.0,
                value=0.5,
                log_prob=-0.5,
                done=(i == 9),
            )
        
        buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)
        
        assert buffer.advantages[:10].sum() != 0
        assert buffer.returns[:10].sum() != 0

    def test_get_batches(self, buffer: RolloutBuffer) -> None:
        """Test batch generation."""
        for i in range(64):
            buffer.add(
                obs=np.random.randn(4).astype(np.float32),
                action=np.array(i % 2),
                reward=1.0,
                value=0.5,
                log_prob=-0.5,
                done=False,
            )
        
        buffer.compute_returns_and_advantages(last_value=0.0, last_done=False)
        
        batches = list(buffer.get(batch_size=16))
        
        assert len(batches) == 4
        for batch in batches:
            assert "observations" in batch
            assert "actions" in batch
            assert "advantages" in batch
            assert "returns" in batch

    def test_reset(self, buffer: RolloutBuffer) -> None:
        """Test buffer reset."""
        buffer.add(
            obs=np.random.randn(4).astype(np.float32),
            action=np.array(0),
            reward=1.0,
            value=0.5,
            log_prob=-0.5,
            done=False,
        )
        
        buffer.reset()
        
        assert len(buffer) == 0
        assert buffer.ptr == 0


class TestPPOTrainer:
    """Tests for PPOTrainer."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def config(self) -> Config:
        """Create test config."""
        config = Config()
        config.model.hidden_dim = 32
        config.model.latent_dim = 16
        config.model.num_gru_layers = 1
        config.memory.memory_size = 8
        config.memory.memory_dim = 8
        config.memory.num_read_heads = 1
        config.ppo.rollout_steps = 64
        config.ppo.batch_size = 16
        config.ppo.num_epochs = 2
        config.train.checkpoint_dir = "/tmp/cyborg_test"
        return config

    @pytest.fixture
    def env(self, device: torch.device) -> GymAdapter:
        """Create test environment."""
        return GymAdapter(
            env_name="CartPole-v1",
            device=device,
            seed=42,
        )

    @pytest.fixture
    def agent(self, config: Config, env: GymAdapter, device: torch.device) -> PPOAgent:
        """Create test agent."""
        return PPOAgent(
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=config,
            is_discrete=env.is_discrete,
            device=device,
        )

    @pytest.fixture
    def trainer(
        self, env: GymAdapter, agent: PPOAgent, config: Config
    ) -> PPOTrainer:
        """Create trainer."""
        return PPOTrainer(env=env, agent=agent, config=config)

    def test_collect_rollouts(self, trainer: PPOTrainer) -> None:
        """Test rollout collection."""
        stats = trainer.collect_rollouts()
        
        assert "mean_reward" in stats
        assert "mean_length" in stats
        assert "num_episodes" in stats
        assert len(trainer.rollout_buffer) == trainer.config.ppo.rollout_steps

    def test_train_step(self, trainer: PPOTrainer) -> None:
        """Test single training step."""
        trainer.collect_rollouts()
        stats = trainer.train_step()
        
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy_loss" in stats
        assert "approx_kl" in stats

    def test_gradient_update(self, trainer: PPOTrainer) -> None:
        """Test that gradients are applied."""
        # Get initial parameters
        initial_params = {
            name: p.clone() for name, p in trainer.agent.named_parameters()
        }
        
        # Train
        trainer.collect_rollouts()
        trainer.train_step()
        
        # Check parameters changed
        params_changed = False
        for name, p in trainer.agent.named_parameters():
            if not torch.allclose(initial_params[name], p):
                params_changed = True
                break
        
        assert params_changed

    def test_cleanup(self, env: GymAdapter) -> None:
        """Cleanup."""
        env.close()

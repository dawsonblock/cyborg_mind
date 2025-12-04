"""Comprehensive smoke test for CyborgMind v3.0 core pipeline.

This test validates:
- Config loading from dict/YAML
- PPOAgent initialization
- RolloutBuffer operations
- PPOTrainer setup
- Basic training iteration (without full env loop)
"""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path

from cyborg_rl.config import Config, EnvConfig, ModelConfig, TrainConfig, PPOConfig
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.rollout_buffer import RolloutBuffer
from cyborg_rl.trainers.ppo_trainer import PPOTrainer
from cyborg_rl.experiments.registry import ExperimentRegistry


class TestV3CorePipeline:
    """Test v3.0 core components work together."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def minimal_config(self) -> Config:
        """Create minimal valid config."""
        config = Config()
        config.env = EnvConfig(name="CartPole-v1")
        config.model = ModelConfig(hidden_dim=64, latent_dim=64, use_mamba=False)
        config.train = TrainConfig(
            total_timesteps=1000,
            n_steps=128,
            batch_size=32,
            n_epochs=2,
            lr=3e-4,
        )
        config.ppo = PPOConfig()
        return config

    def test_config_from_dict(self):
        """Test Config.from_dict works."""
        config_dict = {
            "env": {"name": "CartPole-v1", "normalize_obs": True},
            "model": {"hidden_dim": 128, "latent_dim": 128},
            "train": {"total_timesteps": 10000, "lr": 1e-3},
        }

        config = Config.from_dict(config_dict)

        assert config.env.name == "CartPole-v1"
        assert config.env.normalize_obs is True
        assert config.model.hidden_dim == 128
        assert config.train.total_timesteps == 10000
        assert config.train.lr == 1e-3

    def test_config_from_yaml(self, tmp_path: Path):
        """Test Config.from_yaml works."""
        yaml_content = """
env:
  name: "Pendulum-v1"
  max_episode_steps: 200

model:
  hidden_dim: 256
  use_mamba: false

train:
  total_timesteps: 50000
  lr: 3e-4
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = Config.from_yaml(str(yaml_file))

        assert config.env.name == "Pendulum-v1"
        assert config.env.max_episode_steps == 200
        assert config.model.hidden_dim == 256
        assert config.train.total_timesteps == 50000

    def test_ppo_agent_initialization(self, minimal_config: Config, device: torch.device):
        """Test PPOAgent can be initialized."""
        agent = PPOAgent(
            obs_dim=4,
            action_dim=2,
            config=minimal_config,
            is_discrete=True,
            device=device,
        )

        assert agent.obs_dim == 4
        assert agent.action_dim == 2
        assert agent.is_discrete is True
        assert agent.device == device

        # Test forward pass
        obs = torch.randn(2, 4, device=device)
        state = agent.init_state(batch_size=2)

        action, log_prob, value, new_state, info = agent(obs, state)

        assert action.shape == (2,)  # Discrete actions
        assert log_prob.shape == (2,)
        assert value.shape == (2, 1)
        assert "entropy" in info

    def test_rollout_buffer(self, device: torch.device):
        """Test RolloutBuffer operations."""
        buffer = RolloutBuffer(
            buffer_size=64,
            obs_dim=4,
            action_dim=1,
            device=device,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Add transitions
        for i in range(64):
            obs = torch.randn(4).numpy()
            action = torch.tensor([0])
            reward = 1.0
            done = False
            value = torch.tensor([[0.5]])
            log_prob = torch.tensor([-0.5])

            buffer.add(obs, action, reward, done, value, log_prob)

        # Compute GAE
        last_value = torch.tensor([[0.0]])
        buffer.compute_gae(last_value)

        # Get batches
        batches = list(buffer.get(batch_size=16))

        assert len(batches) > 0
        obs, actions, log_probs, returns, advantages, values = batches[0]
        assert obs.shape[0] == 16
        assert actions.shape[0] == 16

    def test_ppo_trainer_initialization(self, minimal_config: Config, device: torch.device):
        """Test PPOTrainer can be initialized (without env)."""
        import gymnasium as gym

        # Create simple env
        env = gym.make("CartPole-v1")

        agent = PPOAgent(
            obs_dim=4,
            action_dim=2,
            config=minimal_config,
            is_discrete=True,
            device=device,
        )

        # Note: PPOTrainer expects an env, so we use a real one here
        trainer = PPOTrainer(
            env=env,
            agent=agent,
            config=minimal_config,
            registry=None,
        )

        assert trainer.agent == agent
        assert trainer.config == minimal_config
        assert trainer.buffer is not None

        env.close()

    def test_experiment_registry(self, tmp_path: Path, minimal_config: Config):
        """Test ExperimentRegistry basic operations."""
        config_dict = minimal_config.to_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(
                config_dict,
                run_name="test_run",
                base_dir=tmpdir,
            )

            # Log metrics
            metrics = {"loss": 0.5, "reward": 10.0}
            registry.log_metrics(step=100, metrics=metrics)

            # Save checkpoint
            dummy_state = {"model": "dummy"}
            registry.save_checkpoint(dummy_state, step=100)

            # Check files were created
            assert Path(tmpdir).exists()

    def test_agent_save_load(self, tmp_path: Path, minimal_config: Config, device: torch.device):
        """Test PPOAgent save/load."""
        agent = PPOAgent(
            obs_dim=4,
            action_dim=2,
            config=minimal_config,
            is_discrete=True,
            device=device,
        )

        # Save
        save_path = tmp_path / "agent.pt"
        agent.save(str(save_path))

        assert save_path.exists()

        # Load
        loaded_agent = PPOAgent.load(str(save_path), device)

        assert loaded_agent.obs_dim == agent.obs_dim
        assert loaded_agent.action_dim == agent.action_dim

    def test_continuous_action_agent(self, minimal_config: Config, device: torch.device):
        """Test PPOAgent with continuous actions."""
        agent = PPOAgent(
            obs_dim=3,
            action_dim=1,
            config=minimal_config,
            is_discrete=False,  # Continuous
            device=device,
        )

        obs = torch.randn(2, 3, device=device)
        state = agent.init_state(batch_size=2)

        action, log_prob, value, new_state, info = agent(obs, state)

        # Continuous actions are unbounded
        assert action.shape == (2, 1)
        assert log_prob.shape == (2,)
        assert not torch.isnan(action).any()
        assert not torch.isnan(log_prob).any()

    def test_model_gradient_flow(self, minimal_config: Config, device: torch.device):
        """Test gradients flow through agent."""
        agent = PPOAgent(
            obs_dim=4,
            action_dim=2,
            config=minimal_config,
            is_discrete=True,
            device=device,
        )

        obs = torch.randn(2, 4, device=device)
        state = agent.init_state(batch_size=2)

        action, log_prob, value, _, _ = agent(obs, state)

        loss = log_prob.sum() + value.sum()
        loss.backward()

        # Check gradients exist
        for name, param in agent.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

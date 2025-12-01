import torch
import pytest
from unittest.mock import MagicMock, patch
from cyborg_mind_v2.training.trainer import CyborgTrainer
from cyborg_mind_v2.configs.schema import Config, EnvConfig, PMMConfig, ModelConfig, PPOConfig, TrainConfig, WandBConfig
from cyborg_mind_v2.capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind
from cyborg_mind_v2.envs.base_adapter import BrainInputs

def test_trainer_collect_rollouts():
    """Test that trainer collects rollouts correctly."""
    # Mock Env
    env = MagicMock()
    env.scalar_dim = 20
    env.goal_dim = 4
    env.action_space_size = 5
    
    # Mock Reset
    env.reset.return_value = BrainInputs(
        pixels=torch.randn(3, 128, 128),
        scalars=torch.randn(20),
        goal=torch.randn(4)
    )
    
    # Mock Step
    env.step.return_value = (
        BrainInputs(
            pixels=torch.randn(3, 128, 128),
            scalars=torch.randn(20),
            goal=torch.randn(4)
        ),
        1.0, # Reward
        False, # Done
        {} # Info
    )
    
    # Config
    cfg = Config()
    cfg.ppo.rollout_steps = 10
    cfg.train.device = "cpu"
    cfg.model.hidden_dim = 32
    cfg.model.thought_dim = 32
    cfg.model.emotion_dim = 8
    cfg.model.workspace_dim = 64
    cfg.model.vision_dim = 32
    cfg.model.emb_dim = 32
    
    # Brain
    brain = BrainCyborgMind(
        scalar_dim=20,
        goal_dim=4,
        num_actions=5,
        vision_dim=32, # Small for speed
        hidden_dim=32,
        emb_dim=32,
        thought_dim=32,
        emotion_dim=8,
        workspace_dim=64
    )
    
    # Trainer
    with patch("cyborg_mind_v2.training.trainer.wandb") as mock_wandb:
        trainer = CyborgTrainer(env, brain, cfg, device="cpu")
        
        # Run collect
        stats = trainer.collect_rollouts()
        
        assert len(trainer.rollout_buffer) == 10
        assert "mean_reward" in stats
        assert stats["mean_reward"] == 0.0 # No episode finished in 10 steps if done=False

def test_trainer_train_step():
    """Test that trainer performs optimization step."""
    # Setup similar to above
    env = MagicMock()
    cfg = Config()
    cfg.ppo.rollout_steps = 2
    cfg.ppo.batch_size = 2
    cfg.ppo.num_epochs = 1
    cfg.train.device = "cpu"
    
    brain = BrainCyborgMind(
        scalar_dim=20,
        goal_dim=4,
        num_actions=5,
        vision_dim=32,
        hidden_dim=32,
        emb_dim=32
    )
    
    with patch("cyborg_mind_v2.training.trainer.wandb") as mock_wandb:
        trainer = CyborgTrainer(env, brain, cfg, device="cpu")
        
        # Manually populate buffer
        for _ in range(2):
            trainer.rollout_buffer.append({
                "pixels": torch.randn(1, 3, 128, 128),
                "scalars": torch.randn(1, 20),
                "goal": torch.randn(1, 4),
                "state": {
                    "thought": torch.randn(1, 32),
                    "emotion": torch.randn(1, 8),
                    "workspace": torch.randn(1, 64),
                    "hidden": (torch.randn(1, 1, 32), torch.randn(1, 1, 32))
                },
                "action": torch.tensor([0]),
                "log_prob": torch.tensor([0.0]),
                "value": torch.tensor([0.0]),
                "reward": 1.0,
                "done": False,
                "advantage": torch.tensor([0.0]),
                "return": torch.tensor([0.0])
            })
            
        # Run train step
        initial_weights = brain.action_head.weight.clone()
        trainer.train_step()
        
        # Check weights changed (optimizer step)
        # Note: might not change if loss is 0, but PPO usually has some loss.
        # With random data, it should change.
        assert not torch.allclose(brain.action_head.weight, initial_weights)

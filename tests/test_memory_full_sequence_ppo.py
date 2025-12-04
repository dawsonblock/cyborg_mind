#!/usr/bin/env python3
"""
Test suite for MemoryPPOTrainer with full-sequence BPTT.

Verifies:
1. MemoryPPOTrainer can complete training on Delayed Cue task
2. Agent learns (reward improvement over training)
3. Full-sequence BPTT wiring (gradients flow through sequences)
"""

import sys
from pathlib import Path
import torch
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyborg_rl.config import Config
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.memory_ppo_trainer import MemoryPPOTrainer
from cyborg_rl.memory_benchmarks.delayed_cue_env import VectorizedDelayedCueEnv
from cyborg_rl.utils.device import get_device
from cyborg_rl.utils.seeding import set_seed
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


@pytest.fixture
def config():
    """Create a minimal config for testing."""
    config_dict = {
        "env": {"name": "delayed_cue_test"},
        "model": {
            "encoder_type": "gru",
            "hidden_dim": 32,
            "latent_dim": 32,
            "num_gru_layers": 1,
            "use_mamba": False,
            "dropout": 0.0,
        },
        "memory": {
            "memory_size": 8,
            "memory_dim": 16,
            "num_read_heads": 2,
            "num_write_heads": 1,
            "sharp_factor": 1.0,
        },
        "train": {
            "total_timesteps": 5000,
            "n_steps": 128,
            "n_epochs": 2,
            "batch_size": 64,
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_amp": False,
            "weight_decay": 0.0,
            "save_freq": 1000,
            "wandb_enabled": False,
        },
        "api": {"enabled": False},
    }
    return Config.from_dict(config_dict)


@pytest.fixture
def env():
    """Create a small vectorized delayed cue environment."""
    return VectorizedDelayedCueEnv(num_envs=4, num_cues=4, horizon=20)


@pytest.fixture
def agent(config, env):
    """Create a PPOAgent."""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = get_device("cpu")

    return PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        is_discrete=True,
        device=device,
    )


def test_memory_trainer_initialization(config, env, agent):
    """Test that MemoryPPOTrainer can be initialized."""
    trainer = MemoryPPOTrainer(
        config=config,
        env_adapter=env,
        agent=agent,
    )

    assert trainer.num_envs == 4
    assert trainer.episode_len > 20  # horizon + slack
    assert trainer.device == agent.device


def test_rollout_collection(config, env, agent):
    """Test that rollout collection produces correct tensor shapes."""
    trainer = MemoryPPOTrainer(
        config=config,
        env_adapter=env,
        agent=agent,
    )

    rollout = trainer.collect_rollout()

    # Check shapes
    assert rollout.observations.shape[1] == trainer.num_envs
    assert rollout.actions.shape[1] == trainer.num_envs
    assert rollout.rewards.shape[1] == trainer.num_envs
    assert rollout.dones.shape[1] == trainer.num_envs
    assert rollout.values.shape[1] == trainer.num_envs
    assert rollout.log_probs.shape[1] == trainer.num_envs
    assert rollout.advantages.shape[1] == trainer.num_envs
    assert rollout.returns.shape[1] == trainer.num_envs

    # All tensors should have same time dimension
    T = rollout.observations.shape[0]
    assert rollout.actions.shape[0] == T
    assert rollout.rewards.shape[0] == T


def test_policy_update(config, env, agent):
    """Test that policy update runs without errors."""
    trainer = MemoryPPOTrainer(
        config=config,
        env_adapter=env,
        agent=agent,
    )

    # Collect rollout
    rollout = trainer.collect_rollout()

    # Update policy
    metrics = trainer.update_policy(rollout)

    # Check metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "mean_reward" in metrics

    # Losses should be finite
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics["value_loss"]))


def test_full_sequence_bptt(config, env, agent):
    """Test that gradients flow through full sequences."""
    set_seed(42)

    trainer = MemoryPPOTrainer(
        config=config,
        env_adapter=env,
        agent=agent,
    )

    # Store initial parameters
    initial_params = {
        name: param.clone()
        for name, param in agent.named_parameters()
    }

    # Run one training cycle
    rollout = trainer.collect_rollout()
    metrics = trainer.update_policy(rollout)

    # Check metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "mean_reward" in metrics

    # Losses should be finite
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics["value_loss"]))
    # Check metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "mean_reward" in metrics
    # Check that parameters changed (gradients flowed)
    changes = 0
    for name, param in agent.named_parameters():
        if not torch.allclose(param, initial_params[name], atol=1e-6):
            changes += 1

    assert changes > 0, "No parameters changed - gradients may not be flowing!"
    logger.info(f"Parameters changed: {changes} / {len(initial_params)}")


def test_learning_on_delayed_cue(config, env, agent):
    """Test that agent learns on Delayed Cue task (reward improvement)."""
    set_seed(42)

    # Extended training for learning
    config.train.total_timesteps = 10000

    trainer = MemoryPPOTrainer(
        config=config,
        env_adapter=env,
        agent=agent,
    )

    # Measure initial performance
    initial_rollout = trainer.collect_rollout()
    initial_reward = initial_rollout.rewards.sum(dim=0).mean().item()

    # Train
    trainer.train()

    # Measure final performance
    final_rollout = trainer.collect_rollout()
    final_reward = final_rollout.rewards.sum(dim=0).mean().item()

    logger.info(f"Initial reward: {initial_reward:.4f}")
    logger.info(f"Final reward: {final_reward:.4f}")
    logger.info(f"Improvement: {final_reward - initial_reward:.4f}")

    # Assert improvement (may be noisy, but should see some learning)
    # On Delayed Cue with horizon=20, random policy gets ~0.25, trained gets closer to 1.0
    assert final_reward > initial_reward or final_reward > 0.3, \
        f"Agent did not learn: initial={initial_reward:.4f}, final={final_reward:.4f}"


def test_forward_sequence_method(config, env, agent):
    """Test that PPOAgent.forward_sequence works correctly."""
    T = 10
    B = 4
    obs_dim = env.observation_space.shape[0]

    # Create random observation sequence
    obs_seq = torch.randn(T, B, obs_dim, device=agent.device)

    # Forward through sequence
    logits_seq, values_seq = agent.forward_sequence(obs_seq, init_state=None)

    # Check shapes
    assert logits_seq.shape == (T, B, agent.action_dim)
    assert values_seq.shape == (T, B)

    # Check values are finite
    assert torch.isfinite(logits_seq).all()
    assert torch.isfinite(values_seq).all()


def test_forward_step_method(config, env, agent):
    """Test that PPOAgent.forward_step works correctly."""
    B = 4
    obs_dim = env.observation_space.shape[0]

    # Create random observation
    obs = torch.randn(B, obs_dim, device=agent.device)
    state = agent.init_state(batch_size=B)

    # Forward step
    logits, value, new_state, pmm_info = agent.forward_step(obs, state)

    # Check shapes
    assert logits.shape == (B, agent.action_dim)
    assert value.shape == (B,)

    # Check state structure
    assert "hidden" in new_state
    assert "memory" in new_state

    # Check PMM info
    assert isinstance(pmm_info, dict)

    # Check values are finite
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


if __name__ == "__main__":
    # Run tests manually if not using pytest
    print("=" * 80)
    print("MEMORY FULL-SEQUENCE PPO TEST SUITE")
    print("=" * 80)

    # Create fixtures manually
    config_dict = {
        "env": {"name": "delayed_cue_test"},
        "model": {
            "encoder_type": "gru",
            "hidden_dim": 32,
            "latent_dim": 32,
            "num_gru_layers": 1,
            "use_mamba": False,
            "dropout": 0.0,
        },
        "memory": {
            "memory_size": 8,
            "memory_dim": 16,
            "num_read_heads": 2,
            "num_write_heads": 1,
            "sharp_factor": 1.0,
        },
        "train": {
            "total_timesteps": 5000,
            "n_steps": 128,
            "n_epochs": 2,
            "batch_size": 64,
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_amp": False,
            "weight_decay": 0.0,
            "save_freq": 1000,
            "wandb_enabled": False,
        },
        "api": {"enabled": False},
    }
    config = Config.from_dict(config_dict)
    env = VectorizedDelayedCueEnv(num_envs=4, num_cues=4, horizon=20)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = get_device("cpu")
    agent = PPOAgent(obs_dim, action_dim, config, is_discrete=True, device=device)

    tests = [
        ("Initialization", lambda: test_memory_trainer_initialization(config, env, agent)),
        ("Rollout Collection", lambda: test_rollout_collection(config, env, agent)),
        ("Policy Update", lambda: test_policy_update(config, env, agent)),
        ("Full-Sequence BPTT", lambda: test_full_sequence_bptt(config, env, agent)),
        ("Forward Sequence Method", lambda: test_forward_sequence_method(config, env, agent)),
        ("Forward Step Method", lambda: test_forward_step_method(config, env, agent)),
        ("Learning on Delayed Cue", lambda: test_learning_on_delayed_cue(config, env, agent)),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\nRunning: {name}")
            test_fn()
            print(f"✓ PASSED: {name}")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    env.close()
    sys.exit(0 if failed == 0 else 1)

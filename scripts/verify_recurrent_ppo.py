#!/usr/bin/env python3
"""
Verification script for recurrent PPO mode.

Tests:
1. Buffer size calculation for vectorized envs
2. State storage and retrieval in recurrent mode
3. Training loop completes without errors
4. Gradients flow properly through recurrent states

Usage:
    python scripts/verify_recurrent_ppo.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyborg_rl.config import Config
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.ppo_trainer import PPOTrainer
from cyborg_rl.memory_benchmarks.delayed_cue_env import VectorizedDelayedCueEnv
from cyborg_rl.utils.device import get_device
from cyborg_rl.utils.seeding import set_seed
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


def get_test_config_dict(n_steps=128):
    return {
        "memory_size": 8,
        "memory_dim": 16,
        "train": {
            "n_steps": n_steps,
            "n_epochs": 1,
            "batch_size": 4,
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "recurrent_mode": "burn_in",
            "use_amp": False,
            "weight_decay": 0.0,
            "save_freq": 10,
        },
        "api": {"enabled": False},
    }
    config = Config.from_dict(config_dict)

    env = VectorizedDelayedCueEnv(num_envs=num_envs, num_cues=4, horizon=10)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = get_device("cpu")
    agent = PPOAgent(obs_dim, action_dim, config, is_discrete=True, device=device)
    trainer = PPOTrainer(config=config, env=env, agent=agent)

    # Collect rollouts
    trainer._collect_rollouts()

    # Check buffer has stored states
    expected_transitions = n_steps * num_envs
    actual_transitions = len(trainer.buffer)

    logger.info(f"  Expected transitions: {expected_transitions}")
    logger.info(f"  Actual transitions: {actual_transitions}")

    assert actual_transitions == expected_transitions, \
        f"Buffer size mismatch: {actual_transitions} != {expected_transitions}"

    # Check states are stored (for recurrent buffer)
    if hasattr(trainer.buffer, 'states'):
        non_none_states = sum(1 for s in trainer.buffer.states if s is not None)
        logger.info(f"  Non-None states stored: {non_none_states} / {actual_transitions}")
        assert non_none_states == actual_transitions, \
            f"Not all states were stored: {non_none_states} != {actual_transitions}"

        # Check state structure
        sample_state = trainer.buffer.states[0]
        logger.info(f"  Sample state keys: {list(sample_state.keys())}")
        assert 'hidden' in sample_state, "State should have 'hidden' key"
        assert 'memory' in sample_state, "State should have 'memory' key"

        logger.info("✓ TEST 2 PASSED: States properly stored\n")
    else:
        logger.warning("⚠ TEST 2 SKIPPED: Buffer doesn't have states attribute (non-recurrent mode?)\n")

    env.close()
    return True


def test_training_loop():
    """Test that a minimal training loop completes without errors."""
    logger.info("=" * 80)
    logger.info("TEST 3: Training Loop Execution")
    logger.info("=" * 80)

    num_envs = 4
    n_steps = 32
    total_timesteps = n_steps * num_envs * 2  # 2 updates

    config_dict = {
        "env": {"name": "test"},
        "model": {
            "encoder_type": "gru",
            "hidden_dim": 64,
            "latent_dim": 64,
            "num_gru_layers": 1,
            "use_mamba": False,
        },
        "memory": {
            "memory_size": 16,
            "memory_dim": 32,
        },
        "train": {
            "total_timesteps": total_timesteps,
            "n_steps": n_steps,
            "n_epochs": 2,
            "batch_size": 32,
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "recurrent_mode": "burn_in",
            "use_amp": False,
            "weight_decay": 0.0,
            "save_freq": 100,
            "wandb_enabled": False,
        },
        "api": {"enabled": False},
    }
    config = Config.from_dict(config_dict)

    env = VectorizedDelayedCueEnv(num_envs=num_envs, num_cues=4, horizon=10)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = get_device("cpu")
    set_seed(42)

    agent = PPOAgent(obs_dim, action_dim, config, is_discrete=True, device=device)
    trainer = PPOTrainer(config=config, env=env, agent=agent)

    logger.info(f"  Training for {total_timesteps} timesteps ({total_timesteps // (n_steps * num_envs)} updates)")

    try:
        trainer.train()
        logger.info(f"  Final global_step: {trainer.global_step}")
        logger.info("✓ TEST 3 PASSED: Training loop completed successfully\n")
        env.close()
        return True
    except Exception as e:
        logger.error(f"✗ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        env.close()
        return False


def test_gradient_flow():
    """Test that gradients flow through recurrent states."""
    logger.info("=" * 80)
    logger.info("TEST 4: Gradient Flow Check")
    logger.info("=" * 80)

    config_dict = {
        "env": {"name": "test"},
        "model": {
            "encoder_type": "gru",
            "hidden_dim": 32,
            "latent_dim": 32,
            "num_gru_layers": 1,
            "use_mamba": False,
        },
        "memory": {
            "memory_size": 8,
            "memory_dim": 16,
        },
        "train": {
            "n_steps": 16,
            "n_epochs": 1,
            "batch_size": 8,
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "recurrent_mode": "burn_in",
            "use_amp": False,
            "weight_decay": 0.0,
            "save_freq": 10,
        },
        "api": {"enabled": False},
    }
    config = Config.from_dict(config_dict)

    env = VectorizedDelayedCueEnv(num_envs=2, num_cues=4, horizon=10)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = get_device("cpu")
    agent = PPOAgent(obs_dim, action_dim, config, is_discrete=True, device=device)

    # Store initial param norms
    initial_norms = {name: param.norm().item() for name, param in agent.named_parameters()}

    trainer = PPOTrainer(config=config, env=env, agent=agent)

    # Run one training cycle
    trainer._collect_rollouts()
    metrics = trainer._update_policy()

    # Check param norms changed (gradients flowed)
    final_norms = {name: param.norm().item() for name, param in agent.named_parameters()}

    changes = 0
    for name in initial_norms:
        if abs(final_norms[name] - initial_norms[name]) > 1e-6:
            changes += 1

    logger.info(f"  Parameters changed: {changes} / {len(initial_norms)}")
    logger.info(f"  Loss: {metrics.get('loss', 0):.4f}")

    assert changes > 0, "No parameters changed - gradients may not be flowing!"
    logger.info("✓ TEST 4 PASSED: Gradients are flowing\n")

    env.close()
    return True


def main():
    """Run all verification tests."""
    logger.info("\n" + "=" * 80)
    logger.info("RECURRENT PPO VERIFICATION SUITE")
    logger.info("=" * 80 + "\n")

    tests = [
        ("Buffer Size Calculation", test_buffer_sizes),
        ("State Storage", test_state_storage),
        ("Training Loop", test_training_loop),
        ("Gradient Flow", test_gradient_flow),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"✗ TEST FAILED: {name}")
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"  {name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Recurrent PPO is working correctly.")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} TEST(S) FAILED. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

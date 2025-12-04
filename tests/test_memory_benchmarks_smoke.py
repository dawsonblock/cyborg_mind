#!/usr/bin/env python
"""Smoke test for memory benchmark suite.

Quick verification that memory benchmarks can be instantiated and run.
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from cyborg_rl.memory_benchmarks import DelayedCueEnv, VectorizedDelayedCueEnv


def test_delayed_cue_env():
    """Test delayed cue environment."""
    print("Testing DelayedCueEnv...")
    
    env = DelayedCueEnv(num_cues=4, horizon=10)
    
    # Reset
    obs, info = env.reset(seed=42)
    expected_shape = (4,)
    assert obs.shape == expected_shape, "Expected obs shape {}, got {}".format(expected_shape, obs.shape)
    assert info["phase"] == "cue"
    
    # Step through episode
    total_reward = 0
    for _ in range(15):  # horizon=10 + cue + query
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print("  [OK] Single environment completed (reward: {})".format(total_reward))
    env.close()


def test_vectorized_delayed_cue():
    """Test vectorized delayed cue environment."""
    print("Testing VectorizedDelayedCueEnv...")
    
    num_envs = 4
    env = VectorizedDelayedCueEnv(num_envs=num_envs, num_cues=4, horizon=10)
    
    # Reset
    obs, infos = env.reset(seed=42)
    expected_shape = (num_envs, 4)
    assert obs.shape == expected_shape, "Expected obs shape {}, got {}".format(expected_shape, obs.shape)
    
    # Step multiple times
    for _ in range(15):
        actions = np.random.randint(0, 4, size=num_envs)
        # VectorEnv.step() takes actions directly
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        assert obs.shape == (num_envs, 4)
        assert rewards.shape == (num_envs,)
    
    print("  [OK] Vectorized environment completed ({} envs)".format(num_envs))
    env.close()


def test_memory_benchmark_import():
    """Test that memory benchmark suite can be imported."""
    print("Testing import of memory benchmark suite...")
    
    try:
        from cyborg_rl.memory_benchmarks import pseudo_mamba_memory_suite
        print("  [OK] Memory benchmark suite imported successfully")
    except Exception as e:
        print("  [FAIL] Failed to import: {}".format(e))
        raise


def test_encoder_types():
    """Test that different encoder types can be selected."""
    print("Testing encoder type selection...")
    
    from cyborg_rl.config import Config
    from cyborg_rl.agents import PPOAgent
    
    obs_dim = 10
    action_dim = 4
    device = torch.device("cpu")
    
    # Test GRU encoder
    config_gru = Config()
    config_gru.model.encoder_type = "gru"
    config_gru.model.use_mamba = False
    
    agent_gru = PPOAgent(obs_dim, action_dim, config_gru, device=device)
    print("  [OK] GRU encoder created")
    
    # Test Mamba+GRU encoder
    config_mamba = Config()
    config_mamba.model.encoder_type = "mamba_gru"
    config_mamba.model.use_mamba = False  # Will use GRU fallback
    
    agent_mamba = PPOAgent(obs_dim, action_dim, config_mamba, device=device)
    print("  [OK] Mamba+GRU encoder created")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Memory Benchmark Smoke Tests")
    print("=" * 60)
    
    try:
        test_delayed_cue_env()
        test_vectorized_delayed_cue()
        test_memory_benchmark_import()
        test_encoder_types()
        
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("=" * 60)
        print("Tests failed: {}".format(e))
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

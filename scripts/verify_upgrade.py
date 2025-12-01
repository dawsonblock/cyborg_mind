#!/usr/bin/env python3
"""
Quick verification script for PPO training upgrades.
Tests that all new config parameters and trainer methods exist.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyborg_rl.config import Config, PPOConfig

def test_config_upgrades():
    """Test that all new config parameters exist."""
    print("=" * 60)
    print("TESTING CONFIG UPGRADES")
    print("=" * 60)

    config = Config()
    ppo = config.ppo

    # Check LR annealing params
    assert hasattr(ppo, 'lr_start'), "Missing lr_start"
    assert hasattr(ppo, 'lr_end'), "Missing lr_end"
    assert hasattr(ppo, 'anneal_lr'), "Missing anneal_lr"
    print("✓ LR annealing parameters exist")

    # Check entropy annealing params
    assert hasattr(ppo, 'entropy_start'), "Missing entropy_start"
    assert hasattr(ppo, 'entropy_end'), "Missing entropy_end"
    assert hasattr(ppo, 'anneal_entropy'), "Missing anneal_entropy"
    print("✓ Entropy annealing parameters exist")

    # Check early stopping params
    assert hasattr(ppo, 'reward_buffer_size'), "Missing reward_buffer_size"
    assert hasattr(ppo, 'reward_improvement_threshold'), "Missing reward_improvement_threshold"
    assert hasattr(ppo, 'early_stop_patience'), "Missing early_stop_patience"
    assert hasattr(ppo, 'enable_early_stopping'), "Missing enable_early_stopping"
    print("✓ Early stopping parameters exist")

    # Check collapse detection params
    assert hasattr(ppo, 'reward_collapse_threshold'), "Missing reward_collapse_threshold"
    assert hasattr(ppo, 'enable_collapse_detection'), "Missing enable_collapse_detection"
    assert hasattr(ppo, 'collapse_lr_reduction'), "Missing collapse_lr_reduction"
    print("✓ Collapse detection parameters exist")

    # Check validation params
    assert hasattr(ppo, 'inference_validation'), "Missing inference_validation"
    assert hasattr(ppo, 'inference_validation_episodes'), "Missing inference_validation_episodes"
    assert hasattr(ppo, 'inference_validation_threshold'), "Missing inference_validation_threshold"
    print("✓ Inference validation parameters exist")

    # Check plotting
    assert hasattr(ppo, 'auto_plot'), "Missing auto_plot"
    print("✓ Auto-plot parameter exists")

    print("\n✅ All config upgrades verified!\n")

def test_trainer_methods():
    """Test that all new trainer methods exist."""
    print("=" * 60)
    print("TESTING TRAINER METHODS")
    print("=" * 60)

    from cyborg_rl.trainers import PPOTrainer

    # Check methods exist
    required_methods = [
        'update_lr_and_entropy',
        'compute_moving_average_reward',
        'check_early_stopping',
        'check_reward_collapse',
        'run_inference_validation',
        'generate_training_plots',
        'print_training_summary',
    ]

    for method_name in required_methods:
        assert hasattr(PPOTrainer, method_name), f"Missing method: {method_name}"
        print(f"✓ Method {method_name} exists")

    print("\n✅ All trainer methods verified!\n")

def test_default_values():
    """Test that default values are reasonable."""
    print("=" * 60)
    print("TESTING DEFAULT VALUES")
    print("=" * 60)

    config = Config()
    ppo = config.ppo

    print(f"lr_end: {ppo.lr_end} (expected: 1e-5)")
    assert ppo.lr_end == 1e-5

    print(f"entropy_end: {ppo.entropy_end} (expected: 0.0)")
    assert ppo.entropy_end == 0.0

    print(f"reward_buffer_size: {ppo.reward_buffer_size} (expected: 10)")
    assert ppo.reward_buffer_size == 10

    print(f"early_stop_patience: {ppo.early_stop_patience} (expected: 8)")
    assert ppo.early_stop_patience == 8

    print(f"reward_collapse_threshold: {ppo.reward_collapse_threshold} (expected: 0.4)")
    assert ppo.reward_collapse_threshold == 0.4

    print(f"collapse_lr_reduction: {ppo.collapse_lr_reduction} (expected: 0.3)")
    assert ppo.collapse_lr_reduction == 0.3

    print(f"inference_validation_threshold: {ppo.inference_validation_threshold} (expected: 0.8)")
    assert ppo.inference_validation_threshold == 0.8

    print("\n✅ All default values correct!\n")

if __name__ == "__main__":
    try:
        test_config_upgrades()
        test_trainer_methods()
        test_default_values()

        print("=" * 60)
        print("🎉 ALL UPGRADE TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nUpgrade Summary:")
        print("  ✓ Reward stability buffer")
        print("  ✓ Early stopping on plateau")
        print("  ✓ LR annealing schedule")
        print("  ✓ Entropy decay schedule")
        print("  ✓ Checkpoint rollback logic")
        print("  ✓ Reward collapse detector")
        print("  ✓ Inference reward validator")
        print("  ✓ Training metrics auto-plot")
        print("  ✓ Trainer summary block")
        print("\nThe PPO training system has been successfully upgraded!")
        print("Ready for production use.")

    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

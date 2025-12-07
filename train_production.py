#!/usr/bin/env python3
"""
MineRL Production Training Script

Entry point for production MineRL training with full validation.

Usage:
    python train_production.py --config configs/minerl_memory_ppo.yaml
    python train_production.py --config configs/minerl_memory_ppo.yaml --steps 100000
"""

import argparse
import sys
from pathlib import Path

import yaml


def validate_config(config: dict) -> None:
    """Validate configuration for MineRL-only build."""
    # ==================== CRITICAL ASSERTION ====================
    adapter = config.get("env", {}).get("adapter", "")
    assert adapter == "minerl", (
        f"This build is MineRL-only. Got adapter='{adapter}'. "
        "Set env.adapter='minerl' in your config."
    )

    # Validate required fields
    required_paths = [
        ("env", "name"),
        ("model", "encoder"),
        ("model", "hidden_dim"),
        ("train", "num_envs"),
        ("train", "horizon"),
    ]

    for path in required_paths:
        current = config
        for key in path:
            if key not in current:
                raise ValueError(f"Missing required config: {'.'.join(path)}")
            current = current[key]

    print("✓ Configuration validated for MineRL-only build")


def main():
    parser = argparse.ArgumentParser(description="MineRL Production Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minerl_memory_ppo.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total training steps",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test with mock environment",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override steps if provided
    if args.steps:
        config["train"]["total_timesteps"] = args.steps

    # Override wandb if provided
    if args.wandb:
        config["logging"]["wandb"] = True

    # Validate config (skip for smoke test)
    if not args.smoke_test:
        validate_config(config)

    # Import trainer (after validation to catch import errors separately)
    from cyborg_rl.trainers.ppo_trainer import PPOTrainer

    # Create trainer
    trainer = PPOTrainer(
        config_dict=config,
        use_wandb=config.get("logging", {}).get("wandb", False),
    )

    print(f"🚀 Starting Training on {trainer.device}")
    print(f"   Environment: {config['env']['name']}")
    print(f"   Encoder: {config['model']['encoder']}")
    print(f"   PMM: {'enabled' if config.get('pmm', {}).get('enabled') else 'disabled'}")
    print(f"   Total Steps: {config['train']['total_timesteps']:,}")

    # Train
    trainer.train()

    print("✅ Training Complete")


if __name__ == "__main__":
    main()

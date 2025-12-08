#!/usr/bin/env python3
"""
train_v7.py - V7 Training CLI

Usage:
    python train_v7.py --env minerl_treechop --encoder mamba --memory pmm
    python train_v7.py --config configs/memory_ppo_v7.yaml --wandb
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch


def main():
    parser = argparse.ArgumentParser(
        description="CyborgMind V7 Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Environment
    parser.add_argument("--env", type=str, default="minerl_treechop",
                        help="Environment name")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments")

    # Model
    parser.add_argument("--encoder", type=str, default="mamba",
                        choices=["gru", "mamba", "hybrid", "fusion"],
                        help="Encoder type")
    parser.add_argument("--hidden-dim", type=int, default=384,
                        help="Hidden dimension")
    parser.add_argument("--latent-dim", type=int, default=256,
                        help="Latent dimension")

    # Memory
    parser.add_argument("--memory", type=str, default="pmm",
                        choices=["pmm", "slot", "kv", "ring"],
                        help="Memory type")
    parser.add_argument("--memory-slots", type=int, default=16,
                        help="Number of memory slots")

    # Training
    parser.add_argument("--total-steps", type=int, default=1_000_000,
                        help="Total training steps")
    parser.add_argument("--horizon", type=int, default=1024,
                        help="Rollout horizon")
    parser.add_argument("--burn-in", type=int, default=128,
                        help="Burn-in length for recurrent training")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Sequence length for training")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (number of sequences)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO epochs per update")

    # Features
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP (automatic mixed precision)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable WandB logging")

    # Config
    parser.add_argument("--config", type=str,
                        help="Load config from YAML file")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config not found: {config_path}")
            sys.exit(1)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"📋 Loaded config: {config_path}")
    else:
        # Build config from CLI args
        config = {
            "env": args.env,
            "num_envs": args.num_envs,
            "encoder": args.encoder,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "memory": {
                "type": args.memory,
                "num_slots": args.memory_slots,
                "memory_dim": args.latent_dim,
            },
            "total_steps": args.total_steps,
            "horizon": args.horizon,
            "burn_in": args.burn_in,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "ppo_epochs": args.ppo_epochs,
            "amp": args.amp,
            "compile": args.compile,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_clip": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
        }

    # Print config summary
    print("\n" + "=" * 60)
    print("🚀 CyborgMind V7 Training")
    print("=" * 60)
    print(f"  Environment: {config.get('env')}")
    print(f"  Encoder: {config.get('encoder')}")
    print(f"  Memory: {config.get('memory', {}).get('type')}")
    print(f"  Num Envs: {config.get('num_envs')}")
    print(f"  Horizon: {config.get('horizon')}")
    print(f"  Burn-in: {config.get('burn_in')}")
    print(f"  Total Steps: {config.get('total_steps', args.total_steps):,}")
    print(f"  AMP: {config.get('amp')}")
    print(f"  Compile: {config.get('compile')}")
    print(f"  WandB: {args.wandb}")
    print("=" * 60 + "\n")

    # Create trainer
    from cyborg_rl.trainers.trainer_v7 import TrainerV7

    trainer = TrainerV7(
        config=config,
        use_wandb=args.wandb,
    )

    # Train
    trainer.train(total_timesteps=config.get("total_steps", args.total_steps))

    # Save final checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(checkpoint_dir / "final_v7.pt")

    print(f"\n✅ Training complete! Checkpoint: {checkpoint_dir / 'final_v7.pt'}")


if __name__ == "__main__":
    main()

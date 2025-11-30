#!/usr/bin/env python3
"""
Train a PPO agent on CartPole-v1.

This is a minimal working example that demonstrates the full pipeline:
- Environment setup
- Agent creation with PMM memory
- PPO training with GAE
- Checkpoint saving
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyborg_rl import Config, get_device, set_seed, get_logger
from cyborg_rl.envs import GymAdapter
from cyborg_rl.agents import PPOAgent
from cyborg_rl.trainers import PPOTrainer

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO on CartPole")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/cartpole",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable Prometheus metrics server",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Setup
    device = get_device(args.device)
    set_seed(args.seed)
    logger.info(f"Using device: {device}")

    # Configuration
    config = Config()
    config.env.name = "CartPole-v1"
    config.env.max_episode_steps = 500
    config.train.total_timesteps = args.total_timesteps
    config.train.seed = args.seed
    config.train.checkpoint_dir = args.checkpoint_dir
    config.train.log_frequency = 2048
    config.train.save_frequency = 10_000

    # Smaller model for CartPole
    config.model.hidden_dim = 128
    config.model.latent_dim = 64
    config.model.num_gru_layers = 1
    config.model.use_mamba = False

    # Smaller memory for CartPole
    config.memory.memory_size = 32
    config.memory.memory_dim = 32
    config.memory.num_read_heads = 2

    # PPO hyperparameters
    config.ppo.rollout_steps = 2048
    config.ppo.batch_size = 64
    config.ppo.num_epochs = 10
    config.ppo.learning_rate = 3e-4

    # Create checkpoint directory
    Path(config.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config.to_yaml(Path(config.train.checkpoint_dir) / "config.yaml")

    # Environment
    env = GymAdapter(
        env_name=config.env.name,
        device=device,
        seed=args.seed,
        max_episode_steps=config.env.max_episode_steps,
        normalize_obs=False,
    )

    # Agent
    agent = PPOAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        config=config,
        is_discrete=env.is_discrete,
        device=device,
    )

    # Metrics (optional)
    metrics = None
    if not args.no_metrics:
        try:
            from cyborg_rl.metrics import PrometheusMetrics
            metrics = PrometheusMetrics(port=8000, enable_server=True)
        except Exception as e:
            logger.warning(f"Could not start metrics server: {e}")

    # Trainer
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        config=config,
        metrics=metrics,
    )

    # Train
    logger.info("Starting CartPole training...")
    trainer.train()

    # Cleanup
    env.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

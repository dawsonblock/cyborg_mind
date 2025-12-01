#!/usr/bin/env python3
"""Production training script using cyborg_rl core."""

import argparse
import yaml
import torch
import logging
from pathlib import Path

from cyborg_rl.config import Config
from cyborg_rl.envs.gym_adapter import GymAdapter
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.ppo_trainer import PPOTrainer
from cyborg_rl.utils.logging import get_logger
from cyborg_rl.utils.seeding import set_seed

logger = get_logger(__name__)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return Config.from_dict(cfg_dict)


def main():
    parser = argparse.ArgumentParser(description="Train CyborgMind RL Agent")
    parser.add_argument("--config", type=str, default="configs/envs/gym_cartpole.yaml",
                        help="Path to config file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.device != "auto":
        config.train.device = args.device
    if args.seed is not None:
        config.train.seed = args.seed
    
    # Set device
    device = config.train.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {config}")
    
    # Set seed
    set_seed(config.train.seed)
    
    # Create environment
    env = GymAdapter(
        env_name=config.env.name,
        max_episode_steps=config.env.max_episode_steps,
        normalize_obs=config.env.normalize_obs,
        clip_obs=config.env.clip_obs
    )
    
    # Create agent
    agent = PPOAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        config=config,
        device=device
    )
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        config=config
    )
    
    # Resume if requested
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

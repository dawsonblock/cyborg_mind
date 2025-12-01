#!/usr/bin/env python3
"""Production training script using cyborg_rl core."""

import argparse
import yaml
import torch
from cyborg_rl.config import Config
from cyborg_rl.envs.gym_adapter import GymAdapter
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.ppo_trainer import PPOTrainer
from cyborg_rl.utils.logging import get_logger
from cyborg_rl.utils.seeding import set_seed

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train CyborgMind RL Agent")
    parser.add_argument("--config", type=str, default="configs/envs/gym_cartpole.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    # Load config from YAML
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    config = Config.from_dict(cfg_dict)
    
    # Set device
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed
    if args.seed:
        set_seed(args.seed)
    
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
    trainer = PPOTrainer(env=env, agent=agent, config=config)
    
    # Resume if requested
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()

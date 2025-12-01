#!/usr/bin/env python3
"""Production evaluation script using cyborg_rl core."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from cyborg_rl.config import Config
from cyborg_rl.envs.gym_adapter import GymAdapter
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return Config.from_dict(cfg_dict)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CyborgMind RL Agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="configs/envs/gym_cartpole.yaml",
                        help="Path to config file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent.policy.load_state_dict(checkpoint['policy'])
    agent.value_fn.load_state_dict(checkpoint['value'])
    logger.info("Checkpoint loaded successfully")
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get action from agent
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, _ = agent.predict(obs_tensor, deterministic=True)
            
            # Step environment
            obs, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_reward += reward
            episode_length += 1
            
            if args.render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        logger.info(f"Episode {ep+1}/{args.episodes}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    # Summary
    logger.info(f"\nEvaluation Summary:")
    logger.info(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"  Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    logger.info(f"  Min/Max Reward: {np.min(episode_rewards):.2f}/{np.max(episode_rewards):.2f}")


if __name__ == "__main__":
    main()

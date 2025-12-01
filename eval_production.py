#!/usr/bin/env python3
"""Production evaluation script using cyborg_rl core."""

import argparse
import yaml
import torch
import numpy as np
from cyborg_rl.config import Config
from cyborg_rl.envs.gym_adapter import GymAdapter
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate CyborgMind RL Agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/envs/gym_cartpole.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    config = Config.from_dict(cfg_dict)
    
    # Set device
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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
    logger.info("Checkpoint loaded")
    
    # Evaluate
    episode_rewards = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        reward_sum = 0
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, _ = agent.predict(obs_tensor, deterministic=True)
            obs, reward, done, _ = env.step(action.cpu().numpy()[0])
            reward_sum += reward
        
        episode_rewards.append(reward_sum)
        logger.info(f"Episode {ep+1}: {reward_sum:.2f}")
    
    logger.info(f"Mean: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

if __name__ == "__main__":
    main()

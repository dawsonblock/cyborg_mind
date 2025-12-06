#!/usr/bin/env python3
"""
Production Training Entry Point for CyborgMind.

Usage:
    python train_production.py --config configs/envs/gym_cartpole.yaml --run-name my-experiment
"""

import argparse
import yaml
import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from cyborg_rl.config import Config
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.ppo_trainer import PPOTrainer
from cyborg_rl.experiments.registry import ExperimentRegistry
from cyborg_rl.utils.logging import get_logger
from cyborg_rl.utils.seeding import set_seed

logger = get_logger(__name__)

def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def main():
    parser = argparse.ArgumentParser(description="CyborgMind Production Training")
    parser.add_argument("--config", type=str, default="configs/envs/gym_cartpole.yaml", help="Path to config YAML")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the experiment run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda)")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of vectorized environments")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_dict(config_dict)
    
    # 2. Setup Registry
    registry = ExperimentRegistry(config_dict, run_name=args.run_name)
    
    # 3. Setup Device & Seed
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    set_seed(args.seed)
    logger.info(f"Device: {device} | Seed: {args.seed}")

    # 4. Create Vectorized Environment
    # Note: For custom envs (MineRL, Trading), we'd swap this factory logic
    envs = AsyncVectorEnv([
        make_env(config.env.name, args.seed + i, i, False, registry.run_name)
        for i in range(args.num_envs)
    ])
    
    # 5. Initialize Agent
    # Handle obs/action shapes from vector env
    obs_dim = envs.single_observation_space.shape[0]
    is_discrete = isinstance(envs.single_action_space, gym.spaces.Discrete)
    action_dim = envs.single_action_space.n if is_discrete else envs.single_action_space.shape[0]
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        is_discrete=is_discrete,
        device=device
    )
    
    # 6. Initialize Trainer
    trainer = PPOTrainer(
        env=envs,
        agent=agent,
        config=config,
        registry=registry
    )
    
    # 7. Start Training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving final checkpoint...")
        registry.save_checkpoint(agent.state_dict(), trainer.global_step)
        envs.close()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        envs.close()
        raise

if __name__ == "__main__":
    main()

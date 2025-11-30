#!/usr/bin/env python3
"""
Run the full MineRL Treechop training pipeline.

Steps:
1. Behavior Cloning (BC) warm-start (simulated here via PPO on expert data if available, or just PPO)
2. PPO Training with PMM
3. Evaluation
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyborg_rl import Config, get_device, set_seed, get_logger
from cyborg_rl.envs import MineRLAdapter
from cyborg_rl.agents import PPOAgent
from cyborg_rl.trainers import PPOTrainer

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device("auto")
    set_seed(args.seed)

    logger.info("=== Starting Treechop Pipeline ===")

    # Config
    config = Config()
    config.env.name = "MineRLTreechop-v0"
    config.train.total_timesteps = args.steps
    config.train.checkpoint_dir = "checkpoints/treechop_pipeline"
    
    # Step 1: Initialize Agent
    logger.info("Step 1: Initializing Agent & Environment")
    try:
        env = MineRLAdapter(env_name=config.env.name, device=device)
    except ImportError:
        logger.error("MineRL not installed.")
        return

    agent = PPOAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        config=config,
        is_discrete=env.is_discrete,
        device=device
    )

    # Step 2: Train
    logger.info("Step 2: Starting PPO Training")
    trainer = PPOTrainer(env=env, agent=agent, config=config)
    trainer.train()

    # Step 3: Evaluate (Placeholder)
    logger.info("Step 3: Evaluation")
    # In a real pipeline, run evaluation episodes here and log results

    env.close()
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()

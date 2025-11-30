#!/usr/bin/env python3
"""
Colab-optimized training script for MineRL Treechop.
Designed to run in a single cell or via command line in Colab.
"""

import os
import sys
from pathlib import Path

# Ensure python path includes repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import wandb
from cyborg_rl import Config, get_device, set_seed, get_logger
from cyborg_rl.envs import MineRLAdapter
from cyborg_rl.agents import PPOAgent
from cyborg_rl.trainers import PPOTrainer

logger = get_logger(__name__)

def train_colab():
    # 1. Setup
    print("=== CyborgMind v2.8 Colab Training ===")
    device = get_device("auto")
    print(f"Device: {device}")
    
    # 2. Config
    config = Config()
    config.env.name = "MineRLTreechop-v0"
    config.env.max_episode_steps = 2000
    config.train.total_timesteps = 500_000
    config.train.checkpoint_dir = "/content/drive/MyDrive/CyborgMind/checkpoints/treechop"
    
    # PMM Config for MineRL
    config.model.hidden_dim = 256
    config.memory.memory_size = 64
    config.memory.use_intrinsic_reward = True
    
    # 3. W&B Init (Optional)
    if "WANDB_API_KEY" in os.environ:
        wandb.init(project="cyborg-minerl", config=config.__dict__)
    
    # 4. Environment
    print("Initializing MineRL environment...")
    try:
        env = MineRLAdapter(
            env_name=config.env.name,
            device=device,
            image_size=(64, 64)
        )
    except Exception as e:
        print(f"Failed to load MineRL: {e}")
        print("Did you install JDK 8? (!apt-get install openjdk-8-jdk)")
        return

    # 5. Agent
    agent = PPOAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        config=config,
        is_discrete=env.is_discrete,
        device=device
    )
    
    # 6. Trainer
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        config=config
    )
    
    # 7. Train
    print("Starting training loop...")
    trainer.train()
    
    env.close()
    print("Training complete. Checkpoints saved to Drive.")

if __name__ == "__main__":
    train_colab()

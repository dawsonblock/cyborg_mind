#!/usr/bin/env python3
"""
Verify Simulation Script.

Runs a very short training loop (100 steps) on CartPole to ensure
the entire pipeline (Env -> Agent -> PMM -> Trainer) is functional.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyborg_rl import Config, get_device, set_seed, get_logger
from cyborg_rl.envs import GymAdapter
from cyborg_rl.agents import PPOAgent
from cyborg_rl.trainers import PPOTrainer

logger = get_logger(__name__)

def main():
    print("=== CyborgMind Simulation Verification ===")
    
    # 1. Setup
    device = get_device("cpu") # Use CPU for quick verification
    set_seed(42)
    
    # 2. Config (Minimal)
    config = Config()
    config.env.name = "CartPole-v1"
    config.train.total_timesteps = 200  # Very short run
    config.ppo.rollout_steps = 100
    config.ppo.batch_size = 10
    config.ppo.num_epochs = 2
    
    # 3. Environment
    print("Initializing Environment...")
    env = GymAdapter(env_name="CartPole-v1", device=device)
    
    # 4. Agent
    print("Initializing Agent (Mamba/GRU + PMM)...")
    agent = PPOAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        config=config,
        is_discrete=env.is_discrete,
        device=device
    )
    
    # 5. Trainer
    print("Initializing Trainer...")
    trainer = PPOTrainer(env=env, agent=agent, config=config)
    
    # 6. Run Sim
    print("Running Training Sim...")
    try:
        trainer.train()
        print("\n✅ Simulation Successful! The pipeline is fully operational.")
    except Exception as e:
        print(f"\n❌ Simulation Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        env.close()

if __name__ == "__main__":
    main()

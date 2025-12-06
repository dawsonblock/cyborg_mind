# -*- coding: utf-8 -*-
import argparse
import sys
import yaml
import torch
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# from cyborg_rl.utils.config import load_config
from cyborg_rl.trainers.ppo_trainer import PPOTrainer

def main():
    parser = argparse.ArgumentParser(description="Cyborg MineRL Training v3.0")
    
    # Core args overlapping with config for quick overrides
    parser.add_argument("--env", type=str, help="Environment ID")
    parser.add_argument("--encoder", type=str, choices=["gru", "mamba", "mamba_gru"], help="Encoder type")
    parser.add_argument("--memory", type=str, choices=["pmm", "none"], help="Memory type")
    parser.add_argument("--num-envs", type=int, help="Number of vectorized environments")
    parser.add_argument("--horizon", type=int, help="Episode horizon / Trace length")
    parser.add_argument("--burn-in", type=int, help="Burn-in length for truncated BPTT")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--steps", type=int, help="Total training timesteps")
    
    # Flags
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--config", type=str, default="configs/unified_config.yaml", help="Path to base config")

    args = parser.parse_args()

    # Load Base Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    if args.env: config['env']['name'] = args.env
    if args.encoder: config['model']['encoder'] = args.encoder
    if args.memory == "none": config['pmm']['enabled'] = False
    elif args.memory == "pmm": config['pmm']['enabled'] = True
    if args.num_envs: config['train']['num_envs'] = args.num_envs
    if args.horizon: config['train']['horizon'] = args.horizon
    if args.burn_in: config['train']['burn_in'] = args.burn_in
    if args.lr: config['train']['learning_rate'] = args.lr
    if args.steps: config['train']['total_timesteps'] = args.steps
    
    # Smoke Test Overrides
    if args.smoke_test:
        print("RUNNING SMOKE TEST")
        class MockEnv:
            def reset(self): return None
        # Should we mock? 
        # No, let's just adjust params.
        # But if MINERL missing, adapter fails.
        config['train']['num_envs'] = 2
        config['train']['batch_size'] = 256 # Must be > seq_len (128)
        config['train']['total_timesteps'] = 1000
        config['train']['log_interval'] = 1
        config['train']['save_interval'] = 10
        config['env']['max_steps'] = 100
        config['train']['compile'] = False # Disable compile for smoke test (can fail on some envs)
        
        # Mock Adapter for CI/Smoke Test without MineRL
        class MockMineRLAdapter:
            def __init__(self, *args, **kwargs):
                self.observation_dim = (3 * 4 * 64 * 64) + 1
                self.action_dim = 10
                self.is_discrete = True
                self.frame_stack = 4
                self.image_size = (64, 64)
            
            def reset(self):
                return torch.zeros(2, self.observation_dim)
            
            def step(self, action):
                import numpy as np
                obs = torch.zeros(2, self.observation_dim)
                rew = np.zeros(2, dtype=np.float32)
                done = np.zeros(2, dtype=bool)
                info = [{} for _ in range(2)]
                return obs, rew, done, info
            
            def close(self): pass

        # Patch the imported class in the trainer module
        import cyborg_rl.trainers.ppo_trainer
        cyborg_rl.trainers.ppo_trainer.MineRLAdapter = MockMineRLAdapter

    # Initialize Trainer
    trainer = PPOTrainer(config, use_wandb=args.wandb)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()

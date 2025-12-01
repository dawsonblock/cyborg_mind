#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="CyborgMind Launcher")
    parser.add_argument("mode", choices=["train", "eval"], help="Mode to run: train or eval")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--steps", type=int, default=1000000, help="Total training steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--overrides", nargs="*", help="Additional Hydra overrides (e.g. ppo.learning_rate=0.001)")
    
    args = parser.parse_args()
    
    cmd = [sys.executable]
    
    if args.mode == "train":
        cmd.append("train.py")
    else:
        cmd.append("eval.py")
        
    # Build Hydra arguments
    cmd.append(f"env.name={args.env}")
    cmd.append(f"train.total_timesteps={args.steps}")
    cmd.append(f"train.device={args.device}")
    
    if args.resume:
        cmd.append(f"train.resume_path={args.resume}")
        
    if args.overrides:
        cmd.extend(args.overrides)
        
    print(f"🚀 Launching CyborgMind {args.mode.upper()}...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Execution failed with code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()

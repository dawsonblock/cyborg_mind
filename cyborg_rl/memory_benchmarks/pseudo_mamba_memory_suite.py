#!/usr/bin/env python3
"""
Pseudo-Mamba Memory Benchmark Suite.

Canonical memory benchmarking system for comparing GRU, Mamba, and Pseudo-Mamba
backbones on long-horizon memory tasks.

Usage:
    python -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
        --task delayed_cue \
        --backbone mamba \
        --horizon 1000 \
        --num-envs 64 \
        --steps 100000 \
        --device cuda \
        --seed 42 \
        --run-name delayed_cue_mamba_h1000
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cyborg_rl.config import Config
from cyborg_rl.memory_benchmarks import (
    DelayedCueEnv,
    VectorizedDelayedCueEnv,
    CopyMemoryEnv,
    AssociativeRecallEnv,
)
from cyborg_rl.agents import PPOAgent
from cyborg_rl.trainers import PPOTrainer
from cyborg_rl.experiments import ExperimentRegistry
from cyborg_rl.utils import get_device, set_seed, get_logger

logger = get_logger(__name__)


def create_env(task: str, horizon: int, num_envs: int):
    """
    Create environment based on task name.
    
    Args:
        task: Task name (delayed_cue, copy, assoc_recall)
        horizon: Horizon/delay length
        num_envs: Number of parallel environments
        
    Returns:
        Environment instance
    """
    if task == "delayed_cue":
        return VectorizedDelayedCueEnv(
            num_envs=num_envs,
            num_cues=4,
            horizon=horizon,
        )
    elif task == "copy":
        logger.warning("Copy-Memory is a stub implementation")
        # For now, return a list of single envs (no vectorization)
        return [CopyMemoryEnv(sequence_length=3, delay_length=horizon) for _ in range(num_envs)]
    elif task == "assoc_recall":
        logger.warning("Associative Recall is a stub implementation")
        return [AssociativeRecallEnv(num_keys=5) for _ in range(num_envs)]
    else:
        raise ValueError(f"Unknown task: {task}")


def create_agent(
    obs_dim: int,
    action_dim: int,
    backbone: str,
    device: torch.device,
) -> PPOAgent:
    """
    Create PPO agent with specified backbone.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        backbone: Backbone type (gru, mamba, mamba_gru, pseudo_mamba)
        device: Torch device
        
    Returns:
        PPOAgent instance
    """
    config = Config()
    
    # Set encoder type based on backbone
    if backbone == "gru":
        config.model.encoder_type = "gru"
        config.model.use_mamba = False
    elif backbone == "mamba":
        config.model.encoder_type = "mamba_gru"
        config.model.use_mamba = True
    elif backbone == "mamba_gru":
        config.model.encoder_type = "mamba_gru"
        config.model.use_mamba = True
    elif backbone == "pseudo_mamba":
        logger.warning("Pseudo-Mamba not yet implemented, using Mamba+GRU")
        config.model.encoder_type = "mamba_gru"
        config.model.use_mamba = True
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    # Memory benchmark specific config
    config.model.hidden_dim = 256
    config.model.latent_dim = 128
    config.model.num_gru_layers = 2
    
    # PPO config
    config.ppo.learning_rate = 3e-4
    config.ppo.gamma = 0.99
    config.ppo.gae_lambda = 0.95
    config.ppo.clip_epsilon = 0.2
    config.ppo.entropy_coef = 0.01
    config.ppo.value_coef = 0.5
    config.ppo.max_grad_norm = 0.5
    config.ppo.rollout_steps = 2048
    config.ppo.batch_size = 64
    config.ppo.num_epochs = 10
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        is_discrete=True,
        device=device,
    )
    
    return agent


def compute_memory_stats(agent: PPOAgent, obs: torch.Tensor, state: dict) -> dict:
    """
    Compute memory usage statistics.
    
    Args:
        agent: PPO agent
        obs: Observation tensor
        state: Agent state dict
        
    Returns:
        Dictionary of statistics
    """
    with torch.no_grad():
        # Get PMM info
        _, _, _, new_state, info = agent.forward(obs, state, deterministic=True)
        
        pmm_info = info.get("pmm_info", {})
        
        stats = {
            "memory_usage": pmm_info.get("memory_usage", 0.0).item() if torch.is_tensor(pmm_info.get("memory_usage")) else pmm_info.get("memory_usage", 0.0),
            "pressure": pmm_info.get("pressure", torch.tensor(0.0)).mean().item() if torch.is_tensor(pmm_info.get("pressure")) else 0.0,
            "read_entropy": pmm_info.get("read_entropy", torch.tensor(0.0)).mean().item() if torch.is_tensor(pmm_info.get("read_entropy")) else 0.0,
            "resonance": pmm_info.get("resonance", torch.tensor(0.0)).mean().item() if torch.is_tensor(pmm_info.get("resonance")) else 0.0,
        }
        
    return stats


def main():
    parser = argparse.ArgumentParser(description="Memory Benchmark Suite")
    
    # Task and backbone selection
    parser.add_argument(
        "--task",
        type=str,
        default="delayed_cue",
        choices=["delayed_cue", "copy", "assoc_recall"],
        help="Memory task to benchmark",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="gru",
        choices=["gru", "mamba", "mamba_gru", "pseudo_mamba"],
        help="Encoder backbone to use",
    )
    
    # Environment parameters
    parser.add_argument("--horizon", type=int, default=100, help="Delay/horizon length")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Experiment tracking
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    parser.add_argument("--base-dir", type=str, default="experiments/memory_benchmarks", help="Base directory for experiments")
    
    args = parser.parse_args()
    
    # Setup
    device = get_device(args.device)
    set_seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("Memory Benchmark Suite")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Horizon: {args.horizon}")
    logger.info(f"Num Envs: {args.num_envs}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 80)
    
    # Create environment
    logger.info("Creating environment...")
    env = create_env(args.task, args.horizon, args.num_envs)
    
    # Get observation and action dimensions
    if isinstance(env, VectorizedDelayedCueEnv):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    else:
        # Non-vectorized envs
        obs_dim = env[0].observation_space.shape[0]
        action_dim = env[0].action_space.n
    
    logger.info(f"Observation dim: {obs_dim}")
    logger.info(f"Action dim: {action_dim}")
    
    # Create agent
    logger.info("Creating agent...")
    agent = create_agent(obs_dim, action_dim, args.backbone, device)
    logger.info(f"Agent parameters: {agent.count_parameters():,}")
    
    # Create experiment registry
    run_name = args.run_name or f"{args.task}_{args.backbone}_h{args.horizon}"
    registry = ExperimentRegistry(
        config={
            "task": args.task,
            "backbone": args.backbone,
            "horizon": args.horizon,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
        },
        run_name=run_name,
        base_dir=args.base_dir,
    )
    
    logger.info(f"Experiment: {registry.run_name}")
    logger.info(f"Artifacts: {registry.base_dir}")
    
    # Create trainer config
    config = agent.config
    config.train.total_timesteps = args.steps
    config.train.device = str(device)
    config.train.seed = args.seed
    
    # Train
    logger.info("Starting training...")
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        config=config,
        registry=registry,
    )
    
    trainer.train()
    
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Results saved to: {registry.base_dir}")
    logger.info("=" * 80)
    
    # Close environment
    if hasattr(env, "close"):
        env.close()
    else:
        for e in env:
            e.close()


if __name__ == "__main__":
    main()

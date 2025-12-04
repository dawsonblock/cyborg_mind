#!/usr/bin/env python3
"""
pseudo_mamba_memory_suite.py

Canonical memory benchmark entrypoint for Cyborg Mind.

Supports:
    - Tasks: delayed_cue, copy_memory, associative_recall
    - Backbones: gru, pseudo_mamba, mamba_gru
    - Horizon, num_envs, total_timesteps, device, seed, etc.

Provides:
    1) CLI to run a single experiment.
    2) `run_single_experiment(...)` callable for sweep scripts.

All runs are logged via ExperimentRegistry, and the function returns a
dict of scalar metrics for sweep aggregation.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch

from cyborg_rl.config import Config
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.memory_ppo_trainer import MemoryPPOTrainer
from cyborg_rl.experiments.registry import ExperimentRegistry
from cyborg_rl.utils.device import get_device
from cyborg_rl.utils.seeding import set_seed
from cyborg_rl.utils.logging import get_logger

from cyborg_rl.memory_benchmarks.delayed_cue_env import DelayedCueEnv, VectorizedDelayedCueEnv
from cyborg_rl.memory_benchmarks.copy_memory_env import CopyMemoryEnv
from cyborg_rl.memory_benchmarks.associative_recall_env import AssociativeRecallEnv

logger = get_logger(__name__)


@dataclass
class MemorySuiteConfig:
    # High-level identifiers
    task: str
    backbone: str
    horizon: int
    run_name: str

    # Env / training settings
    num_envs: int = 64
    total_timesteps: int = 200_000
    rollout_steps: int = 512
    update_epochs: int = 4
    minibatch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4

    # PMM / model
    encoder_type: str = "gru"  # overridden by `backbone`
    hidden_dim: int = 128
    latent_dim: int = 128
    num_gru_layers: int = 1
    memory_size: int = 32
    memory_dim: int = 64

    # Misc
    device: str = "cuda"
    seed: int = 42
    base_dir: str = "experiments/runs"
    log_interval: int = 10  # in PPO updates

    # Evaluation
    eval_episodes: int = 64


def _make_env(task: str, horizon: int, num_envs: int):
    """Create vectorized environment for the specified task."""
    if task == "delayed_cue":
        return VectorizedDelayedCueEnv(
            num_envs=num_envs,
            num_cues=4,
            horizon=horizon,
        )
    elif task == "copy_memory":
        # Wrap single envs in a vectorized wrapper
        import gymnasium as gym
        envs = [
            lambda seq_len=3, delay=horizon: CopyMemoryEnv(
                sequence_length=seq_len,
                delay_length=delay,
            )
            for _ in range(num_envs)
        ]
        return gym.vector.SyncVectorEnv(envs)
    elif task == "associative_recall":
        import gymnasium as gym
        envs = [
            lambda: AssociativeRecallEnv(num_keys=5)
            for _ in range(num_envs)
        ]
        return gym.vector.SyncVectorEnv(envs)
    else:
        raise ValueError(f"Unknown task: {task}")


def _backbone_to_encoder_type(backbone: str) -> str:
    backbone = backbone.lower()
    if backbone == "gru":
        return "gru"
    elif backbone == "pseudo_mamba":
        return "pseudo_mamba"
    elif backbone in ("mamba", "mamba_gru"):
        return "mamba_gru"
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def _build_config(msc: MemorySuiteConfig, obs_dim: int, action_dim: int) -> Config:
    """
    Build a Config object that matches the MemorySuiteConfig.
    This keeps it aligned with the global PPO / model config system.
    """
    encoder_type = _backbone_to_encoder_type(msc.backbone)

    cfg_dict: Dict[str, Any] = {
        "env": {
            "name": f"{msc.task}_h{msc.horizon}",
            "max_episode_steps": msc.horizon + 10,
        },
        "model": {
            "encoder_type": encoder_type,
            "hidden_dim": msc.hidden_dim,
            "latent_dim": msc.latent_dim,
            "num_gru_layers": msc.num_gru_layers,
            "use_mamba": encoder_type == "mamba_gru",
            "dropout": 0.0,
        },
        "memory": {
            "memory_size": msc.memory_size,
            "memory_dim": msc.memory_dim,
            "num_read_heads": 4,
            "num_write_heads": 1,
            "sharp_factor": 1.0,
        },
        "train": {
            "total_timesteps": msc.total_timesteps,
            "n_steps": msc.rollout_steps,
            "n_epochs": msc.update_epochs,
            "batch_size": msc.minibatch_size,
            "gamma": msc.gamma,
            "gae_lambda": msc.gae_lambda,
            "clip_range": msc.clip_range,
            "entropy_coef": msc.ent_coef,
            "value_coef": msc.vf_coef,
            "max_grad_norm": msc.max_grad_norm,
            "lr": msc.learning_rate,
            "device": msc.device,
            "seed": msc.seed,
            "use_amp": False,
            "weight_decay": 0.0,
            "save_freq": 100,
            "wandb_enabled": False,
        },
        "api": {
            "enabled": False,
        },
    }

    return Config.from_dict(cfg_dict)


def run_single_experiment(msc: MemorySuiteConfig) -> Dict[str, Any]:
    """
    Run a single memory experiment and return summary metrics.

    Returns dict with at least:
        {
            "task": ...,
            "backbone": ...,
            "horizon": ...,
            "success_rate": ...,
            "mean_reward": ...,
            "steps_per_second": ...,
            "final_memory_saturation": ...,
            "final_memory_mean_norm": ...,
            "total_updates": ...,
            "total_seconds": ...,
        }
    """
    # Seed + device
    device = get_device(msc.device)
    set_seed(msc.seed)

    # Build vectorized env
    env = _make_env(msc.task, msc.horizon, msc.num_envs)

    # Get obs/action dims from the vectorized env
    obs_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
        is_discrete = True
    else:
        action_dim = env.action_space.shape[0]
        is_discrete = False

    # Build Config
    config = _build_config(msc, obs_dim, action_dim)

    # Registry
    registry = ExperimentRegistry(
        base_dir=msc.base_dir,
        run_name=msc.run_name,
        config=config.to_dict(),
        tags={
            "suite": "memory_benchmarks",
            "task": msc.task,
            "backbone": msc.backbone,
            "horizon": msc.horizon,
        },
    )

    # Agent + trainer
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        is_discrete=is_discrete,
        device=device,
    )

    # Use MemoryPPOTrainer for full-sequence BPTT on memory tasks
    trainer = MemoryPPOTrainer(
        config=config,
        env_adapter=env,
        agent=agent,
        registry=registry,
    )

    start_time = time.time()

    # Main training loop
    trainer.train()

    total_seconds = time.time() - start_time
    steps_per_second = trainer.global_step / max(total_seconds, 1e-6)

    # Evaluate success rate on the memory task
    success_rate, mean_reward, memory_stats = _evaluate_memory_task(
        env=env,
        agent=agent,
        horizon=msc.horizon,
        num_episodes=msc.eval_episodes,
        device=device,
    )

    summary = {
        "task": msc.task,
        "backbone": msc.backbone,
        "horizon": msc.horizon,
        "success_rate": float(success_rate),
        "mean_reward": float(mean_reward),
        "steps_per_second": float(steps_per_second),
        "total_updates": int(trainer.global_step // msc.rollout_steps),
        "total_seconds": float(total_seconds),
        "final_memory_saturation": float(memory_stats.get("memory_saturation", np.nan)),
        "final_memory_mean_norm": float(memory_stats.get("memory_mean_norm", np.nan)),
    }

    # Save summary into registry for future reference
    registry.log_metrics(trainer.global_step, summary)
    registry.save_manifest()

    # Close env
    env.close()

    return summary


@torch.no_grad()
def _evaluate_memory_task(
    env,
    agent: PPOAgent,
    horizon: int,
    num_episodes: int,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Roll out the trained agent on the memory task and compute:
        - success_rate: fraction of episodes with reward > 0.5.
        - mean_reward: average return.
        - memory_stats: averaged PMM stats over episodes.
    """
    agent.eval()

    total_success = 0
    total_reward = 0.0
    all_saturation = []
    all_mean_norm = []

    # We'll run single-env episodes since vectorized eval is complex for episodic metrics
    # Create a single env instance
    single_env = None
    if hasattr(env, 'envs'):
        # It's a vectorized env, get one of the underlying envs
        single_env = env.envs[0]
    else:
        # Fallback: create a fresh single env
        task_type = getattr(agent.config.env, "task_type", None)
        if task_type == "delayed_cue":
            single_env = DelayedCueEnv(num_cues=4, horizon=horizon)
        elif task_type == "copy_memory":
            single_env = CopyMemoryEnv(sequence_length=3, delay_length=horizon)
        elif task_type == "associative_recall":
            single_env = AssociativeRecallEnv(num_keys=5)

    if single_env is None:
        logger.warning("Could not create single env for evaluation")
        return 0.0, 0.0, {}

    for ep in range(num_episodes):
        obs, info = single_env.reset(seed=42 + ep)
        state = agent.init_state(batch_size=1)
        done = False
        ep_reward = 0.0
        ep_success = 0

        t = 0
        max_steps = horizon + 10
        while not done and t < max_steps:
            obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)

            action, log_prob, value, state, extra = agent(
                obs_tensor, state, deterministic=True
            )
            pmm_info = extra.get("pmm_info", {})
            sat = pmm_info.get("memory_saturation", None)
            if sat is not None:
                all_saturation.append(float(sat))

            # Extract scalar action
            action_scalar = action.item() if action.numel() == 1 else action[0].item()

            next_obs, reward, terminated, truncated, info = single_env.step(action_scalar)
            ep_reward += float(reward)

            if info.get("success", False):
                ep_success = 1

            obs = next_obs
            done = terminated or truncated
            t += 1

        total_reward += ep_reward
        total_success += ep_success

    success_rate = total_success / max(num_episodes, 1)
    mean_reward = total_reward / max(num_episodes, 1)
    memory_stats = {
        "memory_saturation": float(np.mean(all_saturation)) if all_saturation else float("nan"),
        "memory_mean_norm": float("nan"),  # Can add if needed
    }

    agent.train()
    return success_rate, mean_reward, memory_stats


def parse_args() -> MemorySuiteConfig:
    parser = argparse.ArgumentParser(description="Pseudo-Mamba Memory Benchmark Suite")
    parser.add_argument("--task", type=str, required=True,
                        choices=["delayed_cue", "copy_memory", "associative_recall"])
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["gru", "pseudo_mamba", "mamba_gru"])
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dir", type=str, default="experiments/runs")
    parser.add_argument("--eval-episodes", type=int, default=64)

    args = parser.parse_args()

    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.task}_{args.backbone}_H{args.horizon}_T{args.total_timesteps}"

    msc = MemorySuiteConfig(
        task=args.task,
        backbone=args.backbone,
        horizon=args.horizon,
        run_name=run_name,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        device=args.device,
        seed=args.seed,
        base_dir=args.base_dir,
        eval_episodes=args.eval_episodes,
    )
    return msc


def main() -> None:
    msc = parse_args()
    summary = run_single_experiment(msc)
    print("=== Memory Benchmark Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

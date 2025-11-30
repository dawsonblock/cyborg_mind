#!/usr/bin/env python3
"""
Run inference with a trained PPO agent.

Loads a checkpoint and runs the agent in the environment,
optionally rendering and recording episodes.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyborg_rl import get_device, set_seed, get_logger
from cyborg_rl.envs import GymAdapter
from cyborg_rl.agents import PPOAgent

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with trained agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to agent checkpoint",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Environment name",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Path to save video recording",
    )
    return parser.parse_args()


def run_episode(
    env: GymAdapter,
    agent: PPOAgent,
    deterministic: bool = True,
) -> Tuple[float, int]:
    """
    Run a single episode.

    Args:
        env: Environment adapter.
        agent: PPO agent.
        deterministic: Use deterministic actions.

    Returns:
        Tuple of (total_reward, episode_length).
    """
    obs = env.reset()
    state = agent.init_state(batch_size=1)

    total_reward = 0.0
    episode_length = 0
    done = False

    while not done:
        with torch.no_grad():
            obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs
            action, _, _, state, _ = agent(
                obs_tensor, state, deterministic=deterministic
            )

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        episode_length += 1

    return total_reward, episode_length


def main() -> None:
    """Main inference function."""
    args = parse_args()

    # Setup
    device = get_device(args.device)
    set_seed(args.seed)
    logger.info(f"Using device: {device}")

    # Load agent
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Loading agent from {checkpoint_path}")
    agent = PPOAgent.load(str(checkpoint_path), device)
    agent.eval()

    # Create environment
    import gymnasium as gym

    render_mode = "human" if args.render else None

    if args.save_video:
        env_raw = gym.make(args.env, render_mode="rgb_array")
        env_raw = gym.wrappers.RecordVideo(
            env_raw,
            args.save_video,
            episode_trigger=lambda x: True,
        )
    elif args.render:
        env_raw = gym.make(args.env, render_mode="human")
    else:
        env_raw = gym.make(args.env)

    env = GymAdapter(
        env_name=args.env,
        device=device,
        seed=args.seed,
        normalize_obs=False,
    )
    env._env = env_raw

    # Run episodes
    logger.info(f"Running {args.episodes} episodes...")

    rewards = []
    lengths = []

    for ep in range(args.episodes):
        reward, length = run_episode(
            env=env,
            agent=agent,
            deterministic=args.deterministic,
        )
        rewards.append(reward)
        lengths.append(length)
        logger.info(f"Episode {ep + 1}: reward={reward:.2f}, length={length}")

    # Summary
    logger.info("=" * 50)
    logger.info(f"Results over {args.episodes} episodes:")
    logger.info(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    logger.info(f"  Mean length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    logger.info(f"  Max reward:  {np.max(rewards):.2f}")
    logger.info(f"  Min reward:  {np.min(rewards):.2f}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()

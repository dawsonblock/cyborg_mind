#!/usr/bin/env python3
"""
MineRL Agent Evaluation Script

Performs deterministic policy evaluation on a trained MineRL agent.
Logs success rate, survival time, and memory usage curves.

Usage:
    python evaluate_minerl_agent.py --checkpoint checkpoints/best.pt --episodes 20
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cyborg_rl.envs.minerl_adapter import MineRLAdapter, MINERL_AVAILABLE
from cyborg_rl.models.encoder import UnifiedEncoder
from cyborg_rl.models.policy import DiscretePolicy
from cyborg_rl.memory.pmm import PMM
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


class AgentEvaluator:
    """Deterministic evaluation of trained MineRL agents."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
            deterministic: Use argmax action selection
        """
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.checkpoint_path = Path(checkpoint_path)

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )

        self.config = self.checkpoint.get("config", {})
        self._build_models()

    def _build_models(self) -> None:
        """Build models from checkpoint."""
        cfg = self.config

        # Get dimensions from checkpoint
        obs_dim = self.checkpoint.get("obs_dim", 49153)  # Default for 64x64x3x4 + compass
        action_dim = self.checkpoint.get("action_dim", 10)

        # Encoder
        self.encoder = UnifiedEncoder(
            encoder_type=cfg.get("model", {}).get("encoder", "gru"),
            input_dim=obs_dim,
            hidden_dim=cfg.get("model", {}).get("hidden_dim", 384),
            latent_dim=cfg.get("model", {}).get("vision_dim", 256),
            device=self.device.type,
        ).to(self.device)

        # PMM
        self.use_pmm = cfg.get("pmm", {}).get("enabled", False)
        if self.use_pmm:
            self.pmm = PMM(
                memory_dim=cfg.get("pmm", {}).get("memory_dim", 256),
                num_slots=cfg.get("pmm", {}).get("num_slots", 16),
            ).to(self.device)
            policy_dim = cfg.get("pmm", {}).get("memory_dim", 256) + cfg.get("model", {}).get("vision_dim", 256)
            self.pmm_proj = torch.nn.Linear(
                cfg.get("model", {}).get("vision_dim", 256),
                cfg.get("pmm", {}).get("memory_dim", 256),
            ).to(self.device)
        else:
            self.pmm = None
            policy_dim = cfg.get("model", {}).get("vision_dim", 256)

        # Policy
        self.policy = DiscretePolicy(policy_dim, action_dim).to(self.device)

        # Load weights
        self.encoder.load_state_dict(self.checkpoint["encoder_state_dict"])
        self.policy.load_state_dict(self.checkpoint["policy_state_dict"])
        if self.use_pmm:
            self.pmm.load_state_dict(self.checkpoint["pmm_state_dict"])
            self.pmm_proj.load_state_dict(self.checkpoint["pmm_proj_state_dict"])

        # Set to eval mode
        self.encoder.eval()
        self.policy.eval()
        if self.pmm:
            self.pmm.eval()

        logger.info("Models loaded successfully")

    @torch.no_grad()
    def evaluate(
        self,
        env_name: str = "MineRLTreechop-v0",
        num_episodes: int = 20,
        seeds: Optional[List[int]] = None,
        max_steps: int = 18000,
    ) -> Dict[str, Any]:
        """
        Run deterministic evaluation.

        Args:
            env_name: MineRL environment ID
            num_episodes: Number of evaluation episodes
            seeds: Fixed seeds for reproducibility (one per episode)
            max_steps: Max steps per episode

        Returns:
            Evaluation results dictionary
        """
        if not MINERL_AVAILABLE:
            logger.error("MineRL not installed. Cannot run evaluation.")
            return {"error": "MineRL not available"}

        # Create environment
        env = MineRLAdapter(
            env_name=env_name,
            num_envs=1,
            max_steps=max_steps,
            sticky_action_prob=0.0,  # Disable sticky actions for eval
            normalize_rewards=False,  # Raw rewards for eval
        )

        # Default seeds
        if seeds is None:
            seeds = list(range(1000, 1000 + num_episodes))

        results = {
            "episodes": [],
            "total_episodes": num_episodes,
            "env_name": env_name,
            "checkpoint": str(self.checkpoint_path),
        }

        episode_rewards = []
        episode_lengths = []
        memory_usage_curves = []

        for ep_idx, seed in enumerate(seeds[:num_episodes]):
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Reset
            obs = env.reset()
            obs = obs.to(self.device)

            # Initialize states
            encoder_state = self.encoder.get_initial_state(1, self.device)
            if self.use_pmm:
                pmm_memory = self.pmm.get_initial_state(1, self.device)

            total_reward = 0.0
            step_count = 0
            memory_usage = []
            done = False

            while not done and step_count < max_steps:
                # Encode observation
                obs_flat = obs.view(1, -1)
                vision_emb, encoder_state = self.encoder(obs_flat, encoder_state)
                vision_emb = vision_emb.squeeze(1)  # (1, D)

                # PMM
                if self.use_pmm:
                    pmm_input = self.pmm_proj(vision_emb)
                    read_vec, pmm_memory, pmm_logs = self.pmm(
                        pmm_memory, pmm_input, torch.ones(1, 1, device=self.device)
                    )
                    policy_input = torch.cat([vision_emb, read_vec], dim=-1)
                    memory_usage.append(pmm_logs.get("write_strength", 0.0))
                else:
                    policy_input = vision_emb

                # Get action
                action_dist = self.policy(policy_input)
                if self.deterministic:
                    action = action_dist.logits.argmax(dim=-1)
                else:
                    action = action_dist.sample()

                # Step environment
                obs, reward, done_arr, info = env.step(action.cpu().numpy())
                obs = obs.to(self.device)
                done = done_arr[0]

                total_reward += reward[0]
                step_count += 1

            # Log episode
            episode_results = {
                "episode": ep_idx,
                "seed": seed,
                "reward": float(total_reward),
                "length": step_count,
                "success": total_reward > 0,  # Simple success metric
            }
            results["episodes"].append(episode_results)
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            memory_usage_curves.append(memory_usage)

            logger.info(
                f"Episode {ep_idx + 1}/{num_episodes}: "
                f"reward={total_reward:.2f}, length={step_count}"
            )

        # Aggregate statistics
        results["summary"] = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
            "success_rate": float(np.mean([e["success"] for e in results["episodes"]])),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
        }

        if self.use_pmm and memory_usage_curves:
            # Aggregate memory usage
            all_usage = [u for curve in memory_usage_curves for u in curve]
            results["memory_stats"] = {
                "mean_write_strength": float(np.mean(all_usage)) if all_usage else 0.0,
                "std_write_strength": float(np.std(all_usage)) if all_usage else 0.0,
            }

        env.close()
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MineRL Agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MineRLTreechop-v0",
        help="MineRL environment ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action selection",
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = AgentEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        deterministic=not args.stochastic,
    )

    results = evaluator.evaluate(
        env_name=args.env,
        num_episodes=args.episodes,
    )

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    if "summary" in results:
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Episodes: {results['total_episodes']}")
        print(f"Mean Reward: {results['summary']['mean_reward']:.2f} ± {results['summary']['std_reward']:.2f}")
        print(f"Mean Length: {results['summary']['mean_length']:.0f} ± {results['summary']['std_length']:.0f}")
        print(f"Success Rate: {results['summary']['success_rate']:.1%}")
        print("=" * 50)


if __name__ == "__main__":
    main()

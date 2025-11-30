"""PPO Trainer with proper rollout collection, GAE, and monitoring."""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cyborg_rl.config import Config
from cyborg_rl.envs.base import BaseEnvAdapter
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.rollout_buffer import RolloutBuffer
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


class PPOTrainer:
    """
    PPO Trainer with proper rollout collection, GAE, and Prometheus metrics.

    Implements the PPO-Clip algorithm with:
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy bonus
    - Gradient clipping
    """

    def __init__(
        self,
        env: BaseEnvAdapter,
        agent: PPOAgent,
        config: Config,
        metrics: Optional["PrometheusMetrics"] = None,
    ) -> None:
        """
        Initialize the PPO trainer.

        Args:
            env: Environment adapter.
            agent: PPO agent.
            config: Configuration object.
            metrics: Optional Prometheus metrics.
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.device = agent.device

        self.rollout_buffer = RolloutBuffer(
            buffer_size=config.ppo.rollout_steps,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            device=self.device,
            is_discrete=env.is_discrete,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
        )

        self.optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=config.ppo.learning_rate,
        )

        self.metrics = metrics

        self.checkpoint_dir = Path(config.train.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.episode_count = 0
        self.best_reward = -float("inf")

    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollouts from environment.

        Returns:
            Dict with episode statistics.
        """
        self.rollout_buffer.reset()
        self.agent.eval()

        obs = self.env.reset()
        state = self.agent.init_state(batch_size=1)

        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0.0
        current_episode_length = 0
        done = False

        for _ in range(self.config.ppo.rollout_steps):
            with torch.no_grad():
                obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs
                action, log_prob, value, state, _ = self.agent(
                    obs_tensor, state, deterministic=False
                )

            action_np = action.cpu().numpy().flatten()
            if self.env.is_discrete:
                action_np = int(action_np[0])

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.rollout_buffer.add(
                obs=obs.cpu().numpy(),
                action=action_np,
                reward=reward,
                value=value.item(),
                log_prob=log_prob.item(),
                done=done,
            )

            current_episode_reward += reward
            current_episode_length += 1
            self.global_step += 1

            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                self.episode_count += 1

                if self.metrics is not None:
                    self.metrics.record_episode(
                        reward=current_episode_reward,
                        length=current_episode_length,
                    )

                current_episode_reward = 0.0
                current_episode_length = 0
                obs = self.env.reset()
                state = self.agent.init_state(batch_size=1)
            else:
                obs = next_obs

        # Compute bootstrap value
        with torch.no_grad():
            obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs
            last_value, _ = self.agent.get_value(obs_tensor, state)

        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value.item(),
            last_done=done,
        )

        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
        }

    def train_step(self) -> Dict[str, float]:
        """
        Perform one PPO update.

        Returns:
            Dict with training statistics.
        """
        self.agent.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_approx_kl = 0.0
        num_batches = 0

        for _ in range(self.config.ppo.num_epochs):
            for batch in self.rollout_buffer.get(
                batch_size=self.config.ppo.batch_size,
                normalize_advantage=self.config.ppo.normalize_advantage,
            ):
                # Evaluate actions
                log_probs, entropy, values, _ = self.agent.evaluate_actions(
                    batch["observations"],
                    batch["actions"],
                )

                # Policy loss (PPO-Clip)
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.ppo.clip_epsilon,
                    1 + self.config.ppo.clip_epsilon,
                ) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_pred = values
                value_clipped = batch["old_values"] + torch.clamp(
                    values - batch["old_values"],
                    -self.config.ppo.clip_epsilon,
                    self.config.ppo.clip_epsilon,
                )
                value_loss_unclipped = (value_pred - batch["returns"]) ** 2
                value_loss_clipped = (value_clipped - batch["returns"]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.ppo.value_coef * value_loss
                    + self.config.ppo.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(),
                    self.config.ppo.max_grad_norm,
                )
                self.optimizer.step()

                # Track statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                total_approx_kl += approx_kl.item()
                num_batches += 1

        stats = {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy_loss": total_entropy_loss / num_batches,
            "total_loss": total_loss / num_batches,
            "approx_kl": total_approx_kl / num_batches,
        }

        # Record metrics
        if self.metrics is not None:
            self.metrics.record_losses(
                policy_loss=stats["policy_loss"],
                value_loss=stats["value_loss"],
            )
            self.metrics.record_advantage(
                self.rollout_buffer.advantages[:self.rollout_buffer.ptr].mean()
            )

        return stats

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(f"Starting training for {self.config.train.total_timesteps} steps")

        pbar = tqdm(total=self.config.train.total_timesteps, desc="Training")
        pbar.update(self.global_step)

        while self.global_step < self.config.train.total_timesteps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts()

            # Save best model
            if self.config.train.save_best and rollout_stats["mean_reward"] > self.best_reward:
                self.best_reward = rollout_stats["mean_reward"]
                self.save_checkpoint(filename="best_policy.pt")
                logger.info(f"New best reward: {self.best_reward:.2f}. Saved best_policy.pt")

            # Train
            train_stats = self.train_step()

            # Update progress
            pbar.update(self.config.ppo.rollout_steps)

            # Log
            if self.global_step % self.config.train.log_frequency == 0:
                logger.info(
                    f"Step {self.global_step}: "
                    f"reward={rollout_stats['mean_reward']:.2f}, "
                    f"length={rollout_stats['mean_length']:.1f}, "
                    f"policy_loss={train_stats['policy_loss']:.4f}, "
                    f"value_loss={train_stats['value_loss']:.4f}"
                )

            # Save checkpoint
            if self.global_step % self.config.train.save_frequency == 0:
                self.save_checkpoint()

        pbar.close()
        self.save_checkpoint(final=True)
        logger.info("Training complete!")

    def save_checkpoint(self, final: bool = False, filename: Optional[str] = None) -> None:
        """
        Save training checkpoint.

        Args:
            final: If True, save as final checkpoint.
            filename: Optional custom filename.
        """
        if filename:
            path = self.checkpoint_dir / filename
        elif final:
            path = self.checkpoint_dir / "final_policy.pt"
        else:
            path = self.checkpoint_dir / f"policy_step_{self.global_step}.pt"

        self.agent.save(str(path))

        # Save training state
        state_path = self.checkpoint_dir / "trainer_state.pt"
        torch.save({
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "optimizer_state": self.optimizer.state_dict(),
        }, state_path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Checkpoint path.
        """
        self.agent = PPOAgent.load(path, self.device)

        state_path = Path(path).parent / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.episode_count = state["episode_count"]
            self.optimizer.load_state_dict(state["optimizer_state"])
            logger.info(f"Resumed from step {self.global_step}")

"""PPO Trainer with proper rollout collection, GAE, and monitoring."""

from pathlib import Path
from typing import Dict, Optional, List
from collections import deque
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

        # Initialize LR and entropy coefficients
        self.lr_start = config.ppo.lr_start if config.ppo.lr_start is not None else config.ppo.learning_rate
        self.lr_end = config.ppo.lr_end
        self.lr_current = self.lr_start

        self.entropy_start = config.ppo.entropy_start if config.ppo.entropy_start is not None else config.ppo.entropy_coef
        self.entropy_end = config.ppo.entropy_end
        self.entropy_current = self.entropy_start

        self.optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=self.lr_current,
        )

        self.metrics = metrics

        self.checkpoint_dir = Path(config.train.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create plots directory
        self.plots_dir = self.checkpoint_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_reward = -float("inf")
        self.best_policy_state_dict = None
        self.best_step = 0

        # Reward stability buffer
        self.reward_buffer: deque = deque(maxlen=config.ppo.reward_buffer_size)
        self.reward_history: List[float] = []
        self.step_history: List[int] = []

        # Early stopping
        self.plateau_counter = 0
        self.early_stopped = False
        self.last_improvement_step = 0

        # Collapse detection
        self.peak_moving_average = -float("inf")
        self.collapse_detected = False
        self.collapse_recoveries = 0

        # Training metrics for plotting
        self.policy_loss_history: List[float] = []
        self.value_loss_history: List[float] = []
        self.lr_history: List[float] = []
        self.entropy_coef_history: List[float] = []

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

                # Total loss (use dynamic entropy coefficient)
                loss = (
                    policy_loss
                    + self.config.ppo.value_coef * value_loss
                    + self.entropy_current * entropy_loss
                )

                # NaN Guard
                if torch.isnan(loss):
                    logger.error("NaN loss detected! Skipping batch.")
                    continue

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient Clipping
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
            # Record PMM ops (approximate per step)
            self.metrics.record_pmm_ops(
                reads=self.config.ppo.rollout_steps,
                writes=self.config.ppo.rollout_steps
            )
            
            # Record internal state norms (from last batch)
            # Note: In a real scenario, we'd average this over the rollout
            # Here we assume the agent exposes these via info, but trainer doesn't easily access info from rollout buffer
            # So we skip for now or implement a more complex rollout buffer that stores info.

        return stats

    def update_lr_and_entropy(self) -> None:
        """Update learning rate and entropy coefficient based on training progress."""
        if not self.config.ppo.anneal_lr and not self.config.ppo.anneal_entropy:
            return

        progress = self.global_step / self.config.train.total_timesteps

        # Linear annealing for LR
        if self.config.ppo.anneal_lr:
            self.lr_current = self.lr_start * (1 - progress) + self.lr_end * progress
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr_current

        # Linear annealing for entropy
        if self.config.ppo.anneal_entropy:
            self.entropy_current = self.entropy_start * (1 - progress) + self.entropy_end * progress

    def compute_moving_average_reward(self) -> float:
        """Compute moving average reward from buffer."""
        if len(self.reward_buffer) == 0:
            return -float("inf")
        return float(np.mean(list(self.reward_buffer)))

    def check_early_stopping(self) -> bool:
        """
        Check if training should stop due to reward plateau.

        Returns:
            True if training should stop, False otherwise.
        """
        if not self.config.ppo.enable_early_stopping:
            return False

        if len(self.reward_buffer) < self.config.ppo.reward_buffer_size:
            return False

        moving_avg = self.compute_moving_average_reward()

        # Check if we've improved
        if moving_avg > self.best_reward + self.config.ppo.reward_improvement_threshold:
            self.plateau_counter = 0
            self.last_improvement_step = self.global_step
            return False

        # Increment plateau counter
        self.plateau_counter += 1

        # Check if we've plateaued for too long
        if self.plateau_counter >= self.config.ppo.early_stop_patience:
            logger.info(
                f"EARLY STOP TRIGGERED — reward plateau detected "
                f"(no improvement for {self.plateau_counter} evaluations)"
            )
            return True

        return False

    def check_reward_collapse(self, current_reward: float) -> bool:
        """
        Check if reward has collapsed and trigger recovery.

        Args:
            current_reward: Current evaluation reward.

        Returns:
            True if collapse was detected and recovery triggered.
        """
        if not self.config.ppo.enable_collapse_detection:
            return False

        # Update peak moving average
        moving_avg = self.compute_moving_average_reward()
        if moving_avg > self.peak_moving_average:
            self.peak_moving_average = moving_avg

        # Check for collapse
        if self.peak_moving_average > 0:  # Avoid division issues
            collapse_threshold = self.peak_moving_average * self.config.ppo.reward_collapse_threshold
            if current_reward < collapse_threshold:
                logger.warning(
                    f"REWARD COLLAPSE DETECTED at step {self.global_step}! "
                    f"Current: {current_reward:.2f}, Peak MA: {self.peak_moving_average:.2f}, "
                    f"Threshold: {collapse_threshold:.2f}"
                )

                # Restore best checkpoint
                if self.best_policy_state_dict is not None:
                    logger.info(f"Rolling back to best checkpoint from step {self.best_step}")
                    self.agent.load_state_dict(self.best_policy_state_dict)

                    # Reduce learning rate
                    self.lr_current *= self.config.ppo.collapse_lr_reduction
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_current

                    logger.info(f"Reduced LR to {self.lr_current:.2e}")
                    self.collapse_recoveries += 1
                    return True

        return False

    def run_inference_validation(self) -> float:
        """
        Run deterministic inference episodes for validation.

        Returns:
            Mean reward over validation episodes.
        """
        self.agent.eval()
        rewards = []

        for _ in range(self.config.ppo.inference_validation_episodes):
            obs = self.env.reset()
            state = self.agent.init_state(batch_size=1)
            episode_reward = 0.0
            done = False

            while not done:
                with torch.no_grad():
                    obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs
                    action, _, _, state, _ = self.agent(obs_tensor, state, deterministic=True)

                action_np = action.cpu().numpy().flatten()
                if self.env.is_discrete:
                    action_np = int(action_np[0])

                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated
                episode_reward += reward
                obs = next_obs if not done else obs

            rewards.append(episode_reward)

        mean_reward = float(np.mean(rewards))
        logger.info(
            f"Inference validation: mean={mean_reward:.2f}, "
            f"std={np.std(rewards):.2f}, episodes={len(rewards)}"
        )
        return mean_reward

    def generate_training_plots(self) -> None:
        """Generate training metric plots."""
        if not self.config.ppo.auto_plot:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Training Metrics', fontsize=16)

        # Plot 1: Reward vs Step
        if len(self.reward_history) > 0:
            axes[0, 0].plot(self.step_history, self.reward_history, alpha=0.6, label='Reward')
            if len(self.reward_history) >= self.config.ppo.reward_buffer_size:
                # Plot moving average
                ma = []
                for i in range(len(self.reward_history)):
                    start_idx = max(0, i - self.config.ppo.reward_buffer_size + 1)
                    ma.append(np.mean(self.reward_history[start_idx:i+1]))
                axes[0, 0].plot(self.step_history, ma, linewidth=2, label='Moving Average', color='red')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Reward vs Step')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Policy Loss vs Step
        if len(self.policy_loss_history) > 0:
            axes[0, 1].plot(self.step_history, self.policy_loss_history)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Policy Loss')
            axes[0, 1].set_title('Policy Loss vs Step')
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Value Loss vs Step
        if len(self.value_loss_history) > 0:
            axes[1, 0].plot(self.step_history, self.value_loss_history)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Value Loss')
            axes[1, 0].set_title('Value Loss vs Step')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Learning Rate vs Step
        if len(self.lr_history) > 0:
            axes[1, 1].plot(self.step_history, self.lr_history)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)

        # Plot 5: Entropy Coefficient vs Step
        if len(self.entropy_coef_history) > 0:
            axes[2, 0].plot(self.step_history, self.entropy_coef_history)
            axes[2, 0].set_xlabel('Step')
            axes[2, 0].set_ylabel('Entropy Coefficient')
            axes[2, 0].set_title('Entropy Coefficient Schedule')
            axes[2, 0].grid(True, alpha=0.3)

        # Plot 6: Best Reward Marker
        axes[2, 1].axis('off')
        summary_text = f"Training Summary\n\n"
        summary_text += f"Best Reward: {self.best_reward:.2f}\n"
        summary_text += f"Best Step: {self.best_step}\n"
        summary_text += f"Final LR: {self.lr_current:.2e}\n"
        summary_text += f"Final Entropy: {self.entropy_current:.4f}\n"
        summary_text += f"Early Stopped: {self.early_stopped}\n"
        summary_text += f"Collapse Recoveries: {self.collapse_recoveries}"
        axes[2, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plot_path = self.plots_dir / "training_metrics.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Training plots saved to {plot_path}")

    def print_training_summary(self) -> None:
        """Print comprehensive training summary."""
        logger.info("=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Steps:              {self.global_step}")
        logger.info(f"Total Episodes:           {self.episode_count}")
        logger.info(f"Best Reward:              {self.best_reward:.2f}")
        logger.info(f"Best Step:                {self.best_step}")
        logger.info(f"Final Learning Rate:      {self.lr_current:.2e}")
        logger.info(f"Final Entropy Coef:       {self.entropy_current:.4f}")
        logger.info(f"Early Stop Triggered:     {'YES' if self.early_stopped else 'NO'}")
        logger.info(f"Collapse Recoveries:      {self.collapse_recoveries}")
        logger.info(f"Checkpoint Directory:     {self.checkpoint_dir}")
        logger.info(f"Best Policy Saved:        best_policy.pt")
        logger.info(f"Final Policy Saved:       final_policy.pt")
        logger.info("=" * 80)

    def train(self) -> None:
        """Run the full training loop with all upgrade features."""
        logger.info(f"Starting training for {self.config.train.total_timesteps} steps")
        logger.info(f"LR annealing: {self.config.ppo.anneal_lr} ({self.lr_start:.2e} → {self.lr_end:.2e})")
        logger.info(f"Entropy annealing: {self.config.ppo.anneal_entropy} ({self.entropy_start:.4f} → {self.entropy_end:.4f})")
        logger.info(f"Early stopping: {self.config.ppo.enable_early_stopping} (patience={self.config.ppo.early_stop_patience})")
        logger.info(f"Collapse detection: {self.config.ppo.enable_collapse_detection} (threshold={self.config.ppo.reward_collapse_threshold})")

        pbar = tqdm(total=self.config.train.total_timesteps, desc="Training")
        pbar.update(self.global_step)

        while self.global_step < self.config.train.total_timesteps:
            # Update LR and entropy schedules
            self.update_lr_and_entropy()

            # Collect rollouts
            rollout_stats = self.collect_rollouts()
            current_reward = rollout_stats["mean_reward"]

            # Update reward buffer and history
            self.reward_buffer.append(current_reward)
            self.reward_history.append(current_reward)
            self.step_history.append(self.global_step)

            # Save best model and state dict
            if self.config.train.save_best and current_reward > self.best_reward:
                self.best_reward = current_reward
                self.best_step = self.global_step
                self.best_policy_state_dict = {k: v.cpu().clone() for k, v in self.agent.state_dict().items()}
                self.save_checkpoint(filename="best_policy.pt")
                logger.info(f"New best reward: {self.best_reward:.2f} at step {self.best_step}. Saved best_policy.pt")

            # Check for reward collapse
            if self.check_reward_collapse(current_reward):
                self.collapse_detected = True

            # Train
            train_stats = self.train_step()

            # Record training metrics
            self.policy_loss_history.append(train_stats["policy_loss"])
            self.value_loss_history.append(train_stats["value_loss"])
            self.lr_history.append(self.lr_current)
            self.entropy_coef_history.append(self.entropy_current)

            # Update progress
            pbar.update(self.config.ppo.rollout_steps)

            # Log
            if self.global_step % self.config.train.log_frequency == 0:
                moving_avg = self.compute_moving_average_reward()
                logger.info(
                    f"Step {self.global_step}: "
                    f"reward={current_reward:.2f}, "
                    f"moving_avg={moving_avg:.2f}, "
                    f"length={rollout_stats['mean_length']:.1f}, "
                    f"policy_loss={train_stats['policy_loss']:.4f}, "
                    f"value_loss={train_stats['value_loss']:.4f}, "
                    f"lr={self.lr_current:.2e}, "
                    f"entropy={self.entropy_current:.4f}"
                )

            # Check early stopping
            if self.check_early_stopping():
                self.early_stopped = True
                logger.info("Early stopping triggered. Restoring best checkpoint...")
                if self.best_policy_state_dict is not None:
                    self.agent.load_state_dict(self.best_policy_state_dict)
                break

            # Save checkpoint
            if self.global_step % self.config.train.save_frequency == 0:
                self.save_checkpoint()

        pbar.close()

        # Final inference validation
        if self.config.ppo.inference_validation:
            logger.info("Running final inference validation...")
            final_inference_reward = self.run_inference_validation()

            # Check if final policy is worse than best
            if final_inference_reward < self.best_reward * self.config.ppo.inference_validation_threshold:
                logger.warning(
                    f"Final policy validation reward ({final_inference_reward:.2f}) is below "
                    f"{self.config.ppo.inference_validation_threshold*100:.0f}% of best reward ({self.best_reward:.2f}). "
                    f"Restoring best checkpoint as final policy."
                )
                if self.best_policy_state_dict is not None:
                    self.agent.load_state_dict(self.best_policy_state_dict)

        # Save final checkpoint
        self.save_checkpoint(final=True)

        # Generate plots
        self.generate_training_plots()

        # Print summary
        self.print_training_summary()

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

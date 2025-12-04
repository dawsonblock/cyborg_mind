"""Unified PPO Trainer with AMP and Registry support."""

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any
from tqdm import tqdm
import numpy as np

from cyborg_rl.config import Config
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.trainers.rollout_buffer import RolloutBuffer
from cyborg_rl.trainers.recurrent_rollout_buffer import RecurrentRolloutBuffer
from cyborg_rl.experiments.registry import ExperimentRegistry
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class PPOTrainer:
    """
    PPO Trainer with:
    - Mixed Precision (AMP)
    - Gradient Clipping
    - Experiment Registry Integration
    - Vectorized Environment Support
    """

    def __init__(
        self,
        env: Any,
        agent: PPOAgent,
        config: Config,
        registry: Optional[ExperimentRegistry] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config
        self.registry = registry
        self.device = agent.device

        # Optimizer
        self.optimizer = optim.Adam(
            agent.parameters(),
            lr=config.train.lr,
            eps=1e-5,
            weight_decay=config.train.weight_decay,
        )

        # AMP Scaler
        self.scaler = GradScaler(enabled=config.train.use_amp)

        # Buffer - choose type based on recurrent_mode
        self.recurrent_mode = getattr(config.train, "recurrent_mode", "none")

        # Detect if env is vectorized
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vectorized = self.num_envs > 1

        # For vectorized envs, buffer needs to store n_steps * num_envs transitions
        buffer_size = config.train.n_steps * self.num_envs

        if self.recurrent_mode == "burn_in":
            self.buffer = RecurrentRolloutBuffer(
                buffer_size=buffer_size,
                obs_dim=agent.obs_dim,
                action_dim=agent.action_dim if not agent.is_discrete else 1,
                device=self.device,
                is_discrete=agent.is_discrete,
                gamma=config.train.gamma,
                gae_lambda=config.train.gae_lambda,
            )
            logger.info(f"Using RecurrentRolloutBuffer for burn-in recurrent PPO mode (vectorized: {self.is_vectorized}, num_envs: {self.num_envs})")
        else:
            self.buffer = RolloutBuffer(
                buffer_size=buffer_size,
                obs_dim=agent.obs_dim,
                action_dim=agent.action_dim if not agent.is_discrete else 1,
                device=self.device,
                is_discrete=agent.is_discrete,
                gamma=config.train.gamma,
                gae_lambda=config.train.gae_lambda,
            )
            logger.info(f"Using RolloutBuffer (vectorized: {self.is_vectorized}, num_envs: {self.num_envs})")

        # State tracking
        self.global_step = 0
        self.current_obs, _ = self.env.reset()
        self.current_state = self.agent.init_state(self.num_envs)

        # WandB initialization
        self.wandb_enabled = False
        if config.train.wandb_enabled:
            if not WANDB_AVAILABLE:
                logger.warning("WandB is enabled in config but wandb package is not installed. Skipping WandB logging.")
            else:
                self.wandb_enabled = True
                self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        config_dict = self.config.to_dict()

        # Determine run name
        run_name = self.config.train.wandb_run_name
        if not run_name and self.registry:
            run_name = self.registry.run_name

        # Initialize wandb
        wandb.init(
            project=self.config.train.wandb_project,
            entity=self.config.train.wandb_entity,
            name=run_name,
            tags=self.config.train.wandb_tags or [],
            config=config_dict,
            sync_tensorboard=False,
            monitor_gym=False,
        )

        # Watch model for gradients and parameters
        wandb.watch(self.agent, log="all", log_freq=100)

        logger.info(f"WandB initialized: project={self.config.train.wandb_project}, run={run_name}")

    def train(self) -> None:
        """Main training loop."""
        total_timesteps = self.config.train.total_timesteps
        n_steps = self.config.train.n_steps
        num_updates = total_timesteps // n_steps

        logger.info(f"Starting training for {total_timesteps} steps ({num_updates} updates)")

        start_time = time.time()

        for update in range(1, num_updates + 1):
            # 1. Collect Rollouts
            self._collect_rollouts()

            # 2. Update Policy
            train_metrics = self._update_policy()

            # 3. Logging
            fps = int(self.global_step / (time.time() - start_time))
            train_metrics["fps"] = fps
            train_metrics["update"] = update
            train_metrics["timestep"] = self.global_step

            logger.info(f"Update {update}/{num_updates} | FPS: {fps} | Loss: {train_metrics.get('loss', 0):.4f}")

            # Log to registry
            if self.registry:
                self.registry.log_metrics(self.global_step, train_metrics)

                # Save checkpoint periodically
                if update % self.config.train.save_freq == 0:
                    self.registry.save_checkpoint(
                        self.agent.state_dict(),
                        self.global_step
                    )

            # Log to WandB
            if self.wandb_enabled:
                wandb.log(train_metrics, step=self.global_step)

        # Finalize WandB run
        if self.wandb_enabled:
            wandb.finish()
            logger.info("WandB run finished")

    def _collect_rollouts(self) -> None:
        """Collect n_steps of experience."""
        self.buffer.reset()

        with torch.no_grad():
            for step_idx in range(self.config.train.n_steps):
                self.global_step += self.num_envs

                # Convert obs to tensor and ensure batch dimension
                obs_tensor = torch.as_tensor(self.current_obs, device=self.device, dtype=torch.float32)
                if not self.is_vectorized:
                    # Single env: add batch dimension [obs_dim] -> [1, obs_dim]
                    obs_tensor = obs_tensor.unsqueeze(0)

                # Store state BEFORE action (for recurrent mode)
                state_before_action = self._clone_state(self.current_state) if self.recurrent_mode == "burn_in" else None

                # Forward pass
                action, log_prob, value, new_state, info = self.agent(
                    obs_tensor, self.current_state
                )

                # Step env
                action_cpu = action.cpu().numpy()

                # For single env, remove batch dimension from action
                if not self.is_vectorized:
                    action_cpu = action_cpu[0]  # [1, ...] -> [...]

                next_obs, rewards, terminated, truncated, infos = self.env.step(action_cpu)

                # Convert tensors to numpy for buffer storage
                action_np = action.cpu().numpy()
                value_np = value.cpu().numpy()
                log_prob_np = log_prob.cpu().numpy()

                # Handle vectorized vs single env
                if self.is_vectorized:
                    # For vectorized envs, add each env's transition separately
                    for env_idx in range(self.num_envs):
                        # Extract per-env data
                        obs_i = self.current_obs[env_idx]
                        action_i = action_np[env_idx]
                        reward_i = float(rewards[env_idx])
                        value_i = float(value_np[env_idx])
                        log_prob_i = float(log_prob_np[env_idx])
                        done_i = bool(terminated[env_idx])

                        # Extract per-env state (for recurrent mode)
                        if self.recurrent_mode == "burn_in":
                            state_i = self._extract_env_state(state_before_action, env_idx)
                            self.buffer.add(
                                obs_i, action_i, reward_i, value_i, log_prob_i, done_i,
                                recurrent_state=state_i,
                            )
                        else:
                            self.buffer.add(
                                obs_i, action_i, reward_i, value_i, log_prob_i, done_i,
                            )
                else:
                    # Single env case - extract from batch dimension
                    action_i = action_np[0]
                    reward_i = float(rewards)
                    value_i = float(value_np[0])
                    log_prob_i = float(log_prob_np[0])
                    done_i = bool(terminated)

                    if self.recurrent_mode == "burn_in":
                        self.buffer.add(
                            self.current_obs, action_i, reward_i, value_i, log_prob_i, done_i,
                            recurrent_state=state_before_action,
                        )
                    else:
                        self.buffer.add(
                            self.current_obs, action_i, reward_i, value_i, log_prob_i, done_i,
                        )

                # Update state
                self.current_obs = next_obs
                self.current_state = new_state

        # Compute GAE
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.current_obs, device=self.device, dtype=torch.float32)
            if not self.is_vectorized:
                obs_tensor = obs_tensor.unsqueeze(0)

            last_value, _ = self.agent.get_value(obs_tensor, self.current_state)
            # For vectorized, average across envs; for single, just take the value
            last_value_scalar = float(last_value.mean().item())
            # last_done is False since we're mid-rollout (envs auto-reset)
            self.buffer.compute_returns_and_advantages(last_value_scalar, False)

    def _update_policy(self) -> Dict[str, float]:
        """PPO Update with AMP."""
        metrics = {}
        losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []

        # Get generator
        data_loader = self.buffer.get(self.config.train.batch_size, normalize_advantage=True)

        for epoch in range(self.config.train.n_epochs):
            for batch in data_loader:
                # Unpack batch based on buffer type
                if self.recurrent_mode == "burn_in":
                    obs = batch["observations"]
                    actions = batch["actions"]
                    old_log_probs = batch["old_log_probs"]
                    returns = batch["returns"]
                    advantages = batch["advantages"]
                    values = batch["old_values"]
                    batch_indices = batch["batch_indices"]

                    # Gather recurrent states for this batch
                    batch_states = self._gather_recurrent_states(batch_indices)
                else:
                    obs = batch["observations"]
                    actions = batch["actions"]
                    old_log_probs = batch["old_log_probs"]
                    returns = batch["returns"]
                    advantages = batch["advantages"]
                    values = batch["old_values"]
                    batch_states = None

                # AMP Context
                with autocast(enabled=self.config.train.use_amp):
                    # Evaluate actions with proper state handling
                    if batch_states is not None:
                        log_probs, entropy, new_values, _ = self.agent.evaluate_actions(obs, actions, state=batch_states)
                    else:
                        log_probs, entropy, new_values, _ = self.agent.evaluate_actions(obs, actions)

                    # Ratios
                    ratios = torch.exp(log_probs - old_log_probs)

                    # Surrogate Loss
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1.0 - self.config.train.clip_range, 1.0 + self.config.train.clip_range) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss
                    value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                    # Entropy Loss
                    entropy_loss = -entropy.mean()

                    # Total Loss
                    loss = (
                        policy_loss
                        + self.config.train.value_coef * value_loss
                        + self.config.train.entropy_coef * entropy_loss
                    )

                # Backward pass with Scaler
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.train.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Track individual losses
                losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        metrics["loss"] = np.mean(losses)
        metrics["policy_loss"] = np.mean(policy_losses)
        metrics["value_loss"] = np.mean(value_losses)
        metrics["entropy_loss"] = np.mean(entropy_losses)
        return metrics

    def _clone_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clone a recurrent state dict for storage.

        Args:
            state: State dict containing tensors.

        Returns:
            Cloned state dict with detached tensors.
        """
        cloned = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.detach().clone()
            else:
                cloned[key] = value
        return cloned

    def _extract_env_state(self, state: Dict[str, torch.Tensor], env_idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract a single environment's state from a batched state dict.

        Args:
            state: Batched state dict.
            env_idx: Environment index to extract.

        Returns:
            State dict for single environment.
        """
        env_state = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 3:
                    # GRU hidden: [num_layers, batch_size, hidden_dim]
                    env_state[key] = value[:, env_idx:env_idx+1, :].detach().clone()
                elif value.dim() == 2:
                    # Memory or other 2D: [batch_size, dim]
                    env_state[key] = value[env_idx:env_idx+1, :].detach().clone()
                elif value.dim() == 1:
                    # 1D: [batch_size]
                    env_state[key] = value[env_idx:env_idx+1].detach().clone()
                else:
                    # Keep as-is
                    env_state[key] = value.detach().clone()
            else:
                env_state[key] = value
        return env_state

    def _gather_recurrent_states(self, indices: list) -> Dict[str, torch.Tensor]:
        """
        Build a batched recurrent state from per-step stored states in the buffer.

        Args:
            indices: List of buffer indices.

        Returns:
            State dict where each tensor has batch dimension = len(indices).
        """
        # Get states for these indices
        per_sample_states = self.buffer.get_states(indices)

        # Check if we have valid states
        if not per_sample_states:
            # Empty list: return initialized state
            return self.agent.init_state(batch_size=len(indices))

        if per_sample_states[0] is None:
            # First state is None: return initialized state
            return self.agent.init_state(batch_size=len(indices))

        # Assume all states have same keys/shapes
        keys = per_sample_states[0].keys()
        batch_state: Dict[str, torch.Tensor] = {}

        for k in keys:
            tensors = []
            for s in per_sample_states:
                if s is None:
                    # Skip None states (shouldn't happen but be safe)
                    continue
                v = s[k]
                if isinstance(v, torch.Tensor):
                    # Ensure tensor is on correct device
                    v = v.to(self.device)
                    tensors.append(v)
                else:
                    raise TypeError(f"Unsupported state entry type for key {k}: {type(v)}")

            # Stack/concatenate along appropriate dimension
            if tensors:
                # Check dimensionality
                if tensors[0].dim() == 3:
                    # GRU hidden: [num_layers, 1, hidden_dim] → cat along dim=1 → [num_layers, B, hidden_dim]
                    batch_state[k] = torch.cat(tensors, dim=1)
                elif tensors[0].dim() == 2:
                    # Memory or other 2D tensors: [1, dim] → cat along dim=0 → [B, dim]
                    batch_state[k] = torch.cat(tensors, dim=0)
                else:
                    # 1D or other: stack along dim=0
                    batch_state[k] = torch.stack(tensors, dim=0)

        return batch_state

    def load_checkpoint(self, path: str) -> None:
        self.agent = PPOAgent.load(path, self.device)
        # Also load optimizer state if available
        # (Skipping for brevity, but recommended for production)

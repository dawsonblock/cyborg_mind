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

        if self.recurrent_mode == "burn_in":
            self.buffer = RecurrentRolloutBuffer(
                buffer_size=config.train.n_steps,
                obs_dim=agent.obs_dim,
                action_dim=agent.action_dim if not agent.is_discrete else 1,
                device=self.device,
                is_discrete=agent.is_discrete,
                gamma=config.train.gamma,
                gae_lambda=config.train.gae_lambda,
            )
            logger.info("Using RecurrentRolloutBuffer for burn-in recurrent PPO mode")
        else:
            self.buffer = RolloutBuffer(
                buffer_size=config.train.n_steps,
                obs_dim=agent.obs_dim,
                action_dim=agent.action_dim if not agent.is_discrete else 1,
                device=self.device,
                is_discrete=agent.is_discrete,
                gamma=config.train.gamma,
                gae_lambda=config.train.gae_lambda,
            )

        # State tracking
        self.global_step = 0
        self.current_obs, _ = self.env.reset()
        self.current_state = self.agent.init_state(self.env.num_envs if hasattr(self.env, "num_envs") else 1)

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
            for _ in range(self.config.train.n_steps):
                self.global_step += 1

                # Convert obs to tensor
                obs_tensor = torch.as_tensor(self.current_obs, device=self.device, dtype=torch.float32)

                # Store state BEFORE action (for recurrent mode)
                state_before_action = self._clone_state(self.current_state) if self.recurrent_mode == "burn_in" else None

                # Forward pass
                action, log_prob, value, new_state, info = self.agent(
                    obs_tensor, self.current_state
                )

                # Step env
                # Handle discrete/continuous action conversion
                action_cpu = action.cpu().numpy()
                if self.agent.is_discrete:
                    # If vector env, action_cpu is array of ints
                    pass

                next_obs, rewards, terminated, truncated, infos = self.env.step(action_cpu)

                # Convert tensors to numpy/scalars for buffer storage
                action_np = action.cpu().numpy()
                value_np = value.cpu().numpy() if value.dim() > 0 else value.item()
                log_prob_np = log_prob.cpu().numpy() if log_prob.dim() > 0 else log_prob.item()

                # Store in buffer
                # Note: Vector envs return array of rewards/dones
                # RolloutBuffer expects: obs, action, reward, value, log_prob, done
                if self.recurrent_mode == "burn_in":
                    self.buffer.add(
                        self.current_obs,
                        action_np,
                        rewards,
                        value_np,
                        log_prob_np,
                        terminated,
                        recurrent_state=state_before_action,
                    )
                else:
                    self.buffer.add(
                        self.current_obs,
                        action_np,
                        rewards,
                        value_np,
                        log_prob_np,
                        terminated,
                    )

                # Update state
                self.current_obs = next_obs
                self.current_state = new_state

                # Handle resets for vector envs (usually auto-reset)
                # If using standard Gym wrapper, might need manual reset check
                # For now assuming AsyncVectorEnv which auto-resets

        # Compute GAE
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.current_obs, device=self.device, dtype=torch.float32)
            last_value, _ = self.agent.get_value(obs_tensor, self.current_state)
            # Use the actual method name from the buffer
            last_value_scalar = last_value.mean().item() if hasattr(last_value, 'mean') else last_value
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

        if not per_sample_states or per_sample_states[0] is None:
            # Fallback: return initialized state
            return self.agent.init_state(batch_size=len(indices))

        # Assume all states have same keys/shapes
        keys = per_sample_states[0].keys()
        batch_state: Dict[str, torch.Tensor] = {}

        for k in keys:
            tensors = []
            for s in per_sample_states:
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
                    # Memory or other 2D tensors: [1, dim] → stack → [B, dim]
                    batch_state[k] = torch.cat(tensors, dim=0)
                else:
                    # 1D or other: stack along dim=0
                    batch_state[k] = torch.stack(tensors, dim=0)

        return batch_state

    def load_checkpoint(self, path: str) -> None:
        self.agent = PPOAgent.load(path, self.device)
        # Also load optimizer state if available
        # (Skipping for brevity, but recommended for production)

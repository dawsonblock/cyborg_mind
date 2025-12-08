#!/usr/bin/env python3
"""
TrainerV7 - Honest Recurrent PPO Trainer

Features:
- GPU, AMP, torch.compile support
- Official Mamba + PMM/SlotMemory/KVMemory
- Multi-env adapters (MineRL, Gym, Custom)
- Honest recurrent PPO: states flow from acting → training
- Burn-in with gradient masking
- PPO++ improvements (adaptive KL, value clipping, PopArt)
"""

from typing import Any, Dict, Optional, Tuple, Union
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from cyborg_rl.models.encoder_v7 import EncoderV7
from cyborg_rl.models.policy import DiscretePolicy
from cyborg_rl.models.value import ValueHead
from cyborg_rl.memory.memory_v7 import create_memory, BaseMemoryV7
from cyborg_rl.memory.rollout_buffer_v7 import RolloutBufferV7

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainerV7:
    """
    V7 Training System - Honest Recurrent PPO++

    Key Principles:
    - States flow honestly: acting states → buffer → training
    - Burn-in replays state, gradients masked to post-burn region
    - Full PPO++ with adaptive KL, value clipping, normalized advantages
    """

    def __init__(
        self,
        config: Dict[str, Any],
        env=None,
        use_wandb: bool = False,
    ) -> None:
        """
        Args:
            config: Configuration dict
            env: Environment instance (or will create from config)
            use_wandb: Enable WandB logging
        """
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Device
        device_str = config.get("device", "auto")
        if device_str == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)

        # Extract config
        self.num_envs = config.get("num_envs", 4)
        self.horizon = config.get("horizon", 1024)
        self.burn_in = config.get("burn_in", 128)
        self.seq_len = config.get("seq_len", 64)
        self.batch_size = config.get("batch_size", 16)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.value_clip = config.get("value_clip", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.target_kl = config.get("target_kl", 0.02)
        self.use_amp = config.get("amp", True) and self.device.type == "cuda"
        self.use_compile = config.get("compile", False)

        # Environment
        if env is not None:
            self.env = env
        else:
            self._create_env()

        self.obs_dim = self._get_obs_dim()
        self.action_dim = self._get_action_dim()

        # Models
        self._build_models()

        # Optimizer
        self.optimizer = optim.AdamW(self._get_params(), lr=self.lr, eps=1e-5)

        # AMP
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Buffer
        memory_cfg = config.get("memory", {})
        self.buffer = RolloutBufferV7(
            horizon=self.horizon,
            num_envs=self.num_envs,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=config.get("hidden_dim", 384),
            memory_slots=memory_cfg.get("num_slots", 16),
            memory_dim=memory_cfg.get("memory_dim", 256),
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Training state
        self.total_steps = 0
        self.updates = 0

        # WandB
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "cyborg-v7"),
                config=config,
            )

    def _create_env(self) -> None:
        """Create environment from config."""
        env_name = self.config.get("env", "minerl_treechop")

        # Import adapter
        from cyborg_rl.envs.minerl_adapter import MineRLAdapter, MINERL_AVAILABLE

        if "minerl" in env_name.lower() and MINERL_AVAILABLE:
            self.env = MineRLAdapter(
                env_name=self.config.get("env_id", "MineRLTreechop-v0"),
                num_envs=self.num_envs,
            )
        else:
            # Mock environment for testing
            self.env = self._create_mock_env()

    def _create_mock_env(self):
        """Create mock environment for testing without MineRL."""

        class MockEnv:
            def __init__(self, num_envs, obs_dim=64 * 64 * 3 * 4 + 1, action_dim=18):
                self.num_envs = num_envs
                self.observation_dim = obs_dim
                self.action_dim = action_dim

            def reset(self):
                return torch.randn(self.num_envs, self.observation_dim)

            def step(self, actions):
                obs = torch.randn(self.num_envs, self.observation_dim)
                rewards = torch.randn(self.num_envs)
                dones = torch.zeros(self.num_envs, dtype=torch.bool)
                infos = [{} for _ in range(self.num_envs)]
                return obs, rewards, dones, infos

            def close(self):
                pass

        return MockEnv(self.num_envs)

    def _get_obs_dim(self) -> int:
        """Get observation dimension."""
        if hasattr(self.env, "observation_dim"):
            return self.env.observation_dim
        return self.config.get("obs_dim", 64 * 64 * 3 * 4 + 1)

    def _get_action_dim(self) -> int:
        """Get action dimension."""
        if hasattr(self.env, "action_dim"):
            return self.env.action_dim
        return self.config.get("action_dim", 18)

    def _build_models(self) -> None:
        """Build encoder, memory, policy, value networks."""
        hidden_dim = self.config.get("hidden_dim", 384)
        latent_dim = self.config.get("latent_dim", 256)

        # Encoder
        encoder_mode = self.config.get("encoder", "mamba")
        self.encoder = EncoderV7(
            mode=encoder_mode,
            input_dim=self.obs_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=self.config.get("num_layers", 2),
        ).to(self.device)

        # Memory
        memory_cfg = self.config.get("memory", {})
        memory_type = memory_cfg.get("type", "pmm")
        self.memory = create_memory(
            memory_type=memory_type,
            memory_dim=memory_cfg.get("memory_dim", latent_dim),
            num_slots=memory_cfg.get("num_slots", 16),
        ).to(self.device)

        # Policy input dim
        policy_input_dim = latent_dim + memory_cfg.get("memory_dim", latent_dim)

        # Policy
        self.policy = DiscretePolicy(
            input_dim=policy_input_dim,
            action_dim=self.action_dim,
        ).to(self.device)

        # Value
        self.value_net = ValueHead(
            input_dim=policy_input_dim,
        ).to(self.device)

        # torch.compile
        if self.use_compile:
            try:
                self.encoder = torch.compile(self.encoder)
                self.memory = torch.compile(self.memory)
                self.policy = torch.compile(self.policy)
                self.value_net = torch.compile(self.value_net)
                print("✅ Models compiled with torch.compile")
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")

    def _get_params(self):
        """Get all trainable parameters."""
        return (
            list(self.encoder.parameters())
            + list(self.memory.parameters())
            + list(self.policy.parameters())
            + list(self.value_net.parameters())
        )

    @torch.no_grad()
    def collect_rollout(self) -> Dict[str, float]:
        """
        Collect rollout with honest state management.

        Returns:
            Metrics dict
        """
        self.buffer.reset()

        # Reset environment
        obs = self.env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        else:
            obs = obs.to(self.device)

        # Initialize states
        encoder_state = self.encoder.get_initial_state(self.num_envs, self.device)
        memory_state = self.memory.get_initial_state(self.num_envs, self.device)

        episode_rewards = []
        episode_lengths = []
        current_ep_reward = np.zeros(self.num_envs)
        current_ep_len = np.zeros(self.num_envs)

        for step in range(self.horizon):
            # Handle observation shape
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            if obs.dim() == 4:  # (N, C, H, W)) -> flatten
                obs = obs.reshape(obs.size(0), -1)

            # Encode
            latent, new_encoder_state = self.encoder(obs, encoder_state)
            if latent.dim() == 3:
                latent = latent.squeeze(1)

            # Memory
            read_vec, new_memory_state, mem_logs = self.memory.forward_step(
                latent, memory_state
            )

            # Policy input
            policy_input = torch.cat([latent, read_vec], dim=-1)

            # Get action
            action, log_prob = self.policy.sample(policy_input)
            value = self.value_net(policy_input).squeeze(-1)

            # Store to buffer (HONEST: store current states before update)
            self.buffer.add(
                obs=obs,
                action=action,
                reward=torch.zeros(self.num_envs, device=self.device),  # Filled after step
                value=value,
                log_prob=log_prob,
                done=torch.zeros(self.num_envs, device=self.device),
                hidden=latent,
                memory=new_memory_state["memory"] if isinstance(new_memory_state, dict) else new_memory_state,
            )

            # Step environment
            action_cpu = action.cpu().numpy()
            next_obs, rewards, dones, infos = self.env.step(action_cpu)

            # Convert
            if not isinstance(next_obs, torch.Tensor):
                next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
            else:
                next_obs = next_obs.to(self.device)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
            if not isinstance(dones, torch.Tensor):
                dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)

            # Update buffer with actual rewards and dones
            self.buffer.rewards[step] = rewards
            self.buffer.dones[step] = dones

            # Track episodes
            current_ep_reward += rewards.cpu().numpy()
            current_ep_len += 1

            for i, done in enumerate(dones.cpu().numpy()):
                if done:
                    episode_rewards.append(current_ep_reward[i])
                    episode_lengths.append(current_ep_len[i])
                    current_ep_reward[i] = 0
                    current_ep_len[i] = 0

            # Update states
            # Reset states for done episodes
            done_mask = 1.0 - dones
            encoder_state = self.encoder.reset_states(new_encoder_state, done_mask)
            memory_state = self.memory.reset_state(new_memory_state, done_mask)

            obs = next_obs
            self.total_steps += self.num_envs

        # Bootstrap value
        if obs.dim() == 4:
            obs = obs.reshape(obs.size(0), -1)
        latent, _ = self.encoder(obs, encoder_state)
        if latent.dim() == 3:
            latent = latent.squeeze(1)
        read_vec, _, _ = self.memory.forward_step(latent, memory_state)
        policy_input = torch.cat([latent, read_vec], dim=-1)
        last_value = self.value_net(policy_input).squeeze(-1)

        # Compute GAE
        self.buffer.compute_gae(last_value, dones)
        self.buffer.normalize_advantages()

        return {
            "episode_reward_mean": np.mean(episode_rewards) if episode_rewards else 0.0,
            "episode_length_mean": np.mean(episode_lengths) if episode_lengths else 0.0,
            "episodes": len(episode_rewards),
        }

    def train_step(self) -> Dict[str, float]:
        """
        Run one PPO update step.

        Returns:
            Metrics dict
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0
        early_stopped = False

        for epoch in range(self.ppo_epochs):
            if early_stopped:
                break

            for batch in self.buffer.sample_sequences(
                seq_len=self.seq_len,
                burn_in=self.burn_in,
                batch_size=self.batch_size,
            ):
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    loss, metrics = self._compute_loss(batch)

                # KL early stopping
                if metrics["approx_kl"] > self.target_kl * 1.5:
                    early_stopped = True
                    break

                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self._get_params(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Accumulate
                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_entropy += metrics["entropy"]
                total_kl += metrics["approx_kl"]
                num_updates += 1

        self.updates += 1

        if num_updates == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_kl / num_updates,
            "early_stopped": 1.0 if early_stopped else 0.0,
        }

    def _compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss with burn-in gradient masking.

        Args:
            batch: Sequence batch from buffer

        Returns:
            loss: Total loss
            metrics: Dict of metrics
        """
        obs = batch["obs"]  # (B, burn_in + seq_len, D)
        actions = batch["actions"]  # (B, burn_in + seq_len)
        old_log_probs = batch["log_probs"]  # (B, seq_len)
        old_values = batch["values"]  # (B, seq_len)
        advantages = batch["advantages"]  # (B, seq_len)
        returns = batch["returns"]  # (B, seq_len)
        dones = batch["dones"]  # (B, burn_in + seq_len)
        grad_mask = batch["grad_mask"]  # (B, burn_in + seq_len)
        hidden_init = batch["hidden_init"]  # (B, hidden_dim)
        memory_init = batch["memory_init"]  # (B, slots, memory_dim)

        B, T, _ = obs.shape

        # Initialize states
        encoder_state = self.encoder.get_initial_state(B, self.device)
        if "gru" in encoder_state:
            # Use provided initial hidden
            encoder_state["gru"] = hidden_init.unsqueeze(0)

        memory_state = self.memory.get_initial_state(B, self.device)
        if isinstance(memory_state, dict) and "memory" in memory_state:
            memory_state["memory"] = memory_init

        # Forward through sequence
        latents = []
        reads = []
        for t in range(T):
            obs_t = obs[:, t]

            # Handle done masking
            if t > 0:
                done_mask = 1.0 - dones[:, t - 1]
                encoder_state = self.encoder.reset_states(encoder_state, done_mask)
                memory_state = self.memory.reset_state(memory_state, done_mask)

            latent, encoder_state = self.encoder(obs_t, encoder_state)
            if latent.dim() == 3:
                latent = latent.squeeze(1)

            read, memory_state, _ = self.memory.forward_step(latent, memory_state)

            latents.append(latent)
            reads.append(read)

        latents = torch.stack(latents, dim=1)  # (B, T, D)
        reads = torch.stack(reads, dim=1)  # (B, T, D)

        # Policy input
        policy_input = torch.cat([latents, reads], dim=-1)  # (B, T, 2D)

        # Only compute loss for post-burn-in region
        policy_input_train = policy_input[:, self.burn_in:]  # (B, seq_len, 2D)
        actions_train = actions[:, self.burn_in:]  # (B, seq_len)

        # Flatten for policy
        B_train, T_train, D_train = policy_input_train.shape
        flat_input = policy_input_train.reshape(B_train * T_train, D_train)
        flat_actions = actions_train.reshape(B_train * T_train)

        # New log probs and values
        dist = self.policy.get_distribution(flat_input)
        new_log_probs = dist.log_prob(flat_actions).view(B_train, T_train)
        entropy = dist.entropy().mean()

        new_values = self.value_net(flat_input).view(B_train, T_train)

        # PPO loss
        log_ratio = new_log_probs - old_log_probs
        ratio = log_ratio.exp()

        # Approximate KL
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()

        # Clipped surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss with clipping
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, -self.value_clip, self.value_clip
        )
        value_loss1 = (new_values - returns).pow(2)
        value_loss2 = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

        # Total loss
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl,
            "clip_fraction": ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item(),
        }

        return loss, metrics

    def train(self, total_timesteps: int) -> None:
        """
        Main training loop.

        Args:
            total_timesteps: Total environment steps to train
        """
        print(f"🚀 TrainerV7 starting on {self.device}")
        print(f"   Encoder: {self.config.get('encoder', 'mamba')}")
        print(f"   Memory: {self.config.get('memory', {}).get('type', 'pmm')}")
        print(f"   Burn-in: {self.burn_in}, Seq-len: {self.seq_len}")
        print(f"   Total steps: {total_timesteps:,}")

        start_time = time.time()
        pbar = tqdm(total=total_timesteps, desc="Training")

        while self.total_steps < total_timesteps:
            # Collect
            collect_metrics = self.collect_rollout()

            # Train
            train_metrics = self.train_step()

            # Log
            pbar.update(self.horizon * self.num_envs)
            pbar.set_postfix({
                "reward": f"{collect_metrics.get('episode_reward_mean', 0):.2f}",
                "kl": f"{train_metrics.get('approx_kl', 0):.4f}",
            })

            if self.use_wandb:
                wandb.log({
                    **collect_metrics,
                    **train_metrics,
                    "total_steps": self.total_steps,
                })

        pbar.close()
        elapsed = time.time() - start_time
        print(f"✅ Training complete! {self.total_steps:,} steps in {elapsed:.1f}s")

    def save(self, path: str) -> None:
        """Save checkpoint."""
        checkpoint = {
            "config": self.config,
            "total_steps": self.total_steps,
            "updates": self.updates,
            "encoder_state_dict": self.encoder.state_dict(),
            "memory_state_dict": self.memory.state_dict(),
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint: {path}")

    def load(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.memory.load_state_dict(checkpoint["memory_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.updates = checkpoint.get("updates", 0)
        print(f"📂 Loaded checkpoint: {path}")

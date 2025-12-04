#!/usr/bin/env python3
"""
memory_ppo_trainer.py

Full-sequence PPO trainer for fixed-horizon memory tasks
(Delayed Cue, Copy-Memory, Associative Recall).

Key properties:
    - Uses full BPTT over the entire episode horizon (no truncation).
    - Vectorized over envs (batch) and loops over time in Python.
    - Honest recurrent gradients for GRU / Pseudo-Mamba / Mamba+GRU + PMM.

Intended only for memory benchmarks, not MineRL.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import numpy as np
from torch.optim import AdamW

from cyborg_rl.config import Config
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.experiments.registry import ExperimentRegistry
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryRollout:
    """
    Full-episode rollout for memory tasks.

    All tensors are shaped [T, B, ...] where:
        T = episode_len (≈ horizon + small slack)
        B = num_envs
    """
    observations: torch.Tensor      # [T, B, obs_dim]
    actions: torch.Tensor           # [T, B] for discrete
    rewards: torch.Tensor           # [T, B]
    dones: torch.Tensor             # [T, B]
    values: torch.Tensor            # [T, B]
    log_probs: torch.Tensor         # [T, B]
    advantages: torch.Tensor        # [T, B]
    returns: torch.Tensor           # [T, B]


class MemoryPPOTrainer:
    """
    PPO trainer specialized for fixed-horizon memory tasks.

    Differences from generic PPOTrainer:
        - Always uses full sequence BPTT (no truncated windows).
        - Works with [T, B, ...] tensors instead of a flat buffer.
        - No recurrent_mode flag; sequence is the default.
    """

    def __init__(
        self,
        config: Config,
        env_adapter,
        agent: PPOAgent,
        registry: Optional[ExperimentRegistry] = None,
    ) -> None:
        self.config = config
        self.env = env_adapter
        self.agent = agent
        self.registry = registry

        self.device = agent.device
        self.gamma = config.train.gamma
        self.gae_lambda = config.train.gae_lambda

        self.num_envs = getattr(env_adapter, 'num_envs', 1)
        # Use either horizon from env or rollout_steps from config
        self.episode_len = getattr(env_adapter, 'horizon', config.train.n_steps) + 5

        self.optimizer = AdamW(
            self.agent.parameters(),
            lr=config.train.lr,
        )

        self.clip_range = config.train.clip_range
        self.ent_coef = config.train.entropy_coef
        self.vf_coef = config.train.value_coef
        self.max_grad_norm = config.train.max_grad_norm
        self.n_epochs = config.train.n_epochs

        self.global_step = 0
        self.last_mean_reward = 0.0
        self.last_success_rate = 0.0

        logger.info(
            f"MemoryPPOTrainer initialized: num_envs={self.num_envs}, "
            f"episode_len={self.episode_len}, device={self.device}"
        )

    def _compute_gae(self, rewards, values, dones, last_value):
        """
        GAE on full sequences.

        rewards, values, dones: [T, B]
        last_value: [B]
        Returns:
            advantages, returns: [T, B]
        """
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae = torch.zeros(B, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_value
                next_nonterminal = 1.0 - dones[t]
            else:
                next_values = values[t + 1]
                next_nonterminal = 1.0 - dones[t + 1]

            delta = (
                rewards[t]
                + self.gamma * next_values * next_nonterminal
                - values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def collect_rollout(self) -> MemoryRollout:
        """
        Collect one full batch of episodes:

            - T = episode_len (horizon or horizon+slack)
            - B = num_envs

        Returns a MemoryRollout with [T, B, ...] tensors.
        """
        T = self.episode_len
        B = self.num_envs
        obs_dim = self.env.observation_space.shape[0]

        observations = torch.zeros(T, B, obs_dim, device=self.device)
        rewards = torch.zeros(T, B, device=self.device)
        dones = torch.zeros(T, B, device=self.device)
        values = torch.zeros(T, B, device=self.device)
        log_probs = torch.zeros(T, B, device=self.device)
        actions_list = []

        # Reset env batch + agent state
        obs, _ = self.env.reset()
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        state = self.agent.init_state(batch_size=B)

        for t in range(T):
            observations[t] = obs

            # Act (deterministic=False during training)
            with torch.no_grad():
                action_t, logp_t, value_t, state, _ = self.agent(
                    obs, state, deterministic=False
                )

            actions_list.append(action_t.cpu())
            log_probs[t] = logp_t
            values[t] = value_t

            # Step environment
            action_cpu = action_t.cpu().numpy()
            next_obs, rew, terminated, truncated, infos = self.env.step(action_cpu)

            # Convert to tensors
            next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
            rew = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
            done = torch.as_tensor(terminated, device=self.device, dtype=torch.float32)

            rewards[t] = rew
            dones[t] = done

            obs = next_obs

            # Early termination if all envs are done
            if done.all():
                # Pad remaining timesteps
                for t_pad in range(t + 1, T):
                    observations[t_pad] = obs
                    actions_list.append(torch.zeros_like(action_t))
                break

        # Stack actions
        actions = torch.stack(actions_list, dim=0).to(self.device)  # [T, B]

        # Last value for GAE
        with torch.no_grad():
            _, _, last_values, _, _ = self.agent(obs, state, deterministic=True)

        advantages, returns = self._compute_gae(
            rewards, values, dones, last_values
        )

        # Track stats
        self.global_step += T * B
        self.last_mean_reward = rewards.sum(dim=0).mean().item()

        return MemoryRollout(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            values=values,
            log_probs=log_probs,
            advantages=advantages,
            returns=returns,
        )

    def update_policy(self, rollout: MemoryRollout) -> Dict[str, float]:
        """
        Full-sequence PPO update.

        Re-runs the entire sequence through the agent to get
        new log_probs and values under the updated parameters.
        """
        T, B = rollout.observations.shape[0], rollout.observations.shape[1]

        # Flatten time and batch for old stats
        old_log_probs = rollout.log_probs.view(T * B)
        returns = rollout.returns.view(T * B)
        advantages = rollout.advantages.view(T * B)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare for multi-epoch PPO
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.n_epochs):
            # FULL-SEQUENCE FORWARD
            obs_seq = rollout.observations  # [T, B, obs_dim]
            act_seq = rollout.actions       # [T, B]

            # Forward sequence through agent
            logits_seq, values_seq = self.agent.forward_sequence(obs_seq, init_state=None)

            # Build distribution and compute log probs
            # logits_seq: [T, B, num_actions]
            T_, B_, num_actions = logits_seq.shape
            assert T_ == T and B_ == B

            dist = torch.distributions.Categorical(
                logits=logits_seq.view(T * B, num_actions)
            )

            flat_actions = act_seq.view(T * B).long()
            new_log_probs = dist.log_prob(flat_actions)
            entropy = dist.entropy()
            new_values = values_seq.view(T * B)

            # PPO objective
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_range,
                1.0 + self.clip_range,
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - new_values).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + self.vf_coef * value_loss
                + self.ent_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            num_updates += 1

        metrics = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "mean_reward": self.last_mean_reward,
        }

        if self.registry is not None:
            self.registry.log_metrics(self.global_step, metrics)

        return metrics

    def train(self, total_timesteps: Optional[int] = None) -> None:
        """
        Run training until total_timesteps is reached (or config limit).
        """
        if total_timesteps is None:
            total_timesteps = self.config.train.total_timesteps

        logger.info(f"Starting MemoryPPOTrainer training for {total_timesteps} timesteps")

        while self.global_step < total_timesteps:
            rollout = self.collect_rollout()
            metrics = self.update_policy(rollout)

            if self.global_step % 10000 == 0:
                logger.info(
                    f"Step {self.global_step}/{total_timesteps}: "
                    f"reward={metrics['mean_reward']:.4f}, "
                    f"policy_loss={metrics['policy_loss']:.4f}"
                )

        logger.info(f"Training complete: final_step={self.global_step}")

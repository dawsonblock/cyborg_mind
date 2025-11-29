# cyborg_mind_v2/training/train_cyborg_mind_ppo.py

import os
import dataclasses
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gym
import minerl

from torch.utils.tensorboard import SummaryWriter

from ..capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind
from ..envs.minerl_obs_adapter import obs_to_brain
from ..envs.action_mapping import NUM_ACTIONS, index_to_minerl_action


@dataclasses.dataclass
class PPOConfig:
    env_name: str = "MineRLTreechop-v0"
    total_steps: int = 200_000
    steps_per_update: int = 4096
    minibatch_size: int = 256
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cuda"
    log_interval: int = 10_000
    ckpt_path: str = "checkpoints/cyborg_mind_ppo.pt"
    tb_log_dir: str = "runs/cyborg_mind_ppo"


class RolloutBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], scalar_dim: int, goal_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.full = False

        self.pixels = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.scalars = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.goals = np.zeros((capacity, goal_dim), dtype=np.float32)

        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        pixels: np.ndarray,
        scalars: np.ndarray,
        goal: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        self.pixels[self.ptr] = pixels
        self.scalars[self.ptr] = scalars
        self.goals[self.ptr] = goal
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True
            self.ptr = 0

    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        size = self.size()
        returns = np.zeros(size, dtype=np.float32)
        advantages = np.zeros(size, dtype=np.float32)

        last_adv = 0.0
        for t in reversed(range(size)):
            # BUG FIX: Use correct index for done flag
            # OLD: next_non_terminal = 1.0 - (self.dones[t] if t < size - 1 else 0.0)
            # The done flag at t indicates if episode ended AFTER taking action at t
            next_non_terminal = 1.0 - self.dones[t]
            next_value = self.values[t + 1] if t < size - 1 else last_value
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
            advantages[t] = last_adv
            returns[t] = advantages[t] + self.values[t]

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages


def ppo_update(brain: BrainCyborgMind, optimizer, buffer: RolloutBuffer, returns, advantages, cfg: PPOConfig, writer: SummaryWriter, update_idx: int):
    device = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    brain.to(device)
    brain.train()

    size = buffer.size()
    indices = np.arange(size)

    for epoch in range(cfg.ppo_epochs):
        np.random.shuffle(indices)

        for start in range(0, size, cfg.minibatch_size):
            end = start + cfg.minibatch_size
            mb_idx = indices[start:end]

            pixels = torch.from_numpy(buffer.pixels[mb_idx]).to(device)
            scalars = torch.from_numpy(buffer.scalars[mb_idx]).to(device)
            goals = torch.from_numpy(buffer.goals[mb_idx]).to(device)

            actions = torch.from_numpy(buffer.actions[mb_idx]).to(device)
            old_log_probs = torch.from_numpy(buffer.log_probs[mb_idx]).to(device)
            returns_t = torch.from_numpy(returns[mb_idx]).to(device)
            adv_t = torch.from_numpy(advantages[mb_idx]).to(device)

            B = pixels.size(0)
            thought = torch.zeros(B, 32, device=device)
            emotion = torch.zeros(B, 8, device=device)
            workspace = torch.zeros(B, 64, device=device)
            h0 = torch.zeros(1, B, 512, device=device)
            c0 = torch.zeros(1, B, 512, device=device)

            out = brain(
                pixels=pixels,
                scalars=scalars,
                goal=goals,
                thought=thought,
                emotion=emotion,
                workspace=workspace,
                hidden=(h0, c0),
            )

            logits = out["action_logits"]
            values = out["value"].squeeze(-1)

            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (returns_t - values).pow(2).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(brain.parameters(), cfg.max_grad_norm)
            optimizer.step()

        # Log at end of each epoch over this buffer
        writer.add_scalar("ppo/policy_loss", float(policy_loss.item()), update_idx * cfg.ppo_epochs + epoch)
        writer.add_scalar("ppo/value_loss", float(value_loss.item()), update_idx * cfg.ppo_epochs + epoch)
        writer.add_scalar("ppo/entropy", float(entropy.item()), update_idx * cfg.ppo_epochs + epoch)


def train_cyborg_mind_ppo(cfg: PPOConfig):
    device = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    print(f"[PPO] Using device: {device}")

    env = gym.make(cfg.env_name)
    brain = BrainCyborgMind().to(device)
    optimizer = optim.Adam(brain.parameters(), lr=cfg.lr)

    writer = SummaryWriter(log_dir=cfg.tb_log_dir)

    obs_shape = (3, 128, 128)
    scalar_dim = 20
    goal_dim = 4

    buffer = RolloutBuffer(cfg.steps_per_update, obs_shape, scalar_dim, goal_dim)

    obs = env.reset()
    total_steps = 0
    episode_reward = 0.0
    episode_idx = 0
    update_idx = 0

    h = torch.zeros(1, 1, 512, device=device)
    c = torch.zeros(1, 1, 512, device=device)
    thought = torch.zeros(1, 32, device=device)
    emotion = torch.zeros(1, 8, device=device)
    workspace = torch.zeros(1, 64, device=device)

    while total_steps < cfg.total_steps:
        buffer.ptr = 0
        buffer.full = False

        while buffer.size() < cfg.steps_per_update:
            pixels_np, scalars_np, goal_np = obs_to_brain(obs)

            pixels_t = torch.from_numpy(pixels_np).unsqueeze(0).to(device)
            scalars_t = torch.from_numpy(scalars_np).unsqueeze(0).to(device)
            goals_t = torch.from_numpy(goal_np).unsqueeze(0).to(device)

            with torch.no_grad():
                out = brain(
                    pixels=pixels_t,
                    scalars=scalars_t,
                    goal=goals_t,
                    thought=thought,
                    emotion=emotion,
                    workspace=workspace,
                    hidden=(h, c),
                )

            logits = out["action_logits"]
            value = out["value"].item()

            dist = Categorical(logits=logits)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).item()
            action_idx = int(action_t.item())

            thought = out["thought"].detach()
            emotion = out["emotion"].detach()
            workspace = out["workspace"].detach()
            h = out["hidden_h"].detach()
            c = out["hidden_c"].detach()

            minerl_action = index_to_minerl_action(action_idx)
            next_obs, reward, done, info = env.step(minerl_action)
            episode_reward += float(reward)

            buffer.add(
                pixels=pixels_np,
                scalars=scalars_np,
                goal=goal_np,
                action=action_idx,
                log_prob=log_prob,
                reward=float(reward),
                done=done,
                value=value,
            )

            obs = next_obs
            total_steps += 1

            if done:
                writer.add_scalar("env/episode_reward", episode_reward, episode_idx)
                print(f"[PPO] Episode {episode_idx} reward: {episode_reward:.2f}")
                obs = env.reset()
                episode_idx += 1
                episode_reward = 0.0
                h = torch.zeros(1, 1, 512, device=device)
                c = torch.zeros(1, 1, 512, device=device)
                thought = torch.zeros(1, 32, device=device)
                emotion = torch.zeros(1, 8, device=device)
                workspace = torch.zeros(1, 64, device=device)

            if total_steps >= cfg.total_steps:
                break

        pixels_np, scalars_np, goal_np = obs_to_brain(obs)
        with torch.no_grad():
            pixels_t = torch.from_numpy(pixels_np).unsqueeze(0).to(device)
            scalars_t = torch.from_numpy(scalars_np).unsqueeze(0).to(device)
            goals_t = torch.from_numpy(goal_np).unsqueeze(0).to(device)

            out = brain(
                pixels=pixels_t,
                scalars=scalars_t,
                goal=goals_t,
                thought=thought,
                emotion=emotion,
                workspace=workspace,
                hidden=(h, c),
            )
            last_value = out["value"].item()

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        writer.add_scalar("ppo/returns_mean", float(returns.mean()), update_idx)
        writer.add_scalar("ppo/advantages_mean", float(advantages.mean()), update_idx)

        ppo_update(brain, optimizer, buffer, returns, advantages, cfg, writer, update_idx)
        update_idx += 1

        if total_steps % cfg.log_interval < cfg.steps_per_update:
            print(f"[PPO] steps={total_steps} done, saving checkpoint.")
            os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)
            torch.save(
                {
                    "state_dict": brain.state_dict(),
                    "cfg": dataclasses.asdict(cfg),
                },
                cfg.ckpt_path,
            )
            print(f"[PPO] Saved PPO checkpoint to {cfg.ckpt_path}")

    env.close()
    writer.close()
    print("[PPO] Training complete.")


if __name__ == "__main__":
    cfg = PPOConfig()
    train_cyborg_mind_ppo(cfg)

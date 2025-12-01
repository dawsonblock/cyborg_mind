import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
from pathlib import Path
import logging
from tqdm import tqdm

from cyborg_mind_v2.configs.schema import Config
from cyborg_mind_v2.envs.base_adapter import BrainEnvAdapter, BrainInputs
from cyborg_mind_v2.capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind

logger = logging.getLogger(__name__)

import wandb
import time
# Use torch.amp for mixed precision (PyTorch 2.x+)
from omegaconf import OmegaConf

class CyborgTrainer:
    """
    Trainer for BrainCyborgMind using PPO.
    Handles BrainInputs (pixels, scalars, goal) and complex brain state (thought, emotion, workspace, memory).
    """
    def __init__(
        self,
        env: BrainEnvAdapter,
        brain: BrainCyborgMind,
        config: Config,
        device: str = "cuda"
    ):
        self.env = env
        self.brain = brain
        self.config = config
        self.device = torch.device(device)
        self.brain.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.brain.parameters(),
            lr=config.ppo.learning_rate,
            eps=1e-5
        )
        
        # Mixed Precision
        self.use_amp = config.train.use_amp and self.device.type == "cuda"
        # Fix for PyTorch 2.x deprecation warning
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_reward = -float("inf")
        
        # Buffers
        self.rollout_buffer = []
        self.reward_buffer = deque(maxlen=100)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.train.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # WandB Init
        if self.config.wandb.mode != "disabled":
            # Convert OmegaConf config to primitive dict for WandB serialization
            wandb_config = OmegaConf.to_container(self.config, resolve=True)
            
            try:
                wandb.init(
                    project=self.config.wandb.project,
                    entity=self.config.wandb.entity,
                    group=self.config.wandb.group,
                    name=self.config.wandb.name,
                    mode=self.config.wandb.mode,
                    config=wandb_config
                )
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}. Proceeding with WandB disabled.")
                self.config.wandb.mode = "disabled"

    def _init_brain_state(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Initialize the recurrent state of the brain."""
        return {
            "thought": torch.zeros(batch_size, self.config.model.thought_dim, device=self.device),
            "emotion": torch.zeros(batch_size, self.config.model.emotion_dim, device=self.device),
            "workspace": torch.zeros(batch_size, self.config.model.workspace_dim, device=self.device),
            "hidden": (
                torch.zeros(1, batch_size, self.config.model.hidden_dim, device=self.device),
                torch.zeros(1, batch_size, self.config.model.hidden_dim, device=self.device)
            )
        }

    def collect_rollouts(self) -> Dict[str, float]:
        """Collect experience from the environment."""
        start_time = time.time()
        self.brain.eval()
        self.rollout_buffer = []
        
        obs = self.env.reset()
        state = self._init_brain_state()
        
        episode_rewards = []
        current_episode_reward = 0.0
        
        # Storage for GAE
        rewards = []
        values = []
        dones = []
        
        for _ in range(self.config.ppo.rollout_steps):
            with torch.no_grad():
                # Prepare inputs
                pixels = obs.pixels.unsqueeze(0) if obs.pixels.dim() == 3 else obs.pixels
                scalars = obs.scalars.unsqueeze(0) if obs.scalars.dim() == 1 else obs.scalars
                goal = obs.goal.unsqueeze(0) if obs.goal.dim() == 1 else obs.goal
                
                # Store pre-update state for training
                pre_state = {}
                for k, v in state.items():
                    if k == "hidden" and isinstance(v, tuple):
                        pre_state[k] = (v[0].clone(), v[1].clone())
                    elif isinstance(v, torch.Tensor):
                        pre_state[k] = v.clone()
                    else:
                        pre_state[k] = v

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    output = self.brain(
                        pixels=pixels,
                        scalars=scalars,
                        goal=goal,
                        thought=state["thought"],
                        emotion=state["emotion"],
                        workspace=state["workspace"],
                        hidden=state["hidden"]
                    )
                
                # Sample action
                action_logits = output["action_logits"]
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = output["value"]
                
                # Update state for next step
                state = {
                    "thought": output["thought"],
                    "emotion": output["emotion"],
                    "workspace": output["workspace"],
                    "hidden": (output["hidden_h"], output["hidden_c"])
                }
            
            # Execute action
            action_idx = action.item()
            next_obs, reward, done, info = self.env.step(action_idx)
            
            # Store transition data
            self.rollout_buffer.append({
                "pixels": pixels,
                "scalars": scalars,
                "goal": goal,
                "state": pre_state,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": reward,
                "done": done
            })
            
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            
            current_episode_reward += reward
            self.global_step += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                self.reward_buffer.append(current_episode_reward)
                current_episode_reward = 0.0
                obs = self.env.reset()
                state = self._init_brain_state()
                self.episode_count += 1
            else:
                obs = next_obs
                
        # Compute GAE
        with torch.no_grad():
            # Get value of last state
            pixels = obs.pixels.unsqueeze(0) if obs.pixels.dim() == 3 else obs.pixels
            scalars = obs.scalars.unsqueeze(0) if obs.scalars.dim() == 1 else obs.scalars
            goal = obs.goal.unsqueeze(0) if obs.goal.dim() == 1 else obs.goal
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                next_val_out = self.brain(
                    pixels=pixels,
                    scalars=scalars,
                    goal=goal,
                    thought=state["thought"],
                    emotion=state["emotion"],
                    workspace=state["workspace"],
                    hidden=state["hidden"]
                )
            next_value = next_val_out["value"].item()

        advantages = np.zeros(len(rewards))
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
                
            delta = rewards[t] + self.config.ppo.gamma * next_val * next_non_terminal - values[t]
            last_gae_lam = delta + self.config.ppo.gamma * self.config.ppo.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + values
        
        # Add advantages and returns to buffer
        for i, t in enumerate(self.rollout_buffer):
            t["advantage"] = torch.tensor(advantages[i], dtype=torch.float32, device=self.device)
            t["return"] = torch.tensor(returns[i], dtype=torch.float32, device=self.device)
        
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        
        # Calculate FPS
        duration = time.time() - start_time
        fps = self.config.ppo.rollout_steps / duration
        
        if self.config.wandb.mode != "disabled":
            wandb.log({
                "rollout/mean_reward": mean_reward,
                "rollout/episodes": len(episode_rewards),
                "rollout/fps": fps,
                "pmm/usage_mean": self.brain.pmm.usage.mean().item(),
                "pmm/usage_std": self.brain.pmm.usage.std().item(),
                "global_step": self.global_step
            })
            
        return {
            "mean_reward": mean_reward,
            "fps": fps
        }

    def train_step(self):
        """Perform PPO update."""
        self.brain.train()
        
        indices = np.arange(len(self.rollout_buffer))
        np.random.shuffle(indices)
        
        batch_size = self.config.ppo.batch_size
        num_batches = len(self.rollout_buffer) // batch_size
        
        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(self.config.ppo.num_epochs):
            for i in range(num_batches):
                batch_indices = indices[i * batch_size : (i + 1) * batch_size]
                batch = [self.rollout_buffer[idx] for idx in batch_indices]
                
                # Stack inputs
                pixels = torch.cat([b["pixels"] for b in batch])
                scalars = torch.cat([b["scalars"] for b in batch])
                goal = torch.cat([b["goal"] for b in batch])
                actions = torch.cat([b["action"] for b in batch])
                old_log_probs = torch.cat([b["log_prob"] for b in batch])
                advantages = torch.stack([b["advantage"] for b in batch]).unsqueeze(1)
                returns = torch.stack([b["return"] for b in batch]).unsqueeze(1)
                
                # Stack states
                thought = torch.cat([b["state"]["thought"] for b in batch])
                emotion = torch.cat([b["state"]["emotion"] for b in batch])
                workspace = torch.cat([b["state"]["workspace"] for b in batch])
                
                h_list = [b["state"]["hidden"][0] for b in batch]
                c_list = [b["state"]["hidden"][1] for b in batch]
                hidden = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1))
                
                self.optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    # Forward pass
                    output = self.brain(
                        pixels=pixels,
                        scalars=scalars,
                        goal=goal,
                        thought=thought,
                        emotion=emotion,
                        workspace=workspace,
                        hidden=hidden
                    )
                    
                    # Evaluate actions
                    action_logits = output["action_logits"]
                    dist = torch.distributions.Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()
                    values = output["value"]
                    
                    # PPO Loss
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages.squeeze()
                    surr2 = torch.clamp(ratio, 1.0 - self.config.ppo.clip_epsilon, 1.0 + self.config.ppo.clip_epsilon) * advantages.squeeze()
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = 0.5 * (returns - values).pow(2).mean()
                    
                    loss = policy_loss + self.config.ppo.value_coef * value_loss - self.config.ppo.entropy_coef * entropy
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.brain.parameters(), self.config.ppo.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        if self.config.wandb.mode != "disabled":
            wandb.log({
                "train/loss": np.mean(losses),
                "train/policy_loss": np.mean(policy_losses),
                "train/value_loss": np.mean(value_losses),
                "train/entropy": np.mean(entropies),
                "global_step": self.global_step
            })

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training on {self.device}")
        
        pbar = tqdm(total=self.config.train.total_timesteps)
        
        while self.global_step < self.config.train.total_timesteps:
            stats = self.collect_rollouts()
            self.train_step()
            
            if self.global_step % self.config.train.log_frequency == 0:
                logger.info(f"Step {self.global_step}: Reward {stats['mean_reward']:.2f}")
                
            if self.global_step % self.config.train.save_frequency == 0:
                self.save_checkpoint()
                
            pbar.update(self.config.ppo.rollout_steps)
            
        pbar.close()

    def save_checkpoint(self):
        path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save({
            "brain": self.brain.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "episode_count": self.episode_count
        }, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load checkpoint to resume training."""
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.brain.load_state_dict(checkpoint["brain"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["step"]
        self.episode_count = checkpoint.get("episode_count", 0)
        
        logger.info(f"Resumed from step {self.global_step}")

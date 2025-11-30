"""Configuration management for CyborgMind RL."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class EnvConfig:
    """Environment configuration."""
    name: str = "CartPole-v1"
    max_episode_steps: int = 500
    normalize_obs: bool = True
    clip_obs: float = 10.0


@dataclass
class MemoryConfig:
    """PMM Memory configuration."""
    memory_size: int = 128
    memory_dim: int = 64
    num_read_heads: int = 4
    num_write_heads: int = 1
    sharp_factor: float = 1.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = 256
    latent_dim: int = 128
    num_gru_layers: int = 2
    use_mamba: bool = False
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    dropout: float = 0.0


@dataclass
class PPOConfig:
    """PPO training configuration."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_epochs: int = 10
    batch_size: int = 64
    rollout_steps: int = 2048
    normalize_advantage: bool = True


@dataclass
class TrainConfig:
    """Training loop configuration."""
    total_timesteps: int = 1_000_000
    eval_frequency: int = 10_000
    save_frequency: int = 50_000
    log_frequency: int = 1000
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class Config:
    """Master configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            env=EnvConfig(**data.get("env", {})),
            memory=MemoryConfig(**data.get("memory", {})),
            model=ModelConfig(**data.get("model", {})),
            ppo=PPOConfig(**data.get("ppo", {})),
            train=TrainConfig(**data.get("train", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "env": self.env.__dict__,
            "memory": self.memory.__dict__,
            "model": self.model.__dict__,
            "ppo": self.ppo.__dict__,
            "train": self.train.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

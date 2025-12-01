"""Configuration management for CyborgMind RL."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict
import yaml


@dataclass
class EnvConfig:
    """Environment configuration."""
    name: str = "CartPole-v1"
    max_episode_steps: Optional[int] = None
    image_size: tuple = (64, 64)
    frame_stack: int = 1
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
    use_intrinsic_reward: bool = False
    intrinsic_reward_coef: float = 0.01


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = 256
    latent_dim: int = 128
    num_gru_layers: int = 2
    use_mamba: bool = True
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    dropout: float = 0.0


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    # Core PPO hyperparameters
    learning_rate: float = 3e-4  # Kept for backward compatibility
    lr_start: Optional[float] = None  # If None, uses learning_rate
    lr_end: float = 1e-5
    anneal_lr: bool = False

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2

    entropy_coef: float = 0.01  # Kept for backward compatibility
    entropy_start: Optional[float] = None  # If None, uses entropy_coef
    entropy_end: float = 0.0
    anneal_entropy: bool = True

    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    batch_size: int = 64
    num_epochs: int = 10
    normalize_advantage: bool = True

    # Reward stability and early stopping
    reward_buffer_size: int = 10
    reward_improvement_threshold: float = 1.0
    early_stop_patience: int = 8
    enable_early_stopping: bool = False

    # Collapse detection and recovery
    reward_collapse_threshold: float = 0.4  # 40% drop from peak
    enable_collapse_detection: bool = True
    collapse_lr_multiplier: float = 0.3  # On collapse, keep 30% of current LR (i.e., reduce by 70%)

    # Validation and diagnostics
    inference_validation: bool = False
    inference_validation_episodes: int = 5
    inference_validation_threshold: float = 0.8  # 80% of best reward
    auto_plot: bool = False


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
    save_best: bool = True


@dataclass
class APIConfig:
    """API Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    auth_token: str = "cyborg-secret-v2"
    enable_metrics: bool = True


@dataclass
class Config:
    """Global configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "env": self.env.__dict__,
            "memory": self.memory.__dict__,
            "model": self.model.__dict__,
            "ppo": self.ppo.__dict__,
            "train": self.train.__dict__,
            "api": self.api.__dict__,
        }

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        config = cls()
        if "env" in data:
            config.env = EnvConfig(**data["env"])
        if "memory" in data:
            config.memory = MemoryConfig(**data["memory"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "ppo" in data:
            config.ppo = PPOConfig(**data["ppo"])
        if "train" in data:
            config.train = TrainConfig(**data["train"])
        if "api" in data:
            config.api = APIConfig(**data["api"])
            
        return config

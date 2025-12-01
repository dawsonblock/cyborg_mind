from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import MISSING

@dataclass
class EnvConfig:
    name: str = "CartPole-v1"
    max_episode_steps: int = 500
    image_size: List[int] = field(default_factory=lambda: [64, 64])
    frame_stack: int = 1
    normalize_obs: bool = True
    clip_obs: float = 10.0
    num_actions: int = 2  # Added for Brain init

@dataclass
class PMMConfig:
    memory_size: int = 128
    memory_dim: int = 64
    key_dim: int = 64
    num_read_heads: int = 4
    num_write_heads: int = 1
    sharp_factor: float = 1.0
    use_intrinsic_reward: bool = False
    intrinsic_reward_coef: float = 0.01
    start_slots: int = 256 # Added for Brain init

@dataclass
class ModelConfig:
    # BrainCyborgMind specific
    scalar_dim: int = 20
    goal_dim: int = 4
    thought_dim: int = 32
    emotion_dim: int = 8
    workspace_dim: int = 64
    vision_dim: int = 512
    emb_dim: int = 256
    hidden_dim: int = 512
    
    # Legacy/Other
    latent_dim: int = 128
    num_gru_layers: int = 2
    use_mamba: bool = True
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    dropout: float = 0.0

@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    lr_start: Optional[float] = None
    lr_end: float = 1e-5
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    entropy_start: Optional[float] = None
    entropy_end: float = 0.0
    anneal_entropy: bool = True
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    batch_size: int = 64
    num_epochs: int = 10
    normalize_advantage: bool = True
    reward_buffer_size: int = 10
    reward_improvement_threshold: float = 1.0
    early_stop_patience: int = 8
    enable_early_stopping: bool = False
    reward_collapse_threshold: float = 0.4
    enable_collapse_detection: bool = True
    collapse_lr_multiplier: float = 0.3
    inference_validation: bool = False
    inference_validation_episodes: int = 5
    inference_validation_threshold: float = 0.8
    auto_plot: bool = False

@dataclass
class TrainConfig:
    total_timesteps: int = 1_000_000
    eval_frequency: int = 10_000
    save_frequency: int = 50_000
    log_frequency: int = 1000
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_best: bool = True
    resume_path: Optional[str] = None
    use_amp: bool = False

@dataclass
class WandBConfig:
    project: str = "cyborg-mind"
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    mode: str = "online" # online, offline, disabled

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    pmm: PMMConfig = field(default_factory=PMMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

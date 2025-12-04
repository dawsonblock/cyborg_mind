# How to Train CyborgMind v3.0

This guide shows you how to train RL agents using the production `train_production.py` entry point.

## Prerequisites

### 1. Install Dependencies

```bash
# Standard installation (Gym only)
pip install -e .

# Or use environment-specific setup scripts
./setup_gym.sh              # Gym environments
./setup_mamba_gpu.sh        # Mamba + GPU support
./setup_minerl.sh           # MineRL environments
```

### 2. Verify Installation

```bash
python quick_verify.py
```

Expected output:
```
✓ PyTorch installation
✓ CUDA availability
✓ CyborgRL imports
✓ Configuration system
```

---

## Quick Start

### Train on CartPole

```bash
python train_production.py \
    --config configs/envs/gym_cartpole.yaml \
    --run-name cartpole-exp-01
```

### Train with Custom Options

```bash
python train_production.py \
    --config configs/envs/gym_pendulum.yaml \
    --run-name pendulum-test \
    --num-envs 8 \
    --device cuda \
    --seed 42
```

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/envs/gym_cartpole.yaml` | Path to config YAML |
| `--run-name` | Auto-generated | Experiment name (timestamp + ID appended) |
| `--num-envs` | 4 | Number of parallel environments |
| `--device` | `auto` | Device (`auto`, `cpu`, `cuda`) |
| `--seed` | 42 | Random seed |

---

## Configuration Files

Configs are YAML files in `configs/envs/`. Structure:

### Example Config (CartPole)

```yaml
env:
  name: "CartPole-v1"
  max_episode_steps: 500
  normalize_obs: true

model:
  hidden_dim: 64
  latent_dim: 64
  num_gru_layers: 2
  use_mamba: false

ppo:
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2

train:
  total_timesteps: 100000
  n_steps: 2048
  batch_size: 64
```

See existing configs in `configs/envs/` for complete examples.

---

## Output Structure

Training artifacts are saved to `experiments/runs/<run_name>/`:

```
experiments/runs/cartpole-exp-01_20251203-235959_a1b2/
├── config.yaml              # Config used for this run
├── manifest.json            # Git hash, system info, timestamps
├── checkpoints/
│   ├── checkpoint_1000.pt
│   ├── latest.pt            # Most recent
│   └── best.pt              # Best performing
└── logs/
    └── metrics.csv          # step, loss, fps, etc.
```

---

## Monitoring Training

### Console Output

```
INFO: Update 1/48 | FPS: 1243 | Loss: 0.5234
INFO: Update 2/48 | FPS: 1389 | Loss: 0.4721
```

### Metrics CSV

File: `experiments/runs/<run_name>/logs/metrics.csv`

---

## Weights & Biases Integration

CyborgMind supports automatic metric logging to [Weights & Biases](https://wandb.ai/).

### Setup

1. **Install WandB:**
   ```bash
   pip install wandb
   ```

2. **Login:**
   ```bash
   wandb login
   ```

3. **Enable in Config:**
   ```yaml
   train:
     wandb_enabled: true
     wandb_project: "cyborg-mind"
     wandb_entity: "your-username"  # Optional
     wandb_tags: ["cartpole", "ppo"]
     wandb_run_name: null  # Auto-generated if not set
   ```

### Logged Metrics

WandB automatically tracks:
- **Loss Metrics**: `loss`, `policy_loss`, `value_loss`, `entropy_loss`
- **Performance**: `fps`, `timestep`, `update`
- **Model**: Gradients and parameters (via `wandb.watch`)

### Example Run

```bash
python train_production.py \
    --config configs/envs/gym_cartpole.yaml \
    --run-name wandb-test
```

View results at: `https://wandb.ai/<entity>/<project>/runs/<run_name>`

---

## Tips

**CartPole (simple):**
- `hidden_dim: 64`, `use_mamba: false`

**Continuous Control:**
- `hidden_dim: 128`, `use_mamba: true`

**GPU Training:**
```yaml
train:
  use_amp: true
  device: "cuda"
```

**WandB Sweeps:**
```bash
wandb sweep configs/sweep.yaml
wandb agent <sweep_id>
```

---

For API usage, see [docs/API.md](API.md).

# CyborgMind v5.0 - MineRL Rebuild

> **Minimal, Ultra-Efficient MineRL-Only RL Engine**

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red)
![License](https://img.shields.io/badge/license-MIT-green)

CyborgMind v5.0 is a complete rebuild focused exclusively on MineRL environments, featuring:
- **PMM v2.0**: Honest Memory Engine with controlled write gates and zero leakage
- **UnifiedEncoder**: Runtime switching between GRU, Mamba, and Mamba-GRU
- **PPO v3.0**: Production-grade trainer with vectorized environments and recurrent support

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt

# For MineRL training (requires JDK 8)
pip install minerl

# For Mamba encoder (GPU only)
pip install mamba-ssm causal-conv1d
```

### Smoke Test (No MineRL Required)
```bash
python3 mine_rl_train.py --smoke-test
```

### Train MineRL Agent
```bash
python3 mine_rl_train.py \
    --env MineRLTreechop-v0 \
    --encoder gru \
    --memory pmm \
    --num-envs 4 \
    --wandb
```

### CLI Arguments
| Argument | Description |
|----------|-------------|
| `--env` | Environment ID (e.g., `MineRLTreechop-v0`) |
| `--encoder` | `gru`, `mamba`, or `mamba_gru` |
| `--memory` | `pmm` or `none` |
| `--num-envs` | Number of parallel environments |
| `--steps` | Total training timesteps |
| `--wandb` | Enable WandB logging |
| `--smoke-test` | Run mock verification |

---

## Architecture

```
cyborg_mind/
├── configs/
│   └── unified_config.yaml   # Primary Config
├── cyborg_rl/                # Core Package
│   ├── envs/                 # MineRL Adapter
│   ├── memory/               # PMM & RecurrentBuffer
│   ├── models/               # UnifiedEncoder, Policy, Value
│   ├── trainers/             # PPOTrainer
│   └── utils/                # Config, Logging, Device
├── scripts/                  # Launcher Scripts
│   ├── train_treechop.sh
│   └── verify_treechop.sh
└── mine_rl_train.py          # Unified Entry Point
```

### Components

| Component | Description |
|-----------|-------------|
| **UnifiedEncoder** | GRU/Mamba/Mamba-GRU with state caching |
| **PMM v2.0** | Memory with controlled write gate, zero leakage, cosine addressing |
| **PPO v3.0** | Vectorized collection, truncated BPTT, AMP support |
| **RecurrentRolloutBuffer** | Sequence-preserving storage for recurrent training |

---

## Configuration

Edit `configs/unified_config.yaml`:

```yaml
env:
  name: MineRLTreechop-v0
  image_size: [64, 64]
  frame_stack: 4

model:
  encoder: gru  # gru | mamba | mamba_gru
  hidden_dim: 512
  latent_dim: 256

pmm:
  enabled: true
  num_slots: 16
  memory_dim: 512

train:
  num_envs: 4
  horizon: 2048
  total_timesteps: 1000000
  learning_rate: 0.0003
```

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Training Guide](docs/HOW_TO_TRAIN.md)
- [API Reference](docs/API.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

---

## Legacy

The following components have been moved to `legacy/`:
- Trading, EEG, Lab adapters
- Old experiment scripts
- Monitoring dashboards

To use legacy components, install `requirements-experimental.txt`.

---

## License

MIT License - see [LICENSE](LICENSE).

# CyborgMind RL

Production-grade Reinforcement Learning system with Predictive Memory Module (PMM) integration.

## Features

- **PPO Algorithm**: Proximal Policy Optimization with GAE, value clipping, and entropy bonus
- **PMM Memory**: Differentiable external memory with content-based addressing
- **Mamba/GRU Encoder**: Hybrid sequence model for efficient temporal processing
- **Multi-Environment**: Support for Gymnasium and MineRL environments
- **GPU-Ready**: Full CUDA support with proper device management
- **Monitoring**: Prometheus metrics with Grafana dashboards
- **Production Docker**: Multi-stage builds with GPU support

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cyborg_mind.git
cd cyborg_mind

# Install dependencies
pip install -e .

# Or with development tools
pip install -e ".[dev]"
```

### Train CartPole

```bash
# CPU training
python scripts/train_gym_cartpole.py --total-timesteps 100000

# GPU training
python scripts/train_gym_cartpole.py --total-timesteps 500000 --device cuda
```

### Run Inference

```bash
python scripts/inference.py \
    --checkpoint checkpoints/cartpole/final_policy.pt \
    --episodes 10 \
    --render
```

### Docker Training

```bash
# Build and run
docker-compose up trainer

# With GPU
docker-compose --profile gpu up trainer-gpu

# With monitoring
docker-compose --profile monitoring up
```

## Architecture

```
obs → MambaGRU Encoder → latent → PMM (read/write) → memory_augmented
                                                            ↓
                                                     Policy Head → action
                                                     Value Head → value
```

### Components

- **MambaGRUEncoder**: Processes observations into latent representations
- **PredictiveMemoryModule**: External memory bank with attention-based read/write
- **DiscretePolicy/ContinuousPolicy**: Action distribution heads
- **ValueHead**: State value estimation for advantage computation
- **PPOTrainer**: Training loop with rollout collection and GAE

## Configuration

Configuration is managed via dataclasses in `cyborg_rl/config.py`:

```python
from cyborg_rl import Config

config = Config()
config.model.hidden_dim = 256
config.model.latent_dim = 128
config.memory.memory_size = 128
config.ppo.learning_rate = 3e-4

# Save/load from YAML
config.to_yaml("config.yaml")
config = Config.from_yaml("config.yaml")
```

## Monitoring

Access dashboards after starting the monitoring stack:

- **Grafana**: http://localhost:3000 (admin/cyborgmind)
- **Prometheus**: http://localhost:9090

Metrics exposed:
- `cyborg_rl_episode_reward` - Episode reward
- `cyborg_rl_episode_length` - Episode length
- `cyborg_rl_loss_policy` - Policy loss
- `cyborg_rl_loss_value` - Value loss
- `cyborg_rl_advantage_mean` - Mean advantage
- `cyborg_rl_pmm_memory_saturation` - PMM memory usage

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=cyborg_rl --cov-report=html
```

## Project Structure

```
cyborg_mind/
├── cyborg_rl/
│   ├── agents/          # PPO agent implementation
│   ├── envs/            # Environment adapters
│   ├── memory/          # PMM and replay buffer
│   ├── models/          # Neural network architectures
│   ├── trainers/        # PPO trainer and rollout buffer
│   ├── metrics/         # Prometheus metrics
│   └── utils/           # Device, logging, seeding utilities
├── scripts/
│   ├── train_gym_cartpole.py
│   ├── train_minerl_navigate.py
│   └── inference.py
├── tests/               # Unit tests
├── monitoring/          # Prometheus & Grafana configs
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## License

MIT License

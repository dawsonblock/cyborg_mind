# 🧠 CyborgMind V2.6

> **Production-Grade Game AI & RL Brain System**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Training](https://img.shields.io/badge/Training-MineRL%20%7C%20Gym-orange)]()
[![Deploy](https://img.shields.io/badge/Deploy-Docker%20%7C%20FastAPI-green)]()
[![Version](https://img.shields.io/badge/version-2.6.0-brightgreen)]()

**Production-hardened RL brain for game AI, NPC control, and autonomous agents.** Features universal environment adapters, dynamic memory, recurrent processing, and enterprise deployment infrastructure.

### 🆕 V2.6 Highlights
- ✅ **CC3D Removed** - Pure game/RL focus
- ✅ **Hardened Adapters** - Production validation for Gym/MineRL
- ✅ **Docker Deployment** - Multi-stage builds with GPU
- ✅ **Monitoring** - Prometheus + Grafana dashboards
- ✅ **Type Safe** - Full validation with clean errors

**[📖 V2.6 Release Notes](https://github.com/dawsonblock/cyborg_mind/blob/main/docs/V2.6_RELEASE_NOTES.md)** | **[🔄 Migration Guide](https://github.com/dawsonblock/cyborg_mind/blob/main/docs/V2.6_MIGRATION_GUIDE.md)** | **[🏗️ V2.6 Architecture](https://github.com/dawsonblock/cyborg_mind/blob/main/docs/V2.6_ARCHITECTURE.md)**

---

## ✨ Highlights

- 🎯 **Universal Adapters**: Works with MineRL, Gym, custom envs - one brain, any task
- 🧠 **Emotion-Consciousness**: 8-channel emotions + 32D thoughts + 64D workspace
- 💾 **Dynamic Memory**: Auto-expanding PMM (256→2048 slots) with garbage collection
- 🔄 **Recurrent Processing**: LSTM + FRNN for temporal coherence
- 🚀 **Production Ready**: FastAPI server + web visualizer + Docker
- 📊 **Full Observability**: TensorBoard + real-time emotion visualization

---

## 🎬 Quick Start

### Option 1: Docker (Recommended for V2.6)

```bash
# Clone and build
git clone https://github.com/dawsonblock/cyborg_mind.git
cd cyborg_mind
docker-compose up --build

# Access services
open http://localhost:8000/docs    # API documentation
open http://localhost:3000         # Grafana dashboards (admin/admin)
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/dawsonblock/cyborg_mind.git
cd cyborg_mind

# Install dependencies
pip install -e .

# Verify V2.6 installation
python -c "from cyborg_mind_v2.envs import GymAdapter; print('✓ CyborgMind V2.6 ready!')"
```

### Run Demo

```bash
# 1. Start API server
uvicorn cyborg_mind_v2.deployment.api_server:app --host 0.0.0.0 --port 8000

# 2. Open web visualizer
cd frontend/demo && python -m http.server 8080

# 3. Visit http://localhost:8080 and click "Connect"
```

### Train on MineRL

```bash
# Complete pipeline: BC → Distillation → PPO
bash experiments/run_full_pipeline.sh

# Or run individually:
bash experiments/run_treechop_teacher_bc.sh  # Teacher BC
bash experiments/run_treechop_ppo.sh          # PPO training
```

### Train on Gym

```bash
# CartPole demo
python -c "
from cyborg_mind_v2.envs import create_adapter
from cyborg_mind_v2.integration import CyborgMindController

adapter = create_adapter('gym', 'CartPole-v1')
controller = CyborgMindController()

for ep in range(10):
    obs = adapter.reset()
    done = False
    reward_sum = 0
    while not done:
        action = controller.step(['agent_0'], obs.pixels.unsqueeze(0),
                                obs.scalars.unsqueeze(0), obs.goal.unsqueeze(0))[0]
        obs, reward, done, _ = adapter.step(action)
        reward_sum += reward
    print(f'Episode {ep+1}: Reward = {reward_sum}')
"
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CyborgMind V2 System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   MineRL     │    │  Gymnasium   │    │  Synthetic   │  │
│  │   Adapter    │    │   Adapter    │    │   Dataset    │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         └───────────────────┴───────────────────┘           │
│                            │                                │
│         ┌──────────────────▼─────────────────────┐          │
│         │  (pixels, scalars, goal) → action_idx  │          │
│         └──────────────────┬─────────────────────┘          │
│                            │                                │
│  ╔═════════════════════════▼═════════════════════════════╗  │
│  ║              BrainCyborgMind (2.3M params)           ║  │
│  ║                                                       ║  │
│  ║  Vision → PMM → LSTM → [Action, Value, Emotion,     ║  │
│  ║                         Thought, Workspace]          ║  │
│  ╚═══════════════════════════════════════════════════════╝  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Brain Components

| Component | Description | Dimensions |
|-----------|-------------|------------|
| **Vision Adapter** | CNN encoder for RGB images | 512 |
| **Dynamic PMM** | Content-addressable memory | 256-2048 slots × 128 dims |
| **LSTM Core** | Temporal processing | 512 hidden |
| **FRNN Workspace** | Global consciousness (GWT) | 64 |
| **Emotion System** | Affect model (valence, arousal, etc.) | 8 channels |
| **Thought Vector** | Persistent cognition | 32 |

**Total Parameters**: 2.3M (brain) + 87M (optional teacher)

---

## 🎓 Training Pipelines

### 1. Behavior Cloning (BC)

Train from expert demonstrations:

```bash
bash experiments/run_treechop_teacher_bc.sh
```

**Results**: ~75% action prediction accuracy on MineRL dataset

### 2. Proximal Policy Optimization (PPO)

Reinforcement learning from scratch or fine-tuning:

```bash
bash experiments/run_treechop_ppo.sh
```

**Results**: See [docs/MINERL_RESULTS.md](docs/MINERL_RESULTS.md)

### 3. Full Pipeline (Recommended)

Teacher BC → Student Distillation → PPO Fine-tuning:

```bash
bash experiments/run_full_pipeline.sh
```

This combines the best of imitation and reinforcement learning!

---

## 🌍 Universal Environment Adapters

CyborgMind works with **any** environment via adapters:

```python
from cyborg_mind_v2.envs import create_adapter

# MineRL
adapter = create_adapter("minerl", "MineRLTreechop-v0")

# Gym
adapter = create_adapter("gym", "CartPole-v1")

# Custom (implement BrainEnvAdapter protocol)
class MyAdapter:
    def reset(self) -> BrainInputs: ...
    def step(self, action_idx: int) -> Tuple[BrainInputs, float, bool, Dict]: ...
```

All adapters provide the same interface:
- **Input**: `(pixels, scalars, goal)` → unified brain format
- **Output**: `action_idx` → environment-specific action

See [docs/ADAPTER_SYSTEM.md](docs/ADAPTER_SYSTEM.md) for details.

---

## 🚀 Deployment

### Docker

```bash
# Build
docker build -t cyborgmind:latest .

# Run
docker run -d --gpus all -p 8000:8000 cyborgmind:latest
```

### FastAPI Server

```bash
# Start server
uvicorn cyborg_mind_v2.deployment.api_server:app --host 0.0.0.0 --port 8000

# Test
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
  -d '{"agent_id": "test_agent"}'
```

**Endpoints**:
- `POST /reset` - Initialize agent
- `POST /step` - Get action from observation
- `GET /state/{agent_id}` - Query brain state
- `GET /metrics` - Performance metrics

### Web Visualizer

Open `frontend/demo/index.html` in browser:
- Real-time emotion visualization (8 channels)
- Thought vector heatmap (32D)
- Memory pressure tracking
- Action/value display

---

## 📊 Benchmarks

### MineRL TreeChop-v0

| Method | Mean Reward | Training Time | Checkpoint |
|--------|-------------|---------------|------------|
| **BC (Teacher)** | TBD | 1-2 hours | `real_teacher_treechop.pt` |
| **PPO (from scratch)** | TBD | 2-4 hours | `treechop_brain.pt` |
| **Pipeline (BC→PPO)** | TBD | 3-6 hours | `treechop_brain.pt` |

### Gym CartPole-v1

| Method | Mean Reward | Solved? |
|--------|-------------|---------|
| **PPO** | 195+ | ✅ |

See detailed results in [docs/MINERL_RESULTS.md](docs/MINERL_RESULTS.md)

---

## 📂 Project Structure

```
cyborg_mind/
├── cyborg_mind_v2/
│   ├── capsule_brain/       # Brain architecture
│   │   └── policy/
│   │       └── brain_cyborg_mind.py
│   ├── envs/                # Environment adapters
│   │   ├── base_adapter.py
│   │   ├── minerl_adapter.py
│   │   └── gym_adapter.py
│   ├── integration/         # Controller
│   │   └── cyborg_mind_controller.py
│   ├── training/            # Training scripts
│   │   ├── train_cyborg_mind_ppo.py
│   │   ├── train_real_teacher_bc.py
│   │   └── dist/            # Distributed training
│   ├── deployment/          # API & monitoring
│   │   └── api_server.py
│   └── data/                # Datasets
├── configs/                 # YAML configurations
│   ├── treechop_ppo.yaml
│   ├── treechop_bc.yaml
│   └── gym_cartpole.yaml
├── experiments/             # Training scripts
│   ├── run_treechop_ppo.sh
│   ├── run_treechop_teacher_bc.sh
│   └── run_full_pipeline.sh
├── frontend/demo/           # Web visualizer
├── notebooks/               # Colab demos
│   └── cyborg_mind_quickstart.ipynb
├── docs/                    # Documentation
│   ├── ARCHITECTURE_V3.md
│   ├── DEPLOYMENT.md
│   └── MINERL_RESULTS.md
└── checkpoints/             # Saved models
```

---

## 🔧 Configuration

Use YAML configs for reproducible experiments:

```yaml
# configs/treechop_ppo.yaml
env:
  adapter: "minerl"
  name: "MineRLTreechop-v0"

ppo:
  learning_rate: 3e-4
  gamma: 0.99
  clip_epsilon: 0.2

training:
  device: "cuda"
  num_episodes: 1000
```

Load and use:

```python
import yaml

with open("configs/treechop_ppo.yaml") as f:
    config = yaml.safe_load(f)
```

---

## 🧪 Notebooks

Interactive demos in `notebooks/`:

**Quickstart**: [`cyborg_mind_quickstart.ipynb`](notebooks/cyborg_mind_quickstart.ipynb)
- Train on synthetic data
- Run Gym CartPole demo
- Visualize brain state (emotions, thoughts, workspace)

**Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dawsonblock/cyborg_mind/blob/main/notebooks/cyborg_mind_quickstart.ipynb)

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [V2.6_ARCHITECTURE.md](docs/V2.6_ARCHITECTURE.md) | **V2.6 system architecture** |
| [V2.6_RELEASE_NOTES.md](docs/V2.6_RELEASE_NOTES.md) | **V2.6 release notes** |
| [V2.6_MIGRATION_GUIDE.md](docs/V2.6_MIGRATION_GUIDE.md) | **V2.5 → V2.6 migration** |
| [ARCHITECTURE_V3.md](docs/ARCHITECTURE_V3.md) | Legacy architecture docs |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment guide |
| [MINERL_RESULTS.md](docs/MINERL_RESULTS.md) | Training results & benchmarks |
| [ADAPTER_SYSTEM.md](docs/ADAPTER_SYSTEM.md) | Environment adapter guide |

---

## 🔬 Research Foundations

CyborgMind V2 builds on:

- **Global Workspace Theory** (Baars, 1988): FRNN workspace for consciousness
- **Emotion as Computation** (Minsky, 2006): 8-channel affect model
- **PMM Memory** (Graves et al.): Content-addressable episodic storage
- **PPO** (Schulman et al., 2017): Robust policy optimization
- **Behavior Cloning** (Pomerleau, 1989): Learning from demonstrations

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- New environment adapters (Unity, Unreal, robotics)
- Improved memory systems (Transformer-XL, MERLIN)
- Multi-agent communication
- Language grounding
- Benchmarks on more tasks

---

## 🎯 Roadmap

### V2.6 (Current - Production Hardening)
- ✅ CC3D removal (game/RL focus)
- ✅ Hardened adapters with validation
- ✅ Docker deployment infrastructure
- ✅ Prometheus + Grafana monitoring
- ✅ Full type safety and error handling

### V2.7 (Next)
- [ ] Multi-agent coordination
- [ ] Hierarchical RL with options
- [ ] Transformer-based world models
- [ ] Real-time environment streaming
- [ ] Advanced curriculum learning

### V3.0 (Future)
- [ ] Language instruction following
- [ ] Open-ended exploration
- [ ] Meta-learning
- [ ] Unity/Unreal integration

---

## 📜 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **MineRL Team** for the benchmark environment
- **OpenAI** for Gym and PPO
- **Anthropic** for Claude AI development assistance
- **PyTorch Team** for the ML framework

---

## 📬 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/dawsonblock/cyborg_mind/issues)
- **Author**: Dawson Block
- **Email**: [Your Email]

---

## 📊 Citation

If you use CyborgMind V2 in your research:

```bibtex
@software{cyborgmind_v26,
  title={CyborgMind V2.6: Production-Grade Game AI & RL Brain System},
  author={Block, Dawson},
  year={2025},
  version={2.6.0},
  url={https://github.com/dawsonblock/cyborg_mind}
}
```

---

<div align="center">

**Built with 🧠 and ❤️**

[Documentation](docs/) • [Quickstart](notebooks/cyborg_mind_quickstart.ipynb) • [Results](docs/MINERL_RESULTS.md)

</div>

# Quick Start Guide

## Installation

### Prerequisites
- Python 3.9+ (3.9-3.10 for MineRL)
- CUDA 12.1+ (for GPU support)

### Basic Install
```bash
git clone https://github.com/yourusername/cyborg_mind.git
cd cyborg_mind
pip install -e .
```

### With MineRL
```bash
pip install -e ".[minerl]"
```

## Verification

Run the quick verification script to ensure your environment is ready:

```bash
python scripts/quick_verify.py
```

## Training

### CartPole (CPU)
```bash
python scripts/train_gym_cartpole.py
```

### MineRL Treechop (GPU)
```bash
python scripts/run_treechop_pipeline.py
```

## Monitoring

1. Start the monitoring stack:
   ```bash
   docker-compose --profile monitoring up -d
   ```
2. Open Grafana at http://localhost:3000
3. Login with `admin` / `cyborgmind`

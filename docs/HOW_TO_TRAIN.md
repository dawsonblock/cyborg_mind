# How to Train

## Prerequisites

Ensure you have set up the environment:
```bash
./setup_mamba_gpu.sh
source venv_mamba/bin/activate
```

## Running a Training Job

Use the `train_production.py` entry point.

```bash
python train_production.py \
    --config configs/envs/gym_cartpole.yaml \
    --run-name experiment-01 \
    --num-envs 8 \
    --device cuda
```

## Configuration

Configs are YAML files located in `configs/`.

```yaml
env:
  name: "CartPole-v1"
  max_episode_steps: 500

model:
  hidden_dim: 256
  use_mamba: true

train:
  total_timesteps: 1000000
  lr: 3e-4
  use_amp: true
```

## Monitoring

1. **Console**: Shows FPS and Loss.
2. **Local CSV**: Saved to `experiments/runs/<run_name>/logs/metrics.csv`.
3. **Grafana**: If running the monitoring stack, view the "Training Dashboard".

## Checkpoints

Checkpoints are saved to `experiments/runs/<run_name>/checkpoints/`.
- `latest.pt`: Most recent save.
- `best.pt`: Best performing model.

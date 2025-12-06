# Quick Start Guide

Get CyborgMind running in 5 minutes.

## Prerequisites

- Python 3.10+
- 4GB RAM
- CPU (GPU optional)

## Installation

```bash
# Clone repository
cd cyborg_mind

# Install dependencies
pip install -r requirements.txt
pip install minigrid slowapi

# Install package
pip install -e .

# Verify setup
python scripts/verify_setup.py
```

## Your First Training Run

### 1. Basic Training (2 minutes)

```bash
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
    --task delayed_cue \
    --backbone gru \
    --horizon 10 \
    --num-envs 4 \
    --total-timesteps 10000 \
    --device cpu \
    --run-name my_first_run
```

**Expected output**:
```
[INFO] Using GRU encoder
[INFO] Initialized PPOAgent: params=1,550,729
[INFO] Starting training for 10000 timesteps
[INFO] Step 10200/10000: reward=2.27, success=0.000
[INFO] Training complete

=== Memory Benchmark Summary ===
success_rate: 0.297
mean_reward: 2.27
```

### 2. Better Results (5 minutes)

```bash
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
    --task delayed_cue \
    --backbone gru \
    --horizon 10 \
    --num-envs 8 \
    --total-timesteps 30000 \
    --device cpu \
    --run-name better_run
```

**Expected**: 40-50% success rate

### 3. High Accuracy (15 minutes)

```bash
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
    --task delayed_cue \
    --backbone gru \
    --horizon 10 \
    --num-envs 16 \
    --total-timesteps 100000 \
    --device cpu \
    --run-name high_accuracy
```

**Expected**: 70-80% success rate

## Understanding Results

### Success Rate
- **25%**: Random (baseline)
- **30-40%**: Learning started
- **50-60%**: Good progress
- **70-80%**: Well trained
- **>80%**: Excellent

### Mean Reward
- **0-2**: Random exploration
- **2-5**: Learning phase
- **5-8**: Good performance
- **8-9.5**: Near optimal

## Common Commands

### Train with Different Horizons

```bash
# Easy (5-step delay)
--horizon 5 --total-timesteps 20000

# Medium (20-step delay)
--horizon 20 --total-timesteps 50000

# Hard (50-step delay)
--horizon 50 --total-timesteps 150000
```

### Use GPU

```bash
--device cuda
```

### More Parallel Environments

```bash
--num-envs 32  # Faster training
```

### Different Backbones

```bash
--backbone gru           # Standard (recommended)
--backbone pseudo_mamba  # Experimental
```

## View Results

```bash
# Check output directory
ls experiments/runs/my_first_run/

# View metrics
cat experiments/runs/my_first_run/logs/metrics.csv
```

## Troubleshooting

### "No module named 'cyborg_rl'"

```bash
pip install -e .
```

### "No module named 'minigrid'"

```bash
pip install minigrid
```

### Low success rate

```bash
# Train longer
--total-timesteps 100000

# More environments
--num-envs 16
```

### Out of memory

```bash
# Fewer environments
--num-envs 4

# Smaller model (edit config)
hidden_dim = 128
```

## Next Steps

1. **Read**: `docs/HOW_IT_WORKS.md` - Understand the system
2. **Experiment**: Try different horizons and hyperparameters
3. **Deploy**: `docs/DEPLOYMENT.md` - Production setup
4. **Advanced**: `docs/MEMORY_ARCHITECTURE.md` - Deep dive

## Quick Reference

### Minimal Command
```bash
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
    --task delayed_cue --backbone gru --horizon 10 \
    --num-envs 4 --total-timesteps 10000 --run-name test
```

### Recommended Command
```bash
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
    --task delayed_cue --backbone gru --horizon 20 \
    --num-envs 8 --total-timesteps 50000 --run-name standard
```

### High Performance Command
```bash
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
    --task delayed_cue --backbone gru --horizon 10 \
    --num-envs 16 --total-timesteps 100000 --device cuda --run-name best
```

## Help

```bash
# Get all options
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite --help

# Verify installation
python scripts/verify_setup.py

# Check build
python scripts/build_verify.py
```

---

**Time to first result**: 2 minutes  
**Time to good result**: 15 minutes  
**Time to expert**: 1 hour

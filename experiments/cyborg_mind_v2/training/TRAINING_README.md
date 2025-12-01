# Cyborg Mind v2 Training Guide

This directory contains training scripts for the Cyborg Mind v2 project, enabling you to train both the RealTeacher (via behavioral cloning) and the BrainCyborgMind (via PPO) on MineRL environments.

## Quick Start

### 1. Verify Your Setup

**Run this first before attempting any training:**

```bash
python -m cyborg_mind_v2.training.verify_training_setup
```

This script checks:
- ✓ Java installation (required for MineRL)
- ✓ Python package versions (gym==0.21.0, minerl==0.4.4, etc.)
- ✓ CUDA/GPU availability
- ✓ Project file structure
- ✓ BrainCyborgMind API compatibility
- ✓ Action mapping consistency
- ✓ Observation adapter functionality
- ✓ MineRL environment creation

**All checks must pass** before proceeding to training.

---

## Training Scripts

### 1. RealTeacher Behavioral Cloning (`train_real_teacher_bc.py`)

Trains the RealTeacher model using behavioral cloning on the MineRL dataset.

**Purpose:** Create a pretrained teacher that can provide demonstrations for student distillation.

**Usage:**
```bash
# Basic training (1 epoch)
python -m cyborg_mind_v2.training.train_real_teacher_bc \
    --env-name MineRLTreechop-v0 \
    --epochs 1 \
    --batch-size 64 \
    --lr 3e-4

# Monitor training with TensorBoard
tensorboard --logdir runs/real_teacher_bc
```

**Key Arguments:**
- `--env-name`: MineRL environment (default: `MineRLTreechop-v0`)
- `--data-dir`: MineRL data directory (default: `~/.minerl`)
- `--output`: Checkpoint output path (default: `checkpoints/real_teacher_bc.pt`)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--lr`: Learning rate
- `--device`: Device to use (`cuda` or `cpu`)
- `--log-dir`: TensorBoard log directory

**Output:**
- `checkpoints/real_teacher_bc.pt` - Trained teacher checkpoint
- `runs/real_teacher_bc/` - TensorBoard logs

**Important:** This checkpoint is **required** before running distillation training in `teacher_student_trainer_real.py`.

---

### 2. PPO Training - Raw Brain (`train_cyborg_mind_ppo.py`)

Trains BrainCyborgMind directly using Proximal Policy Optimization (PPO) in a MineRL environment.

**Purpose:** Train the brain end-to-end with reinforcement learning.

**Usage:**
```bash
# Start PPO training (default 200k steps)
python -m cyborg_mind_v2.training.train_cyborg_mind_ppo

# Monitor with TensorBoard
tensorboard --logdir runs/cyborg_mind_ppo
```

**Configuration:**
Edit the `PPOConfig` dataclass in the script to adjust:
- `total_steps`: Total environment steps (default: 200,000)
- `steps_per_update`: Rollout buffer size (default: 4096)
- `minibatch_size`: PPO minibatch size (default: 256)
- `ppo_epochs`: PPO update epochs per rollout (default: 4)
- `gamma`: Discount factor (default: 0.99)
- `clip_eps`: PPO clipping epsilon (default: 0.2)
- `lr`: Learning rate (default: 3e-4)

**Output:**
- `checkpoints/cyborg_mind_ppo.pt` - Trained brain checkpoint
- `runs/cyborg_mind_ppo/` - TensorBoard logs

**TensorBoard Metrics:**
- `env/episode_reward` - Episode returns
- `ppo/policy_loss` - Policy gradient loss
- `ppo/value_loss` - Value function loss
- `ppo/entropy` - Policy entropy
- `ppo/returns_mean` - Mean returns
- `ppo/advantages_mean` - Mean advantages

---

### 3. PPO Training - Controller Style (`train_cyborg_mind_ppo_controller.py`)

Similar to raw brain PPO, but uses controller-style state management (thought, emotion, workspace, hidden states) for easier multi-agent extension.

**Purpose:** Train with explicit state tracking for future multi-agent scenarios.

**Usage:**
```bash
# Start controller-style PPO training
python -m cyborg_mind_v2.training.train_cyborg_mind_ppo_controller

# Monitor with TensorBoard
tensorboard --logdir runs/cyborg_mind_ppo_controller
```

**Differences from raw brain PPO:**
- Maintains explicit `thought`, `emotion`, `workspace` state throughout rollouts
- Single agent ID (`"agent0"`) for now
- Easier to extend to multi-agent by copying state per agent

**Configuration:** Similar to `train_cyborg_mind_ppo.py`, edit `PPOControllerCfg` dataclass.

**Output:**
- `checkpoints/cyborg_mind_ppo_controller.pt`
- `runs/cyborg_mind_ppo_controller/`

---

## Complete Training Pipeline

### Step 1: Verify Setup
```bash
python -m cyborg_mind_v2.training.verify_training_setup
```

### Step 2: Train Teacher (Behavioral Cloning)
```bash
python -m cyborg_mind_v2.training.train_real_teacher_bc \
    --env-name MineRLTreechop-v0 \
    --epochs 1 \
    --output checkpoints/real_teacher_bc.pt
```

### Step 3: Train Brain (PPO)
```bash
# Option A: Raw brain
python -m cyborg_mind_v2.training.train_cyborg_mind_ppo

# Option B: Controller-style
python -m cyborg_mind_v2.training.train_cyborg_mind_ppo_controller
```

### Step 4: Monitor Training
```bash
# In separate terminals
tensorboard --logdir runs/real_teacher_bc
tensorboard --logdir runs/cyborg_mind_ppo
tensorboard --logdir runs/cyborg_mind_ppo_controller
```

---

## Troubleshooting

### Java Issues
**Problem:** MineRL hangs on `env.reset()` or Java errors appear

**Solutions:**
1. Verify Java installation: `java -version`
2. Install Java JDK 8 or 11 if missing
3. Ensure Java is in PATH

### Package Version Conflicts
**Problem:** Import errors or gym API errors

**Solutions:**
```bash
# Install exact versions
pip install gym==0.21.0
pip install minerl==0.4.4
```

**Note:** Newer `gymnasium` versions are incompatible with MineRL 0.4.4.

### CUDA Not Available
**Problem:** Training uses CPU instead of GPU

**Solutions:**
1. Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Update NVIDIA drivers

### NaN Losses
**Problem:** Loss becomes NaN during training

**Solutions:**
1. Reduce learning rate (`--lr 1e-4`)
2. Reduce PPO clip epsilon (`clip_eps=0.1`)
3. Add gradient clipping (already implemented: `max_grad_norm=0.5`)
4. Check for inf/nan in observations or rewards

### Low Episode Rewards
**Problem:** PPO doesn't learn, rewards stay low

**Solutions:**
1. **Add reward shaping** - MineRL rewards are sparse, consider:
   ```python
   # In training loop after env.step()
   shaped_reward = reward
   if 'logs_collected' in info:
       shaped_reward += 0.1 * info['logs_collected']
   ```
2. Increase training steps (`total_steps=1_000_000`)
3. Adjust hyperparameters (learning rate, entropy coefficient)
4. Verify action mapping is correct

---

## Directory Structure

```
cyborg_mind_v2/training/
├── TRAINING_README.md                      # This file
├── verify_training_setup.py                # Setup verification script
├── train_real_teacher_bc.py                # Teacher BC training
├── train_cyborg_mind_ppo.py                # PPO training (raw brain)
├── train_cyborg_mind_ppo_controller.py     # PPO training (controller-style)
├── real_teacher.py                          # RealTeacher model
└── teacher_student_trainer_real.py         # Distillation training

checkpoints/
├── real_teacher_bc.pt                      # Teacher checkpoint
├── cyborg_mind_ppo.pt                      # PPO checkpoint (raw)
└── cyborg_mind_ppo_controller.pt           # PPO checkpoint (controller)

runs/
├── real_teacher_bc/                        # BC TensorBoard logs
├── cyborg_mind_ppo/                        # PPO TensorBoard logs
└── cyborg_mind_ppo_controller/             # Controller PPO logs
```

---

## Key Configuration Points

### Action Mapping
Defined in `cyborg_mind_v2/envs/action_mapping.py`:
- `NUM_ACTIONS` - Total discrete actions (default: 14)
- `index_to_minerl_action()` - Convert discrete index to MineRL action dict
- `minerl_action_to_index()` - Convert MineRL action dict to discrete index (BC training)

**Important:** These mappings must be **consistent** across all scripts.

### Observation Adapter
Defined in `cyborg_mind_v2/envs/minerl_obs_adapter.py`:
- `obs_to_brain()` - Convert MineRL obs dict to (pixels, scalars, goal) tuple
  - `pixels`: [3, 128, 128], float32, normalized to [0, 1]
  - `scalars`: [20], float32 (currently zeros, extend with game state)
  - `goal`: [4], float32 (currently zeros, extend with task goals)

**TODO:** Populate `scalars` with meaningful features:
- Health, hunger, y-level
- Inventory counts (logs, tools)
- Distance to nearest tree
- Time step fraction

### Brain API
`BrainCyborgMind.forward()` **must** return a dict with these keys:
- `action_logits`: [B, NUM_ACTIONS]
- `value`: [B, 1]
- `thought`: [B, 32]
- `emotion`: [B, 8]
- `workspace`: [B, 64]
- `hidden_h`: [1, B, 512]
- `hidden_c`: [1, B, 512]
- `mem_write`: [B, mem_dim]
- `pressure`: [B] or scalar

---

## Next Steps

1. **Run verification script** to ensure all dependencies are correct
2. **Train teacher** with BC on MineRL dataset (1-5 epochs)
3. **Train brain** with PPO (start with small `total_steps=50_000` for testing)
4. **Monitor TensorBoard** for loss curves and episode rewards
5. **Refine action mapping** based on dataset analysis
6. **Add scalar features** (health, inventory, etc.) to observation adapter
7. **Implement reward shaping** for better PPO learning
8. **Scale up training** once smoke tests pass

---

## Common Commands Cheat Sheet

```bash
# Verify setup
python -m cyborg_mind_v2.training.verify_training_setup

# Train teacher (1 epoch test)
python -m cyborg_mind_v2.training.train_real_teacher_bc --epochs 1

# Train PPO (short test)
# Edit PPOConfig: total_steps=50_000 first
python -m cyborg_mind_v2.training.train_cyborg_mind_ppo

# Monitor all training
tensorboard --logdir runs

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check Java
java -version

# Check gym version (must be 0.21.0)
python -c "import gym; print(gym.__version__)"

# Check minerl version (must be 0.4.4)
python -c "import minerl; print(minerl.__version__)"
```

---

## Performance Tips

### For 3080 Ti (12GB VRAM)
- **Batch size:** 64-128 for BC, 256 for PPO minibatch
- **Buffer size:** 4096 steps per PPO update
- **Use mixed precision:** Add `torch.cuda.amp.autocast()` to forward passes
- **Monitor VRAM:** `nvidia-smi -l 1`

### For CPU Training (not recommended)
- Reduce batch sizes significantly (16-32)
- Reduce buffer sizes (1024 steps)
- Expect 10-50x slower training
- Use for debugging only

---

## License & Attribution

Part of the Cyborg Mind v2 project. Training scripts integrate:
- MineRL dataset (Guss et al., 2019)
- CLIP vision encoder (Radford et al., 2021)
- PPO algorithm (Schulman et al., 2017)

See main project README for full citations.

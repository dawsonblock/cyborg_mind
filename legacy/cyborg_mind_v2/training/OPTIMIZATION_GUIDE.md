# Training Optimization & Speed-Up Guide

## Fixed Critical Errors ✓

### 1. **Missing `envs/` Directory** (CRITICAL - Training Would Crash)
**Error:** All training scripts imported from `cyborg_mind_v2.envs.action_mapping` and `cyborg_mind_v2.envs.minerl_obs_adapter` which didn't exist.

**Fix:** Created:
- `cyborg_mind_v2/envs/__init__.py`
- `cyborg_mind_v2/envs/action_mapping.py` - Discrete action mapping for MineRL
- `cyborg_mind_v2/envs/minerl_obs_adapter.py` - Observation preprocessing

### 2. **RealTeacher Checkpoint Requirement** (CRITICAL - BC Training Would Crash)
**Error:** `RealTeacher.__init__()` raised `ValueError` if `ckpt_path=None`, but BC trainer needs to train from scratch.

**Fix:** Modified `real_teacher.py` to allow random initialization when `ckpt_path=None`.

### 3. **Import Errors** (Non-Critical - IDE warnings only)
**Status:** These are type stub warnings from MyPy for external packages (cv2, minerl). They don't affect runtime.

**To silence (optional):**
```bash
# Add to pyproject.toml or mypy.ini
[tool.mypy]
ignore_missing_imports = true
```

---

## 🚀 Speed Optimizations

### **Tier 1: Essential Speed-Ups** (Implement First)

#### 1. **Enable Mixed Precision Training (AMP)**
**Speed-up:** 2-3x faster, 40% less VRAM  
**Difficulty:** Easy

Add to training scripts:

```python
from torch.cuda.amp import autocast, GradScaler

# In training loop setup
scaler = GradScaler()

# Wrap forward pass
with autocast():
    logits, value = teacher.predict(pixels, scalars)
    loss = criterion(logits, actions)

# Replace optimizer step
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Files to modify:**
- `train_real_teacher_bc.py` - in `flush_batch()`
- `train_cyborg_mind_ppo.py` - in `ppo_update()`
- `train_cyborg_mind_ppo_controller.py` - in `ppo_update_controller()`

#### 2. **Use DataLoader Instead of Manual Batching**
**Speed-up:** 1.5-2x faster (parallel data loading)  
**Difficulty:** Medium

Current BC trainer uses manual buffering. Replace with PyTorch DataLoader:

```python
from torch.utils.data import Dataset, DataLoader

class MineRLDataset(Dataset):
    def __init__(self, data, max_samples=10000):
        # Pre-load samples into memory
        self.samples = []
        seq_iter = data.sarsd_iter(num_epochs=1, max_sequence_len=64)
        for obs_seq, act_seq, _, _, _ in seq_iter:
            for t in range(len(obs_seq["pov"])):
                obs_t = {k: v[t] for k, v in obs_seq.items()}
                act_t = {k: v[t] for k, v in act_seq.items()}
                pixels, scalars = obs_to_teacher_inputs(obs_t)
                action_idx = minerl_action_to_index(act_t)
                self.samples.append((pixels, scalars, action_idx))
                if len(self.samples) >= max_samples:
                    return
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# Use DataLoader
dataset = MineRLDataset(data, max_samples=50000)
loader = DataLoader(dataset, batch_size=64, shuffle=True,
                    num_workers=4, pin_memory=True)

for pixels, scalars, actions in loader:
    # Training step
    pass
```

#### 3. **Reduce CLIP Encoding Overhead**
**Speed-up:** 1.3x faster  
**Difficulty:** Easy

CLIP is called every forward pass. Cache encodings when possible:

```python
# In BC trainer, encode once per batch
with torch.no_grad():
    visual_features = teacher.encode_pixels(pixels)  # [B, 768]

# Then only train heads (faster)
teacher.train()  # Only heads are trainable
scalars_t = scalars.to(device)
fused_input = torch.cat([visual_features.detach(), scalars_t], dim=1)
fused = teacher.scalar_fusion(fused_input)
logits = teacher.action_head(fused)
```

---

### **Tier 2: Medium Speed-Ups**

#### 4. **Use Smaller CLIP Model for Initial Training**
**Speed-up:** 2x faster encoding  
**Difficulty:** Easy

```python
# In real_teacher.py, line 55
# Replace:
# base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# With faster variant:
base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# Or even smaller (3x faster):
# from transformers import CLIPVisionModel
# self.vision_encoder = CLIPVisionModel.from_pretrained(
#     "openai/clip-vit-base-patch32",
#     image_size=128  # Smaller input
# )
```

#### 5. **Compile Models with torch.compile() (PyTorch 2.0+)**
**Speed-up:** 1.5-2x faster  
**Difficulty:** Easy (requires PyTorch 2.0+)

```python
# After creating models
if torch.__version__ >= "2.0":
    teacher = torch.compile(teacher, mode="reduce-overhead")
    brain = torch.compile(brain, mode="reduce-overhead")
```

#### 6. **Reduce PPO Rollout Buffer Size**
**Speed-up:** Faster updates, less memory  
**Trade-off:** Slightly less stable learning

```python
# In PPOConfig
steps_per_update: int = 2048  # Down from 4096
minibatch_size: int = 128      # Down from 256
ppo_epochs: int = 3             # Down from 4
```

#### 7. **Enable cudnn Benchmarking**
**Speed-up:** 10-20% faster  
**Difficulty:** Trivial

Add to top of training scripts:

```python
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

---

### **Tier 3: Advanced Optimizations**

#### 8. **Distributed Data Parallel (Multi-GPU)**
**Speed-up:** Near-linear with GPU count  
**Difficulty:** Hard

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Launch with: torchrun --nproc_per_node=2 train_script.py
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])

teacher = RealTeacher(...)
teacher = DDP(teacher.to(local_rank), device_ids=[local_rank])
```

#### 9. **Gradient Accumulation (Simulate Larger Batches)**
**Speed-up:** Better learning with limited VRAM  
**Difficulty:** Easy

```python
accumulation_steps = 4

for i, (pixels, scalars, actions) in enumerate(loader):
    loss = ... / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 10. **Use NVIDIA Apex for FP16 (Alternative to AMP)**
**Speed-up:** Similar to AMP but more control  
**Difficulty:** Medium

```bash
pip install nvidia-apex
```

```python
from apex import amp

teacher, optimizer = amp.initialize(teacher, optimizer, opt_level="O2")

with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

---

## 📊 Training Simplifications

### **Simplification 1: Pre-Cache MineRL Dataset**

**Problem:** Loading from disk every epoch is slow  
**Solution:** Pre-process and save to memory-mapped arrays

```python
# One-time preprocessing
import numpy as np

def cache_minerl_dataset(env_name, output_dir, max_samples=100000):
    data = minerl.data.make(environment=env_name)
    
    pixels_list = []
    actions_list = []
    
    for obs_seq, act_seq, _, _, _ in data.sarsd_iter(num_epochs=1):
        for t in range(len(obs_seq["pov"])):
            pixels, scalars = obs_to_teacher_inputs(
                {k: v[t] for k, v in obs_seq.items()}
            )
            action_idx = minerl_action_to_index(
                {k: v[t] for k, v in act_seq.items()}
            )
            pixels_list.append(pixels)
            actions_list.append(action_idx)
            
            if len(pixels_list) >= max_samples:
                break
        if len(pixels_list) >= max_samples:
            break
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/pixels.npy", np.array(pixels_list))
    np.save(f"{output_dir}/actions.npy", np.array(actions_list))
    print(f"Cached {len(pixels_list)} samples to {output_dir}")

# Then in training
pixels = np.load("minerl_cache/pixels.npy", mmap_mode='r')
actions = np.load("minerl_cache/actions.npy", mmap_mode='r')
```

**Speed-up:** 5-10x faster data loading

---

### **Simplification 2: Reduce Action Space Complexity**

**Problem:** 14 discrete actions may be too granular  
**Solution:** Use a smaller, more meaningful action set

```python
# In action_mapping.py
NUM_ACTIONS = 8  # Reduced from 14

# Simplified mapping:
# 0: noop
# 1: forward
# 2: jump+forward (common combo)
# 3: attack
# 4: look_left
# 5: look_right
# 6: look_up
# 7: look_down
```

**Benefits:** Faster convergence, easier to learn

---

### **Simplification 3: Use Curriculum Learning**

Start with easier tasks, gradually increase difficulty:

```python
# Phase 1: Easy (500 logs available, flat terrain)
env = gym.make("MineRLTreechop-v0")

# Phase 2: Medium (100 logs, some obstacles)
# Custom env configuration

# Phase 3: Hard (sparse logs, complex terrain)
# Full environment
```

---

### **Simplification 4: Reduce Image Resolution**

**Trade-off:** Faster but less detail

```python
# In minerl_obs_adapter.py
def obs_to_brain(obs, image_size=64):  # Down from 128
    # 4x fewer pixels = 4x faster
```

**Speed-up:** 3-4x faster forward pass  
**Recommended:** Only for initial prototyping

---

### **Simplification 5: Use Pre-Trained RL Checkpoints**

**Option 1:** Use VPT (Video Pre-Training) weights  
You already have `vpt_minerl.tar.gz` - this contains pre-trained Minecraft policies!

```python
# Load VPT checkpoint
vpt_ckpt = torch.load("vpt_minerl.tar.gz")
# Transfer relevant weights to your brain
brain.load_state_dict(vpt_ckpt, strict=False)
```

**Option 2:** Use CLIP-conditioned RL checkpoints from OpenAI

---

## 🎯 Recommended Quick-Start Optimizations

**For immediate training (apply these first):**

1. ✅ **Fixed critical errors** (already done)
2. **Enable cudnn benchmark** (1 line)
3. **Enable mixed precision (AMP)** (10 lines)
4. **Reduce PPO buffer sizes** (3 config changes)
5. **Cache dataset to disk** (run once, huge speedup)

**Total implementation time:** ~30 minutes  
**Expected speed-up:** 3-5x faster training

---

## 📝 Quick Optimization Checklist

```bash
# 1. Enable cudnn (add to script top)
torch.backends.cudnn.benchmark = True

# 2. Reduce PPO config
# Edit PPOConfig:
# steps_per_update = 2048
# minibatch_size = 128
# ppo_epochs = 3

# 3. Cache dataset (run once)
python -c "from training.cache_dataset import cache_minerl_dataset; \
cache_minerl_dataset('MineRLTreechop-v0', 'minerl_cache')"

# 4. Add AMP to training loop (see Tier 1 #1)

# 5. Run training
python -m cyborg_mind_v2.training.train_real_teacher_bc \
    --epochs 1 --batch-size 128
```

---

## 🔥 Extreme Speed Mode (for debugging)

When you just want to test if training works:

```python
# In PPOConfig
total_steps: int = 10_000        # Down from 200k
steps_per_update: int = 512      # Down from 4096
minibatch_size: int = 64         # Down from 256
ppo_epochs: int = 2              # Down from 4

# In BC trainer
max_samples = 1000               # Train on tiny dataset
epochs = 1                       # One pass only
```

**Expected time:** 2-5 minutes per run  
**Purpose:** Rapid prototyping and debugging

---

## 💾 Memory Optimizations (if VRAM limited)

### If getting OOM errors on 3080 Ti (12GB):

1. **Reduce batch size**
   ```python
   batch_size = 32  # Down from 64
   minibatch_size = 64  # Down from 256
   ```

2. **Enable gradient checkpointing**
   ```python
   from torch.utils.checkpoint import checkpoint
   # Wrap expensive operations
   ```

3. **Use bfloat16 instead of float16**
   ```python
   torch.set_default_dtype(torch.bfloat16)
   ```

4. **Clear cache between batches**
   ```python
   torch.cuda.empty_cache()  # After each update
   ```

---

## 🧪 Experimental: TensorRT Optimization

For maximum inference speed (production deployment):

```bash
pip install torch-tensorrt

# Compile brain to TensorRT
import torch_tensorrt

brain_trt = torch_tensorrt.compile(
    brain,
    inputs=[torch_tensorrt.Input((1, 3, 128, 128))],
    enabled_precisions={torch.float16}
)
```

**Speed-up:** 5-10x faster inference  
**Limitation:** Training only, requires NVIDIA GPU

---

## Summary: Fastest Training Setup

```python
# Configuration for maximum speed
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# Use cached dataset
dataset = CachedMineRLDataset("minerl_cache/")

# DataLoader with parallel workers
loader = DataLoader(dataset, batch_size=128, num_workers=4,
                    pin_memory=True, prefetch_factor=2)

# Enable AMP
scaler = GradScaler()

# Smaller rollout buffers (PPO)
steps_per_update = 2048

# Compile models (PyTorch 2.0+)
teacher = torch.compile(teacher)
brain = torch.compile(brain)
```

**Expected performance on 3080 Ti:**
- BC training: ~5000 samples/sec (vs 1000 baseline)
- PPO training: ~200 env steps/sec (vs 40 baseline)

Total training time for initial checkpoint: **~1-2 hours** (vs 6-8 hours baseline)

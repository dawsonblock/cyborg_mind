# CyborgMind V7 Architecture

## Overview

CyborgMind V7 is a production-grade reinforcement learning framework featuring:

- **Honest Recurrent PPO**: States flow from acting to training without recomputation
- **Multi-Mode Encoder**: GRU, Mamba, Hybrid, Fusion architectures
- **Advanced Memory**: PMM++, Slot, KV, Ring memory modules
- **Burn-In Training**: Gradient masking for proper recurrent learning

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TrainerV7                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Collect Rollout                      │   │
│  │  ┌─────────┐   ┌──────────┐   ┌─────────┐           │   │
│  │  │ Encoder │ → │  Memory  │ → │ Policy  │ → Action  │   │
│  │  │   V7    │   │    V7    │   │  Head   │           │   │
│  │  └─────────┘   └──────────┘   └─────────┘           │   │
│  │       ↓             ↓                                │   │
│  │  [States stored honestly in RolloutBufferV7]        │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  PPO Update                           │   │
│  │  sample_sequences(seq_len, burn_in)                  │   │
│  │       ↓                                               │   │
│  │  [Burn-in: replay state, mask gradients]            │   │
│  │  [Training: compute loss, backprop]                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### EncoderV7

Multi-mode recurrent encoder supporting:

| Mode | Description |
|------|-------------|
| `gru` | Standard GRU encoder |
| `mamba` | Official Mamba SSM (CUDA) or PseudoMamba (fallback) |
| `hybrid` | Mamba → GRU with residual connection |
| `fusion` | Parallel Mamba + GRU, concatenated outputs |

**State Management:**
- `get_initial_state(batch_size, device)` - Create zero states
- `reset_states(state, mask)` - Reset done episodes
- `detach_states(state)` - Detach for TBPTT

### MemoryV7

Unified memory interface with multiple implementations:

| Type | Description |
|------|-------------|
| `pmm` | Multi-head attention with usage-based eviction |
| `slot` | Simple slot-based memory with learned addressing |
| `kv` | Key-value store with circular write pointer |
| `ring` | Circular buffer for temporal tasks |

**Interface:**
```python
read, new_state, logs = memory.forward_step(hidden, state, mask)
read_seq, final_state, logs = memory.forward_sequence(hidden_seq, state, masks)
```

### RolloutBufferV7

Batched rollout buffer with [T, N, ...] layout:

**Features:**
- Stores obs, actions, values, log_probs, dones, hidden states, memory
- GAE computation with proper discounting
- Sequence sampling with burn-in gradient masking

**Sampling:**
```python
for batch in buffer.sample_sequences(seq_len=64, burn_in=128):
    # batch["obs"]: (B, burn_in + seq_len, D)
    # batch["grad_mask"]: 0 for burn-in, 1 for training
```

### TrainerV7

Honest recurrent PPO trainer:

**Principles:**
1. States flow from acting → buffer → training (no hidden recomputation)
2. Burn-in replays state, gradients masked to post-burn region
3. State slicing correct for GRU + Mamba caches

**PPO++ Features:**
- Value clipping
- Normalized advantages
- Adaptive KL early stopping
- Gradient norm clipping

---

## Usage

### CLI

```bash
# Quick start
python train_v7.py --env minerl_treechop --encoder mamba --memory pmm

# With config
python train_v7.py --config configs/memory_ppo_v7.yaml --wandb

# Custom settings
python train_v7.py \
    --env CartPole-v1 \
    --encoder fusion \
    --memory kv \
    --num-envs 8 \
    --horizon 2048 \
    --burn-in 256 \
    --amp
```

### Python API

```python
from cyborg_rl.trainers.trainer_v7 import TrainerV7

config = {
    "env": "minerl_treechop",
    "encoder": "mamba",
    "memory": {"type": "pmm", "num_slots": 16},
    "horizon": 1024,
    "burn_in": 128,
}

trainer = TrainerV7(config)
trainer.train(total_timesteps=1_000_000)
trainer.save("checkpoints/model.pt")
```

---

## File Structure

```
cyborg_rl/
├── models/
│   └── encoder_v7.py      # Multi-mode encoder
├── memory/
│   ├── memory_v7.py       # PMM, Slot, KV, Ring
│   └── rollout_buffer_v7.py
├── trainers/
│   └── trainer_v7.py      # Honest recurrent PPO
├── envs/
│   └── universal_adapter.py
configs/
└── memory_ppo_v7.yaml
train_v7.py
```

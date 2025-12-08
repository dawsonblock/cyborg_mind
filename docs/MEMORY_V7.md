# Memory Systems V7

## Overview

CyborgMind V7 provides four memory architectures, each suited for different use cases.

---

## Memory Types

| Type | Best For | State Size | Complexity |
|------|----------|------------|------------|
| **PMM** | Long-term storage, multi-step reasoning | O(slots × dim) | High |
| **Slot** | Simple fixed addressing | O(slots × dim) | Low |
| **KV** | Episodic memory, retrieval | O(slots × (key + dim)) | Medium |
| **Ring** | Temporal patterns, recent history | O(buffer × dim) | Low |

---

## PMM (Predictive Memory Module)

### Architecture
```
         ┌─────────────────────────────────────┐
         │           Multi-Head Attention       │
         │  ┌─────┐  ┌─────┐  ┌─────┐         │
hidden ──┼──│ Q   │  │ K   │  │ V   │         │
         │  └──┬──┘  └──┬──┘  └──┬──┘         │
         │     └────────┴───────┘              │
         │              │                       │
         │              ▼                       │
         │     ┌─────────────────┐             │
         │     │   Memory Slots   │◄── Write   │
         │     │   (usage-based   │    Gate    │
         │     │    eviction)     │             │
         │     └────────┬────────┘             │
         │              │                       │
         │              ▼                       │
         │         Read Output                  │
         └─────────────────────────────────────┘
```

### Features
- **Multi-head addressing**: Query memory with multiple attention heads
- **Usage-based eviction**: Write to least-used slots
- **Temporal decay**: Old memories fade
- **Write conflict detection**: Penalize overwriting during reads

### Usage
```python
from cyborg_rl.memory.memory_v7 import PMMV7

pmm = PMMV7(
    memory_dim=256,
    num_slots=16,
    num_heads=4,
    decay_rate=0.99,
)

state = pmm.get_initial_state(batch_size=32, device="cuda")
read, state, logs = pmm.forward_step(hidden, state)
```

### Logs
- `write_strength`: Average write gate activation
- `read_entropy`: Entropy of attention weights
- `overwrite_conflict`: Overlap between read/write attention
- `usage_mean`: Average slot usage

---

## Slot Memory

### Architecture
Simple fixed-slot memory with learned addressing.

```
hidden ──► address_net ──► softmax ──► slot weights
                                           │
                                           ▼
                               ┌───────────────────┐
                               │  Slot 0 │ Slot 1  │
                               │  Slot 2 │ Slot 3  │
                               └───────────────────┘
                                           │
                                    read & write
```

### Usage
```python
from cyborg_rl.memory.memory_v7 import SlotMemoryV7

memory = SlotMemoryV7(memory_dim=256, num_slots=8)
state = memory.get_initial_state(32, "cuda")
read, state, logs = memory.forward_step(hidden, state)
```

---

## KV Memory

### Architecture
Key-value store with circular write pointer.

```
hidden ──► key_proj ──► new_key ──┐
      │                           ▼
      └──► value_proj ──► new_val ──► Memory[write_ptr]
                                           │
      ┌────────────────────────────────────┘
      │
query ──► attention over keys ──► weighted sum of values ──► read
```

### Features
- Episodic storage (one key-value pair per step)
- FIFO eviction with circular pointer
- Retrieval by similarity

---

## Ring Memory

### Architecture
Fixed-size circular buffer for recent history.

```
t=0  t=1  t=2  t=3  t=4  ...  t=N-1
 │    │    │    │    │         │
 ▼    ▼    ▼    ▼    ▼         ▼
┌────────────────────────────────┐
│  Ring Buffer (N slots)         │
│  ptr ──► oldest entry          │
└────────────────────────────────┘
            │
            ▼
     query ──► attention ──► read
```

### Use Cases
- Short-term temporal patterns
- Moving average of recent observations
- Sequence modeling without full history

---

## Memory Factory

```python
from cyborg_rl.memory.memory_v7 import create_memory

# Create any memory type
memory = create_memory(
    memory_type="pmm",  # or "slot", "kv", "ring"
    memory_dim=256,
    num_slots=16,
)
```

---

## State Management

All memories share a unified interface:

```python
# Initial state
state = memory.get_initial_state(batch_size, device)

# Forward step
read, state, logs = memory.forward_step(hidden, state, mask=done_mask)

# Forward sequence
read_seq, final_state, logs = memory.forward_sequence(hidden_seq, state, masks)

# Reset on episode done
state = memory.reset_state(state, ~dones)
```

---

## Integration with Trainer

```python
# In TrainerV7
memory = create_memory(config["memory"]["type"], **config["memory"])
mem_state = memory.get_initial_state(num_envs, device)

for t in range(horizon):
    # Encode
    latent, enc_state = encoder(obs, enc_state)
    
    # Memory read
    read, mem_state, mem_logs = memory.forward_step(latent, mem_state)
    
    # Combine for policy
    policy_input = torch.cat([latent, read], dim=-1)
    
    # Reset on done
    mem_state = memory.reset_state(mem_state, ~dones)
```

---

## Choosing a Memory Type

| Use Case | Recommended |
|----------|-------------|
| General RL | PMM |
| Fast prototyping | Slot |
| Few-shot learning | KV |
| Temporal prediction | Ring |
| Long episodes | PMM with many slots |
| Memory-constrained | Slot or Ring |

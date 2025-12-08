# Recurrent PPO V7 - Technical Deep Dive

## Overview

CyborgMind V7 implements **Honest Recurrent PPO** - a variant of PPO that correctly handles recurrent state transitions without cheating by recomputing hidden states.

---

## The State Cheating Problem

### Common Mistake (Many RL Libraries)
```python
# WRONG: Recompute hidden states from scratch during training
for batch in buffer.sample():
    hidden = encoder.get_initial_state()
    for t in range(T):
        output, hidden = encoder(obs[t], hidden)
    # Hidden states don't match acting phase!
```

### Honest Approach (V7)
```python
# CORRECT: Store and reuse actual hidden states
for batch in buffer.sample():
    hidden = batch["initial_hidden"]  # Stored during acting
    for t in range(T):
        output, hidden = encoder(obs[t], hidden)
    # Hidden states match acting phase exactly
```

---

## Burn-In Mechanism

### Purpose
Recurrent networks need time to "warm up" their hidden states. Burn-in provides this warm-up period while masking gradients.

### Implementation
```
Sequence: [---burn_in---][---training---]
Gradients:      0               1
```

```python
# RolloutBufferV7.sample_sequences
total_len = burn_in + seq_len
grad_mask = torch.cat([
    torch.zeros(B, burn_in),
    torch.ones(B, seq_len),
], dim=1)

# During loss computation
loss = (pred - target).pow(2) * grad_mask
```

---

## State Flow Diagram

```
ACTING PHASE                        TRAINING PHASE
─────────────                       ──────────────
                                    
env.reset()                         buffer.sample_sequences()
    │                                       │
    ▼                                       ▼
encoder.get_initial_state() ────► stored as batch["init_hidden"]
    │                                       │
    ▼                                       │
for t in horizon:                          │
    │                                       │
    obs[t] ─────────────────────► batch["obs"][:, burn_in:]
    │                                       │
    encoder(obs[t], hidden[t-1])           encoder(batch["obs"], init_hidden)
    │         │                                    │
    │         ▼                                    ▼
    │    hidden[t] ──────────────► replay burn-in, then train
    │         │
    ▼         ▼
   action   stored in buffer
```

---

## GAE Computation

Generalized Advantage Estimation with proper discounting:

```python
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(T)):
        next_value = values[t+1] if t < T-1 else last_value
        next_done = dones[t+1] if t < T-1 else last_done
        
        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns
```

---

## PPO++ Features

### 1. Value Clipping
```python
value_pred_clipped = old_values + torch.clamp(
    new_values - old_values,
    -clip_range,
    clip_range,
)
value_loss = max(
    (new_values - returns).pow(2),
    (value_pred_clipped - returns).pow(2),
)
```

### 2. Adaptive KL Early Stopping
```python
for epoch in range(ppo_epochs):
    # ... compute loss ...
    approx_kl = ((ratio - 1) - ratio.log()).mean()
    
    if approx_kl > target_kl * 1.5:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 3. PopArt Value Normalization
```python
from cyborg_rl.models.popart import PopArtValueHead

value_head = PopArtValueHead(input_dim=256)

# Training
value_head.update(returns)  # Update running stats
normalized_value = value_head.forward_normalized(features)
normalized_target = value_head.normalize(returns)
value_loss = (normalized_value - normalized_target).pow(2)

# Acting
denormalized_value = value_head(features)  # For logging
```

### 4. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
```

---

## Memory Integration

### PMM Flow
```
hidden ──► query_proj ──► attention ──► read
                              │
                              ▼
                        write_gate ──► eviction ──► new_memory
```

### State Management
```python
# Get initial memory state
mem_state = memory.get_initial_state(batch_size, device)

# Forward step (single timestep)
read, mem_state, logs = memory.forward_step(hidden, mem_state)

# Reset on done
mem_state = memory.reset_state(mem_state, ~dones)
```

---

## Configuration Reference

```yaml
# Recurrent settings
burn_in: 128        # Warm-up steps (gradients masked)
seq_len: 64         # Training sequence length
horizon: 1024       # Rollout length

# PPO settings
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
value_clip: 0.2
entropy_coef: 0.01
value_coef: 0.5
ppo_epochs: 4
target_kl: 0.02
max_grad_norm: 0.5
```

---

## Debugging Tips

### 1. KL Exploding
- Reduce learning rate
- Increase burn-in length
- Check for NaN in hidden states

### 2. Value Loss Not Decreasing
- Enable PopArt
- Check reward normalization
- Verify GAE computation

### 3. Policy Collapse
- Check entropy coefficient
- Verify action distribution logging
- Look for mode collapse in action histogram

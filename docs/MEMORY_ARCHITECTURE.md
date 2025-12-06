# Memory Architecture Deep Dive

## Neural Network Components

### 1. GRU Encoder

**Purpose**: Maintain temporal context across timesteps

**Architecture**:
```python
GRU(
    input_size=obs_dim,      # 4 (one-hot cue)
    hidden_size=256,         # Hidden state dimension
    num_layers=2,            # Stacked layers
    batch_first=True
)
```

**How It Works**:
- Takes observation at each timestep
- Updates hidden state: `h_t = GRU(obs_t, h_{t-1})`
- Hidden state carries information forward
- 2 layers allow hierarchical representations

**Memory Capacity**: ~131k parameters

### 2. Predictive Memory Module (PMM)

**Purpose**: External differentiable memory for long-term storage

**Architecture**:
```python
Memory Matrix: [batch_size, 64 slots, 128 dimensions]
Read Heads: 4 attention-based readers
Write Heads: 1 attention-based writer
```

**Operations**:

**Write**:
```python
# Compute attention weights over memory slots
write_weights = softmax(query @ memory.T)
# Update memory
memory = memory + write_weights @ write_vector
```

**Read**:
```python
# Compute attention for each read head
read_weights = softmax(query @ memory.T)
# Retrieve information
read_vector = read_weights @ memory
```

**Memory Capacity**: ~1.2M parameters

### 3. Policy Head

**Purpose**: Map memory state to action distribution

**Architecture**:
```python
Linear(latent_dim=256, action_dim=4)
→ Gaussian(mean=output, std=learned)
```

**Output**: 4 continuous values (one per cue)
**Interpretation**: Agent uses argmax to select discrete action

### 4. Value Head

**Purpose**: Estimate expected future reward

**Architecture**:
```python
Linear(latent_dim=256, 1)
```

**Output**: Scalar value estimate
**Use**: Compute advantages for PPO

---

## Information Flow

### Forward Pass

```
Observation [4]
    ↓
GRU Layer 1 [256]
    ↓
GRU Layer 2 [256]
    ↓
PMM Write (store cue)
    ↓
PMM Read (retrieve when needed)
    ↓
Latent State [256]
    ↓
    ├─→ Policy Head → Action [4]
    └─→ Value Head → Value [1]
```

### Backward Pass (BPTT)

```
Loss (at query step)
    ↓
∇Policy Head
    ↓
∇PMM (read/write gradients)
    ↓
∇GRU Layer 2
    ↓
∇GRU Layer 1
    ↓
∇Observation Encoding
```

Gradients flow through entire sequence (22 steps for horizon=20)

---

## Memory Mechanisms

### Short-Term Memory (GRU)

**Capacity**: Limited by hidden dimension (256)
**Duration**: Degrades over time without reinforcement
**Best for**: Horizons <20 steps

**Equations**:
```
Reset gate:  r_t = σ(W_r·[h_{t-1}, x_t])
Update gate: z_t = σ(W_z·[h_{t-1}, x_t])
Candidate:   h̃_t = tanh(W·[r_t ⊙ h_{t-1}, x_t])
Hidden:      h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### Long-Term Memory (PMM)

**Capacity**: 64 slots × 128 dim = 8,192 values
**Duration**: Persistent until overwritten
**Best for**: Horizons >20 steps

**Attention Mechanism**:
```python
# Similarity-based addressing
attention = softmax(query @ memory.T / sqrt(dim))
# Weighted retrieval
output = attention @ memory
```

---

## Training Dynamics

### Gradient Flow

**Challenge**: Vanishing gradients over long sequences

**Solutions**:
1. **GRU gates**: Preserve gradients better than vanilla RNN
2. **Gradient clipping**: `max_norm=1.0` prevents explosion
3. **Residual connections**: In PMM read/write
4. **Layer normalization**: Stabilizes activations

### Loss Components

**Total Loss**:
```python
loss = policy_loss + 0.5 * value_loss + 0.05 * entropy_loss
```

**Policy Loss (PPO)**:
```python
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1-ε, 1+ε) * advantage
policy_loss = -min(surr1, surr2)
```

**Value Loss**:
```python
value_loss = (return - value_pred)²
```

**Entropy Loss**:
```python
entropy_loss = -entropy(action_distribution)
```

---

## Memory Encoding Strategies

### What the Agent Learns

**Phase 1: Cue Encoding**
- GRU learns to detect one-hot patterns
- PMM writes cue to specific memory slot
- Value head learns cue phase has no immediate reward

**Phase 2: Delay Maintenance**
- GRU maintains stable hidden state
- PMM preserves written information
- Agent learns to "do nothing" during delay

**Phase 3: Cue Retrieval**
- GRU detects query phase (all zeros after delay)
- PMM reads stored cue information
- Policy head decodes memory to action

### Visualization

```
Step 0 (Cue):
  Input:  [0, 1, 0, 0]
  GRU:    [0.2, 0.8, 0.1, ...]  ← Encodes cue
  Memory: Write to slot 5
  Action: [0.1, 0.3, 0.2, 0.4]  (random)

Steps 1-20 (Delay):
  Input:  [0, 0, 0, 0]
  GRU:    [0.2, 0.7, 0.1, ...]  ← Maintains state
  Memory: Slot 5 preserved
  Action: [0.2, 0.3, 0.3, 0.2]  (random)

Step 21 (Query):
  Input:  [0, 0, 0, 0]
  GRU:    [0.2, 0.7, 0.1, ...]  ← Stable
  Memory: Read from slot 5 → [0, 1, 0, 0]
  Action: [0.1, 0.9, 0.0, 0.0]  ← Correct!
```

---

## Hyperparameter Impact

### Hidden Dimension

| Size | Capacity | Speed | Best For |
|------|----------|-------|----------|
| 128  | Low      | Fast  | Horizon <10 |
| 256  | Medium   | Medium| Horizon 10-30 |
| 512  | High     | Slow  | Horizon >30 |

### Number of GRU Layers

| Layers | Capacity | Stability | Best For |
|--------|----------|-----------|----------|
| 1      | Low      | High      | Simple tasks |
| 2      | Medium   | Medium    | Standard |
| 3+     | High     | Low       | Complex tasks |

### Memory Size

| Slots | Capacity | Overhead | Best For |
|-------|----------|----------|----------|
| 32    | 4,096    | Low      | Few cues |
| 64    | 8,192    | Medium   | Standard |
| 128   | 16,384   | High     | Many cues |

### Learning Rate

| LR    | Convergence | Stability | Best For |
|-------|-------------|-----------|----------|
| 1e-4  | Slow        | High      | Fine-tuning |
| 1e-3  | Fast        | Medium    | Standard |
| 1e-2  | Very Fast   | Low       | Exploration |

---

## Advanced Techniques

### Curriculum Learning

Start with easy tasks, gradually increase difficulty:

```bash
# Stage 1: Short horizon
--horizon 5 --total-timesteps 20000

# Stage 2: Medium horizon
--horizon 10 --total-timesteps 30000

# Stage 3: Long horizon
--horizon 20 --total-timesteps 50000
```

### Learning Rate Scheduling

```python
# Cosine annealing
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))

# Step decay
lr_t = lr_0 * 0.5^(epoch / 10)
```

### Entropy Annealing

Reduce exploration over time:

```python
entropy_coef_t = entropy_start * (entropy_end / entropy_start)^(t / T)
# Start: 0.05, End: 0.01
```

---

## Debugging Memory

### Check Memory Utilization

```python
# In agent forward pass
memory_norm = torch.norm(memory, dim=-1).mean()
memory_saturation = (memory.abs() > 0.1).float().mean()

print(f"Memory norm: {memory_norm:.3f}")
print(f"Memory saturation: {memory_saturation:.1%}")
```

**Healthy values**:
- Norm: 0.5-2.0
- Saturation: 20-80%

### Visualize Attention

```python
import matplotlib.pyplot as plt

# Get attention weights during query phase
attention_weights = agent.get_attention_weights(obs, state)

plt.imshow(attention_weights, cmap='hot')
plt.xlabel('Memory Slots')
plt.ylabel('Read Heads')
plt.colorbar()
plt.show()
```

### Track Hidden State

```python
# Collect hidden states over episode
hidden_states = []
for t in range(episode_length):
    _, _, _, state, _ = agent(obs, state)
    hidden_states.append(state['hidden'].cpu().numpy())

# Plot hidden state evolution
plt.plot(hidden_states)
plt.xlabel('Timestep')
plt.ylabel('Hidden State Activation')
plt.show()
```

---

## Comparison with Other Architectures

### vs Transformer

| Feature | GRU+PMM | Transformer |
|---------|---------|-------------|
| Memory | Explicit | Implicit (attention) |
| Complexity | O(n) | O(n²) |
| Long sequences | Good | Excellent |
| Training speed | Fast | Slow |
| Parameters | 1.5M | 10M+ |

### vs LSTM

| Feature | GRU | LSTM |
|---------|-----|------|
| Gates | 2 (reset, update) | 3 (input, forget, output) |
| Parameters | Fewer | More |
| Speed | Faster | Slower |
| Performance | Similar | Similar |

### vs Mamba

| Feature | GRU+PMM | Mamba |
|---------|---------|-------|
| Architecture | RNN + Memory | SSM |
| Complexity | O(n) | O(n) |
| Long sequences | Good | Excellent |
| Hardware | CPU/GPU | GPU only |
| Maturity | Stable | Experimental |

---

## Performance Optimization

### CPU Optimization

```python
# Use smaller batch size
num_envs = 4

# Reduce model size
hidden_dim = 128
num_gru_layers = 1

# Fewer update epochs
update_epochs = 4
```

### GPU Optimization

```python
# Larger batch size
num_envs = 32

# Enable AMP (automatic mixed precision)
use_amp = True

# Larger model
hidden_dim = 512
num_gru_layers = 3
```

### Memory Optimization

```python
# Gradient checkpointing
torch.utils.checkpoint.checkpoint(gru_forward, obs)

# Reduce sequence length
horizon = 10  # instead of 50

# Clear cache periodically
torch.cuda.empty_cache()
```

---

## Research Directions

### 1. Sparse Memory

Use sparse attention to scale to 1000+ memory slots:

```python
# Top-k attention
top_k_indices = torch.topk(scores, k=10).indices
sparse_attention = torch.zeros_like(scores)
sparse_attention.scatter_(1, top_k_indices, 1.0)
```

### 2. Hierarchical Memory

Multiple memory levels (short/medium/long term):

```python
short_term_memory = PMM(slots=16, dim=64)   # Fast access
long_term_memory = PMM(slots=256, dim=128)  # Large capacity
```

### 3. Meta-Learning

Learn to learn memory strategies:

```python
# MAML-style meta-learning
for task in task_distribution:
    fast_adapt(agent, task, steps=5)
    meta_update(agent, task)
```

---

## References

- [GRU Paper](https://arxiv.org/abs/1406.1078)
- [Neural Turing Machines](https://arxiv.org/abs/1410.5401)
- [Differentiable Neural Computer](https://www.nature.com/articles/nature20101)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

---

**See also**: `docs/HOW_IT_WORKS.md` for training guide

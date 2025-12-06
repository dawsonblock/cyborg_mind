# How CyborgMind Memory Benchmarks Work

## Overview

CyborgMind trains reinforcement learning agents to solve **memory tasks** - challenges that require remembering information over long time horizons. This document explains the complete training pipeline.

---

## The Delayed-Cue Memory Task

### Task Structure

The agent must remember a cue shown at the beginning of an episode, wait through a delay period, then recall it:

```
Step 0:     CUE PHASE    → Show one-hot cue [0,1,0,0] (cue #1)
Step 1-N:   DELAY PHASE  → Show zeros [0,0,0,0] (N = horizon)
Step N+1:   QUERY PHASE  → Agent must output action matching cue #1
```

### Rewards

- **Correct recall**: +10.0 reward
- **Wrong recall**: -1.0 penalty
- **All other steps**: 0.0

### Why It's Hard

The agent must:
1. Encode the cue in step 0
2. Maintain it in memory through N steps of neutral input
3. Retrieve and act on it at step N+1

This tests **long-term memory capacity** of the neural architecture.

---

## System Architecture

### 1. Environment (`DelayedCueEnv`)

**Location**: `cyborg_rl/memory_benchmarks/delayed_cue_env.py`

**Key Features**:
- Gymnasium-compatible environment
- Vectorized for parallel training (8-16 envs)
- Continuous action space (argmax'd to discrete choice)
- Configurable horizon length

**Episode Flow**:
```python
obs, info = env.reset()           # Returns one-hot cue
for step in range(horizon + 2):
    action = agent.act(obs)        # Agent decides action
    obs, reward, done, _, info = env.step(action)
```

### 2. Agent (`PPOAgent`)

**Location**: `cyborg_rl/agents/ppo_agent.py`

**Architecture**:
```
Input (obs_dim=4)
    ↓
GRU Encoder (hidden_dim=256, layers=2)
    ↓
Predictive Memory Module (64 slots × 128 dim)
    ↓
Policy Head → Action (4 continuous values)
Value Head  → State value estimate
```

**Key Components**:

- **GRU Encoder**: Recurrent network that maintains hidden state across timesteps
- **PMM (Predictive Memory Module)**: Differentiable memory with read/write heads
- **Policy Head**: Outputs action distribution (Gaussian for continuous actions)
- **Value Head**: Estimates expected future reward

**Parameters**: ~1.5M trainable parameters

### 3. Trainer (`MemoryPPOTrainer`)

**Location**: `cyborg_rl/trainers/memory_ppo_trainer.py`

**Training Algorithm**: Proximal Policy Optimization (PPO)

**Key Features**:
- Full-sequence BPTT (backpropagation through time)
- Vectorized rollout collection
- Multi-epoch policy updates
- Gaussian distribution for continuous actions

**Training Loop**:
```python
while steps < total_timesteps:
    # 1. Collect rollout (full episode × num_envs)
    rollout = collect_rollout()
    
    # 2. Compute advantages using GAE
    advantages = compute_gae(rollout.rewards, rollout.values)
    
    # 3. Update policy for N epochs
    for epoch in range(10):
        loss = ppo_loss(rollout, advantages)
        optimizer.step()
```

---

## Training Process

### Step-by-Step Execution

**Command**:
```bash
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
    --task delayed_cue \
    --backbone gru \
    --horizon 20 \
    --num-envs 8 \
    --total-timesteps 50000 \
    --device cpu
```

**What Happens**:

1. **Initialization** (2 seconds)
   - Create 8 parallel environments
   - Initialize agent with random weights
   - Set up optimizer (AdamW, lr=1e-3)

2. **Rollout Collection** (~30 seconds per rollout)
   - Run 8 environments in parallel for 22 steps each (horizon + 2)
   - Agent maintains hidden state across timesteps
   - Store: observations, actions, rewards, values, log_probs

3. **Advantage Computation**
   - Use Generalized Advantage Estimation (GAE)
   - Compute returns and advantages for each timestep
   - Normalize advantages for stable training

4. **Policy Update** (10 epochs per rollout)
   - Forward pass through entire sequence
   - Compute PPO loss (clipped surrogate objective)
   - Backpropagate through time
   - Clip gradients (max_norm=1.0)
   - Update weights

5. **Repeat** until 50,000 timesteps

6. **Evaluation** (64 episodes)
   - Run agent deterministically
   - Measure success rate (% correct recalls)
   - Compute mean reward

---

## Key Hyperparameters

### Model Architecture
```python
hidden_dim = 256        # GRU hidden size
latent_dim = 256        # Latent representation size
num_gru_layers = 2      # Stacked GRU layers
memory_size = 64        # PMM memory slots
memory_dim = 128        # Memory slot dimension
```

### Training
```python
learning_rate = 1e-3    # Optimizer learning rate
num_envs = 8            # Parallel environments
update_epochs = 10      # PPO epochs per rollout
clip_range = 0.2        # PPO clipping epsilon
entropy_coef = 0.05     # Exploration bonus
value_coef = 0.5        # Value loss weight
max_grad_norm = 1.0     # Gradient clipping
```

### Task
```python
horizon = 20            # Delay length
num_cues = 4            # Number of possible cues
reward_correct = 10.0   # Reward for correct recall
reward_wrong = -1.0     # Penalty for wrong recall
```

---

## How Learning Happens

### Phase 1: Random Exploration (0-5k steps)
- Agent outputs random actions
- Success rate: ~25% (random chance for 4 cues)
- Mean reward: ~0-2

### Phase 2: Pattern Discovery (5k-20k steps)
- Agent learns to encode cue in GRU hidden state
- Begins to maintain information through delay
- Success rate: 30-40%
- Mean reward: 2-4

### Phase 3: Memory Consolidation (20k-50k steps)
- GRU learns robust encoding strategy
- PMM stores cue representation
- Policy learns to decode memory at query time
- Success rate: 50-70%
- Mean reward: 4-7

### Phase 4: Mastery (50k-100k steps)
- Near-perfect recall on short horizons
- Success rate: 70-90%
- Mean reward: 7-9

---

## Why It Works

### 1. Recurrent Architecture
GRU maintains hidden state across timesteps, allowing information to persist through the delay period.

### 2. Predictive Memory Module
Differentiable memory provides additional capacity for storing cue information beyond GRU hidden state.

### 3. Full-Sequence BPTT
Gradients flow through entire episode, allowing the network to learn long-term dependencies.

### 4. Strong Reward Signal
+10 for correct, -1 for wrong creates clear learning signal.

### 5. Vectorized Training
8-16 parallel environments provide diverse experiences and faster learning.

---

## Performance Metrics

### Success Rate
Percentage of episodes where agent correctly recalls the cue:
- **Random**: 25% (1/4 chance)
- **Learning**: 30-50%
- **Trained**: 60-80%
- **Expert**: 80-95%

### Mean Reward
Average reward per episode:
- **Random**: 0-2
- **Learning**: 2-5
- **Trained**: 5-8
- **Expert**: 8-9.5

### Training Speed
- **CPU**: 60-100 steps/second
- **GPU**: 200-500 steps/second

---

## Troubleshooting

### Low Success Rate (<30%)

**Causes**:
- Not enough training steps
- Learning rate too high/low
- Model too small

**Solutions**:
```bash
# Train longer
--total-timesteps 100000

# Increase model size
hidden_dim = 256, num_gru_layers = 2

# Tune learning rate
learning_rate = 1e-3
```

### Training Instability

**Causes**:
- Gradient explosion
- Reward scale too large

**Solutions**:
```python
max_grad_norm = 1.0      # Clip gradients
reward_correct = 10.0    # Moderate reward
```

### Slow Training

**Causes**:
- Too many environments
- Model too large
- CPU bottleneck

**Solutions**:
```bash
# Reduce environments
--num-envs 4

# Use GPU
--device cuda

# Smaller model
hidden_dim = 128
```

---

## Advanced Topics

### Horizon Scaling

Longer horizons require:
- More training steps (2-3× per 10 steps of horizon)
- Larger memory capacity
- Better exploration

**Example**:
```bash
# Horizon 10: 30k steps → 60% success
# Horizon 20: 50k steps → 50% success
# Horizon 50: 100k steps → 40% success
```

### Backbone Comparison

**GRU**:
- Fast, stable
- Good for horizons <50
- 1.5M parameters

**Pseudo-Mamba**:
- Pure PyTorch implementation
- Better for very long horizons
- 2M parameters

**Mamba-SSM** (requires CUDA):
- State-of-the-art performance
- Best for horizons >100
- Requires GPU

---

## Code Flow Diagram

```
main()
  ↓
run_single_experiment()
  ↓
  ├─→ Create Environment (VectorizedDelayedCueEnv)
  ├─→ Create Agent (PPOAgent with GRU + PMM)
  ├─→ Create Trainer (MemoryPPOTrainer)
  ↓
trainer.train()
  ↓
  Loop until total_timesteps:
    ├─→ collect_rollout()
    │     ├─→ Reset envs
    │     ├─→ For each timestep:
    │     │     ├─→ agent.forward(obs, state)
    │     │     ├─→ env.step(action)
    │     │     └─→ Store transition
    │     └─→ compute_gae()
    │
    └─→ update_policy()
          ├─→ For each epoch:
          │     ├─→ agent.forward_sequence()
          │     ├─→ Compute PPO loss
          │     ├─→ Backprop + gradient clip
          │     └─→ optimizer.step()
          └─→ Log metrics
  ↓
evaluate_memory_task()
  ↓
  ├─→ Run 64 deterministic episodes
  ├─→ Compute success rate
  └─→ Return summary
```

---

## Files Reference

### Core Components
- `cyborg_rl/memory_benchmarks/delayed_cue_env.py` - Environment
- `cyborg_rl/agents/ppo_agent.py` - Agent architecture
- `cyborg_rl/trainers/memory_ppo_trainer.py` - Training loop
- `cyborg_rl/models/mamba_gru.py` - GRU encoder
- `cyborg_rl/memory/pmm.py` - Predictive Memory Module

### Entry Point
- `cyborg_rl/memory_benchmarks/pseudo_mamba_memory_suite.py` - Main script

### Configuration
- `cyborg_rl/config.py` - Config dataclasses

---

## Next Steps

1. **Experiment with horizons**: Try 10, 20, 50, 100
2. **Compare backbones**: Test GRU vs Pseudo-Mamba
3. **Tune hyperparameters**: Learning rate, model size
4. **Scale up**: Use GPU, more environments
5. **Analyze results**: Plot learning curves, success rates

---

## References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **GAE Paper**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **Mamba Paper**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

---

**Questions?** Check `docs/HOW_TO_TRAIN.md` or run `python scripts/verify_setup.py`

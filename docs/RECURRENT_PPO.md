# Recurrent PPO Mode

## Overview

Cyborg Mind supports "honest" recurrent PPO that properly handles RNN/PMM/Mamba states during both rollout collection and policy updates. This ensures gradients flow correctly through memory components for long-horizon tasks.

## Problem Solved

Standard PPO implementations typically ignore recurrent state during policy updates, either:
- Re-initializing states to zeros
- Passing `None` for state parameters

This causes **incorrect gradients** for memory-based agents, as the update doesn't reflect the actual state that was used during rollout collection.

## Solution: Burn-In Recurrent Mode

The `recurrent_mode="burn_in"` setting enables:
1. **State Storage**: Stores the exact recurrent state (hidden, PMM memory, etc.) for each timestep during rollout
2. **State Retrieval**: During PPO update, retrieves and uses the stored states when evaluating actions
3. **Honest Gradients**: Backpropagation flows through the actual state that was present during action selection

## Configuration

Enable in your config YAML:

```yaml
train:
  recurrent_mode: "burn_in"  # Options: "none" (default), "burn_in"
```

Example: `configs/memory_delayed_cue_recurrent.yaml`

## Implementation Details

### Buffer Management

- **Non-Recurrent Mode**: `RolloutBuffer` stores obs, actions, rewards, values, log_probs, dones
- **Recurrent Mode**: `RecurrentRolloutBuffer` additionally stores `recurrent_state` per timestep

### Vectorized Environment Support

The trainer automatically detects vectorized environments and:
- Calculates buffer size as `n_steps * num_envs`
- Flattens data from `[num_envs, ...]` to individual transitions
- Extracts per-env states from batched states using `_extract_env_state()`

### State Structure

States are dictionaries containing:
```python
{
    "hidden": torch.Tensor,  # GRU/RNN hidden state [num_layers, batch_size, hidden_dim]
    "memory": torch.Tensor,  # PMM memory [batch_size, memory_size, memory_dim]
    # Future: "mamba_state" for streaming Mamba state
}
```

### Key Methods

**PPOTrainer:**
- `_clone_state(state)`: Deep clones state dict for storage
- `_extract_env_state(state, env_idx)`: Extracts single env's state from batched state
- `_gather_recurrent_states(indices)`: Reconstructs batched state from stored per-step states

**PPOAgent:**
- `evaluate_actions(obs, actions, state=None)`: Evaluates actions with optional state parameter
- State=None falls back to fresh initialization (non-recurrent mode)

## Usage

### Memory Benchmarks

```bash
python -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
  --task delayed_cue \
  --backbone mamba_gru \
  --horizon 1000 \
  --num-envs 64 \
  --total-timesteps 200000 \
  --device cuda
```

With the `memory_delayed_cue_recurrent.yaml` config, this will use recurrent PPO automatically.

### MineRL Treechop

For 8k-step episodes with honest memory gradients:

```yaml
# configs/treechop_ppo_recurrent.yaml
train:
  recurrent_mode: "burn_in"
  n_steps: 2048  # Rollout length
  # ... other PPO params
```

### Verification

Test that recurrent PPO is working correctly:

```bash
python scripts/verify_recurrent_ppo.py
```

This runs 4 tests:
1. Buffer size calculation for vectorized envs
2. State storage and retrieval
3. Full training loop execution
4. Gradient flow verification

## Performance Considerations

### Memory Overhead

Recurrent mode stores states per timestep:
- **Non-Recurrent**: `buffer_size * (obs_dim + action_dim + 5 scalars)`
- **Recurrent**: Above + `buffer_size * state_size`

For typical configs:
- GRU: ~1-2 MB per 100K timesteps
- Mamba+GRU: ~2-4 MB per 100K timesteps

### Computational Overhead

- **Rollout**: ~5-10% slower (state cloning)
- **Update**: ~10-20% slower (state gathering and batching)

This is acceptable for the correctness gain on memory tasks.

## Backward Compatibility

- Default `recurrent_mode="none"` preserves existing behavior
- Existing configs work without modification
- Non-recurrent and recurrent agents can coexist

## Future Enhancements

1. **Non-Zero Burn-In**: Replay K steps before each sampled timestep for better context
2. **Mamba Streaming State**: Expose and cache Mamba SSM state explicitly
3. **Episode-Based Batching**: Batch by full episodes rather than shuffled timesteps
4. **Truncated BPTT**: Implement proper sequence-based backprop through time

## References

- Original issue: PPO update ignoring recurrent state (commit 5eb037b)
- Vectorized env fixes: Proper flattening for multi-env rollouts
- Memory benchmark suite: `cyborg_rl/memory_benchmarks/pseudo_mamba_memory_suite.py`

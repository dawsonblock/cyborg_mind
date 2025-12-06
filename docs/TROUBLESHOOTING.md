# Troubleshooting Guide

## Installation Issues

### "No module named 'cyborg_rl'"

**Cause**: Package not installed

**Solution**:
```bash
cd /path/to/cyborg_mind
pip install -e .
```

### "No module named 'minigrid'"

**Cause**: Missing dependency

**Solution**:
```bash
pip install minigrid==2.3.1
```

### "No module named 'slowapi'"

**Cause**: Missing API dependency

**Solution**:
```bash
pip install slowapi==0.1.9
```

### "mamba-ssm not installed"

**Cause**: Optional dependency missing (GPU only)

**Solution**:
```bash
# If you have CUDA GPU
./setup_mamba_gpu.sh

# Otherwise, use GRU backbone
--backbone gru
```

---

## Training Issues

### Low Success Rate (<30%)

**Symptoms**:
```
success_rate: 0.25
mean_reward: 0.5
```

**Causes & Solutions**:

1. **Not enough training**
   ```bash
   # Increase timesteps
   --total-timesteps 100000  # instead of 10000
   ```

2. **Model too small**
   ```python
   # Edit pseudo_mamba_memory_suite.py
   hidden_dim = 256  # instead of 128
   num_gru_layers = 2  # instead of 1
   ```

3. **Learning rate wrong**
   ```python
   learning_rate = 1e-3  # try 1e-4 or 3e-3
   ```

4. **Horizon too long**
   ```bash
   # Start with shorter horizon
   --horizon 10  # instead of 50
   ```

### Training Crashes

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```bash
# Reduce batch size
--num-envs 4  # instead of 16

# Use CPU
--device cpu

# Reduce model size
hidden_dim = 128
memory_size = 32
```

### NaN Loss

**Symptoms**:
```
[ERROR] NaN loss detected!
policy_loss=nan
```

**Solutions**:
```python
# Reduce learning rate
learning_rate = 1e-4

# Increase gradient clipping
max_grad_norm = 0.5

# Reduce reward scale
reward_correct = 5.0  # instead of 10.0
```

### Slow Training

**Symptoms**:
- <50 steps/second on CPU
- <200 steps/second on GPU

**Solutions**:

1. **Reduce environments**
   ```bash
   --num-envs 4  # instead of 16
   ```

2. **Reduce model size**
   ```python
   hidden_dim = 128
   num_gru_layers = 1
   ```

3. **Use GPU**
   ```bash
   --device cuda
   ```

4. **Reduce update epochs**
   ```python
   update_epochs = 4  # instead of 10
   ```

---

## Runtime Errors

### "Shape mismatch"

**Error**:
```
RuntimeError: The expanded size of the tensor (8) must match 
the existing size (4) at non-singleton dimension 1
```

**Cause**: Observation dimension mismatch

**Solution**:
```python
# In _make_env(), ensure obs_dim matches num_cues
VectorizedDelayedCueEnv(
    num_envs=num_envs,
    num_cues=4,
    obs_dim=4,  # Must match num_cues
    horizon=horizon,
)
```

### "Can only convert array of size 1"

**Error**:
```
ValueError: can only convert an array of size 1 to a Python scalar
```

**Cause**: Action handling issue

**Solution**: Already fixed in `delayed_cue_env.py`:
```python
action_arr = np.asarray(action)
action_val = int(np.argmax(action_arr))
```

### "Attribute 'save_manifest' not found"

**Error**:
```
AttributeError: 'ExperimentRegistry' object has no attribute 'save_manifest'
```

**Solution**: Remove the call (already fixed):
```python
# Remove this line
# registry.save_manifest()
```

---

## Performance Issues

### Memory Usage Too High

**Symptoms**:
- System RAM >8GB
- Swap usage increasing

**Solutions**:
```bash
# Reduce environments
--num-envs 4

# Reduce episode length
--horizon 10

# Smaller model
hidden_dim = 128
memory_size = 32
```

### GPU Not Being Used

**Symptoms**:
```
nvidia-smi  # Shows 0% GPU usage
```

**Solutions**:
```bash
# Ensure CUDA available
python -c "import torch; print(torch.cuda.is_available())"

# Explicitly set device
--device cuda

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

### Training Not Improving

**Symptoms**:
- Success rate stuck at 25%
- No improvement after 50k steps

**Solutions**:

1. **Check reward signal**
   ```python
   # Verify rewards are being given
   print(f"Mean reward: {rollout.rewards.mean()}")
   ```

2. **Increase exploration**
   ```python
   entropy_coef = 0.1  # Higher exploration
   ```

3. **Check gradient flow**
   ```python
   # Monitor gradient norms
   print(f"Grad norm: {grad_norm:.3f}")
   # Should be 0.1-2.0
   ```

4. **Verify environment**
   ```python
   # Test environment manually
   env = DelayedCueEnv(num_cues=4, horizon=10)
   obs, _ = env.reset()
   print(f"Cue: {obs}")  # Should be one-hot
   ```

---

## API Server Issues

### "Port already in use"

**Error**:
```
OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
--port 8001
```

### "Agent not loaded"

**Error**:
```
AttributeError: 'NoneType' object has no attribute 'forward'
```

**Solution**:
```bash
# Ensure checkpoint exists
ls artifacts/minerl_treechop/run_v1/best_model.pt

# Check config path
--config configs/treechop_ppo.yaml
```

### "Authentication failed"

**Error**:
```
{"detail": "Invalid authentication token"}
```

**Solution**:
```bash
# Use correct token
curl -H "Authorization: Bearer cyborg-secret-v2" ...

# Or generate JWT
curl -X POST http://localhost:8000/auth/token \
  -d '{"subject": "user1"}'
```

---

## Environment Issues

### "Episode never ends"

**Symptoms**:
- Training hangs
- Episode length >1000 steps

**Solution**:
```python
# Check termination condition in env
terminated = self.step_count >= self.total_steps
```

### "Rewards always zero"

**Symptoms**:
```
mean_reward: 0.0
success_rate: 0.0
```

**Solution**:
```python
# Verify reward logic
if self.step_count == self.horizon + 1:
    if action_val == self.current_cue:
        reward = 10.0  # Should be positive
```

### "Success rate always 0%"

**Symptoms**:
- Training improves but success_rate stays 0%

**Solution**:
```python
# Check success flag in info dict
info = {
    "success": success,  # Must be set
}
```

---

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Tensor Shapes

```python
print(f"Obs shape: {obs.shape}")
print(f"Action shape: {action.shape}")
print(f"Hidden shape: {state['hidden'].shape}")
```

### Monitor Memory

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Profile Performance

```python
import cProfile
cProfile.run('trainer.train()', 'profile.stats')

# View results
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

### Visualize Training

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load metrics
df = pd.read_csv('experiments/runs/my_run/logs/metrics.csv')

# Plot learning curve
plt.plot(df['step'], df['mean_reward'])
plt.xlabel('Steps')
plt.ylabel('Mean Reward')
plt.show()
```

---

## Common Error Messages

### "Expected observation of length X, got Y"

**Fix**: Ensure `obs_dim` matches `num_cues`

### "Tensor sizes must match"

**Fix**: Check batch dimensions in forward pass

### "CUDA error: out of memory"

**Fix**: Reduce `num_envs` or use CPU

### "No such file or directory"

**Fix**: Check paths are absolute or relative to correct directory

### "Module not found"

**Fix**: Run `pip install -e .` from project root

---

## Getting Help

### Check Documentation

1. `docs/HOW_IT_WORKS.md` - System overview
2. `docs/MEMORY_ARCHITECTURE.md` - Technical details
3. `docs/QUICK_START_GUIDE.md` - Basic usage

### Verify Setup

```bash
python scripts/verify_setup.py
```

### Check Build

```bash
python scripts/build_verify.py
```

### Run Tests

```bash
pytest tests/ -v
```

### Enable Debug Mode

```bash
export CYBORG_DEBUG=1
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite ...
```

---

## Still Having Issues?

1. **Check Python version**: `python --version` (need 3.10+)
2. **Check PyTorch version**: `pip show torch` (need 2.0+)
3. **Check CUDA version**: `nvidia-smi` (if using GPU)
4. **Clear cache**: `rm -rf __pycache__ *.pyc`
5. **Reinstall**: `pip uninstall cyborg_rl && pip install -e .`

---

**Last updated**: 2025-12-05

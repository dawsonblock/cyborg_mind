# Training Diagnosis & Fix

## Problem

Training shows **0% success rate** throughout, even though mean reward increases. Final evaluation shows **21.8% success** (worse than 25% random baseline).

## Root Cause Analysis

### Issue 1: Success Tracking During Training

Looking at metrics.csv:
```
mean_reward,success_rate,grad_norm,step
3.8125,0.0,2.374,1632
1.0625,0.0,0.445,3264
```

**Success rate is always 0.0** - the training loop isn't tracking successes correctly.

### Issue 2: Reward Signal

Rewards are positive (1-5 range) but success isn't being detected. This means:
- Agent IS getting rewards
- But success flag isn't being set properly
- Or success tracking is broken in trainer

### Issue 3: Episode Length Mismatch

```python
episode_len=102  # For horizon=10
# Should be: horizon + 2 = 12
```

The episode is running 102 steps instead of 12! This means:
- Agent sees 100 delay steps instead of 10
- Query phase happens at wrong time
- Memory task becomes much harder

## The Fix

### 1. Fix Episode Length Calculation

The trainer is using wrong horizon value. Need to pass horizon from env to trainer.

### 2. Fix Success Tracking

Success info from environment isn't being collected properly during vectorized rollouts.

### 3. Verify Reward Timing

Ensure rewards are given at correct timestep (query phase).

## Implementation


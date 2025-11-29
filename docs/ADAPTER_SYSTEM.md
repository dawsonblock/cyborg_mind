# Environment Adapter System

## Overview

The CyborgMind V2 adapter system provides a universal interface for integrating any RL environment with the brain. This allows the same brain architecture to work with MineRL, OpenAI Gym, custom simulators, and more.

---

## Design Philosophy

### 1. Environment Agnostic
The brain only sees three inputs regardless of the environment:
- **Pixels**: [3, H, W] RGB images
- **Scalars**: [scalar_dim] numeric features
- **Goal**: [goal_dim] task directive

### 2. Action Abstraction
The brain outputs discrete action indices. Each adapter maps these to environment-specific actions.

### 3. Type Safety
Uses Python `Protocol` for compile-time type checking.

### 4. Extensibility
Adding a new environment requires implementing a single interface.

---

## BrainEnvAdapter Protocol

All adapters must implement this protocol:

```python
from typing import Protocol, Tuple, Dict, Any
from cyborg_mind_v2.envs import BrainInputs

class BrainEnvAdapter(Protocol):
    @property
    def action_space_size(self) -> int:
        """Number of discrete actions."""
        ...

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """Pixel observation shape (C, H, W)."""
        ...

    @property
    def scalar_dim(self) -> int:
        """Dimensionality of scalar features."""
        ...

    @property
    def goal_dim(self) -> int:
        """Dimensionality of goal vector."""
        ...

    def reset(self) -> BrainInputs:
        """Reset environment, return initial observation."""
        ...

    def step(self, action_idx: int) -> Tuple[BrainInputs, float, bool, Dict[str, Any]]:
        """Execute action, return (obs, reward, done, info)."""
        ...

    def render(self, mode: str = "human") -> Any:
        """Optional: render for visualization."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
```

---

## BrainInputs Dataclass

Standard observation format:

```python
@dataclass
class BrainInputs:
    pixels: torch.Tensor  # [3, H, W] RGB image
    scalars: torch.Tensor  # [scalar_dim] numeric features
    goal: torch.Tensor    # [goal_dim] task objective
```

---

## Creating a New Adapter

### Step 1: Inherit from BaseEnvAdapter

```python
from cyborg_mind_v2.envs import BaseEnvAdapter, BrainInputs

class MyCustomAdapter(BaseEnvAdapter):
    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name, image_size=(128, 128), device="cuda")

        # Create your environment
        self.env = create_my_environment(env_name)

        # Define action space size
        self._action_space_size = 10
```

### Step 2: Implement Required Properties

```python
@property
def action_space_size(self) -> int:
    return self._action_space_size

@property
def scalar_dim(self) -> int:
    return 20  # Your scalar feature count

@property
def goal_dim(self) -> int:
    return 4  # Your goal vector size
```

### Step 3: Implement Observation Conversion

```python
def _obs_to_brain_inputs(self, obs: Any) -> BrainInputs:
    # Extract pixels from your observation
    pixels_img = extract_image(obs)
    pixels = self.preprocess_pixels(pixels_img)  # BaseEnvAdapter utility

    # Extract scalars
    scalars_array = extract_features(obs)
    scalars = self.normalize_scalars(scalars_array)  # BaseEnvAdapter utility

    # Create goal
    goal = torch.tensor([1, 0, 0, 0]).float().to(self.device)

    return BrainInputs(pixels=pixels, scalars=scalars, goal=goal)
```

### Step 4: Implement Action Mapping

```python
def _action_idx_to_env_action(self, action_idx: int) -> Any:
    """Map brain action index to environment action."""
    action_map = {
        0: "noop",
        1: "forward",
        2: "backward",
        # ... more actions
    }
    return action_map[action_idx]
```

### Step 5: Implement reset() and step()

```python
def reset(self) -> BrainInputs:
    self.reset_episode_stats()  # BaseEnvAdapter utility
    obs = self.env.reset()
    return self._obs_to_brain_inputs(obs)

def step(self, action_idx: int) -> Tuple[BrainInputs, float, bool, Dict[str, Any]]:
    env_action = self._action_idx_to_env_action(action_idx)
    obs, reward, done, info = self.env.step(env_action)

    self.update_episode_stats(reward)  # BaseEnvAdapter utility

    brain_obs = self._obs_to_brain_inputs(obs)
    info.update(self.episode_stats)  # Add episode statistics

    return brain_obs, float(reward), bool(done), info
```

---

## Example: MineRL Adapter

### Action Mapping

The MineRL adapter defines 19 discrete actions combining movement, camera, and attack:

```python
ACTION_MAP = [
    {"camera": [0, 0], "forward": 0, "attack": 0},  # No-op
    {"camera": [0, 0], "forward": 1, "attack": 0},  # Forward
    {"camera": [0, 0], "forward": 1, "attack": 1},  # Forward + Attack
    # ... 16 more actions
]
```

### Scalar Features

```python
def _extract_scalars(self, obs: Dict[str, Any]) -> np.ndarray:
    scalars = []

    # Inventory counts
    inventory = obs.get("inventory", {})
    for item in ["log", "planks", "stick"]:
        scalars.append(float(inventory.get(item, 0)))

    # Compass angle (normalized)
    compass = obs.get("compassAngle", 0.0)
    scalars.append(float(compass) / 180.0)

    # Step ratio
    step_ratio = self._episode_step / self.max_steps
    scalars.append(float(step_ratio))

    return np.array(scalars, dtype=np.float32)
```

---

## Example: Gym Adapter

### Handling State vs Pixels

The Gym adapter handles both:

```python
def __init__(self, env_name: str, use_pixels: bool = False, **kwargs):
    super().__init__(env_name, **kwargs)

    self.env = gym.make(env_name)
    self.use_pixels = use_pixels

    # Detect if environment has pixel observations
    obs_space = self.env.observation_space
    if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
        self.is_pixel_env = True
        self.use_pixels = True
    else:
        self.is_pixel_env = False
```

### Discretizing Continuous Actions

```python
def _discretize_action_space(self, action_space: gym.spaces.Box) -> int:
    dim = action_space.shape[0]

    if dim == 1:
        # Map to [-1, 0, 1]
        return 3
    elif dim == 2:
        # Map to 9 directions (8-way + no-op)
        return 9
    else:
        return 3 ** min(dim, 3)  # Limit combinatorial explosion
```

---

## Factory Function

Use the factory for easy instantiation:

```python
from cyborg_mind_v2.envs import create_adapter

# MineRL
adapter = create_adapter("minerl", "MineRLTreechop-v0")

# Gym
adapter = create_adapter("gym", "CartPole-v1")

# Custom (after registering)
adapter = create_adapter("myenv", "MyTask-v0")
```

---

## Best Practices

### 1. Normalize Inputs
- **Pixels**: Scale to [0, 1]
- **Scalars**: Use z-score or min-max normalization
- **Goals**: Keep in [-1, 1] or [0, 1] range

### 2. Handle Edge Cases
- Missing observations → use defaults
- Invalid actions → clamp to valid range
- Episode timeouts → set `done=True` and `info["timeout"]=True`

### 3. Provide Metadata
Return useful info in the `info` dict:
```python
info = {
    "episode_step": self._episode_step,
    "episode_reward": self._episode_reward,
    "timeout": done and self._episode_step >= self.max_steps,
    # Environment-specific metrics
    "wood_collected": inventory.get("log", 0),
}
```

### 4. Efficient Preprocessing
- Cache transforms when possible
- Use vectorized operations
- Avoid unnecessary copies

---

## Testing Your Adapter

```python
# Test basic functionality
adapter = MyCustomAdapter("task-v0")

# Reset
obs = adapter.reset()
assert obs.pixels.shape == (3, 128, 128)
assert obs.scalars.shape[0] == adapter.scalar_dim
assert obs.goal.shape[0] == adapter.goal_dim

# Step
for _ in range(10):
    action = np.random.randint(0, adapter.action_space_size)
    obs, reward, done, info = adapter.step(action)

    if done:
        obs = adapter.reset()

adapter.close()
```

---

## Integration with Brain

Once you have an adapter, use it with the controller:

```python
from cyborg_mind_v2.integration import CyborgMindController

adapter = create_adapter("minerl", "MineRLTreechop-v0")
controller = CyborgMindController()

# Training loop
obs = adapter.reset()
done = False

while not done:
    # Prepare inputs
    pixels = obs.pixels.unsqueeze(0)
    scalars = obs.scalars.unsqueeze(0)
    goal = obs.goal.unsqueeze(0)

    # Get action from brain
    actions = controller.step(["agent_0"], pixels, scalars, goal)
    action_idx = actions[0]

    # Execute in environment
    obs, reward, done, info = adapter.step(action_idx)
```

---

## Advanced: Custom Observation Spaces

For complex observations (e.g., point clouds, graphs):

```python
def _obs_to_brain_inputs(self, obs: Any) -> BrainInputs:
    # Render point cloud to 2D image
    pixels_img = render_point_cloud(obs.points, obs.colors)
    pixels = self.preprocess_pixels(pixels_img)

    # Extract graph statistics as scalars
    scalars = [
        obs.num_nodes,
        obs.num_edges,
        obs.average_degree,
        # ...
    ]

    # Task-specific goal
    goal = create_goal_from_task(obs.task_id)

    return BrainInputs(pixels=pixels, scalars=scalars, goal=goal)
```

---

## Troubleshooting

### Issue: "Observation shape mismatch"
- Ensure pixels are [3, H, W] after preprocessing
- Check that image_size matches brain expectation (default: 128x128)

### Issue: "Action out of bounds"
- Validate action_idx before mapping: `0 <= action_idx < action_space_size`
- Add error handling in `_action_idx_to_env_action()`

### Issue: "Scalar normalization errors"
- Check for NaN/Inf values in raw observations
- Use robust normalization (clip outliers)

---

## Future Extensions

- **Multi-modal inputs**: Add audio, text, etc.
- **Hierarchical actions**: Options/skills on top of primitives
- **Reward shaping**: Built-in shaping in adapters
- **Curriculum learning**: Progressive difficulty in adapters

---

For implementation examples, see:
- `cyborg_mind_v2/envs/minerl_adapter.py`
- `cyborg_mind_v2/envs/gym_adapter.py`
- `cyborg_mind_v2/envs/cc3d_adapter.py`

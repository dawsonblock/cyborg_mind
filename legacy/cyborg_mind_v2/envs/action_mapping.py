"""
Action mapping utilities for MineRL environments.

Maps between discrete action indices and MineRL action dictionaries.
"""

import numpy as np
from typing import Dict, Any


# Total number of discrete actions
# Expanded to 20 to match BrainCyborgMind default
NUM_ACTIONS = 20


def index_to_minerl_action(index: int) -> Dict[str, Any]:
    """
    Convert a discrete action index to a MineRL action dictionary.
    
    Args:
        index: Discrete action index in range [0, NUM_ACTIONS)
        
    Returns:
        MineRL action dictionary with keys: forward, back, left, right,
        jump, attack, sneak, sprint, camera, place, etc.
    """
    # Base action template (all zeros/false)
    action = {
        "forward": 0,
        "back": 0,
        "left": 0,
        "right": 0,
        "jump": 0,
        "attack": 0,
        "sneak": 0,
        "sprint": 0,
        "camera": np.array([0.0, 0.0], dtype=np.float32),
        "place": 0,
    }
    
    # Map index to specific action
    if index == 0:
        # No-op
        pass
    elif index == 1:
        # Forward
        action["forward"] = 1
    elif index == 2:
        # Back
        action["back"] = 1
    elif index == 3:
        # Left
        action["left"] = 1
    elif index == 4:
        # Right
        action["right"] = 1
    elif index == 5:
        # Jump
        action["jump"] = 1
    elif index == 6:
        # Attack
        action["attack"] = 1
    elif index == 7:
        # Look right (camera yaw positive)
        action["camera"] = np.array([0.0, 5.0], dtype=np.float32)
    elif index == 8:
        # Look left (camera yaw negative)
        action["camera"] = np.array([0.0, -5.0], dtype=np.float32)
    elif index == 9:
        # Look up (camera pitch negative)
        action["camera"] = np.array([-5.0, 0.0], dtype=np.float32)
    elif index == 10:
        # Look down (camera pitch positive)
        action["camera"] = np.array([5.0, 0.0], dtype=np.float32)
    elif index == 11:
        # Sprint + forward
        action["sprint"] = 1
        action["forward"] = 1
    elif index == 12:
        # Sneak + forward
        action["sneak"] = 1
        action["forward"] = 1
    elif index == 13:
        # Place block
        action["place"] = 1
    elif index == 14:
        # Attack + forward (mining while moving)
        action["attack"] = 1
        action["forward"] = 1
    elif index == 15:
        # Jump + attack (jumping attack)
        action["jump"] = 1
        action["attack"] = 1
    elif index == 16:
        # Sprint + attack (sprinting attack)
        action["sprint"] = 1
        action["attack"] = 1
    elif index == 17:
        # Look up-right diagonal
        action["camera"] = np.array([-3.0, 3.0], dtype=np.float32)
    elif index == 18:
        # Look down-left diagonal
        action["camera"] = np.array([3.0, -3.0], dtype=np.float32)
    elif index == 19:
        # Crouch (sneak without movement)
        action["sneak"] = 1
    else:
        # Fallback to no-op
        pass
    
    return action


def minerl_action_to_index(action: Dict[str, Any]) -> int:
    """
    Convert a MineRL action dictionary to a discrete action index.
    
    This is a heuristic mapping for behavioral cloning - it may not
    perfectly invert index_to_minerl_action but should capture the
    main action categories.
    
    Args:
        action: MineRL action dictionary
        
    Returns:
        Discrete action index in range [0, NUM_ACTIONS)
    """
    forward = int(action.get("forward", 0))
    back = int(action.get("back", 0))
    left = int(action.get("left", 0))
    right = int(action.get("right", 0))
    jump = int(action.get("jump", 0))
    attack = int(action.get("attack", 0))
    sneak = int(action.get("sneak", 0))
    sprint = int(action.get("sprint", 0))
    camera = np.array(
        action.get("camera", np.array([0.0, 0.0], dtype=np.float32)),
        dtype=np.float32
    )
    place = int(action.get("place", 0))
    
    # Priority order for classification (check combos first!)
    
    # Sprint + attack combo
    if sprint and attack and not forward:
        return 16
    
    # Sprint + forward (may include attack)
    if sprint and forward and not attack:
        return 11
    
    # Jump + attack combo
    if jump and attack:
        return 15
    
    # Attack + forward (mining while moving)
    if attack and forward and not jump and not sprint:
        return 14
    
    # Sneak + forward
    if sneak and forward:
        return 12
    
    # Sneak without movement (crouch)
    if sneak and not forward:
        return 19
    
    # Place block
    if place:
        return 13
    
    # Basic movement
    if forward and not any([back, left, right, jump, attack]):
        return 1
    if back and not any([forward, left, right, jump, attack]):
        return 2
    if left and not any([forward, back, right, jump, attack]):
        return 3
    if right and not any([forward, back, left, jump, attack]):
        return 4
    
    # Jump (without attack)
    if jump and not attack:
        return 5
    
    # Attack (basic, no other modifiers)
    if attack and not any([forward, jump, sprint]):
        return 6
    
    # Camera movements (check thresholds and diagonals)
    pitch, yaw = camera
    # Check for diagonal movements first
    if abs(pitch) > 2.0 and abs(yaw) > 2.0:
        if pitch < 0 and yaw > 0:
            return 17  # Up-right diagonal
        elif pitch > 0 and yaw < 0:
            return 18  # Down-left diagonal
    
    # Single-axis camera movements
    if abs(yaw) > abs(pitch):
        if yaw > 2.0:
            return 7  # Look right
        if yaw < -2.0:
            return 8  # Look left
    else:
        if pitch < -2.0:
            return 9  # Look up
        if pitch > 2.0:
            return 10  # Look down
    
    # Default to no-op
    return 0

import hydra
from omegaconf import DictConfig
import torch
import logging
import numpy as np
from pathlib import Path

from experiments.cyborg_mind_v2.configs.schema import Config
from experiments.cyborg_mind_v2.envs.gym_adapter import GymAdapter
from experiments.cyborg_mind_v2.envs.minerl_adapter import MineRLAdapter
from experiments.cyborg_mind_v2.capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind
from experiments.cyborg_mind_v2.training.trainer import CyborgTrainer

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="cyborg_mind_v2/configs", config_name="config")
def main(cfg: Config):
    logging.basicConfig(level=logging.INFO)
    
    # Override config for eval
    device = cfg.train.device if cfg.train.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = cfg.train.checkpoint_dir + "/final_policy.pt" # Default to final, can be overridden
    
    # Create Environment
    adapter_type = "gym"
    if "MineRL" in cfg.env.name:
        adapter_type = "minerl"
        
    env = create_adapter(
        adapter_type=adapter_type,
        env_name=cfg.env.name,
        image_size=tuple(cfg.env.image_size),
        device=device,
        max_steps=cfg.env.max_episode_steps
    )
    
    # Create Brain
    brain = BrainCyborgMind(
        scalar_dim=env.scalar_dim,
        goal_dim=env.goal_dim,
        thought_dim=cfg.model.thought_dim,
        emotion_dim=cfg.model.emotion_dim,
        workspace_dim=cfg.model.workspace_dim,
        vision_dim=cfg.model.vision_dim,
        emb_dim=cfg.model.emb_dim,
        hidden_dim=cfg.model.hidden_dim,
        mem_dim=cfg.pmm.memory_dim,
        num_actions=env.action_space_size,
        start_slots=cfg.pmm.start_slots
    )
    
    # Load Checkpoint
    if Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "brain" in checkpoint:
            brain.load_state_dict(checkpoint["brain"])
        else:
            brain.load_state_dict(checkpoint) # Assume raw state dict
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using random weights")
        
    brain.to(device)
    brain.eval()
    
    # Run Evaluation
    num_episodes = 5
    rewards = []
    
    for i in range(num_episodes):
        obs = env.reset()
        
        # Init state
        state = {
            "thought": torch.zeros(1, cfg.model.thought_dim, device=device),
            "emotion": torch.zeros(1, cfg.model.emotion_dim, device=device),
            "workspace": torch.zeros(1, cfg.model.workspace_dim, device=device),
            "hidden": (
                torch.zeros(1, 1, cfg.model.hidden_dim, device=device),
                torch.zeros(1, 1, cfg.model.hidden_dim, device=device)
            )
        }
        
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                pixels = obs.pixels.unsqueeze(0) if obs.pixels.dim() == 3 else obs.pixels
                scalars = obs.scalars.unsqueeze(0) if obs.scalars.dim() == 1 else obs.scalars
                goal = obs.goal.unsqueeze(0) if obs.goal.dim() == 1 else obs.goal
                
                output = brain(
                    pixels=pixels,
                    scalars=scalars,
                    goal=goal,
                    thought=state["thought"],
                    emotion=state["emotion"],
                    workspace=state["workspace"],
                    hidden=state["hidden"]
                )
                
                # Deterministic action
                action_logits = output["action_logits"]
                action_idx = torch.argmax(action_logits, dim=-1).item()
                
                state = {
                    "thought": output["thought"],
                    "emotion": output["emotion"],
                    "workspace": output["workspace"],
                    "hidden": (output["hidden_h"].unsqueeze(0), output["hidden_c"].unsqueeze(0))
                }
                
                obs, reward, done, info = env.step(action_idx)
                episode_reward += reward
                
        rewards.append(episode_reward)
        logger.info(f"Episode {i+1}: Reward {episode_reward:.2f}")
        
    logger.info(f"Mean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

if __name__ == "__main__":
    main()

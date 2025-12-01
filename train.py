import hydra
from omegaconf import DictConfig
import torch
import logging
from pathlib import Path

from experiments.cyborg_mind_v2.configs.schema import Config
from experiments.cyborg_mind_v2.envs.gym_adapter import GymAdapter
from experiments.cyborg_mind_v2.envs.minerl_adapter import MineRLAdapter
from experiments.cyborg_mind_v2.capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind
from experiments.cyborg_mind_v2.training.trainer import CyborgTrainer

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="cyborg_mind_v2/configs", config_name="config")
def main(cfg: Config):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting training with config:\n{cfg}")
    
    # Create Environment
    # Determine adapter type based on env name or config
    adapter_type = "gym"
    if "MineRL" in cfg.env.name:
        adapter_type = "minerl"
        
    env = create_adapter(
        adapter_type=adapter_type,
        env_name=cfg.env.name,
        image_size=tuple(cfg.env.image_size),
        device=cfg.train.device if cfg.train.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
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
    
    # Create Trainer
    trainer = CyborgTrainer(
        env=env,
        brain=brain,
        config=cfg,
        device=cfg.train.device if cfg.train.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Resume if requested
    if cfg.train.resume_path:
        trainer.load_checkpoint(cfg.train.resume_path)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()

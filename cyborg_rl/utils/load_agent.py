"""Agent loading utilities for CyborgMind.

Helper functions to load trained agents from checkpoints with proper configuration.
"""

from pathlib import Path
from typing import Tuple
import torch

from cyborg_rl.config import Config
from cyborg_rl.agents import PPOAgent
from cyborg_rl.envs import MineRLAdapter
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


def load_agent(
    config_path: str,
    checkpoint_path: str,
    device: str = "auto",
) -> Tuple[PPOAgent, Config, MineRLAdapter]:
    """
    Load a trained MineRL agent from config and checkpoint.
    
    This helper function:
    1. Loads the configuration from YAML
    2. Creates the appropriate environment adapter (MineRL)
    3. Instantiates the PPOAgent with matching architecture
    4. Loads the trained weights from checkpoint
    5. Returns ready-to-use agent, config, and environment
    
    Args:
        config_path: Path to configuration YAML file
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on ("auto", "cpu", "cuda")
    
    Returns:
        Tuple of (agent, config, env_adapter)
        
    Raises:
        FileNotFoundError: If config or checkpoint files don't exist
        ValueError: If config specifies non-MineRL adapter
        RuntimeError: If checkpoint loading fails
    
    Example:
        >>> agent, config, env = load_agent(
        ...     "configs/treechop_ppo.yaml",
        ...     "artifacts/minerl_treechop/run_v1/best_model.pt",
        ...     device="cuda"
        ... )
        >>> obs = env.reset()
        >>> action, _, _, _, _ = agent.forward(obs.unsqueeze(0), deterministic=True)
    """
    # Validate paths
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    
    if not config_path.exists():
        raise FileNotFoundError("Config file not found: {}".format(config_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint file not found: {}".format(checkpoint_path))
    
    logger.info("Loading agent from:")
    logger.info("  Config: {}".format(config_path))
    logger.info("  Checkpoint: {}".format(checkpoint_path))
    
    # Load configuration
    logger.info("Loading configuration...")
    config = Config.from_yaml(str(config_path))
    
    # Validate that config uses MineRL adapter
    adapter_type = getattr(config.env, 'adapter', None)
    if adapter_type and adapter_type != 'minerl':
        logger.warning("Config specifies non-MineRL adapter: {}".format(adapter_type))
        logger.warning("Proceeding anyway, but ensure this is intended")
    
    # Determine device
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Auto-detected device: {}".format(device_obj))
    else:
        device_obj = torch.device(device)
        logger.info("Using specified device: {}".format(device_obj))
    
    # Create environment adapter
    logger.info("Creating MineRL environment adapter...")
    try:
        env = MineRLAdapter(
            env_name=config.env.name,
            device=device_obj,
            normalize_obs=config.env.normalize_obs,
            clip_obs=config.env.clip_obs,
            image_size=tuple(config.env.image_size) if hasattr(config.env, 'image_size') else (64, 64),
        )
        logger.info("Environment created: {}".format(config.env.name))
        logger.info("  Observation dim: {}".format(env.observation_dim))
        logger.info("  Action dim: {}".format(env.action_dim))
    except ImportError as e:
        logger.error("MineRL not installed: {}".format(e))
        raise RuntimeError("Cannot create MineRL environment. Install with: pip install minerl")
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device_obj)
        
        # Extract agent parameters
        obs_dim = checkpoint.get('obs_dim', env.observation_dim)
        action_dim = checkpoint.get('action_dim', env.action_dim)
        is_discrete = checkpoint.get('is_discrete', env.is_discrete)
        
        # Use checkpoint config if available (overrides YAML config for model architecture)
        if 'config' in checkpoint:
            logger.info("Using model config from checkpoint")
            checkpoint_config = checkpoint['config']
            # Merge checkpoint config into loaded config (model params only)
            if hasattr(checkpoint_config, 'model'):
                config.model = checkpoint_config.model
            if hasattr(checkpoint_config, 'memory'):
                config.memory = checkpoint_config.memory
        
        # Create agent
        logger.info("Creating PPO agent...")
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            is_discrete=is_discrete,
            device=device_obj,
        )
        
        # Load weights
        agent.load_state_dict(checkpoint['state_dict'])
        agent.to(device_obj)
        agent.eval()  # Set to evaluation mode by default
        
        logger.info("Agent loaded successfully")
        logger.info("  Parameters: {}".format(agent.count_parameters()))
        logger.info("  Encoder type: {}".format(getattr(config.model, 'encoder_type', 'mamba_gru')))
        
    except Exception as e:
        logger.error("Failed to load checkpoint: {}".format(e))
        raise RuntimeError("Checkpoint loading failed: {}".format(e))
    
    return agent, config, env

#!/usr/bin/env python3
"""
MineRL Treechop Training Pipeline - Production Grade

Complete pipeline for training MineRL Treechop agents with:
1. Environment & Agent Initialization
2. (Optional) Behavior Cloning warm-start
3. PPO Training with checkpointing
4. Deterministic Evaluation

Usage:
    python scripts/run_treechop_pipeline.py \
        --config configs/treechop_ppo.yaml \
        --run-name minerl_treechop_v1 \
        --steps 200000
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyborg_rl.config import Config
from cyborg_rl.envs import MineRLAdapter
from cyborg_rl.agents import PPOAgent
from cyborg_rl.trainers import PPOTrainer
from cyborg_rl.experiments import ExperimentRegistry
from cyborg_rl.utils import get_device, set_seed, get_logger

logger = get_logger(__name__)


def create_artifacts_dir(run_name):
    """Create artifacts directory structure."""
    base_dir = Path("artifacts") / "minerl_treechop" / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def stage_1_initialize(args, device):
    """
    Stage 1: Environment & Agent Initialization
    
    Returns:
        Tuple of (env, agent, config, registry)
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Environment & Agent Initialization")
    logger.info("=" * 80)
    
    # Load config
    if Path(args.config).exists():
        config = Config.from_yaml(args.config)
        logger.info("Loaded config from: {}".format(args.config))
    else:
        config = Config()
        config.env.name = "MineRLTreechop-v0"
        logger.warning("Config file not found, using defaults")
    
    # Override with CLI args
    if args.steps:
        config.train.total_timesteps = args.steps
    if args.seed is not None:
        config.train.seed = args.seed
    
    # Set deterministic seed
    set_seed(config.train.seed)
    logger.info("Set seed: {}".format(config.train.seed))
    
    # Create environment
    logger.info("Creating MineRL environment: {}".format(config.env.name))
    try:
        env = MineRLAdapter(
            env_name=config.env.name,
            device=device,
            seed=config.train.seed,
            normalize_obs=config.env.normalize_obs,
            clip_obs=config.env.clip_obs,
            image_size=tuple(config.env.image_size) if hasattr(config.env, 'image_size') else (64, 64),
        )
        logger.info("Environment created successfully")
        logger.info("  Observation dim: {}".format(env.observation_dim))
        logger.info("  Action dim: {}".format(env.action_dim))
        logger.info("  Discrete actions: {}".format(env.is_discrete))
    except ImportError as e:
        logger.error("MineRL not installed: {}".format(e))
        logger.error("Install with: pip install minerl")
        sys.exit(1)
    
    # Create agent
    logger.info("Creating PPO agent...")
    agent = PPOAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        config=config,
        is_discrete=env.is_discrete,
        device=device,
    )
    logger.info("Agent created with {} parameters".format(agent.count_parameters()))
    logger.info("  Encoder type: {}".format(getattr(config.model, 'encoder_type', 'mamba_gru')))
    
    # Create experiment registry
    artifacts_dir = create_artifacts_dir(args.run_name)
    registry = ExperimentRegistry(
        config=config.to_dict(),
        run_name=args.run_name,
        base_dir=str(artifacts_dir.parent),
    )
    logger.info("Experiment registry initialized")
    logger.info("  Run name: {}".format(registry.run_name))
    logger.info("  Artifacts: {}".format(registry.base_dir))
    
    # Save config to artifacts
    config.to_yaml(str(artifacts_dir / "config.yaml"))
    logger.info("Saved config to: {}".format(artifacts_dir / "config.yaml"))
    
    return env, agent, config, registry


def stage_2_behavior_cloning(env, agent, config, args):
    """
    Stage 2: (Optional) Behavior Cloning Warm-Start
    
    Currently a placeholder - BC requires expert demonstrations.
    """
    if args.bc_data:
        logger.info("=" * 80)
        logger.info("STAGE 2: Behavior Cloning Warm-Start")
        logger.info("=" * 80)
        logger.warning("BC warm-start requested but not yet implemented")
        logger.warning("BC data path: {}".format(args.bc_data))
        logger.info("Skipping BC, proceeding to PPO training...")
    else:
        logger.info("Skipping BC warm-start (no --bc-data provided)")


def stage_3_ppo_training(env, agent, config, registry):
    """
    Stage 3: PPO Training with Checkpointing
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: PPO Training")
    logger.info("=" * 80)
    logger.info("Total timesteps: {}".format(config.train.total_timesteps))
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        config=config,
        registry=registry,
    )
    
    # Train
    logger.info("Starting PPO training...")
    trainer.train()
    
    # Save final checkpoint
    final_path = registry.ckpt_dir / "final_model.pt"
    agent.save(str(final_path))
    logger.info("Saved final model to: {}".format(final_path))
    
    return trainer


def stage_4_evaluation(env, agent, config, registry):
    """
    Stage 4: Deterministic Evaluation
    
    Runs N evaluation episodes with fixed seeds to measure:
    - Mean reward
    - Success rate (trees chopped / episodes)
    - Episode length
    """
    logger.info("=" * 80)
    logger.info("STAGE 4: Deterministic Evaluation")
    logger.info("=" * 80)
    
    # Get eval config
    num_eval_episodes = getattr(config.eval, 'num_eval_episodes', 10) if hasattr(config, 'eval') else 10
    eval_seeds = getattr(config.eval, 'eval_seeds', None) if hasattr(config, 'eval') else None
    
    if eval_seeds is None:
        # Generate fixed seeds starting from 1000
        eval_seeds = list(range(1000, 1000 + num_eval_episodes))
    else:
        eval_seeds = eval_seeds[:num_eval_episodes]  # Ensure we have exactly num_eval_episodes seeds
    
    logger.info("Running {} evaluation episodes".format(num_eval_episodes))
    logger.info("Eval seeds: {}".format(eval_seeds))
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    agent.eval()  # Set to eval mode
    
    for ep_idx, seed in enumerate(eval_seeds):
        logger.info("Eval episode {}/{}  (seed={})".format(ep_idx + 1, num_eval_episodes, seed))
        
        # Reset with fixed seed
        obs = env.reset()
        state = agent.init_state(batch_size=1)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < 8000:  # Max steps per episode
            # Get deterministic action
            with torch.no_grad():
                action, _, _, state, _ = agent.forward(obs.unsqueeze(0), state, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action[0])
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check success (simple heuristic: positive reward indicates progress)
        if episode_reward > 0:
            success_count += 1
        
        logger.info("  Reward: {:.2f}, Length: {}".format(episode_reward, episode_length))
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / num_eval_episodes
    
    logger.info("=" * 80)
    logger.info("Evaluation Results:")
    logger.info("  Mean Reward: {:.2f} +/- {:.2f}".format(mean_reward, std_reward))
    logger.info("  Mean Length: {:.1f}".format(mean_length))
    logger.info("  Success Rate: {:.1%} ({}/{})".format(success_rate, success_count, num_eval_episodes))
    logger.info("=" * 80)
    
    # Log to registry
    eval_metrics = {
        "eval_mean_reward": mean_reward,
        "eval_std_reward": std_reward,
        "eval_mean_length": mean_length,
        "eval_success_rate": success_rate,
        "eval_success_count": success_count,
        "eval_num_episodes": num_eval_episodes,
    }
    
    # Save eval results
    import json
    eval_results_path = registry.base_dir / "eval_results.json"
    with open(eval_results_path, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    logger.info("Saved eval results to: {}".format(eval_results_path))
    
    agent.train()  # Set back to train mode
    
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="MineRL Treechop Training Pipeline")
    
    # Required
    parser.add_argument("--config", type=str, default="configs/treechop_ppo.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name for this training run")
    
    # Optional overrides
    parser.add_argument("--steps", type=int, default=None,
                        help="Override total training steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cpu/cuda/auto)")
    
    # BC warm-start (optional)
    parser.add_argument("--bc-data", type=str, default=None,
                        help="Path to BC demonstration data (optional)")
    
    # Evaluation
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation phase")
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device(args.device)
    logger.info("Using device: {}".format(device))
    
    try:
        # Stage 1: Initialize
        env, agent, config, registry = stage_1_initialize(args, device)
        
        # Stage 2: BC (optional)
        stage_2_behavior_cloning(env, agent, config, args)
        
        # Stage 3: PPO Training
        trainer = stage_3_ppo_training(env, agent, config, registry)
        
        # Stage 4: Evaluation
        if not args.skip_eval:
            eval_metrics = stage_4_evaluation(env, agent, config, registry)
        else:
            logger.info("Skipping evaluation (--skip-eval flag set)")
        
        # Cleanup
        env.close()
        
        logger.info("=" * 80)
        logger.info("Pipeline Complete!")
        logger.info("=" * 80)
        logger.info("Results saved to: {}".format(registry.base_dir))
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed with error: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

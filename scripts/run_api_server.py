#!/usr/bin/env python3
"""
API Server Launcher for MineRL Agents.

Launches the CyborgMind FastAPI server with a trained MineRL agent.

Usage:
    python scripts/run_api_server.py \
        --config configs/treechop_ppo.yaml \
        --checkpoint artifacts/minerl_treechop/run_v1/best_model.pt \
        --host 0.0.0.0 \
        --port 8000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cyborg_rl.utils.load_agent import load_agent
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CyborgMind API Server Launcher")
    
    # Required
    parser.add_argument("--config", type=str, required=True,
                        help="Path to agent config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Server settings
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind server to")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run model on (auto/cpu/cuda)")
    
    # Advanced
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload (development only)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("CyborgMind API Server Launcher")
    logger.info("=" * 80)
    
    # Load agent
    logger.info("Loading agent...")
    try:
        agent, config, env = load_agent(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        logger.info("Agent loaded successfully")
    except Exception as e:
        logger.error("Failed to load agent: {}".format(e))
        sys.exit(1)
    
    # Store paths in environment variables for server to load
    import os
    os.environ["CYBORG_CONFIG_PATH"] = args.config
    os.environ["CYBORG_CHECKPOINT_PATH"] = args.checkpoint
    os.environ["CYBORG_DEVICE"] = args.device
    
    logger.info("Starting FastAPI server...")
    logger.info("  Host: {}".format(args.host))
    logger.info("  Port: {}".format(args.port))
    logger.info("  Workers: {}".format(args.workers))
    logger.info("  Reload: {}".format(args.reload))
    logger.info("=" * 80)
    
    # Launch server
    try:
        import uvicorn
        
        # For single worker mode, we can pass the agent directly
        if args.workers == 1 and not args.reload:
            import cyborg_rl.server as server_module
            server_module.LOADED_AGENT = agent
            server_module.LOADED_CONFIG = config
            server_module.LOADED_ENV = env
        
        uvicorn.run(
            "cyborg_rl.server:create_app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            factory=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.error("Server failed: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

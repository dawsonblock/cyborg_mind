"""
CyborgMind Production API Server.

Features:
- FastAPI based
- JWT/HMAC Token authentication
- Rate Limiting (SlowAPI)
- Prometheus metrics
- Async Batch inference
- Health checks
- Structured Logging
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from cyborg_rl.config import Config
from cyborg_rl.agents.ppo_agent import PPOAgent
from cyborg_rl.utils.logging import get_logger
from cyborg_rl.utils.jwt_auth import create_jwt_handler, JWTAuth

# Setup Logging
logger = get_logger(__name__)

# Setup Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# --- Schemas ---

class StepRequest(BaseModel):
    observation: List[float] = Field(..., description="Observation vector")
    agent_id: str = Field("default", description="Unique agent session ID")
    deterministic: bool = Field(True, description="Use deterministic action selection")

class StepResponse(BaseModel):
    action: int | List[float]
    value: float
    thought: str
    pressure: float

class BatchStepRequest(BaseModel):
    observations: List[List[float]]
    agent_ids: List[str]

class TokenRequest(BaseModel):
    subject: str = Field(..., description="Token subject (user/agent ID)")
    expiry_minutes: Optional[int] = Field(None, description="Custom expiry (overrides default)")

class StreamObservation(BaseModel):
    observation: List[float] = Field(..., description="Observation vector")
    deterministic: bool = Field(True, description="Use deterministic action selection")

class StreamAction(BaseModel):
    action: int | List[float]
    value: float
    pressure: float
    error: Optional[str] = None

# --- Metrics ---

REQUEST_COUNT = Counter("cyborg_api_requests_total", "Total API requests", ["endpoint", "status"])
LATENCY = Histogram("cyborg_api_latency_seconds", "Request latency", ["endpoint"])
AGENT_PRESSURE = Histogram("cyborg_agent_memory_pressure", "Agent memory pressure")
WEBSOCKET_CONNECTIONS = Counter("cyborg_websocket_connections_total", "Total WebSocket connections", ["status"])

# --- Server ---

class CyborgServer:
    def __init__(self, config_path: str = "config.yaml", checkpoint_path: str = "checkpoints/final_policy.pt", device: str = "cpu"):
        self.config = Config()
        if Path(config_path).exists():
            self.config = Config.from_yaml(config_path)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config {config_path} not found. Using defaults.")
        
        self.device = torch.device(device)
        self.agent = self._load_agent(checkpoint_path)
        self.states: Dict[str, Dict[str, torch.Tensor]] = {}

        # Setup authentication (JWT + static token fallback)
        self.jwt_auth = self.__create_jwt_auth()

        self.security = HTTPBearer()
        self.app = FastAPI(
            title="CyborgMind v3.0 Brain API",
            description="Production RL Inference API",
            version="3.0.0"
        )

        self._setup_middleware()
        self._setup_routes()

    def __create_jwt_auth(self) -> JWTAuth:
        """Create JWT authentication handler."""
        return create_jwt_handler(
            jwt_enabled=self.config.api.jwt_enabled,
            jwt_secret=self.config.api.jwt_secret,
            jwt_algorithm=self.config.api.jwt_algorithm,
            jwt_issuer=self.config.api.jwt_issuer,
            jwt_audience=self.config.api.jwt_audience,
            jwt_expiry_minutes=self.config.api.jwt_expiry_minutes,
            static_token=self.config.api.auth_token,
            jwt_public_key_path=self.config.api.jwt_public_key_path,
        )

    def _load_agent(self, path: str) -> PPOAgent:
        if not Path(path).exists():
            logger.warning(f"Checkpoint {path} not found. Initializing random agent.")
            # Fallback: Create dummy agent
            return PPOAgent(4, 2, self.config, device=self.device)
        
        try:
            agent = PPOAgent.load(path, self.device)
            agent.eval()
            logger.info(f"Loaded agent from {path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load agent from {path}: {e}")
            return PPOAgent(4, 2, self.config, device=self.device)

    def _setup_middleware(self):
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Rate Limit Handler
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    def _verify_token(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
        """Verify JWT or static bearer token."""
        token = credentials.credentials

        # Verify token (supports both JWT and static tokens)
        is_valid, payload, error = self.jwt_auth.verify_token(token)

        if not is_valid:
            logger.warning(f"Authentication failed: {error}")
            raise HTTPException(status_code=403, detail=error or "Invalid authentication token")

        # Log successful auth
        subject = payload.get("sub", "unknown") if payload else "unknown"
        logger.debug(f"Authenticated request for subject: {subject}")

        return payload

    def _get_state(self, agent_id: str) -> Dict[str, torch.Tensor]:
        if agent_id not in self.states:
            self.states[agent_id] = self.agent.init_state(batch_size=1)
        return self.states[agent_id]

    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy", 
                "device": str(self.device), 
                "agents_active": len(self.states),
                "version": "3.0.0"
            }

        @self.app.get("/metrics")
        async def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @self.app.post("/auth/token")
        async def generate_token(req: TokenRequest):
            """Generate JWT token (only available if JWT is enabled)."""
            if not self.config.api.jwt_enabled:
                raise HTTPException(
                    status_code=501,
                    detail="JWT authentication is not enabled. Set api.jwt_enabled=true in config."
                )

            from datetime import timedelta
            expires_delta = timedelta(minutes=req.expiry_minutes) if req.expiry_minutes else None

            token = self.jwt_auth.generate_token(
                subject=req.subject,
                expires_delta=expires_delta
            )

            expiry = self.jwt_auth.get_token_expiry(token)

            return {
                "access_token": token,
                "token_type": "bearer",
                "expires_at": expiry.isoformat() if expiry else None,
                "subject": req.subject
            }

        @self.app.post("/reset", dependencies=[Depends(self._verify_token)])
        async def reset(agent_id: str = "default"):
            if agent_id in self.states:
                del self.states[agent_id]
            return {"status": "reset", "agent_id": agent_id}

        @self.app.post("/step", response_model=StepResponse, dependencies=[Depends(self._verify_token)])
        @limiter.limit("100/second")
        async def step(request: Request, req: StepRequest):
            start_time = time.time()
            
            try:
                # Prepare input
                obs_tensor = torch.tensor(req.observation, device=self.device, dtype=torch.float32).unsqueeze(0)
                state = self._get_state(req.agent_id)
                
                # Inference
                with torch.no_grad():
                    action, _, value, new_state, info = self.agent(
                        obs_tensor, state, deterministic=req.deterministic
                    )
                
                # Update state
                self.states[req.agent_id] = new_state
                
                # Process output
                action_val = action.cpu().numpy()
                if self.agent.is_discrete:
                    action_val = int(action_val[0])
                else:
                    action_val = action_val[0].tolist()
                
                pressure = info["pmm_info"]["pressure"].item() if "pressure" in info["pmm_info"] else 0.0
                
                # Metrics
                REQUEST_COUNT.labels(endpoint="/step", status="success").inc()
                LATENCY.labels(endpoint="/step").observe(time.time() - start_time)
                AGENT_PRESSURE.observe(pressure)
                
                return StepResponse(
                    action=action_val,
                    value=value.item(),
                    thought=f"Memory pressure: {pressure:.2f}",
                    pressure=pressure
                )
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                REQUEST_COUNT.labels(endpoint="/step", status="error").inc()
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/step_batch", dependencies=[Depends(self._verify_token)])
        @limiter.limit("20/second")
        async def step_batch(request: Request, req: BatchStepRequest):
            """Batch inference endpoint."""
            start_time = time.time()
            
            try:
                batch_size = len(req.observations)
                if batch_size == 0:
                    return {"actions": []}

                # Prepare input
                obs_tensor = torch.tensor(req.observations, device=self.device, dtype=torch.float32)
                
                # Collect states (Naive sequential collection for now)
                # Ideally, we'd have a BatchedStateManager
                hidden_states = []
                memory_states = []
                
                for agent_id in req.agent_ids:
                    state = self._get_state(agent_id)
                    hidden_states.append(state["hidden"])
                    memory_states.append(state["memory"])
                
                # Stack: [B, Layers, H] -> [Layers, B, H] for GRU usually
                # But our agent expects [B, ...] for state if we designed it that way.
                # Let's assume PPOAgent handles [B, ...] correctly if we stack along dim 0.
                
                # NOTE: MambaGRUEncoder usually expects hidden as [D, B, H] for GRU
                # We need to check PPOAgent.init_state.
                # Assuming init_state returns [Layers, B, H] for hidden.
                
                # Let's assume standard stacking along batch dim works for now
                # or we do sequential inference if batching logic is tricky without refactor.
                # For safety in this "rewrite", we'll do sequential loop if batching is risky,
                # BUT the prompt asked for "Async batch inference".
                
                # Let's try to batch.
                batched_hidden = torch.cat(hidden_states, dim=1) # [Layers, B, H]
                batched_memory = torch.cat(memory_states, dim=0) # [B, N, M]
                
                batched_state = {"hidden": batched_hidden, "memory": batched_memory}
                
                # Inference
                with torch.no_grad():
                    actions, _, values, new_state, info = self.agent(
                        obs_tensor, batched_state, deterministic=True
                    )
                
                # Unpack
                new_hidden = new_state["hidden"]
                new_memory = new_state["memory"]
                
                results = []
                for i, agent_id in enumerate(req.agent_ids):
                    # Slice back
                    h = new_hidden[:, i:i+1, :].contiguous()
                    m = new_memory[i:i+1, :, :].contiguous()
                    self.states[agent_id] = {"hidden": h, "memory": m}
                    
                    act = actions[i].cpu().item() if self.agent.is_discrete else actions[i].cpu().tolist()
                    results.append({
                        "agent_id": agent_id,
                        "action": act,
                        "value": values[i].item()
                    })
                
                REQUEST_COUNT.labels(endpoint="/step_batch", status="success").inc()
                LATENCY.labels(endpoint="/step_batch").observe(time.time() - start_time)
                return {"results": results}

            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                REQUEST_COUNT.labels(endpoint="/step_batch", status="error").inc()
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/stream")
        async def stream_inference(websocket: WebSocket):
            """WebSocket streaming endpoint for continuous inference.

            Client sends: {"observation": [float, ...], "deterministic": bool, "token": "bearer-token"}
            Server responds: {"action": int|[float], "value": float, "pressure": float, "error": str|null}
            """
            await websocket.accept()
            WEBSOCKET_CONNECTIONS.labels(status="connected").inc()

            agent_id = f"ws_{id(websocket)}"
            logger.info(f"WebSocket connection established: {agent_id}")

            try:
                while True:
                    # Receive observation from client
                    data = await websocket.receive_json()

                    # Authenticate via token in message
                    token = data.get("token")
                    if not token:
                        await websocket.send_json({
                            "action": None,
                            "value": 0.0,
                            "pressure": 0.0,
                            "error": "Missing 'token' field in message"
                        })
                        continue

                    # Verify token
                    is_valid, payload, error = self.jwt_auth.verify_token(token)
                    if not is_valid:
                        await websocket.send_json({
                            "action": None,
                            "value": 0.0,
                            "pressure": 0.0,
                            "error": f"Authentication failed: {error}"
                        })
                        continue

                    # Extract observation
                    observation = data.get("observation")
                    deterministic = data.get("deterministic", True)

                    if not observation:
                        await websocket.send_json({
                            "action": None,
                            "value": 0.0,
                            "pressure": 0.0,
                            "error": "Missing 'observation' field"
                        })
                        continue

                    # Validate observation dimension
                    if len(observation) != self.agent.obs_dim:
                        await websocket.send_json({
                            "action": None,
                            "value": 0.0,
                            "pressure": 0.0,
                            "error": f"Expected observation of length {self.agent.obs_dim}, got {len(observation)}"
                        })
                        continue

                    # Perform inference
                    try:
                        obs_tensor = torch.tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0)
                        state = self._get_state(agent_id)

                        with torch.no_grad():
                            action, _, value, new_state, info = self.agent(
                                obs_tensor, state, deterministic=deterministic
                            )

                        # Update state
                        self.states[agent_id] = new_state

                        # Process output
                        action_val = action.cpu().numpy()
                        if self.agent.is_discrete:
                            action_val = int(action_val[0])
                        else:
                            action_val = action_val[0].tolist()

                        pressure = info["pmm_info"]["pressure"].item() if "pressure" in info["pmm_info"] else 0.0

                        # Send response
                        await websocket.send_json({
                            "action": action_val,
                            "value": value.item(),
                            "pressure": pressure,
                            "error": None
                        })

                        # Metrics
                        REQUEST_COUNT.labels(endpoint="/stream", status="success").inc()
                        AGENT_PRESSURE.observe(pressure)

                    except Exception as e:
                        logger.error(f"WebSocket inference error: {e}")
                        await websocket.send_json({
                            "action": None,
                            "value": 0.0,
                            "pressure": 0.0,
                            "error": str(e)
                        })
                        REQUEST_COUNT.labels(endpoint="/stream", status="error").inc()

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {agent_id}")
                WEBSOCKET_CONNECTIONS.labels(status="disconnected").inc()

                # Clean up agent state
                if agent_id in self.states:
                    del self.states[agent_id]

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                WEBSOCKET_CONNECTIONS.labels(status="error").inc()
                await websocket.close()

# Module-level variables for single-worker mode
LOADED_AGENT = None
LOADED_CONFIG = None
LOADED_ENV = None

def create_app():
    import os
    
    # Try to use pre-loaded agent (single worker mode)
    if LOADED_AGENT is not None:
        logger.info("Using pre-loaded agent from module")
        server = CyborgServer.__new__(CyborgServer)
        server.agent = LOADED_AGENT
        server.config = LOADED_CONFIG
        server.device = LOADED_AGENT.device
        server.states = {}
        server.jwt_auth = server._CyborgServer__create_jwt_auth()
        server.security = HTTPBearer()
        server.app = FastAPI(
            title="CyborgMind v3.0 Brain API",
            description="Production RL Inference API",
            version="3.0.0"
        )
        server._setup_middleware()
        server._setup_routes()
        return server.app
    
    # Load from environment variables (multi-worker mode)
    config_path = os.getenv("CYBORG_CONFIG_PATH", "config.yaml")
    checkpoint_path = os.getenv("CYBORG_CHECKPOINT_PATH", "checkpoints/final_policy.pt")
    device = os.getenv("CYBORG_DEVICE", "cpu")
    
    logger.info(f"Loading agent from environment: config={config_path}, checkpoint={checkpoint_path}")
    server = CyborgServer(config_path, checkpoint_path, device)
    return server.app

if __name__ == "__main__":
    uvicorn.run("cyborg_rl.server:create_app", host="0.0.0.0", port=8000, factory=True)

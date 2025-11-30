"""
CyborgMind Production API Server.

Features:
- FastAPI based
- Token authentication
- Prometheus metrics
- Batch inference support
- Health checks
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from cyborg_rl import Config, get_device, get_logger
from cyborg_rl.agents import PPOAgent

logger = get_logger(__name__)

# --- Schemas ---

class StepRequest(BaseModel):
    observation: List[float]
    agent_id: str = "default"
    deterministic: bool = True

class StepResponse(BaseModel):
    action: int | List[float]
    value: float
    thought: str
    pressure: float

class BatchStepRequest(BaseModel):
    observations: List[List[float]]
    agent_ids: List[str]

# --- Metrics ---

REQUEST_COUNT = Counter("cyborg_api_requests_total", "Total API requests", ["endpoint"])
LATENCY = Histogram("cyborg_api_latency_seconds", "Request latency")
AGENT_PRESSURE = Histogram("cyborg_agent_memory_pressure", "Agent memory pressure")

# --- Server ---

class CyborgServer:
    def __init__(self, config_path: str = "config.yaml", checkpoint_path: str = "checkpoints/final_policy.pt"):
        self.config = Config()
        if Path(config_path).exists():
            self.config = Config.from_yaml(config_path)
        
        self.device = get_device("cpu") # Inference usually on CPU for low latency unless batch is huge
        self.agent = self._load_agent(checkpoint_path)
        self.states: Dict[str, Dict[str, torch.Tensor]] = {}
        
        self.security = HTTPBearer()
        self.app = FastAPI(title="CyborgMind v2.8 Brain API")
        self._setup_routes()

    def _load_agent(self, path: str) -> PPOAgent:
        if not Path(path).exists():
            logger.warning(f"Checkpoint {path} not found. Initializing random agent.")
            return PPOAgent(4, 2, self.config, device=self.device) # Fallback
        
        try:
            agent = PPOAgent.load(path, self.device)
            agent.eval()
            return agent
        except Exception as e:
            logger.error(f"Failed to load agent from {path}: {e}")
            # Fallback to random agent to keep server alive
            return PPOAgent(4, 2, self.config, device=self.device)

    def _verify_token(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
        if credentials.credentials != self.config.api.auth_token:
            raise HTTPException(status_code=403, detail="Invalid authentication token")
        return credentials.credentials

    def _get_state(self, agent_id: str) -> Dict[str, torch.Tensor]:
        if agent_id not in self.states:
            self.states[agent_id] = self.agent.init_state(batch_size=1)
        return self.states[agent_id]

    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "device": str(self.device), "agents_active": len(self.states)}

        @self.app.get("/metrics")
        async def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @self.app.post("/reset", dependencies=[Depends(self._verify_token)])
        async def reset(agent_id: str = "default"):
            if agent_id in self.states:
                del self.states[agent_id]
            return {"status": "reset", "agent_id": agent_id}

        @self.app.post("/step", response_model=StepResponse, dependencies=[Depends(self._verify_token)])
        async def step(req: StepRequest):
            start_time = time.time()
            REQUEST_COUNT.labels(endpoint="/step").inc()
            
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
                
                pressure = info["pmm_info"]["pressure"].item()
                AGENT_PRESSURE.observe(pressure)
                
                LATENCY.observe(time.time() - start_time)
                
                return StepResponse(
                    action=action_val,
                    value=value.item(),
                    thought=f"Memory pressure: {pressure:.2f}",
                    pressure=pressure
                )
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

def create_app():
    from pathlib import Path
    server = CyborgServer()
    return server.app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)

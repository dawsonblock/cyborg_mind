"""
CyborgMind V2 - FastAPI NPC/Agent Web API

This module provides a REST API for deploying CyborgMind as an NPC or agent service.
Game engines, simulators, or other applications can interact with the brain via HTTP.

Features:
- /reset: Initialize new agent or reset existing
- /step: Get action from observation
- /state: Query current brain state (thoughts, emotions, workspace)
- /health: API health check
- /metrics: Memory and performance metrics

Usage:
    # Start server
    uvicorn cyborg_mind_v2.deployment.api_server:app --host 0.0.0.0 --port 8000

    # Example client
    import requests
    response = requests.post("http://localhost:8000/reset", json={"agent_id": "npc_1"})
    response = requests.post("http://localhost:8000/step", json={
        "agent_id": "npc_1",
        "pixels": [...],  # Base64 encoded image or array
        "scalars": [0.1, 0.2, ...],
        "goal": [1.0, 0.0, 0.0, 0.0]
    })
"""

from typing import Dict, List, Optional, Any
import base64
import io
import numpy as np
import torch
import os
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind

class ResetRequest(BaseModel):
    agent_id: str = Field(..., description="Unique agent identifier")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint to load")


class StepRequest(BaseModel):
    agent_id: str = Field(..., description="Unique agent identifier")
    pixels: Any = Field(..., description="Image observation (base64 string or array)")
    scalars: List[float] = Field(..., description="Scalar features")
    goal: List[float] = Field(..., description="Goal vector")


class StepResponse(BaseModel):
    action: int = Field(..., description="Selected action index")
    thought: List[float] = Field(..., description="Thought vector")
    emotion: List[float] = Field(..., description="Emotion vector")
    workspace: List[float] = Field(..., description="Workspace vector")
    value: float = Field(..., description="Value estimate")
    pressure: float = Field(..., description="Memory pressure")


class StateResponse(BaseModel):
    agent_id: str
    thought: List[float]
    emotion: List[float]
    workspace: List[float]
    memory_slots: int
    memory_pressure: float


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    cuda_available: bool
    num_agents: int


class MetricsResponse(BaseModel):
    total_agents: int
    total_steps: int
    memory_usage_mb: float
    gpu_memory_mb: Optional[float]


# Agent state manager
class AgentStateManager:
    """
    Manages per-agent persistent state for the CyborgMind brain.

    Each agent has:
    - LSTM hidden state (h, c)
    - Thought vector
    - Emotion vector
    - Workspace vector
    - Step counter
    """

    def __init__(self, brain: BrainCyborgMind, device: str):
        self.brain = brain
        self.device = torch.device(device)
        self.agents: Dict[str, Dict[str, torch.Tensor]] = {}
        self.step_counts: Dict[str, int] = {}

    def reset(self, agent_id: str) -> None:
        """Initialize or reset agent state."""
        self.agents[agent_id] = {
            "h": torch.zeros(1, 1, self.brain.hidden_dim, device=self.device),
            "c": torch.zeros(1, 1, self.brain.hidden_dim, device=self.device),
            "thought": torch.zeros(1, self.brain.thought_dim, device=self.device),
            "emotion": torch.zeros(1, self.brain.emotion_dim, device=self.device),
            "workspace": torch.zeros(1, self.brain.workspace_dim, device=self.device),
        }
        self.step_counts[agent_id] = 0

    def get_state(self, agent_id: str) -> Dict[str, torch.Tensor]:
        """Get agent state, creating if doesn't exist."""
        if agent_id not in self.agents:
            self.reset(agent_id)
        return self.agents[agent_id]

    def update_state(self, agent_id: str, new_state: Dict[str, torch.Tensor]) -> None:
        """Update agent state after brain forward pass."""
        self.agents[agent_id] = new_state
        self.step_counts[agent_id] += 1

    def remove(self, agent_id: str) -> None:
        """Remove agent from manager."""
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.step_counts:
            del self.step_counts[agent_id]

    @property
    def num_agents(self) -> int:
        """Number of active agents."""
        return len(self.agents)


# Create FastAPI app
app = FastAPI(
    title="CyborgMind V2 API",
    description="REST API for CyborgMind emotion-consciousness brain",
    version="2.0.0",
)

# CORS middleware (allow all origins for demo - restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
brain: Optional[BrainCyborgMind] = None
agent_manager: Optional[AgentStateManager] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"
total_steps: int = 0


@app.on_event("startup")
async def startup_event():
    """Initialize brain on startup."""
    global brain, agent_manager, device

    print("="*60)
    print("  CyborgMind V2 API Server - Initializing")
    print("="*60)

    # Create brain
    brain = BrainCyborgMind().to(device)
    brain.eval()  # Inference mode

    # Create agent manager
    agent_manager = AgentStateManager(brain, device)

    print(f"✓ Brain initialized on {device}")
    print(f"✓ Memory slots: {brain.pmm.mem_slots}")
    print(f"✓ Parameters: {sum(p.numel() for p in brain.parameters()):,}")
    print("="*60)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "CyborgMind V2 API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.5.0",
        device=device,
        cuda_available=torch.cuda.is_available(),
        num_agents=agent_manager.num_agents if agent_manager else 0,
    )


@app.post("/reset")
async def reset(request: ResetRequest):
    """
    Reset agent state.

    Initializes or resets an agent's brain state. Optionally loads from checkpoint.
    """
    global brain, agent_manager

    if brain is None or agent_manager is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    # Load checkpoint if provided
    if request.checkpoint_path:
        # Only allow loading from whitelisted checkpoints
        checkpoint_name = os.path.basename(request.checkpoint_path)
        allowed_path = ALLOWED_CHECKPOINTS.get(checkpoint_name)
        if not allowed_path:
            raise HTTPException(
                status_code=400,
                detail="Requested checkpoint path is not allowed"
            )
        try:
            checkpoint = torch.load(allowed_path, map_location=device)
            brain.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
            print(f"Loaded checkpoint for {request.agent_id}: {allowed_path}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load checkpoint: {str(e)}"
            )

    # Reset agent state
    agent_manager.reset(request.agent_id)

    return {
        "status": "success",
        "agent_id": request.agent_id,
        "message": "Agent state reset"
    }


def decode_pixels(pixels: Any) -> torch.Tensor:
    """
    Decode pixels from various formats to tensor.

    Supports:
    - Base64 encoded image string
    - List/array of pixel values
    - Already a tensor
    """
    if isinstance(pixels, str):
        # Base64 encoded image
        img_bytes = base64.b64decode(pixels)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        # HWC -> CHW
        img_array = np.transpose(img_array, (2, 0, 1))
        return torch.from_numpy(img_array).float()
    elif isinstance(pixels, list):
        # List of pixel values
        arr = np.array(pixels, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            # HWC -> CHW
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).float()
    elif isinstance(pixels, torch.Tensor):
        return pixels.float()
    else:
        raise ValueError(f"Unsupported pixels format: {type(pixels)}")


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Execute one brain step.

    Takes observation (pixels, scalars, goal) and returns action + brain state.
    """
    global brain, agent_manager, total_steps

    if brain is None or agent_manager is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    # Get agent state
    state = agent_manager.get_state(request.agent_id)

    try:
        # Prepare inputs
        pixels = decode_pixels(request.pixels).to(device).unsqueeze(0)
        scalars = torch.tensor(request.scalars, dtype=torch.float32).to(device).unsqueeze(0)
        goal = torch.tensor(request.goal, dtype=torch.float32).to(device).unsqueeze(0)

        # Validate shapes
        if pixels.size(1) != 3:
            raise ValueError(f"Pixels must have 3 channels, got {pixels.size(1)}")
        if scalars.size(1) != brain.scalar_dim:
            raise ValueError(f"Scalars must have {brain.scalar_dim} dims, got {scalars.size(1)}")
        if goal.size(1) != brain.goal_dim:
            raise ValueError(f"Goal must have {brain.goal_dim} dims, got {goal.size(1)}")

        # Forward pass
        with torch.no_grad():
            output = brain(
                pixels=pixels,
                scalars=scalars,
                goal=goal,
                thought=state["thought"],
                emotion=state["emotion"],
                workspace=state["workspace"],
                hidden=(state["h"], state["c"]),
            )

        # Extract outputs
        action_logits = output["action_logits"]
        action = torch.argmax(action_logits, dim=1).item()
        value = output["value"].item()
        pressure = output["pressure"].item()

        # Update agent state
        new_state = {
            "h": output["hidden_h"],
            "c": output["hidden_c"],
            "thought": output["thought"],
            "emotion": output["emotion"],
            "workspace": output["workspace"],
        }
        agent_manager.update_state(request.agent_id, new_state)
        total_steps += 1

        # Return response
        return StepResponse(
            action=action,
            thought=output["thought"].cpu().numpy().flatten().tolist(),
            emotion=output["emotion"].cpu().numpy().flatten().tolist(),
            workspace=output["workspace"].cpu().numpy().flatten().tolist(),
            value=value,
            pressure=pressure,
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Step failed: {str(e)}"
        )


@app.get("/state/{agent_id}", response_model=StateResponse)
async def get_state(agent_id: str):
    """
    Get current brain state for an agent.

    Returns thought, emotion, workspace vectors and memory info.
    """
    global brain, agent_manager

    if brain is None or agent_manager is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    if agent_id not in agent_manager.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    state = agent_manager.agents[agent_id]

    return StateResponse(
        agent_id=agent_id,
        thought=state["thought"].cpu().numpy().flatten().tolist(),
        emotion=state["emotion"].cpu().numpy().flatten().tolist(),
        workspace=state["workspace"].cpu().numpy().flatten().tolist(),
        memory_slots=brain.pmm.mem_slots,
        memory_pressure=brain.pmm.get_pressure(),
    )


@app.delete("/agent/{agent_id}")
async def delete_agent(agent_id: str):
    """Remove an agent from the system."""
    global agent_manager

    if agent_manager is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    agent_manager.remove(agent_id)

    return {"status": "success", "message": f"Agent {agent_id} removed"}


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Get system metrics."""
    global brain, agent_manager, total_steps

    if brain is None or agent_manager is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024

    gpu_memory_mb = None
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

    return MetricsResponse(
        total_agents=agent_manager.num_agents,
        total_steps=total_steps,
        memory_usage_mb=memory_mb,
        gpu_memory_mb=gpu_memory_mb,
    )


@app.post("/load_checkpoint")
async def load_checkpoint(checkpoint_path: str):
    """Load brain weights from checkpoint."""
    global brain

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        brain.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
        return {"status": "success", "message": f"Loaded checkpoint: {checkpoint_path}"}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load checkpoint: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "cyborg_mind_v2.deployment.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

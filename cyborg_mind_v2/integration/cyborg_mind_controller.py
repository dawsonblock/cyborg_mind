"""
Cyborg Mind Controller

This controller orchestrates the multi‑agent interactions for the
Cyborg Mind brain defined in
``capsule_brain/policy/brain_cyborg_mind.py``.  It maintains per‑agent
persistent state—including the hidden LSTM state, the thought
vector, the emotion vector and the global workspace—and handles
memory expansion when pressure is high.  It is analogous to the
``CyborgController`` in the original dynamic architecture【940122878883148†L386-L458】
but extended to support the new cognition features.

Usage:

```python
controller = CyborgMindController(device="cuda")
actions = controller.step(agent_ids, pixels, scalars, goals)
```

Each call to `step` returns a list of integer actions for the agents.

"""

from typing import Dict, Tuple, List

import torch

from ..capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind


class CyborgMindController:
    def __init__(self, ckpt_path: str = None, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.brain = BrainCyborgMind().to(self.device)
        # Load checkpoint if provided
        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.brain.load_state_dict(ckpt, strict=False)
                print(
                    f"[System] Mind loaded. Memory Slots: {self.brain.pmm.mem_slots}"
                )
            except Exception as e:
                print(f"[System] Init fresh mind. ({e})")
        self.brain.eval()
        # Per‑agent state containers
        self.hidden_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_thoughts: Dict[str, torch.Tensor] = {}
        self.last_emotions: Dict[str, torch.Tensor] = {}
        self.last_workspaces: Dict[str, torch.Tensor] = {}
        # Emergency fallback
        self.nan_detected: bool = False
        # Track last valid action per agent for graceful fallback
        self.last_actions: Dict[str, int] = {}
        self.fallback_policy = self._create_fallback_policy()

    def _ensure_agent_state(self, aid: str) -> None:
        """Initialise state entries for a new agent."""
        if aid not in self.hidden_states:
            h0 = torch.zeros(1, 1, self.brain.hidden_dim, device=self.device)
            c0 = torch.zeros(1, 1, self.brain.hidden_dim, device=self.device)
            self.hidden_states[aid] = (h0, c0)
            self.last_thoughts[aid] = torch.zeros(1, self.brain.thought_dim, device=self.device)
            self.last_emotions[aid] = torch.zeros(1, self.brain.emotion_dim, device=self.device)
            self.last_workspaces[aid] = torch.zeros(1, self.brain.workspace_dim, device=self.device)

    def step(
        self,
        agent_ids: List[str],
        pixels: torch.Tensor,
        scalars: torch.Tensor,
        goals: torch.Tensor,
    ) -> List[int]:
        """
        Execute one timestep for a batch of agents.

        This method performs inference for each agent, updates their
        persistent state, handles memory expansion when pressure is high
        and falls back to a scripted policy if NaNs are detected in the
        output.  It mirrors the behaviour described in Phase 3 of the
        HTDE plan.

        Parameters
        ----------
        agent_ids : List[str]
            A list of unique identifiers for each agent in the batch.
        pixels : torch.Tensor
            Batch of image observations, shape [B, 3, 128, 128].
        scalars : torch.Tensor
            Batch of scalar state vectors, shape [B, scalar_dim].
        goals : torch.Tensor
            Batch of goal vectors, shape [B, goal_dim].

        Returns
        -------
        List[int]
            The chosen action index for each agent.
        """
        B = len(agent_ids)
        pixels = pixels.to(self.device)
        scalars = scalars.to(self.device)
        goals = goals.to(self.device)

        h_list, c_list, t_list, e_list, w_list = [], [], [], [], []
        for aid in agent_ids:
            self._ensure_agent_state(aid)
            h, c = self.hidden_states[aid]
            h_list.append(h)
            c_list.append(c)
            t_list.append(self.last_thoughts[aid])
            e_list.append(self.last_emotions[aid])
            w_list.append(self.last_workspaces[aid])

        h_batch = torch.cat(h_list, dim=1)
        c_batch = torch.cat(c_list, dim=1)
        prev_thought = torch.cat(t_list, dim=0)
        prev_emotion = torch.cat(e_list, dim=0)
        prev_workspace = torch.cat(w_list, dim=0)

        try:
            with torch.no_grad():
                out = self.brain(
                    pixels,
                    scalars,
                    goals,
                    prev_thought,
                    emotion=prev_emotion,
                    workspace=prev_workspace,
                    hidden=(h_batch, c_batch),
                )
            # Check for NaNs
            if torch.isnan(out["action_logits"]).any():
                raise ValueError("NaN detected in action logits")
            self.nan_detected = False
            # Memory expansion
            # out['pressure'] is a 0‑D tensor
            if out["pressure"].item() > 0.85:
                if self.brain.pmm.expand():
                    print(
                        f"*** CYBORG MIND MEMORY EXPANSION -> {self.brain.pmm.mem_slots} SLOTS ***"
                    )
            actions_tensor = torch.argmax(out["action_logits"], dim=-1)
            actions = actions_tensor.cpu().numpy().tolist()
        except Exception as e:
            # Emergency fallback: repeat last valid action or no‑op (0)
            print(f"[EMERGENCY] Brain failure: {e}")
            self.nan_detected = True
            actions = [self.last_actions.get(aid, 0) for aid in agent_ids]
            # Do not update brain state on failure
            out = None

        # Update per‑agent state if inference succeeded
        if not self.nan_detected:
            for i, aid in enumerate(agent_ids):
                self.hidden_states[aid] = (
                    out["hidden_h"][:, i : i + 1],
                    out["hidden_c"][:, i : i + 1],
                )
                self.last_thoughts[aid] = out["thought"][i : i + 1]
                self.last_emotions[aid] = out["emotion"][i : i + 1]
                self.last_workspaces[aid] = out["workspace"][i : i + 1]
                # Record the chosen action for fallback
                self.last_actions[aid] = actions[i]

        return actions

    def _create_fallback_policy(self):
        """
        Create a simple fallback policy that returns random actions.

        This policy is used when the brain outputs invalid values
        (e.g., NaNs) to keep the simulation running instead of crashing.
        """
        return lambda: torch.randint(0, self.brain.num_actions, (1,)).item()
"""
Production brain wrapper for Cyborg Mind v2.0.

This module provides a thin wrapper around ``BrainCyborgMind`` that
performs JIT scripting and dynamic quantisation to improve
inference speed and reduce memory usage.  Use this class in a
production environment where the model is loaded once and run
repeatedly.  The wrapper exposes a simple ``forward`` method that
expects the same inputs as the original brain but omits optional
state arguments for ease of use.
"""

from __future__ import annotations

import torch
from torch import nn

from ..capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind


class ProductionBrain(nn.Module):
    """
    Optimised inference‑only version of ``BrainCyborgMind``.

    The underlying brain is loaded from a state dictionary, then
    scripted via ``torch.jit.script`` and quantised dynamically on
    linear layers.  Once constructed, the brain is set to ``eval``
    mode and no gradients are tracked.  Only the forward pass is
    exposed.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda") -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Load trained brain
        brain = BrainCyborgMind().to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        brain.load_state_dict(state, strict=False)
        brain.eval()
        # JIT script
        scripted = torch.jit.script(brain)
        # Dynamic quantisation on linear layers
        self.brain = torch.quantization.quantize_dynamic(
            scripted, {nn.Linear}, dtype=torch.qint8
        )
        self.brain.eval()

    @torch.no_grad()
    def forward(self, pixels: torch.Tensor, scalars: torch.Tensor, goals: torch.Tensor) -> any:
        """
        Forward inference through the optimised brain.

        Parameters
        ----------
        pixels : torch.Tensor
            Image observations, shape [B, 3, 128, 128].  Should be on
            the same device as the brain.
        scalars : torch.Tensor
            Scalar state inputs, shape [B, scalar_dim].
        goals : torch.Tensor
            Goal vectors, shape [B, goal_dim].

        Returns
        -------
        any
            Output object with attributes ``action_logits``, ``value``, etc.
        """
        B = pixels.size(0)
        # Initialise zero internal state for one‑off inference
        thoughts = torch.zeros(B, self.brain.thought_dim, device=self.device)
        emotions = torch.zeros(B, self.brain.emotion_dim, device=self.device)
        workspaces = torch.zeros(B, self.brain.workspace_dim, device=self.device)
        return self.brain(
            pixels,
            scalars,
            goals,
            thoughts,
            emotion=emotions,
            workspace=workspaces,
        )
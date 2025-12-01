"""
Real Teacher module for Cyborg Mind v2.0

This teacher uses a frozen CLIP vision encoder (ViT‑B/32) to extract
visual features from 128×128 RGB inputs and trains simple linear
heads to predict discrete actions and a scalar value.  The teacher
ignores scalar state inputs for now but reserves a parameter for
future fusion.  It is intended as a distillation source for the
student brain during Phase 1 of the HTDE roadmap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPModel


class RealTeacher(nn.Module):
    """
    CLIP‑based teacher network.

    The teacher comprises a frozen vision encoder from OpenAI's
    ``clip-vit-base-patch32`` model and two trainable heads: one
    for predicting action logits and one for predicting a scalar
    value.  Currently the scalar state inputs are unused but kept
    for compatibility.

    Parameters
    ----------
    ckpt_path : str, optional
        Path to a checkpoint containing pretrained head weights.
        If provided, ``action_head`` and ``value_head`` will be
        loaded from the checkpoint.  The vision encoder is always
        loaded from HuggingFace.
    device : str, optional
        Device on which to initialise the network.  Defaults to
        ``"cuda"``.
    num_actions : int, optional
        Number of discrete actions.  Defaults to 20.
    """

    def __init__(
        self,
        ckpt_path: str | None = None,
        device: str = "cuda",
        num_actions: int = 20,
        scalar_dim: int = 20,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load CLIP vision encoder (ViT‑B/32) and freeze parameters
        base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = base_model.vision_model.to(self.device)
        self.vision_encoder.eval()
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Dimensions
        embed_dim = self.vision_encoder.config.hidden_size  # typically 768
        self.num_actions = num_actions
        self.scalar_dim = scalar_dim

        # Fusion of visual embedding and scalar state
        # We first reduce the concatenated feature to hidden_dim with a small MLP
        self.scalar_fusion = nn.Sequential(
            nn.Linear(embed_dim + scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(self.device)
        # Heads operating on fused representation
        self.action_head = nn.Linear(hidden_dim, num_actions).to(self.device)
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)

        # Optionally load head weights
        if ckpt_path is not None:
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                # For backwards compatibility we support older checkpoint keys.
                if "state_dict" in ckpt:
                    # New format: full state dict
                    self.load_state_dict(ckpt["state_dict"], strict=False)
                    print(f"[RealTeacher] Loaded from state_dict in {ckpt_path}")
                else:
                    # Old format: individual components
                    if "action_head" in ckpt:
                        self.action_head.load_state_dict(ckpt["action_head"])
                    if "value_head" in ckpt:
                        self.value_head.load_state_dict(ckpt["value_head"])
                    if "scalar_fusion" in ckpt:
                        self.scalar_fusion.load_state_dict(ckpt["scalar_fusion"])
                    print(f"[RealTeacher] Loaded head weights from {ckpt_path}")
            except Exception as e:
                print(f"[RealTeacher] Warning: could not load {ckpt_path}: {e}")
        else:
            # No checkpoint provided - initialize with random weights (for training)
            print("[RealTeacher] No checkpoint provided, using random initialization")

    @torch.no_grad()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode raw images into CLIP embeddings.

        Parameters
        ----------
        pixels : torch.Tensor
            Input tensor of shape [B, 3, 128, 128].  Values may be
            either float in [0,255] or already normalised.

        Returns
        -------
        torch.Tensor
            Feature embeddings of shape [B, embed_dim].
        """
        if pixels.dtype != torch.float32:
            pixels = pixels.float()
        # Bring inputs into [0,1] range if necessary
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        # Apply CLIP standard normalization: subtract mean and divide by std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        pixels = (pixels.to(self.device) - mean) / std
        outputs = self.vision_encoder(pixels)
        features = outputs.pooler_output  # shape [B, embed_dim]
        return features

    @torch.no_grad()
    def predict(self, pixels: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict action logits and state value for a batch of observations.

        Parameters
        ----------
        pixels : torch.Tensor
            Image observations, shape [B, 3, 128, 128].
        scalars : torch.Tensor
            Scalar state vectors, shape [B, S].  (Currently unused.)

        Returns
        -------
        tuple of torch.Tensor
            (action_logits [B, num_actions], value [B, 1])
        """
        features = self.encode_pixels(pixels)
        # Ensure scalars are float and on the same device
        if scalars.dtype != torch.float32:
            scalars = scalars.float()
        scalars = scalars.to(self.device)
        # Concatenate visual and scalar features
        fused_input = torch.cat([features, scalars], dim=1)
        fused = self.scalar_fusion(fused_input)
        logits = self.action_head(fused)
        value = self.value_head(fused)
        return logits, value
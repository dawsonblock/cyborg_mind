"""PPO Agent with PMM memory integration."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from cyborg_rl.config import Config
from cyborg_rl.models.mamba_gru import MambaGRUEncoder
from cyborg_rl.models.policy import DiscretePolicy, ContinuousPolicy
from cyborg_rl.models.value import ValueHead
from cyborg_rl.memory.pmm import PredictiveMemoryModule
from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


class PPOAgent(nn.Module):
    """
    PPO Agent with Mamba/GRU encoder and PMM memory.

    Architecture:
        obs -> Encoder -> latent -> PMM -> memory_augmented -> Policy/Value

    The PMM receives the latent state and outputs a memory-augmented state
    that is then consumed by both the policy and value heads.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Config,
        is_discrete: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize the PPO agent.

        Args:
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            config: Configuration object.
            is_discrete: Whether action space is discrete.
            device: Torch device.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.device = device
        self.config = config

        # Encoder: obs -> latent
        self.encoder = MambaGRUEncoder(
            input_dim=obs_dim,
            hidden_dim=config.model.hidden_dim,
            latent_dim=config.model.latent_dim,
            num_gru_layers=config.model.num_gru_layers,
            use_mamba=config.model.use_mamba,
            mamba_d_state=config.model.mamba_d_state,
            mamba_d_conv=config.model.mamba_d_conv,
            mamba_expand=config.model.mamba_expand,
            dropout=config.model.dropout,
        )

        # PMM: latent -> memory_augmented_state
        self.pmm = PredictiveMemoryModule(
            input_dim=config.model.latent_dim,
            memory_size=config.memory.memory_size,
            memory_dim=config.memory.memory_dim,
            num_read_heads=config.memory.num_read_heads,
            num_write_heads=config.memory.num_write_heads,
            sharp_factor=config.memory.sharp_factor,
        )

        # Policy head: memory_augmented -> action
        if is_discrete:
            self.policy = DiscretePolicy(
                input_dim=config.model.latent_dim,
                action_dim=action_dim,
                hidden_dim=config.model.hidden_dim,
            )
        else:
            self.policy = ContinuousPolicy(
                input_dim=config.model.latent_dim,
                action_dim=action_dim,
                hidden_dim=config.model.hidden_dim,
            )

        # Value head: memory_augmented -> value
        self.value = ValueHead(
            input_dim=config.model.latent_dim,
            hidden_dim=config.model.hidden_dim,
        )

        self.to(device)
        logger.info(
            f"Initialized PPOAgent: obs_dim={obs_dim}, action_dim={action_dim}, "
            f"discrete={is_discrete}, params={self.count_parameters():,}"
        )

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Initialize recurrent and memory state.

        Args:
            batch_size: Batch size.

        Returns:
            Dict with 'hidden' and 'memory' tensors.
        """
        return {
            "hidden": self.encoder.init_hidden(batch_size, self.device),
            "memory": self.pmm.init_memory(batch_size, self.device),
        }

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """
        Full forward pass.

        Args:
            obs: Observation tensor [B, D] or [B, T, D].
            state: Optional dict with 'hidden' and 'memory'.
            deterministic: If True, use deterministic action selection.

        Returns:
            Tuple of:
                - action [B] or [B, A]
                - log_prob [B]
                - value [B]
                - new_state dict
                - info dict with memory stats
        """
        batch_size = obs.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        # Encode observation
        latent, new_hidden = self.encoder(obs, state["hidden"])

        # PMM read/write cycle
        memory_augmented, new_memory, pmm_info = self.pmm(latent, state["memory"])

        # Policy and value from memory-augmented state
        action, log_prob = self.policy.sample(memory_augmented, deterministic=deterministic)
        value = self.value(memory_augmented).squeeze(-1)

        new_state = {
            "hidden": new_hidden,
            "memory": new_memory,
        }

        info = {
            "pmm_info": pmm_info,
            "latent": latent,
            "memory_augmented": memory_augmented,
        }

        return action, log_prob, value, new_state, info

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: Observation tensor [B, D].
            actions: Actions to evaluate [B] or [B, A].
            state: Optional recurrent state.

        Returns:
            Tuple of (log_prob, entropy, value, new_state).
        """
        batch_size = obs.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        # Encode
        latent, new_hidden = self.encoder(obs, state["hidden"])

        # PMM
        memory_augmented, new_memory, _ = self.pmm(latent, state["memory"])

        # Evaluate
        log_prob, entropy = self.policy.evaluate(memory_augmented, actions)
        value = self.value(memory_augmented).squeeze(-1)

        new_state = {
            "hidden": new_hidden,
            "memory": new_memory,
        }

        return log_prob, entropy, value, new_state

    def get_value(
        self,
        obs: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get state value only.

        Args:
            obs: Observation tensor [B, D].
            state: Optional recurrent state.

        Returns:
            Tuple of (value [B], new_state).
        """
        batch_size = obs.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        latent, new_hidden = self.encoder(obs, state["hidden"])
        memory_augmented, new_memory, _ = self.pmm(latent, state["memory"])
        value = self.value(memory_augmented).squeeze(-1)

        new_state = {
            "hidden": new_hidden,
            "memory": new_memory,
        }

        return value, new_state

    def save(self, path: str) -> None:
        """
        Save agent checkpoint.

        Args:
            path: Save path.
        """
        torch.save({
            "state_dict": self.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "is_discrete": self.is_discrete,
            "config": self.config,
        }, path)
        logger.info(f"Saved agent to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> "PPOAgent":
        """
        Load agent from checkpoint.

        Args:
            path: Checkpoint path.
            device: Target device.

        Returns:
            PPOAgent: Loaded agent.
        """
        checkpoint = torch.load(path, map_location=device)
        agent = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            config=checkpoint["config"],
            is_discrete=checkpoint["is_discrete"],
            device=device,
        )
        agent.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Loaded agent from {path}")
        return agent

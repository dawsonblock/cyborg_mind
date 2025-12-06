"""PPO Agent with PMM memory integration and AMP support."""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.amp import autocast

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
        
    Features:
    - Mixed Precision (AMP) safe
    - NaN guarding
    - PMM Memory integration
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Config,
        is_discrete: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.device = device
        self.config = config

        # Encoder: obs -> latent
        # Select encoder based on config.model.encoder_type
        encoder_type = getattr(config.model, "encoder_type", "mamba_gru")
        
        if encoder_type == "gru":
            # Pure GRU encoder
            from cyborg_rl.models.mamba_gru import GRUEncoder
            self.encoder = GRUEncoder(
                input_dim=obs_dim,
                hidden_dim=config.model.hidden_dim,
                latent_dim=config.model.latent_dim,
                num_layers=config.model.num_gru_layers,
                dropout=config.model.dropout,
            )
            logger.info("Using GRU encoder")
        elif encoder_type in ["mamba", "mamba_gru"]:
            # Mamba+GRU hybrid encoder
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
            logger.info(f"Using Mamba+GRU hybrid encoder (use_mamba={config.model.use_mamba})")
        elif encoder_type == "pseudo_mamba":
            # Pseudo-Mamba (Pure PyTorch implementation)
            from cyborg_rl.models.pseudo_mamba import PseudoMambaEncoder
            self.encoder = PseudoMambaEncoder(
                input_dim=obs_dim,
                hidden_dim=config.model.hidden_dim,
                latent_dim=config.model.latent_dim,
                num_layers=getattr(config.model, "num_mamba_layers", 2),
                dropout=config.model.dropout,
            )
            logger.info("Using Pseudo-Mamba encoder (Pure PyTorch)")
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # PMM: latent -> memory_augmented_state
        self.pmm = PredictiveMemoryModule(
            input_dim=config.model.latent_dim,
            memory_size=config.memory.memory_size,
            memory_dim=config.memory.memory_dim,
            num_read_heads=config.memory.num_read_heads,
            num_write_heads=config.memory.num_write_heads,
            sharp_factor=config.memory.sharp_factor,
        )

        # Policy head
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

        # Value head
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
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
        Forward pass with AMP support.
        """
        batch_size = obs.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        # Ensure state is on correct device
        state["hidden"] = state["hidden"].to(self.device)
        state["memory"] = state["memory"].to(self.device)

        # Encode observation
        # Mamba/GRU might need float32 for stability, but we let autocast handle it
        latent, new_hidden = self.encoder(obs, state["hidden"])

        # NaN Guard
        if torch.isnan(new_hidden).any():
            logger.warning("NaN detected in hidden state. Resetting hidden state.")
            new_hidden = self.encoder.init_hidden(batch_size, self.device)

        # PMM read/write cycle
        memory_augmented, new_memory, pmm_info = self.pmm(latent, state["memory"])

        # Policy and value
        action, log_prob = self.policy.sample(memory_augmented, deterministic=deterministic)
        value = self.value(memory_augmented).squeeze(-1)

        # Monitoring stats
        with torch.no_grad():
            latent_norm = latent.norm(dim=-1).mean().item()
            memory_norm = memory_augmented.norm(dim=-1).mean().item()

        new_state = {
            "hidden": new_hidden,
            "memory": new_memory,
        }

        info = {
            "pmm_info": pmm_info,
            "latent_norm": latent_norm,
            "memory_norm": memory_norm,
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
        """
        batch_size = obs.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        latent, new_hidden = self.encoder(obs, state["hidden"])
        memory_augmented, new_memory, _ = self.pmm(latent, state["memory"])

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

    def forward_step(
        self,
        obs_t: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """
        Single recurrent step for sequence processing.

        Used by both act() and forward_sequence() for consistent state updates.

        Args:
            obs_t: Observation [B, obs_dim]
            state: Recurrent state dict with 'hidden' and 'memory'

        Returns:
            logits: Action logits [B, action_dim] (discrete only)
            value: State value [B]
            new_state: Updated recurrent state
            pmm_info: PMM diagnostics
        """
        # Encode observation
        latent, new_hidden = self.encoder(obs_t, state["hidden"])

        # PMM read/write cycle
        memory_augmented, new_memory, pmm_info = self.pmm(latent, state["memory"])

        # Get logits and value
        if self.is_discrete:
            logits = self.policy.forward(memory_augmented)
        else:
            # For continuous policies, return mean as "logits" (less clean but works)
            mean, log_std = self.policy.forward(memory_augmented)
            logits = mean

        value = self.value(memory_augmented).squeeze(-1)

        new_state = {
            "hidden": new_hidden,
            "memory": new_memory,
        }

        return logits, value, new_state, pmm_info

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        init_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full-sequence forward pass for MemoryPPOTrainer.

        Loops over time dimension in Python, maintaining recurrent state
        across timesteps. Enables full BPTT for memory-based agents.

        Args:
            obs_seq: Observation sequence [T, B, obs_dim]
            init_state: Initial recurrent state (or None to initialize zeros)

        Returns:
            logits_seq: Action logits [T, B, action_dim] (discrete only)
            values_seq: State values [T, B]
        """
        T, B, obs_dim = obs_seq.shape

        # Initialize state if not provided
        if init_state is None:
            state = self.init_state(batch_size=B)
        else:
            state = init_state

        logits_list = []
        values_list = []

        # Loop over time
        for t in range(T):
            obs_t = obs_seq[t]  # [B, obs_dim]

            logits_t, value_t, state, _ = self.forward_step(obs_t, state)

            logits_list.append(logits_t)
            values_list.append(value_t)

        # Stack into [T, B, ...] tensors
        logits_seq = torch.stack(logits_list, dim=0)  # [T, B, action_dim]
        values_seq = torch.stack(values_list, dim=0)  # [T, B]

        return logits_seq, values_seq

    def save(self, path: str) -> None:
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
        checkpoint = torch.load(path, map_location=device)
        agent = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            config=checkpoint["config"],
            is_discrete=checkpoint["is_discrete"],
            device=device,
        )
        agent.load_state_dict(checkpoint["state_dict"])
        agent.to(device)
        logger.info(f"Loaded agent from {path}")
        return agent

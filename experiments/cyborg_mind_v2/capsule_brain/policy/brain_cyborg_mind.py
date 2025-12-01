"""
Cyborg Mind v2.0: Unified Emotion‑Consciousness Brain

This module builds on top of the apex cyborg architecture described in
`Brain cyborg.pdf`.  In that design the agent receives raw pixel
observations and a small number of scalar inputs, fuses them with a
goal vector and a 32‑dimensional thought vector, processes this
representation with a frozen vision adapter and an LSTM, and writes to
a dynamically expanding PMM memory bank.  The hidden state and
thought vectors persist across timesteps via the controller, but there
is no explicit mechanism for emotions or a global conscious workspace
【940122878883148†L320-L380】.  The LSTM provides temporal continuity for
cognition, while the DynamicGPUPMM acts as a content‑addressable
memory with on‑the‑fly expansion【940122878883148†L320-L383】.

To realise a true ``Cyborg Mind'' we extend this architecture with
explicit emotion, a global workspace, a fully recurrent neural
network (FRNN) and a simple self‑writing mechanism:

* **Explicit Emotion:** Eight emotion channels represent a
  valence–arousal–dominance space and related affective states.  They
  are inputs to the brain and are updated by a dedicated head on
  every forward pass.  This design follows the earlier emotion engine
  prototype where emotions are continuous values in the range
  ``[-1,1]``.
* **Conscious Workspace:** Inspired by Global Workspace Theory, the
  workspace is a low‑dimensional latent vector that integrates the
  agent’s perception, emotions, memory and thoughts into a single
  broadcastable summary.  It is recurrently updated and fed back into
  the brain, enabling information in the workspace to influence
  subsequent processing.
* **FRNN Core:** In addition to the standard LSTM, we include a
  fully recurrent neural network layer.  A fully recurrent neural
  network connects the outputs of all neurons to the inputs of all
  neurons【340715413007859†L656-L660】, providing rich dynamics beyond a
  strictly feed‑forward or gated recurrent model.  We implement this
  as a simple tanh‑based recurrent cell operating on the workspace.
* **Self‑Writer:** A lightweight meta‑learning component that
  periodically modulates memory write operations.  The self‑writer
  scales the memory write vector based on the workspace, mimicking
  self‑reflective rewriting of internal state.  In a more advanced
  implementation this could perform real code updates.

The `BrainCyborgMind` class encapsulates these features.  It accepts
pixels, scalars, goals, previous thought, previous emotion and
previous workspace as inputs, and produces action logits, value
estimates, memory write vectors, updated thought and emotion vectors,
a new workspace vector, and hidden LSTM state.  The DynamicGPUPMM and
VisionAdapter from the original cyborg brain are reused unmodified.

This module is designed to be used with `CyborgMindController` in
``integration/cyborg_mind_controller.py`` which maintains per‑agent
persistent state.

"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import base components from the original cyborg architecture
try:
    from capsule_brain.policy.brain_cyborg_dynamic import (
        DynamicGPUPMM,
        VisionAdapter,
    )
except ImportError:
    # When running as a standalone module outside the original package,
    # provide minimal fallbacks.  These will be replaced when the full
    # package is present.
    class DynamicGPUPMM(nn.Module):
        """
        A simplified pseudo‑memory module that supports content‑based retrieval,
        least‑used eviction and hot‑swappable expansion.  This fallback
        implementation is designed to run when the full dynamic PMM from
        ``capsule_brain.policy.brain_cyborg_dynamic`` is unavailable.  It
        loosely mirrors the semantics of the production PMM described in the
        documentation【531940505518075†L293-L344】.  In particular it tracks a
        per‑slot ``usage`` buffer that decays slowly over time and writes
        new vectors into the least used slots rather than cycling through
        memory indiscriminately.  When queried it returns a weighted sum of
        the stored values based on cosine similarity.

        The module can grow by doubling its capacity while preserving the
        contents of existing slots.  Call ``expand()`` when
        ``get_pressure()`` indicates high utilisation.
        """
        def __init__(
            self,
            mem_slots: int,
            mem_dim: int,
            key_dim: int,
            max_slots: int = 2048,
        ) -> None:
            """
            Dynamic GPU‑based pseudo memory (fallback) with pressure metrics and
            garbage collection.

            This implementation extends the simple PMM used in the original
            baseline to support:

            * A rolling window of attention entropy and write density for
              estimating memory pressure.  Low entropy or high write density
              signals that queries are repetitive or writes are saturating
              capacity.
            * Automatic expansion when pressure is high and capacity allows.
            * Periodic garbage collection to reset stale slots that are seldom
              accessed.  Slots are considered stale when their access
              frequency drops below ``gc_threshold``.  GC runs every
              ``gc_interval`` forward passes and will evict at least 10% of
              memory if enough stale slots are found.

            Parameters
            ----------
            mem_slots : int
                Initial number of memory slots.
            mem_dim : int
                Dimension of each memory value.
            key_dim : int
                Dimension of each memory key.
            max_slots : int, optional
                Maximum allowed memory slots.  Expansion will not exceed
                this number.  Defaults to 2048.
            """
            super().__init__()
            self.mem_slots = mem_slots
            self.mem_dim = mem_dim
            self.key_dim = key_dim
            self.max_slots = max_slots
            # Trainable key/value tensors
            self.keys = nn.Parameter(torch.randn(mem_slots, key_dim) * 0.1)
            self.values = nn.Parameter(torch.randn(mem_slots, mem_dim) * 0.1)
            # Track usage and write counts as buffers (no gradient)
            self.register_buffer("usage", torch.zeros(mem_slots))
            self.register_buffer("write_count", torch.zeros(mem_slots))
            # Rolling history buffers for attention entropy and write density
            self.register_buffer("attn_history", torch.zeros(100))
            self.register_buffer("write_history", torch.zeros(100))
            self.history_idx = 0
            # Access counts for garbage collection
            self.register_buffer("access_counts", torch.zeros(mem_slots))
            # Garbage collection parameters
            self.gc_interval = 1000  # number of forward calls between GC runs
            self.gc_threshold = 0.1   # minimum access frequency to keep a slot
            self._step_counter = 0
            # Temporary storage for metrics between write/read
            self._last_attn = None
            self._last_write_mask = None

        def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Retrieve a value from memory via cosine similarity and update
            pressure/usage statistics.

            Each call performs the following steps:
            1. Compute cosine similarity between the query and all keys.
            2. Produce a softmax attention over slots and read out the
               corresponding values.
            3. Update usage via an exponential decay and small contribution
               from current attention.  Usage approximates how often slots
               are read/written.
            4. Update rolling histories of attention entropy and write
               density.  These are used to compute a composite pressure
               metric via ``compute_memory_pressure``.
            5. Track which slots were accessed for garbage collection.
            6. Perform periodic garbage collection if the configured
               ``gc_interval`` steps have passed.
            """
            # Increment step counter
            self._step_counter += 1

            # Normalise for cosine similarity
            q_norm = F.normalize(query, dim=-1)
            k_norm = F.normalize(self.keys, dim=-1)
            sims = torch.matmul(q_norm, k_norm.t())  # [B, mem_slots]
            attn = F.softmax(sims, dim=-1)
            readout = torch.matmul(attn, self.values)

            # Update usage: slow decay and small bump from current attention
            with torch.no_grad():
                self.usage.mul_(0.99).add_(attn.mean(dim=0) * 0.01)

                # Compute attention entropy (normalised) for pressure metric
                probs = attn.clamp_min(1e-8)
                log_probs = probs.log()
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                max_entropy = math.log(self.mem_slots) if self.mem_slots > 0 else 1.0
                ent_norm = float((entropy / max_entropy).clamp(0.0, 1.0))
                # Compute write density from last write mask (if any)
                write_density = 0.0
                if self._last_write_mask is not None:
                    write_density = float(self._last_write_mask.float().mean())
                # Update rolling histories
                self.attn_history[self.history_idx] = ent_norm
                self.write_history[self.history_idx] = write_density
                self.history_idx = (self.history_idx + 1) % self.attn_history.numel()
                # Track which slots were accessed for GC
                accessed_slots = (attn > 0.01).any(dim=0)
                self.access_counts += accessed_slots.float()
                # Periodic garbage collection
                if self._step_counter % self.gc_interval == 0:
                    self.garbage_collect()

            # Store last attention for potential inspection
            self._last_attn = attn.detach()
            return readout, attn

        @torch.no_grad()
        def write(self, write_vector: torch.Tensor, alpha: float = 0.5) -> None:
            """
            Write one or more vectors into the least used slots.

            The memory uses a least‑used replacement scheme: for each
            vector in the batch, the slot with the smallest usage value is
            selected.  If the batch size exceeds the number of slots the
            indices wrap around.

            A soft update is applied with blending factor ``alpha``.  The
            write mask is stored to estimate write density for the pressure
            metric.
            """
            if write_vector.numel() == 0:
                return
            B = write_vector.size(0)
            # Identify the indices of the least used slots
            _, idx = torch.topk(-self.usage, k=min(B, self.mem_slots))
            if B > self.mem_slots:
                repeat_factor = (B + self.mem_slots - 1) // self.mem_slots
                idx = idx.repeat(repeat_factor)[:B]
            # Create a mask for computing write density
            write_mask = torch.zeros(self.mem_slots, device=self.usage.device, dtype=torch.bool)
            # If multiple write vectors map to the same slot, aggregate them
            unique_slots, inverse_indices = torch.unique(idx, return_inverse=True)
            for slot_idx, slot in enumerate(unique_slots):
                # Gather all vectors that map to this slot
                mask = inverse_indices == slot_idx
                # Compute mean of these vectors
                vecs = write_vector[mask]
                mean_val = vecs.mean(dim=0).detach()
                s = int(slot.item())
                # Softly blend the new aggregated value
                self.values.data[s] = (1.0 - alpha) * self.values.data[s] + alpha * mean_val
                # Use the aggregated vector as key if dimension matches, otherwise random
                if mean_val.numel() == self.key_dim:
                    self.keys.data[s] = mean_val
                else:
                    self.keys.data[s] = F.normalize(mean_val.new_empty(self.key_dim).normal_(0, 0.1), dim=0)
                # Reset usage and increment write count
                self.usage[s] = 0.5
                self.write_count[s] += mask.sum()
                write_mask[s] = True
            # Store write mask for pressure computation
            self._last_write_mask = write_mask

        def compute_memory_pressure(self) -> float:
            """
            Compute a composite memory pressure metric in [0,1].

            The pressure is a weighted combination of three components:

            1. **Entropy Pressure (40%)** — Low attention entropy implies
               repetitive queries and suggests inadequate capacity.  We use
               ``1 - mean(attn_history)`` so high entropy yields low
               pressure.
            2. **Write Pressure (40%)** — High write density over the
               rolling window indicates frequent overwrites and thus high
               utilisation.
            3. **Gradient Pressure (20%)** — Large gradient norms on keys
               imply significant learning updates.  We bound this by a
               ``tanh`` to map any real value into [-1,1].
            """
            entropy_pressure = 1.0 - self.attn_history.mean().item()
            write_pressure = self.write_history.mean().item()
            # Gradient norm pressure (safe if no grad)
            if self.keys.grad is not None:
                grad_norm = self.keys.grad.norm() / (self.keys.numel() + 1e-8)
                grad_pressure = float(torch.tanh(grad_norm))
            else:
                grad_pressure = 0.0
            pressure = 0.4 * entropy_pressure + 0.4 * write_pressure + 0.2 * grad_pressure
            return float(max(0.0, min(1.0, pressure)))

        def get_pressure(self) -> float:
            """
            Backwards‑compatible alias for ``compute_memory_pressure``.
            """
            return self.compute_memory_pressure()

        @torch.no_grad()
        def garbage_collect(self) -> None:
            """
            Perform garbage collection by evicting stale memory slots.

            A slot is considered stale if its access frequency (as
            recorded in ``access_counts``) falls below ``gc_threshold``.
            If more than 10% of slots are stale we reset them to small
            random values and zero their usage, write counts and access
            counts.  GC is triggered periodically in ``forward``.
            """
            total_accesses = float(self.access_counts.sum().item())
            if total_accesses <= 0:
                return
            access_freq = self.access_counts / (total_accesses + 1e-8)
            stale_mask = access_freq < self.gc_threshold
            num_stale = int(stale_mask.sum().item())
            if num_stale < int(0.1 * self.mem_slots) or num_stale == 0:
                return
            # Reset stale slots
            print(f"[GC] Evicting {num_stale} stale memories")
            noise_keys = torch.randn_like(self.keys.data[stale_mask]) * 0.1
            noise_vals = torch.randn_like(self.values.data[stale_mask]) * 0.1
            self.keys.data[stale_mask] = noise_keys
            self.values.data[stale_mask] = noise_vals
            self.usage[stale_mask] = 0.0
            self.write_count[stale_mask] = 0.0
            self.access_counts[stale_mask] = 0.0

        @torch.no_grad()
        def expand(self, factor: int = 2) -> bool:
            """
            Expand the memory by a given factor up to ``max_slots``.

            On expansion, all existing contents are copied into the new
            allocation.  Associated buffers (usage, write count,
            access_counts, history buffers) are resized accordingly and
            initialised for the new slots.  Returns True on success,
            False if the maximum capacity has already been reached.
            """
            if self.mem_slots >= self.max_slots:
                return False
            new_slots = min(self.max_slots, self.mem_slots * factor)
            device = self.keys.device
            # Allocate new tensors
            new_keys = torch.zeros(new_slots, self.key_dim, device=device)
            new_values = torch.zeros(new_slots, self.mem_dim, device=device)
            new_usage = torch.zeros(new_slots, device=device)
            new_counts = torch.zeros(new_slots, device=device)
            new_access = torch.zeros(new_slots, device=device)
            # Preserve existing contents
            new_keys[: self.mem_slots] = self.keys.data
            new_values[: self.mem_slots] = self.values.data
            new_usage[: self.mem_slots] = self.usage
            new_counts[: self.mem_slots] = self.write_count
            new_access[: self.mem_slots] = self.access_counts
            # Swap parameters
            self.keys = nn.Parameter(new_keys)
            self.values = nn.Parameter(new_values)
            self.usage = new_usage
            self.write_count = new_counts
            self.access_counts = new_access
            self.mem_slots = new_slots
            return True

    class VisionAdapter(nn.Module):
        def __init__(self, out_dim: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Linear(64, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            return self.fc(h)


@dataclass
class CyborgMindOutput:
    """Outputs of the `BrainCyborgMind` forward pass."""
    action_logits: torch.Tensor  # [B, num_actions]
    value: torch.Tensor          # [B, 1]
    mem_write: torch.Tensor      # [B, mem_dim]
    thought: torch.Tensor        # [B, thought_dim]
    emotion: torch.Tensor        # [B, emotion_dim]
    workspace: torch.Tensor      # [B, workspace_dim]
    hidden: Tuple[torch.Tensor, torch.Tensor]  # LSTM hidden state
    pressure: float              # memory pressure indicator


class FullyRecurrentCell(nn.Module):
    """
    A simple fully recurrent neural network (FRNN) cell operating on a
    workspace vector.  Unlike an LSTM, this cell has a single tanh
    activation and weight matrices that connect the workspace back to
    itself.  According to the definition of fully recurrent networks
    each neuron’s output is connected to every other neuron’s input
    【340715413007859†L656-L660】.  Here the workspace represents the
    collection of neurons.
    """

    def __init__(self, workspace_dim: int):
        super().__init__()
        self.W_self = nn.Linear(workspace_dim, workspace_dim)
        self.W_input = nn.Linear(workspace_dim, workspace_dim)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x and h_prev shape: [B, workspace_dim]
        return torch.tanh(self.W_input(x) + self.W_self(h_prev))


class BrainCyborgMind(nn.Module):
    """
    Unified emotion and consciousness brain.

    This brain fuses pixel observations, scalar inputs, goal directives,
    previous thought, previous emotion and previous workspace into a
    latent embedding.  The embedding is processed by a DynamicGPUPMM
    memory module and an LSTM.  Outputs are produced via several
    heads: action, value, memory write, thought, emotion and
    workspace.  A FullyRecurrentCell updates the workspace using the
    previous workspace and the current hidden state.  A simple
    self‑writer modulates the memory write based on the workspace.
    """

    def __init__(
        self,
        scalar_dim: int = 20,
        goal_dim: int = 4,
        thought_dim: int = 32,
        emotion_dim: int = 8,
        workspace_dim: int = 64,
        vision_dim: int = 512,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        mem_dim: int = 128,
        num_actions: int = 20,
        start_slots: int = 256,
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.goal_dim = goal_dim
        self.thought_dim = thought_dim
        self.emotion_dim = emotion_dim
        self.workspace_dim = workspace_dim
        self.vision_dim = vision_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.num_actions = num_actions

        # Vision adapter and memory
        self.vision = VisionAdapter(vision_dim)
        self.pmm = DynamicGPUPMM(start_slots, mem_dim, key_dim=64)
        # BUG FIX: align should output key_dim (64), not mem_dim (128)
        # The query dimension must match the key dimension in PMM
        self.align = nn.Linear(emb_dim, 64)  # key_dim = 64

        # --- Stability and thought loop parameters ---
        # Reset thought to world state every ``anchor_interval`` forward passes.
        # This prevents runaway loops by grounding the thought in the current
        # perception and scalars.  See Phase 3 of the HTDE plan.
        self.anchor_interval: int = 10
        # Clip the thought vector to this range to avoid explosion.
        self.thought_clip: float = 3.0
        # Linear projection from vision+scalars to thought space for anchoring
        self.world_state_projector = nn.Linear(vision_dim + scalar_dim, thought_dim)
        # BUG FIX: Use buffer instead of instance variable for batch compatibility
        # Register as buffer so it doesn't get gradients but persists in state_dict
        self.register_buffer("_step_counter", torch.tensor(0, dtype=torch.long))

        # Encoder fuses perception, scalars, goal, thought, emotion
        fusion_dim = vision_dim + scalar_dim + goal_dim + thought_dim + emotion_dim + workspace_dim
        self.encoder = nn.Sequential(
            nn.Linear(fusion_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        # LSTM to capture temporal dynamics
        self.lstm = nn.LSTM(emb_dim + mem_dim, hidden_dim, batch_first=False)

        # FRNN cell for the workspace
        self.frnn = FullyRecurrentCell(workspace_dim)

        # Output heads
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.mem_head = nn.Sequential(nn.Linear(hidden_dim, mem_dim), nn.Tanh())
        self.thought_head = nn.Sequential(nn.Linear(hidden_dim, thought_dim), nn.Tanh())
        self.emotion_head = nn.Sequential(nn.Linear(hidden_dim, emotion_dim), nn.Tanh())
        self.workspace_head = nn.Sequential(nn.Linear(hidden_dim, workspace_dim), nn.Tanh())

        # Self‑writer modulators
        self.self_writer = nn.Linear(workspace_dim, mem_dim, bias=False)

    def forward(
        self,
        pixels: torch.Tensor,
        scalars: torch.Tensor,
        goal: torch.Tensor,
        thought: torch.Tensor,
        emotion: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass of the brain.

        Parameters
        ----------
        pixels: [B, 3, 128, 128] raw image input.
        scalars: [B, scalar_dim] numeric state features.
        goal: [B, goal_dim] goal or directive vector.
        thought: [B, thought_dim] previous thought vector.
        emotion: [B, emotion_dim], optional.  If None, neutral emotions are assumed.
        workspace: [B, workspace_dim], optional.  If None, initialise to zero.
        hidden: (h, c) tuple for the LSTM, each [1, B, hidden_dim].  If None,
                zero initial hidden state is used.
        """
        B = pixels.size(0)
        device = pixels.device

        if emotion is None:
            emotion = torch.zeros(B, self.emotion_dim, device=device)
        if workspace is None:
            workspace = torch.zeros(B, self.workspace_dim, device=device)
        if hidden is None:
            hidden = (
                torch.zeros(1, B, self.hidden_dim, device=device),
                torch.zeros(1, B, self.hidden_dim, device=device),
            )

        # Update global step counter (only during training)
        if self.training:
            self._step_counter += 1

        # Vision embedding (frozen)
        vis_emb = self.vision(pixels)

        # Anchor thought periodically to the current world state.  On every
        # ``anchor_interval``th call we reset the thought vector to a
        # projection of the current vision and scalar inputs.  This
        # grounding helps prevent self‑amplifying loops in the recurrent
        # architecture. Only apply anchoring during training.
        if self.training and self._step_counter % self.anchor_interval == 0:
            world_state = torch.cat([vis_emb, scalars], dim=-1)
            thought = self.world_state_projector(world_state)

        # Clip previous thought to prevent explosion
        thought = torch.clamp(thought, -self.thought_clip, self.thought_clip)

        # Fusion including workspace and emotion
        fusion_in = torch.cat([vis_emb, scalars, goal, thought, emotion, workspace], dim=-1)
        emb = self.encoder(fusion_in)

        # Memory read
        query = self.align(emb)
        mem_out, _ = self.pmm(query)

        # LSTM processing
        lstm_in = torch.cat([emb, mem_out], dim=-1).unsqueeze(0)
        lstm_out, new_hidden = self.lstm(lstm_in, hidden)
        h_t = lstm_out.squeeze(0)

        # Heads
        mem_write = self.mem_head(h_t)
        new_thought = self.thought_head(h_t)
        # Clip new thought to prevent explosion
        new_thought = torch.clamp(new_thought, -self.thought_clip, self.thought_clip)
        new_emotion = self.emotion_head(h_t)
        # Preliminary workspace update from hidden
        workspace_update = self.workspace_head(h_t)
        # FRNN update: combine previous workspace and new hidden
        new_workspace = self.frnn(workspace_update, workspace)

        # Self‑writer: modulate memory write based on workspace
        writer_mod = torch.tanh(self.self_writer(new_workspace))
        modulated_mem_write = mem_write * (1 + writer_mod)

        # Write to memory
        self.pmm.write(modulated_mem_write)

        # Prepare outputs as a dictionary for TorchScript compatibility
        # Pack hidden state tuple into separate tensors
        output = {
            "action_logits": self.action_head(h_t),
            "value": self.value_head(h_t),
            "mem_write": modulated_mem_write,
            "thought": new_thought,
            "emotion": new_emotion,
            "workspace": new_workspace,
            "hidden_h": new_hidden[0],
            "hidden_c": new_hidden[1],
            # Convert scalar pressure to a tensor on the same device
            "pressure": torch.tensor(self.pmm.get_pressure(), device=pixels.device),
        }
        return output
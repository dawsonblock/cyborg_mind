"""Unit tests for model forward passes."""

import pytest
import torch

from cyborg_rl.config import Config
from cyborg_rl.models.mamba_gru import MambaGRUEncoder
from cyborg_rl.models.policy import DiscretePolicy, ContinuousPolicy
from cyborg_rl.models.value import ValueHead
from cyborg_rl.agents.ppo_agent import PPOAgent


class TestMambaGRUEncoder:
    """Tests for MambaGRUEncoder."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def encoder(self, device: torch.device) -> MambaGRUEncoder:
        """Create encoder instance."""
        return MambaGRUEncoder(
            input_dim=4,
            hidden_dim=64,
            latent_dim=32,
            num_gru_layers=1,
            use_mamba=False,
        ).to(device)

    def test_forward_2d(self, encoder: MambaGRUEncoder, device: torch.device) -> None:
        """Test forward with 2D input."""
        batch_size = 8
        x = torch.randn(batch_size, 4, device=device)
        
        latent, hidden = encoder(x)
        
        assert latent.shape == (batch_size, 32)
        assert hidden.shape == (1, batch_size, 64)

    def test_forward_3d(self, encoder: MambaGRUEncoder, device: torch.device) -> None:
        """Test forward with 3D input (sequence)."""
        batch_size = 8
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 4, device=device)
        
        latent, hidden = encoder(x)
        
        assert latent.shape == (batch_size, 32)

    def test_forward_with_hidden(self, encoder: MambaGRUEncoder, device: torch.device) -> None:
        """Test forward with initial hidden state."""
        batch_size = 8
        x = torch.randn(batch_size, 4, device=device)
        hidden = encoder.init_hidden(batch_size, device)
        
        latent, new_hidden = encoder(x, hidden)
        
        assert latent.shape == (batch_size, 32)
        assert new_hidden.shape == hidden.shape


class TestPolicyHeads:
    """Tests for policy heads."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    def test_discrete_policy(self, device: torch.device) -> None:
        """Test discrete policy."""
        policy = DiscretePolicy(input_dim=32, action_dim=4).to(device)
        x = torch.randn(8, 32, device=device)
        
        action, log_prob = policy.sample(x)
        
        assert action.shape == (8,)
        assert log_prob.shape == (8,)
        assert (action >= 0).all() and (action < 4).all()

    def test_discrete_evaluate(self, device: torch.device) -> None:
        """Test discrete policy evaluation."""
        policy = DiscretePolicy(input_dim=32, action_dim=4).to(device)
        x = torch.randn(8, 32, device=device)
        actions = torch.randint(0, 4, (8,), device=device)
        
        log_prob, entropy = policy.evaluate(x, actions)
        
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)
        assert (entropy >= 0).all()

    def test_continuous_policy(self, device: torch.device) -> None:
        """Test continuous policy."""
        policy = ContinuousPolicy(input_dim=32, action_dim=2).to(device)
        x = torch.randn(8, 32, device=device)
        
        action, log_prob = policy.sample(x)
        
        assert action.shape == (8, 2)
        assert log_prob.shape == (8,)

    def test_continuous_evaluate(self, device: torch.device) -> None:
        """Test continuous policy evaluation."""
        policy = ContinuousPolicy(input_dim=32, action_dim=2).to(device)
        x = torch.randn(8, 32, device=device)
        actions = torch.randn(8, 2, device=device)
        
        log_prob, entropy = policy.evaluate(x, actions)
        
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)


class TestValueHead:
    """Tests for value head."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    def test_forward(self, device: torch.device) -> None:
        """Test value head forward."""
        value_head = ValueHead(input_dim=32).to(device)
        x = torch.randn(8, 32, device=device)
        
        value = value_head(x)
        
        assert value.shape == (8, 1)


class TestPPOAgent:
    """Tests for PPO agent."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def config(self) -> Config:
        """Create test config."""
        config = Config()
        config.model.hidden_dim = 64
        config.model.latent_dim = 32
        config.model.num_gru_layers = 1
        config.memory.memory_size = 16
        config.memory.memory_dim = 16
        config.memory.num_read_heads = 2
        return config

    @pytest.fixture
    def agent(self, config: Config, device: torch.device) -> PPOAgent:
        """Create agent instance."""
        return PPOAgent(
            obs_dim=4,
            action_dim=2,
            config=config,
            is_discrete=True,
            device=device,
        )

    def test_forward(self, agent: PPOAgent, device: torch.device) -> None:
        """Test agent forward pass."""
        batch_size = 8
        obs = torch.randn(batch_size, 4, device=device)
        
        action, log_prob, value, state, info = agent(obs)
        
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size,)
        assert "hidden" in state
        assert "memory" in state

    def test_evaluate_actions(self, agent: PPOAgent, device: torch.device) -> None:
        """Test action evaluation."""
        batch_size = 8
        obs = torch.randn(batch_size, 4, device=device)
        actions = torch.randint(0, 2, (batch_size,), device=device)
        
        log_prob, entropy, value, state = agent.evaluate_actions(obs, actions)
        
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size,)

    def test_get_value(self, agent: PPOAgent, device: torch.device) -> None:
        """Test value estimation."""
        batch_size = 8
        obs = torch.randn(batch_size, 4, device=device)
        
        value, state = agent.get_value(obs)
        
        assert value.shape == (batch_size,)

    def test_state_persistence(self, agent: PPOAgent, device: torch.device) -> None:
        """Test state persistence across steps."""
        obs = torch.randn(1, 4, device=device)
        state = agent.init_state(1)
        
        _, _, _, state1, _ = agent(obs, state)
        _, _, _, state2, _ = agent(obs, state1)
        
        # States should be different
        assert not torch.allclose(state1["hidden"], state2["hidden"])

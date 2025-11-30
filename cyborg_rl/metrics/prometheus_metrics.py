"""Prometheus metrics for PPO trainer monitoring."""

from typing import Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from cyborg_rl.utils.logging import get_logger

logger = get_logger(__name__)


class PrometheusMetrics:
    """
    Prometheus metrics for monitoring RL training.

    Exposes:
    - reward_per_episode
    - loss_policy
    - loss_value
    - advantage_mean
    - episode_length
    - memory_saturation (PMM)
    """

    def __init__(
        self,
        port: int = 8000,
        enable_server: bool = True,
    ) -> None:
        """
        Initialize Prometheus metrics.

        Args:
            port: HTTP port for metrics endpoint.
            enable_server: Whether to start the HTTP server.
        """
        self.port = port

        # Episode metrics
        self.episode_reward = Gauge(
            "cyborg_rl_episode_reward",
            "Reward per episode",
        )
        self.reward_total = Counter(
            "cyborg_rl_reward_total",
            "Total accumulated reward",
        )
        self.episode_length = Gauge(
            "cyborg_rl_episode_length",
            "Length of episode in steps",
        )
        self.episode_count = Counter(
            "cyborg_rl_episode_count",
            "Total number of episodes",
        )

        # Training metrics
        self.policy_loss = Gauge(
            "cyborg_rl_loss_policy",
            "Policy loss",
        )
        self.value_loss = Gauge(
            "cyborg_rl_loss_value",
            "Value loss",
        )
        self.advantage_mean = Gauge(
            "cyborg_rl_advantage_mean",
            "Mean advantage",
        )

        # Memory metrics
        self.memory_saturation = Gauge(
            "cyborg_rl_pmm_memory_saturation",
            "PMM memory saturation (fraction of used slots)",
        )
        self.pmm_write_ops = Counter(
            "cyborg_rl_pmm_write_ops_total",
            "Total PMM write operations",
        )
        self.pmm_read_ops = Counter(
            "cyborg_rl_pmm_read_ops_total",
            "Total PMM read operations",
        )

        # Internal State metrics
        self.emotion_norm = Gauge(
            "cyborg_rl_emotion_norm",
            "Norm of emotion/latent vector",
        )
        self.thought_norm = Gauge(
            "cyborg_rl_thought_norm",
            "Norm of thought/memory vector",
        )

        # Histograms
        self.reward_histogram = Histogram(
            "cyborg_rl_reward_histogram",
            "Distribution of episode rewards",
            buckets=[0, 50, 100, 150, 200, 300, 400, 500],
        )

        if enable_server:
            try:
                start_http_server(port)
                logger.info(f"Prometheus metrics server started on port {port}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")

    def record_episode(self, reward: float, length: int) -> None:
        """
        Record episode metrics.

        Args:
            reward: Total episode reward.
            length: Episode length in steps.
        """
        self.episode_reward.set(reward)
        self.reward_total.inc(reward)
        self.episode_length.set(length)
        self.episode_count.inc()
        self.reward_histogram.observe(reward)

    def record_losses(self, policy_loss: float, value_loss: float) -> None:
        """
        Record training losses.

        Args:
            policy_loss: Policy loss value.
            value_loss: Value loss value.
        """
        self.policy_loss.set(policy_loss)
        self.value_loss.set(value_loss)

    def record_advantage(self, advantage: float) -> None:
        """
        Record advantage statistics.

        Args:
            advantage: Mean advantage value.
        """
        self.advantage_mean.set(advantage)

    def record_memory_stats(self, saturation: float) -> None:
        """
        Record PMM memory statistics.

        Args:
            saturation: Memory saturation (0-1).
        """
        self.memory_saturation.set(saturation)

    def record_pmm_ops(self, reads: int = 1, writes: int = 1) -> None:
        """Record PMM operations."""
        self.pmm_read_ops.inc(reads)
        self.pmm_write_ops.inc(writes)

    def record_internal_state(self, emotion: float, thought: float) -> None:
        """Record internal state norms."""
        self.emotion_norm.set(emotion)
        self.thought_norm.set(thought)

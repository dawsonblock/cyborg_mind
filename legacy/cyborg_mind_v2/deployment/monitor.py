"""
Prometheus monitoring utilities for Cyborg Mind v2.0.

This module defines counters, histograms and gauges for instrumenting
a running controller.  Use these metrics in combination with
``prometheus_client.start_http_server`` to expose them on a given
port.  Example usage:

```
from prometheus_client import start_http_server
from experiments.cyborg_mind_v2.deployment.monitor import inference_count, inference_latency, memory_pressure, active_agents

start_http_server(9090)

while True:
    start = time.time()
    actions = controller.step(agent_ids, pixels, scalars, goals)
    duration = time.time() - start
    inference_latency.observe(duration)
    inference_count.inc()
    memory_pressure.set(controller.brain.pmm.get_pressure())
    active_agents.set(len(agent_ids))
```
```"""

from prometheus_client import Counter, Histogram, Gauge

inference_count = Counter('brain_inferences_total', 'Total inference calls')
inference_latency = Histogram('brain_latency_seconds', 'Inference latency in seconds')
memory_pressure = Gauge('brain_memory_pressure', 'Memory pressure (0-1)')
active_agents = Gauge('brain_active_agents', 'Number of active agents')
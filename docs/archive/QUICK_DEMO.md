# Cyborg Mind v2.8 - Golden Path Demo

This guide takes you from zero to a running brain in 60 seconds.

## 1. Start the Brain

Run the full stack with GPU support and monitoring:

```bash
docker-compose --profile gpu up -d trainer-gpu monitoring
```

## 2. Verify Health

Check if the brain is online:

```bash
curl localhost:8000/health
# {"status": "healthy", "device": "cuda", "agents_active": 0}
```

## 3. Run Demo Client

We provide a python script that acts as a body for the brain, sending random observations and printing the brain's thoughts.

```bash
python scripts/demo_client.py
```

Output:
```text
[Step 1] Action: 0 | Value: 0.12 | Pressure: 0.05 | Thought: Memory pressure: 0.05
[Step 2] Action: 1 | Value: 0.15 | Pressure: 0.08 | Thought: Memory pressure: 0.08
...
```

## 4. View Dashboards

- **Grafana**: [http://localhost:3000](http://localhost:3000) (admin/cyborgmind)
  - View "CyborgMind RL Training Dashboard"
  - Watch "PMM Memory Saturation" in real-time.

## 5. Manual Interaction

You can manually step the brain via curl:

```bash
curl -X POST localhost:8000/step \
  -H "Authorization: Bearer cyborg-secret-v2" \
  -H "Content-Type: application/json" \
  -d '{
    "observation": [0.1, 0.2, 0.3, 0.4],
    "agent_id": "manual_tester"
  }'
```

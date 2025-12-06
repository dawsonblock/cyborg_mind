#!/usr/bin/env python3
"""
Demo client for CyborgMind v2.8 API.
Simulates an environment interacting with the brain.
"""

import time
import random
import requests
import sys

API_URL = "http://localhost:8000"
TOKEN = os.environ.get("CYBORG_AUTH_TOKEN", "")  # Set via environment variable
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def main():
    print(f"Connecting to brain at {API_URL}...")
    
    # 1. Health Check
    try:
        resp = requests.get(f"{API_URL}/health")
        resp.raise_for_status()
        print(f"Brain Online: {resp.json()}")
    except Exception as e:
        print(f"Could not connect to brain: {e}")
        sys.exit(1)

    # 2. Reset Agent
    agent_id = f"demo_agent_{random.randint(1000, 9999)}"
    requests.post(f"{API_URL}/reset", headers=HEADERS, params={"agent_id": agent_id})
    print(f"Spawned agent: {agent_id}")

    # 3. Interaction Loop
    print("\nStarting interaction loop (Ctrl+C to stop)...")
    try:
        for step in range(1, 1000):
            # Simulate observation (e.g., CartPole 4 floats)
            obs = [random.uniform(-1, 1) for _ in range(4)]
            
            payload = {
                "observation": obs,
                "agent_id": agent_id,
                "deterministic": False
            }
            
            start = time.time()
            resp = requests.post(f"{API_URL}/step", headers=HEADERS, json=payload)
            latency = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"[Step {step}] "
                      f"Act: {data['action']} | "
                      f"Val: {data['value']:.2f} | "
                      f"Press: {data['pressure']:.2f} | "
                      f"Lat: {latency:.1f}ms")
            else:
                print(f"Error: {resp.text}")
            
            time.sleep(0.1) # Simulate environment tick
            
    except KeyboardInterrupt:
        print("\nDisconnecting...")

if __name__ == "__main__":
    main()

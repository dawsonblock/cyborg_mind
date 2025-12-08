#!/usr/bin/env python3
"""
Live WebSocket Visualizer - Real-time training metrics streaming.

Streams:
- embeddings
- PMM writes
- recurrent state norms
- action logits

Usage:
    # Start server
    python -m cyborg_rl.diagnostics.live_visualizer --port 8765
    
    # In training code
    from cyborg_rl.diagnostics.live_visualizer import Visualizer
    viz = Visualizer("ws://localhost:8765")
    viz.emit("embedding", embedding_data)
"""

from typing import Any, Dict, List, Optional, Union
import json
import asyncio
import threading
import time
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np

try:
    import websockets
    from websockets.server import serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("websockets not installed. Run: pip install websockets")


@dataclass
class TrainingSnapshot:
    """Single training step snapshot."""
    
    step: int
    timestamp: float
    
    # Embeddings (reduced for transmission)
    embedding_norm: float = 0.0
    embedding_mean: float = 0.0
    embedding_std: float = 0.0
    
    # Memory state
    memory_write_strength: float = 0.0
    memory_read_entropy: float = 0.0
    memory_usage: float = 0.0
    
    # Recurrent state
    hidden_norm: float = 0.0
    hidden_mean: float = 0.0
    
    # Policy
    action_entropy: float = 0.0
    top_action: int = 0
    action_probs: List[float] = None
    
    # Training
    loss: float = 0.0
    reward: float = 0.0
    value: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy types
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


class VisualizerServer:
    """WebSocket server for streaming training metrics."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        max_clients: int = 10,
    ):
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets library not available")
        
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.clients: set = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: deque = deque(maxlen=100)
    
    async def _handler(self, websocket, path):
        """Handle new WebSocket connection."""
        if len(self.clients) >= self.max_clients:
            await websocket.close(1013, "Max clients reached")
            return
        
        self.clients.add(websocket)
        print(f"📡 Client connected ({len(self.clients)} total)")
        
        try:
            async for message in websocket:
                # Handle incoming messages (e.g., configuration)
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"📡 Client disconnected ({len(self.clients)} total)")
    
    async def _broadcast(self, message: str) -> None:
        """Broadcast message to all clients."""
        if not self.clients:
            return
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        self.clients -= disconnected
    
    async def _run_server(self) -> None:
        """Main server loop."""
        async with serve(self._handler, self.host, self.port):
            print(f"🌐 Visualizer server running on ws://{self.host}:{self.port}")
            
            while self._running:
                # Broadcast queued messages
                while self._queue:
                    message = self._queue.popleft()
                    await self._broadcast(message)
                
                await asyncio.sleep(0.05)  # 20 FPS max
    
    def _run_loop(self) -> None:
        """Run event loop in thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run_server())
    
    def start(self) -> None:
        """Start server in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop server."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        print("🌐 Visualizer server stopped")
    
    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Queue data for broadcast."""
        message = json.dumps({
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        })
        self._queue.append(message)
    
    def emit_snapshot(self, snapshot: TrainingSnapshot) -> None:
        """Queue training snapshot for broadcast."""
        self.emit("snapshot", snapshot.to_dict())


class VisualizerClient:
    """WebSocket client for receiving training metrics."""
    
    def __init__(self, url: str = "ws://localhost:8765"):
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets library not available")
        
        self.url = url
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, List[callable]] = {}
        self._latest: Dict[str, Any] = {}
    
    def on(self, event_type: str, callback: callable) -> None:
        """Register callback for event type."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    async def _receive_loop(self) -> None:
        """Receive loop."""
        async with websockets.connect(self.url) as ws:
            while self._running:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(message)
                    event_type = data.get("type", "unknown")
                    self._latest[event_type] = data
                    
                    # Trigger callbacks
                    for callback in self._callbacks.get(event_type, []):
                        callback(data)
                except asyncio.TimeoutError:
                    pass
                except websockets.exceptions.ConnectionClosed:
                    break
    
    def _run_loop(self) -> None:
        """Run event loop in thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._receive_loop())
    
    def start(self) -> None:
        """Start client in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop client."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    @property
    def latest(self) -> Dict[str, Any]:
        """Get latest received data."""
        return self._latest.copy()


# Convenience functions
def create_server(host: str = "localhost", port: int = 8765) -> VisualizerServer:
    """Create and return a visualizer server."""
    return VisualizerServer(host=host, port=port)


def create_client(url: str = "ws://localhost:8765") -> VisualizerClient:
    """Create and return a visualizer client."""
    return VisualizerClient(url=url)


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Visualizer Server")
    parser.add_argument("--host", default="localhost", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    args = parser.parse_args()
    
    server = create_server(host=args.host, port=args.port)
    server.start()
    
    print("Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()

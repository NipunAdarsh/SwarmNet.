"""
SwarmNet — Swarm Node Agent

Represents a single edge compute node in the swarm.  Each node:
- Auto-detects hardware capabilities (CPU cores, NPU, memory)
- Registers with the swarm registry
- Sends heartbeats every 2 seconds
- Reports queue depth and latency
- Processes inference requests forwarded by the registry

For single-machine demos, multiple SwarmNode instances run as threads
within the same server process, each with unique device IDs.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import threading
import time
import uuid
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .registry import SwarmRegistry

logger = logging.getLogger("swarm.node")

# Known NPU device names for different hardware
_NPU_NAMES = ["Qualcomm Hexagon", "AMD XDNA", "Intel Meteor Lake NPU", "Apple Neural Engine"]
_MACHINE_NAMES = [
    "edge-alpha", "edge-bravo", "edge-charlie", "edge-delta", "edge-echo",
    "edge-foxtrot", "edge-golf", "edge-hotel", "edge-india", "edge-juliet",
]


def _detect_hardware() -> dict:
    """Detect real hardware capabilities of the current machine."""
    import psutil  # soft dependency
    cores = os.cpu_count() or 4
    try:
        mem_mb = psutil.virtual_memory().total // (1024 * 1024)
    except Exception:
        mem_mb = 8192

    # Check for NPU via ONNX Runtime providers
    npu_available = False
    accelerators = []
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "QNNExecutionProvider" in available:
            npu_available = True
            accelerators.append("QNNExecutionProvider")
        if "VitisAIExecutionProvider" in available:
            npu_available = True
            accelerators.append("VitisAIExecutionProvider")
        if "DmlExecutionProvider" in available:
            accelerators.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available:
            accelerators.append("CUDAExecutionProvider")
    except ImportError:
        pass

    return {
        "cpu_cores": cores,
        "npu_available": npu_available,
        "gpu_available": "CUDAExecutionProvider" in accelerators or "DmlExecutionProvider" in accelerators,
        "memory_mb": mem_mb,
        "available_accelerators": accelerators,
    }


class SwarmNode:
    """A single edge compute node in the swarm.

    Parameters
    ----------
    registry : SwarmRegistry
        The central registry to register with and heartbeat to.
    device_id : str, optional
        Node identifier.  Auto-generated if not provided.
    ip_address : str
        Advertised IP address.
    simulated : bool
        If True, simulates variable load patterns for demo purposes.
    """

    def __init__(
        self,
        registry: "SwarmRegistry",
        device_id: Optional[str] = None,
        ip_address: str = "127.0.0.1",
        port: int = 8000,
        simulated: bool = False,
        npu_available: bool = False,
    ) -> None:
        self.device_id = device_id or f"node-{uuid.uuid4().hex[:8]}"
        self.ip_address = ip_address
        self.port = port
        self._registry = registry
        self._simulated = simulated
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Detect real hardware or use simulated values
        if simulated:
            self.cpu_cores = random.choice([4, 6, 8, 12])
            self.npu_available = npu_available
            self.gpu_available = False
            self.memory_mb = random.choice([8192, 16384, 32768])
            self.accelerators = ["QNNExecutionProvider"] if npu_available else ["CPUExecutionProvider"]
        else:
            hw = _detect_hardware()
            self.cpu_cores = hw["cpu_cores"]
            self.npu_available = hw["npu_available"]
            self.gpu_available = hw["gpu_available"]
            self.memory_mb = hw["memory_mb"]
            self.accelerators = hw["available_accelerators"]

        # Dynamic state
        self._queue_size = 0
        self._avg_latency_ms = 0.0
        self._inferences_run = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Register with the registry and begin heartbeating."""
        from .models import NodeInfo

        info = NodeInfo(
            device_id=self.device_id,
            ip_address=self.ip_address,
            port=self.port,
            cpu_cores=self.cpu_cores,
            npu_available=self.npu_available,
            gpu_available=self.gpu_available,
            memory_mb=self.memory_mb,
            available_accelerators=self.accelerators,
        )
        self._registry.register_node(info)

        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True,
            name=f"heartbeat-{self.device_id}",
        )
        self._heartbeat_thread.start()
        logger.info("Node %s started (NPU=%s)", self.device_id, self.npu_available)

    def stop(self) -> None:
        """Stop heartbeating."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        logger.info("Node %s stopped", self.device_id)

    # ------------------------------------------------------------------
    # Inference simulation
    # ------------------------------------------------------------------

    def process_inference(self) -> dict:
        """Simulate processing an inference request.

        In a real deployment, this would forward to the local ONNX Runtime
        session.  For the demo, simulates realistic latency and queue behavior.
        """
        with self._lock:
            self._queue_size += 1

        # Simulate NPU inference latency
        if self.npu_available:
            latency = random.uniform(6.0, 14.0)  # ms
        else:
            latency = random.uniform(18.0, 35.0)  # ms

        time.sleep(latency / 1000.0)

        with self._lock:
            self._queue_size = max(0, self._queue_size - 1)
            self._inferences_run += 1
            # Rolling average
            alpha = 0.1
            self._avg_latency_ms = (
                alpha * latency + (1 - alpha) * self._avg_latency_ms
                if self._avg_latency_ms > 0 else latency
            )

        return {
            "node_id": self.device_id,
            "latency_ms": round(latency, 3),
            "npu_used": self.npu_available,
        }

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Send heartbeat every 2 seconds."""
        while self._running:
            with self._lock:
                qs = self._queue_size
                lat = self._avg_latency_ms
                inf = self._inferences_run

            # Simulate varying load for demo
            if self._simulated:
                qs = random.randint(0, 8)
                lat = random.uniform(5.0, 20.0) if self.npu_available else random.uniform(18.0, 40.0)

            self._registry.heartbeat(
                device_id=self.device_id,
                queue_size=qs,
                avg_latency_ms=round(lat, 2),
                inferences_run=inf,
            )
            time.sleep(2.0)

        logger.info("Heartbeat loop ended for %s", self.device_id)

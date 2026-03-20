"""
SwarmNet — Swarm Registry

Thread-safe registry that tracks node health, manages queue-aware routing,
and detects node failures via heartbeat timeouts.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .models import NodeInfo, NodeStatus, SwarmMetrics

logger = logging.getLogger("swarm.registry")

# Node is considered offline if no heartbeat for this many seconds
HEARTBEAT_TIMEOUT_S = 6.0

# How often the monitor thread checks for stale nodes
MONITOR_INTERVAL_S = 2.0


class SwarmRegistry:
    """Central registry for the edge NPU swarm.

    Responsibilities:
    - Register / deregister nodes
    - Process heartbeats
    - Route requests to the best available node (queue-aware)
    - Monitor node health in a background thread
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeInfo] = {}
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

        # Model registry
        self._models: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background heartbeat monitor."""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._heartbeat_monitor, daemon=True, name="registry-monitor"
        )
        self._monitor_thread.start()
        logger.info("Swarm registry started — heartbeat monitor active")

    def stop(self) -> None:
        """Stop the background monitor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Swarm registry stopped")

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def register_node(self, info: NodeInfo) -> None:
        """Add or update a node in the registry."""
        with self._lock:
            info.last_heartbeat = time.time()
            info.registered_at = info.registered_at or time.time()
            info.status = NodeStatus.ONLINE
            self._nodes[info.device_id] = info
        logger.info(
            "Node registered: %s (%s) — NPU=%s, cores=%d, mem=%dMB",
            info.device_id, info.ip_address, info.npu_available,
            info.cpu_cores, info.memory_mb,
        )

    def remove_node(self, device_id: str) -> bool:
        """Mark a node as offline."""
        with self._lock:
            if device_id in self._nodes:
                self._nodes[device_id].status = NodeStatus.OFFLINE
                logger.info("Node removed: %s", device_id)
                return True
        return False

    def disable_node(self, device_id: str) -> bool:
        """Administratively disable a node (simulates failure)."""
        with self._lock:
            if device_id in self._nodes:
                self._nodes[device_id].status = NodeStatus.DISABLED
                logger.warning("Node DISABLED (simulated failure): %s", device_id)
                return True
        return False

    def enable_node(self, device_id: str) -> bool:
        """Re-enable a disabled node."""
        with self._lock:
            if device_id in self._nodes:
                node = self._nodes[device_id]
                node.status = NodeStatus.ONLINE
                node.last_heartbeat = time.time()
                logger.info("Node RE-ENABLED: %s", device_id)
                return True
        return False

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(self, device_id: str, queue_size: int = 0,
                  avg_latency_ms: float = 0.0, inferences_run: int = 0) -> bool:
        """Process a heartbeat from a node."""
        with self._lock:
            node = self._nodes.get(device_id)
            if node is None:
                return False
            if node.status == NodeStatus.DISABLED:
                return False  # Don't accept heartbeats for disabled nodes
            node.last_heartbeat = time.time()
            node.queue_size = queue_size
            node.avg_latency_ms = avg_latency_ms
            node.inferences_run = inferences_run
            # Update status based on queue depth
            if queue_size > 10:
                node.status = NodeStatus.HIGH_LOAD
            else:
                node.status = NodeStatus.ONLINE
        return True

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def select_best_node(self) -> Optional[NodeInfo]:
        """Select the best node using queue-aware routing.

        Primary: lowest queue depth among online nodes.
        Fallback: lowest average latency.
        """
        with self._lock:
            online = [
                n for n in self._nodes.values()
                if n.status in (NodeStatus.ONLINE, NodeStatus.HIGH_LOAD)
            ]
            if not online:
                return None

            # Primary: lowest queue depth
            min_queue = min(n.queue_size for n in online)
            lowest_queue_nodes = [n for n in online if n.queue_size == min_queue]

            if len(lowest_queue_nodes) == 1:
                return lowest_queue_nodes[0]

            # Fallback: lowest latency among tied nodes
            return min(lowest_queue_nodes, key=lambda n: n.avg_latency_ms)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_nodes(self) -> list[NodeInfo]:
        """Return all registered nodes."""
        with self._lock:
            return list(self._nodes.values())

    def get_node(self, device_id: str) -> Optional[NodeInfo]:
        """Return a specific node by ID."""
        with self._lock:
            return self._nodes.get(device_id)

    def get_online_count(self) -> int:
        """Count of online or high-load nodes."""
        with self._lock:
            return sum(
                1 for n in self._nodes.values()
                if n.status in (NodeStatus.ONLINE, NodeStatus.HIGH_LOAD)
            )

    def get_swarm_metrics(self) -> SwarmMetrics:
        """Compute aggregate swarm metrics."""
        with self._lock:
            nodes = list(self._nodes.values())

        online = [n for n in nodes if n.status in (NodeStatus.ONLINE, NodeStatus.HIGH_LOAD)]
        npu_count = sum(1 for n in online if n.npu_available)

        total_throughput = sum(n.inferences_run for n in online)
        avg_lat = (
            sum(n.avg_latency_ms for n in online) / len(online)
            if online else 0.0
        )

        return SwarmMetrics(
            active_nodes=len(online),
            total_nodes=len(nodes),
            total_throughput=total_throughput,
            avg_latency_ms=round(avg_lat, 2),
            routing_strategy="lowest_queue_depth",
            npu_nodes=npu_count,
            cpu_only_nodes=len(online) - npu_count,
        )

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def register_model(self, version: str, name: str, accuracy: float,
                       status: str = "staged", file_path: str = "") -> None:
        self._models.append({
            "version": version,
            "model_name": name,
            "accuracy": accuracy,
            "status": status,
            "created_at": time.time(),
            "file_path": file_path,
        })

    def get_models(self) -> list[dict]:
        return list(self._models)

    def deploy_best_model(self) -> Optional[dict]:
        """Auto-deploy the highest accuracy model."""
        if not self._models:
            return None
        best = max(self._models, key=lambda m: m["accuracy"])
        for m in self._models:
            m["status"] = "retired" if m["version"] != best["version"] else "deployed"
        return best

    # ------------------------------------------------------------------
    # Background monitor
    # ------------------------------------------------------------------

    def _heartbeat_monitor(self) -> None:
        """Background thread that marks nodes as offline on heartbeat timeout."""
        while self._running:
            time.sleep(MONITOR_INTERVAL_S)
            now = time.time()
            with self._lock:
                for node in self._nodes.values():
                    if node.status == NodeStatus.DISABLED:
                        continue
                    if now - node.last_heartbeat > HEARTBEAT_TIMEOUT_S:
                        if node.status != NodeStatus.OFFLINE:
                            logger.warning(
                                "Node %s heartbeat timeout (%.1fs) — marking OFFLINE",
                                node.device_id, now - node.last_heartbeat,
                            )
                            node.status = NodeStatus.OFFLINE

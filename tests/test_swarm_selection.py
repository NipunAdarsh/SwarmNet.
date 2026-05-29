import pytest
from backend.swarm.registry import SwarmRegistry
from backend.swarm.models import NodeInfo

def test_select_best_node(monkeypatch):
    # Initialise registry without starting background thread (override start method)
    registry = SwarmRegistry()
    # Patch the internal thread start to no‑op
    monkeypatch.setattr(registry, "start", lambda: None)
    registry.start()
    # Create two NodeInfo instances with different queue depths
    node_a = NodeInfo(device_id="node-a", ip_address="127.0.0.1", port=8001, cpu_cores=4, npu_available=True, queue_size=5)
    node_b = NodeInfo(device_id="node-b", ip_address="127.0.0.2", port=8002, cpu_cores=4, npu_available=True, queue_size=1)
    # Register nodes
    registry.register_node(node_a)
    registry.register_node(node_b)
    best = registry.select_best_node()
    assert best.device_id == "node-b", "Node with lower queue depth should be selected"

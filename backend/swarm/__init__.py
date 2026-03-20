"""
SwarmNet — Distributed Edge NPU Swarm Package

Provides node discovery, registry, health monitoring, and queue-aware routing
for a decentralized edge AI inference swarm.
"""

from .models import NodeInfo, HeartbeatPayload, SwarmMetrics
from .registry import SwarmRegistry
from .discovery import DiscoveryBroadcaster, DiscoveryListener
from .node import SwarmNode

__all__ = [
    "NodeInfo",
    "HeartbeatPayload",
    "SwarmMetrics",
    "SwarmRegistry",
    "DiscoveryBroadcaster",
    "DiscoveryListener",
    "SwarmNode",
]

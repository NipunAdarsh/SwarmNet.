"""
Pydantic models for swarm node registration, heartbeat, and metrics.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    HIGH_LOAD = "high_load"
    DISABLED = "disabled"


class NodeInfo(BaseModel):
    """Registration data for a single swarm node."""

    device_id: str = Field(..., description="Unique identifier for this node")
    ip_address: str = Field(..., description="Reachable IP address")
    port: int = Field(default=8000, description="Service port")
    cpu_cores: int = Field(default=1, description="Number of CPU cores")
    npu_available: bool = Field(default=False, description="Whether an NPU is present")
    gpu_available: bool = Field(default=False, description="Whether a GPU is present")
    memory_mb: int = Field(default=0, description="Total memory in MB")
    available_accelerators: list[str] = Field(
        default_factory=list,
        description="List of available accelerators (e.g. QNN, VitisAI, CUDA)",
    )

    # Dynamic fields — updated by heartbeats
    queue_size: int = Field(default=0, description="Current inference queue depth")
    avg_latency_ms: float = Field(default=0.0, description="Rolling average latency")
    inferences_run: int = Field(default=0, description="Total inferences completed")
    status: NodeStatus = Field(default=NodeStatus.ONLINE)
    last_heartbeat: float = Field(default_factory=time.time)
    registered_at: float = Field(default_factory=time.time)


class HeartbeatPayload(BaseModel):
    """Periodic health update from a node."""

    device_id: str
    queue_size: int = 0
    avg_latency_ms: float = 0.0
    inferences_run: int = 0


class SwarmMetrics(BaseModel):
    """Aggregate metrics across the entire swarm."""

    active_nodes: int = 0
    total_nodes: int = 0
    total_throughput: int = 0  # total inferences across all nodes
    avg_latency_ms: float = 0.0
    routing_strategy: str = "lowest_queue_depth"
    npu_nodes: int = 0
    cpu_only_nodes: int = 0


class ModelVersion(BaseModel):
    """A versioned model entry in the model registry."""

    version: str
    model_name: str
    accuracy: float = 0.0
    status: str = "staged"  # staged | deployed | retired
    created_at: float = Field(default_factory=time.time)
    file_path: Optional[str] = None


class EnergyBenchmark(BaseModel):
    """Energy consumption comparison per inference."""

    cpu_watts: float = 3.2
    gpu_watts: float = 2.4
    npu_watts: float = 0.8
    cpu_inference_ms: float = 25.0
    gpu_inference_ms: float = 15.0
    npu_inference_ms: float = 8.0


class CloudComparison(BaseModel):
    """Edge swarm vs cloud inference comparison."""

    cloud_latency_ms: float = 300.0
    edge_latency_ms: float = 15.0
    cloud_cost_per_million: float = 4.50  # USD
    edge_cost_per_million: float = 0.02   # electricity only
    cloud_provider: str = "AWS Lambda + SageMaker"
    edge_provider: str = "SwarmNet NPU Swarm"

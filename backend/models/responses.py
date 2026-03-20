"""
Pydantic response models for SwarmNet API.
"""

from pydantic import BaseModel


class InferenceResponse(BaseModel):
    """Response payload for inference endpoints."""
    task_id: str
    status: str
    result: dict
    processing_time_ms: float
    execution_provider: str

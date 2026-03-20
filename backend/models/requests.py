"""
Pydantic request models for SwarmNet API.
"""

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Request payload for inference endpoints."""
    task_id: str = Field(..., description="Unique identifier for this request")
    data_type: str = Field(
        ...,
        description="Type of the payload, e.g. 'base64_image'",
        examples=["base64_image"],
    )
    payload: str = Field(..., description="Base64-encoded data")


class FrameRequest(BaseModel):
    """Request payload for webcam frame inference."""
    frame: str = Field(..., description="Base64-encoded image frame")


class BenchmarkControlledRequest(BaseModel):
    """Request payload for controlled benchmark."""
    runs: int = Field(default=5, ge=1, le=20, description="Number of full benchmark runs")

"""Admin endpoints for queue/task management."""

import os
from typing import List

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from supabase_client import get_supabase_service_client

router = APIRouter(tags=["admin"])


class SeedTasksRequest(BaseModel):
    """Payload for seeding batches of tasks into the queue."""

    model_name: str
    task_type: str
    data_url: str
    total_batches: int = Field(..., ge=1, le=500)
    project_name: str = "general"


def _verify_admin_secret(admin_secret: str) -> None:
    """Validate admin API key to protect privileged routes."""
    expected = os.getenv("ADMIN_SECRET")
    if not expected:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ADMIN_SECRET is not configured")
    if admin_secret != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin secret")


@router.post("/api/admin/tasks/seed")
def seed_tasks(payload: SeedTasksRequest, x_admin_secret: str = Header(default="")):
    """Insert a requested number of pending tasks into the queue table."""
    try:
        _verify_admin_secret(x_admin_secret)
        supabase = get_supabase_service_client()

        rows: List[dict] = []
        for batch_index in range(payload.total_batches):
            rows.append(
                {
                    "model_name": payload.model_name,
                    "task_type": payload.task_type,
                    "data_url": payload.data_url,
                    "status": "pending",
                    "project_name": payload.project_name,
                    "params": {
                        "batch_index": batch_index,
                    },
                }
            )

        response = supabase.table("tasks").insert(rows).execute()
        return {
            "message": "Tasks seeded",
            "inserted_count": len(response.data or []),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to seed tasks: {str(exc)}") from exc

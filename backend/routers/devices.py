"""Device registration and task execution endpoints for desktop agents."""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from routers.auth import verify_user_token
from services.task_service import TaskService
from services.xp_service import XPService
from supabase_client import get_supabase_service_client

router = APIRouter(tags=["devices"])


class DeviceRegisterRequest(BaseModel):
    """Request payload for registering or refreshing a device."""

    device_id: str = Field(..., min_length=3)
    user_token: str = Field(..., min_length=10)
    os: str
    cpu_name: str
    ram_gb: float = Field(..., ge=0)


class TaskCompleteRequest(BaseModel):
    """Request payload for reporting task completion from an agent."""

    device_id: str
    task_id: str
    result_data: Dict[str, Any]
    duration_seconds: int = Field(..., ge=0)


@router.post("/api/device/register")
def register_device(payload: DeviceRegisterRequest) -> Dict[str, Any]:
    """Register a donor device and update heartbeat metadata on repeated calls."""
    try:
        user = verify_user_token(payload.user_token)
        user_id = user["id"]

        supabase = get_supabase_service_client()
        existing = (
            supabase.table("devices")
            .select("id")
            .eq("device_id", payload.device_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )

        if existing.data:
            update_resp = (
                supabase.table("devices")
                .update(
                    {
                        "os": payload.os,
                        "cpu_name": payload.cpu_name,
                        "ram_gb": payload.ram_gb,
                        "last_seen": datetime.now(timezone.utc).isoformat(),
                    }
                )
                .eq("id", existing.data[0]["id"])
                .execute()
            )
            record = update_resp.data[0] if update_resp.data else None
        else:
            insert_resp = (
                supabase.table("devices")
                .insert(
                    {
                        "user_id": user_id,
                        "device_id": payload.device_id,
                        "os": payload.os,
                        "cpu_name": payload.cpu_name,
                        "ram_gb": payload.ram_gb,
                    }
                )
                .execute()
            )
            record = insert_resp.data[0] if insert_resp.data else None

        return {
            "message": "Device registered",
            "device": record,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Device registration failed: {str(exc)}") from exc


@router.get("/api/tasks/next")
def get_next_task(device_id: str) -> Dict[str, Any]:
    """Poll the tasks queue and claim the next pending task for a specific device."""
    try:
        task_service = TaskService()
        task = task_service.get_next_pending_task(device_id=device_id)
        if not task:
            return {"task": None}
        return task
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch next task: {str(exc)}") from exc


@router.post("/api/tasks/complete")
def complete_task(payload: TaskCompleteRequest) -> Dict[str, Any]:
    """Persist task result, mark task as done, and award XP and badges."""
    try:
        supabase = get_supabase_service_client()

        device_resp = (
            supabase.table("devices")
            .select("user_id")
            .eq("device_id", payload.device_id)
            .limit(1)
            .execute()
        )
        if not device_resp.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")

        user_id = device_resp.data[0]["user_id"]

        (
            supabase.table("task_results")
            .insert(
                {
                    "task_id": payload.task_id,
                    "device_id": payload.device_id,
                    "user_id": user_id,
                    "result_data": payload.result_data,
                    "duration_seconds": payload.duration_seconds,
                }
            )
            .execute()
        )

        task_service = TaskService()
        task_service.mark_task_done(payload.task_id)

        xp_service = XPService()
        xp_result = xp_service.award_for_task_completion(user_id=user_id, duration_seconds=payload.duration_seconds)

        return {
            "message": "Task completion recorded",
            "task_id": payload.task_id,
            "xp": xp_result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to complete task: {str(exc)}") from exc

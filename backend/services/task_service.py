"""Task queue operations using the Supabase `tasks` table as a polling queue."""

from typing import Any, Dict, Optional

from supabase_client import get_supabase_service_client


class TaskService:
    """Service methods for selecting, assigning, and completing queued tasks."""

    def __init__(self) -> None:
        self.supabase = get_supabase_service_client()

    def get_next_pending_task(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Claim the next pending task for a device with a safe compare-and-set update."""
        for _ in range(3):
            pending = (
                self.supabase.table("tasks")
                .select("id, model_name, task_type, data_url, status, project_name, params")
                .eq("status", "pending")
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            )
            if not pending.data:
                return None

            candidate = pending.data[0]
            updated = (
                self.supabase.table("tasks")
                .update(
                    {
                        "status": "in_progress",
                        "assigned_device_id": device_id,
                    }
                )
                .eq("id", candidate["id"])
                .eq("status", "pending")
                .execute()
            )
            if updated.data:
                task = updated.data[0]
                return {
                    "task_id": task["id"],
                    "model_url": task.get("model_name"),
                    "data_url": task.get("data_url"),
                    "params": task.get("params") or {},
                    "project_name": task.get("project_name"),
                    "task_type": task.get("task_type"),
                }
        return None

    def mark_task_done(self, task_id: str) -> None:
        """Mark a task as completed."""
        (
            self.supabase.table("tasks")
            .update({"status": "done"})
            .eq("id", task_id)
            .execute()
        )

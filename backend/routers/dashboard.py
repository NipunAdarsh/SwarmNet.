"""Dashboard and analytics endpoints for frontend usage."""

from fastapi import APIRouter, Depends, HTTPException, status

from routers.auth import get_current_user
from supabase_client import get_supabase_service_client

router = APIRouter(tags=["dashboard"])


@router.get("/api/stats")
def get_stats(current_user=Depends(get_current_user)):
    """Return user contribution stats from XP ledger and task results."""
    try:
        user_id = current_user["id"]
        supabase = get_supabase_service_client()

        xp_rows = (
            supabase.table("xp_ledger")
            .select("xp_amount")
            .eq("user_id", user_id)
            .execute()
        )
        task_rows = (
            supabase.table("task_results")
            .select("duration_seconds")
            .eq("user_id", user_id)
            .execute()
        )
        badge_rows = (
            supabase.table("badges")
            .select("badge_name, earned_at")
            .eq("user_id", user_id)
            .order("earned_at", desc=True)
            .execute()
        )

        total_xp = sum((row.get("xp_amount") or 0) for row in (xp_rows.data or []))
        total_tasks = len(task_rows.data or [])
        total_seconds = sum((row.get("duration_seconds") or 0) for row in (task_rows.data or []))

        return {
            "user_id": user_id,
            "total_xp": total_xp,
            "completed_tasks": total_tasks,
            "total_hours": round(total_seconds / 3600, 2),
            "badges": badge_rows.data or [],
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch stats: {str(exc)}") from exc


@router.get("/api/leaderboard")
def get_leaderboard():
    """Return top 50 users ranked by XP totals."""
    try:
        supabase = get_supabase_service_client()

        xp_rows = supabase.table("xp_ledger").select("user_id, xp_amount").execute()

        totals = {}
        for row in xp_rows.data or []:
            uid = row.get("user_id")
            totals[uid] = totals.get(uid, 0) + (row.get("xp_amount") or 0)

        ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:50]

        profile_ids = [uid for uid, _ in ranked]
        username_map = {}
        if profile_ids:
            profiles = (
                supabase.table("profiles")
                .select("id, username")
                .in_("id", profile_ids)
                .execute()
            )
            username_map = {row["id"]: row.get("username") for row in (profiles.data or [])}

        return {
            "leaderboard": [
                {
                    "rank": idx + 1,
                    "user_id": uid,
                    "username": username_map.get(uid),
                    "total_xp": xp,
                }
                for idx, (uid, xp) in enumerate(ranked)
            ]
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch leaderboard: {str(exc)}") from exc


@router.get("/api/science/progress")
def get_science_progress():
    """Aggregate task completion counts by project for scientific progress tracking."""
    try:
        supabase = get_supabase_service_client()
        task_rows = supabase.table("tasks").select("project_name, status").execute()

        progress = {}
        for row in task_rows.data or []:
            project_name = row.get("project_name") or "unassigned"
            if project_name not in progress:
                progress[project_name] = {
                    "project_name": project_name,
                    "total_tasks": 0,
                    "done_tasks": 0,
                }
            progress[project_name]["total_tasks"] += 1
            if row.get("status") == "done":
                progress[project_name]["done_tasks"] += 1

        items = []
        for value in progress.values():
            total = value["total_tasks"]
            done = value["done_tasks"]
            value["completion_percent"] = round((done / total) * 100, 2) if total else 0
            items.append(value)

        items.sort(key=lambda x: x["completion_percent"], reverse=True)
        return {"projects": items}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch science progress: {str(exc)}") from exc

"""XP and badge award logic for SwarmNet contribution events."""

from datetime import datetime, timedelta, timezone
from typing import Dict

from supabase_client import get_supabase_service_client


class XPService:
    """Encapsulates XP grants and badge evaluation operations."""

    TASK_COMPLETION_XP = 1
    UPTIME_HOURLY_XP = 5
    REFERRAL_XP = 25

    def __init__(self) -> None:
        self.supabase = get_supabase_service_client()

    def award_for_task_completion(self, user_id: str, duration_seconds: int) -> Dict[str, int]:
        """Grant XP for a completed task and any earned uptime bonus, then evaluate badges."""
        xp_total = 0

        xp_total += self._insert_xp(user_id, self.TASK_COMPLETION_XP, "Task completed")

        uptime_hours = max(duration_seconds, 0) // 3600
        if uptime_hours > 0:
            uptime_xp = uptime_hours * self.UPTIME_HOURLY_XP
            xp_total += self._insert_xp(user_id, uptime_xp, f"Uptime bonus: {uptime_hours}h")

        self._evaluate_badges(user_id)

        return {
            "xp_awarded": xp_total,
            "uptime_hours_counted": uptime_hours,
        }

    def award_referral_bonus(self, user_id: str) -> int:
        """Grant referral XP when a user successfully refers another new user."""
        return self._insert_xp(user_id, self.REFERRAL_XP, "Referral bonus")

    def _insert_xp(self, user_id: str, amount: int, reason: str) -> int:
        """Insert one XP ledger row and return inserted amount."""
        (
            self.supabase.table("xp_ledger")
            .insert(
                {
                    "user_id": user_id,
                    "xp_amount": amount,
                    "reason": reason,
                }
            )
            .execute()
        )
        return amount

    def _badge_exists(self, user_id: str, badge_name: str) -> bool:
        """Check whether a badge has already been awarded to prevent duplicates."""
        result = (
            self.supabase.table("badges")
            .select("id")
            .eq("user_id", user_id)
            .eq("badge_name", badge_name)
            .limit(1)
            .execute()
        )
        return bool(result.data)

    def _award_badge_if_missing(self, user_id: str, badge_name: str) -> None:
        """Insert a badge record only if it does not already exist."""
        if self._badge_exists(user_id, badge_name):
            return
        (
            self.supabase.table("badges")
            .insert(
                {
                    "user_id": user_id,
                    "badge_name": badge_name,
                }
            )
            .execute()
        )

    def _evaluate_badges(self, user_id: str) -> None:
        """Evaluate all current badge rules and persist newly earned badges."""
        # First gradient: first successful task completion.
        task_count_resp = (
            self.supabase.table("task_results")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        completed_tasks = task_count_resp.count or 0
        if completed_tasks >= 1:
            self._award_badge_if_missing(user_id, "First gradient")

        # 100h club: total donated duration reaches at least 100 hours.
        durations_resp = (
            self.supabase.table("task_results")
            .select("duration_seconds")
            .eq("user_id", user_id)
            .execute()
        )
        total_seconds = sum((row.get("duration_seconds") or 0) for row in (durations_resp.data or []))
        if total_seconds >= 100 * 3600:
            self._award_badge_if_missing(user_id, "100h club")

        # 24h streak: donated at least 24 hours within the last 24-hour window.
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_resp = (
            self.supabase.table("task_results")
            .select("duration_seconds")
            .eq("user_id", user_id)
            .gte("completed_at", since.isoformat())
            .execute()
        )
        recent_seconds = sum((row.get("duration_seconds") or 0) for row in (recent_resp.data or []))
        if recent_seconds >= 24 * 3600:
            self._award_badge_if_missing(user_id, "24h streak")

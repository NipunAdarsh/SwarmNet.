"""Supabase client initialization helpers."""

import os
from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables from .env when present.
load_dotenv()


class SupabaseConfigError(RuntimeError):
    """Raised when required Supabase environment variables are missing."""


def _require_env(name: str) -> str:
    """Read a required environment variable or fail fast with a clear message."""
    value = os.getenv(name)
    if not value:
        raise SupabaseConfigError(f"Missing required environment variable: {name}")
    return value


def get_supabase_anon_client() -> Client:
    """Create a Supabase client using anon key for user-scoped operations."""
    url = _require_env("SUPABASE_URL")
    anon_key = _require_env("SUPABASE_ANON_KEY")
    return create_client(url, anon_key)


def get_supabase_service_client() -> Client:
    """Create a Supabase client using service role key for privileged operations."""
    url = _require_env("SUPABASE_URL")
    service_key = _require_env("SUPABASE_SERVICE_KEY")
    return create_client(url, service_key)

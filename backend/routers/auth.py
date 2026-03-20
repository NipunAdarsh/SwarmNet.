"""Authentication helpers and auth routes backed by Supabase Auth."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, EmailStr

from supabase_client import get_supabase_anon_client, get_supabase_service_client

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    """Request payload for user registration."""

    email: EmailStr
    password: str
    username: Optional[str] = None


class LoginRequest(BaseModel):
    """Request payload for user login."""

    email: EmailStr
    password: str


def _extract_bearer_token(authorization: Optional[str]) -> str:
    """Extract a JWT token from a Bearer Authorization header."""
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return token


def verify_user_token(token: str) -> Dict[str, Any]:
    """Verify a Supabase JWT and return user payload."""
    try:
        supabase = get_supabase_anon_client()
        response = supabase.auth.get_user(token)
        user = response.user
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
        return {
            "id": user.id,
            "email": user.email,
            "raw_user_meta_data": getattr(user, "user_metadata", {}) or {},
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token verification failed: {str(exc)}") from exc


def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """FastAPI dependency to resolve the authenticated user from bearer token."""
    token = _extract_bearer_token(authorization)
    return verify_user_token(token)


@router.post("/register")
def register_user(payload: RegisterRequest) -> Dict[str, Any]:
    """Create a new Supabase Auth user account."""
    try:
        supabase = get_supabase_anon_client()
        response = supabase.auth.sign_up(
            {
                "email": payload.email,
                "password": payload.password,
                "options": {
                    "data": {
                        "username": payload.username or payload.email.split("@")[0],
                    }
                },
            }
        )
        user = response.user
        if not user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to register user")
        return {
            "message": "Registration successful",
            "user": {
                "id": user.id,
                "email": user.email,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Registration failed: {str(exc)}") from exc


@router.post("/login")
def login_user(payload: LoginRequest) -> Dict[str, Any]:
    """Authenticate a user and return the Supabase JWT access token."""
    try:
        supabase = get_supabase_anon_client()
        response = supabase.auth.sign_in_with_password(
            {
                "email": payload.email,
                "password": payload.password,
            }
        )
        session = response.session
        user = response.user
        if not session or not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid login credentials")
        return {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Login failed: {str(exc)}") from exc


@router.get("/me")
def get_me(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Verify token and return the authenticated user profile information."""
    try:
        supabase = get_supabase_service_client()
        profile_resp = (
            supabase.table("profiles")
            .select("id, username, created_at")
            .eq("id", current_user["id"])
            .limit(1)
            .execute()
        )
        profile = profile_resp.data[0] if profile_resp.data else None
        return {"user": current_user, "profile": profile}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unable to fetch profile: {str(exc)}") from exc

# Load centralized JSON logging configuration
import backend.logging_config  # noqa: F401
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.config import config

# Optional slowapi imports
try:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False

# Frontend directory mapping
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

from backend.routers.inference import lifespan as legacy_lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with legacy_lifespan(app):
        yield

app = FastAPI(
    title="SwarmNet Ultimate",
    description="SwarmNet FastAPI Backend",
    version="2.0.0",
    lifespan=lifespan,
)

@app.on_event("startup")
async def _validate_startup_config():
    # Ensure ADMIN_SECRET is set (the validator already raises on import, but double‑check here for clarity)
    from backend.config import config
    try:
        # Accessing config will trigger the SecurityConfig validator above
        _ = config.security.ADMIN_SECRET
    except RuntimeError as exc:
        import logging, sys
        logging.getLogger("swarm-backend").error(str(exc))
        sys.exit(1)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.origins_list,
    allow_credentials=config.cors.ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _HAS_SLOWAPI:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

from fastapi import HTTPException
from fastapi.responses import JSONResponse
@app.exception_handler(HTTPException)
async def structured_http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)}},
    )

# Health endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0"}

# Register Supabase API route groups
from backend.routers.auth import router as auth_router
from backend.routers.admin import router as admin_router
from backend.routers.dashboard import router as dashboard_router
from backend.routers.devices import router as devices_router
from backend.routers.inference import router as inference_router

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(dashboard_router)
app.include_router(devices_router)
app.include_router(inference_router)

# Mount static directories
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "dist" / "assets"), name="assets")

# Include static frontend routes (instead of mounting the whole directory)
from backend.static_routes import router as static_router
app.include_router(static_router, include_in_schema=False)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.server.HOST,
        port=config.server.PORT,
        reload=config.server.DEBUG,
    )

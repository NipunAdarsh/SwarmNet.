import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from config import config

# Optional slowapi imports
try:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False

# Frontend directory mapping
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

from routers.inference import lifespan as legacy_lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with legacy_lifespan(app):
        yield

app = FastAPI(
    title="SwarmNet Combined Backend API",
    description="Distributed NPU compute orchestration and local NPU Inference",
    version="2.0.0",
    lifespan=lifespan,
)

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
from routers.auth import router as auth_router
from routers.admin import router as admin_router
from routers.dashboard import router as dashboard_router
from routers.devices import router as devices_router

from routers.inference import router as inference_router
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(dashboard_router)
app.include_router(devices_router)
app.include_router(inference_router)

# Mount frontend at the root (must be done last so it doesn't swallow API routes)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.server.HOST,
        port=config.server.PORT,
        reload=config.server.DEBUG,
    )

# Static file routes for SwarmNet frontend

"""Provides the FastAPI router that serves the static HTML pages and the
frontend assets.  The original implementation lived in ``backend.routers.inference``;
we extract it here for clarity.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

# Determine the frontend directory relative to this file
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

def get_index_response():
    index_path = FRONTEND_DIR / "dist" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    # Fallback to root index.html if dist doesn't exist
    index_path_root = FRONTEND_DIR / "index.html"
    if index_path_root.exists():
        return FileResponse(str(index_path_root), media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend index.html not found. Run 'npm run build' inside frontend directory first.")

@router.get("/")
async def serve_index():
    return get_index_response()

# Catch-all route to serve static files in dist or fall back to index.html for SPA routing
@router.get("/{path:path}")
async def serve_static_or_spa(path: str):
    # Construct potential static file path in dist directory
    static_file = FRONTEND_DIR / "dist" / path
    
    # Resolve and check for path traversal vulnerability
    try:
        resolved_file = static_file.resolve()
        resolved_dist = (FRONTEND_DIR / "dist").resolve()
        
        # Check if the resolved file is inside the dist directory and is indeed a file
        if resolved_dist in resolved_file.parents or resolved_file == resolved_dist:
            if resolved_file.is_file():
                # Map standard content types
                suffix = resolved_file.suffix.lower()
                media_types = {
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".ico": "image/x-icon",
                    ".json": "application/json",
                    ".js": "application/javascript",
                    ".css": "text/css",
                    ".html": "text/html"
                }
                return FileResponse(str(resolved_file), media_type=media_types.get(suffix))
    except Exception:
        pass

    # For SPA routing, fall back to index.html
    return get_index_response()

__all__ = ["router"]

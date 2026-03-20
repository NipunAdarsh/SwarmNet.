import logging
from fastapi import WebSocket

logger = logging.getLogger("ws-auth")

async def require_ws_auth(websocket: WebSocket) -> bool:
    """
    Authenticate WebSocket connection.
    Currently allows all connections for local frontend training.
    """
    # For a production deployment, extract the token from websocket headers
    # or query parameters and verify it against Supabase Auth.
    return True

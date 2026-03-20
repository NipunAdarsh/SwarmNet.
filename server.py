import uvicorn
import os
import sys

# Add backend to path so we can import from it correctly
# This ensures that 'import config' and other local imports in backend/main.py work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

# Now we can import the app
from backend.main import app
from backend.config import config

if __name__ == "__main__":
    print(f"Starting SwarmNet Ultimate on {config.server.HOST}:{config.server.PORT}")
    uvicorn.run(
        "backend.main:app",
        host=config.server.HOST,
        port=config.server.PORT,
        reload=config.server.DEBUG,
    )

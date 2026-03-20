"""Configuration constants for the SwarmNet agent."""

import os
from pathlib import Path

# The base URL of the SwarmNet backend API.
# Change this to your real Railway URL after deployment, e.g., "https://swarmnet.up.railway.app"
BASE_URL = "http://localhost:8000"

# Application data directory
APP_DIR = Path.home() / ".swarmnet"
DATA_FILE = APP_DIR / "agent_data.json"
LOG_FILE = APP_DIR / "agent.log"

# Idle detection constants
IDLE_CPU_THRESHOLD = 20.0  # CPU must be below this percentage
IDLE_DURATION_REQUIRED = 30  # Seconds of continuous idle CPU required
MONITOR_INTERVAL = 10  # Seconds between CPU checks
POLL_INTERVAL_EMPTY = 60  # Seconds to wait if no tasks available
POLL_INTERVAL_ERROR = 300  # Seconds to wait if server unreachable

# Data required for tasks
MODELS_DIR = APP_DIR / "models"

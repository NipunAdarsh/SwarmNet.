"""Handles all HTTP API communication with the SwarmNet backend."""

import logging
import requests
from typing import Dict, Any, Optional

from config import BASE_URL

logger = logging.getLogger(__name__)

def register_device(device_id: str, user_token: str, os_name: str, cpu_name: str, ram_gb: float) -> Optional[Dict[str, Any]]:
    """Register the device with the backend."""
    url = f"{BASE_URL}/api/device/register"
    payload = {
        "device_id": device_id,
        "user_token": user_token,
        "os": os_name,
        "cpu_name": cpu_name,
        "ram_gb": ram_gb
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to register device: {e}")
        return None

def get_next_task(device_id: str) -> Optional[Dict[str, Any]]:
    """Poll for the next available task."""
    url = f"{BASE_URL}/api/tasks/next"
    params = {"device_id": device_id}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("task") if "task" in data else data
    except Exception as e:
        logger.error(f"Failed to fetch next task: {e}")
        return None

def complete_task(device_id: str, task_id: str, result_data: Dict[str, Any], duration_seconds: int) -> bool:
    """Report task completion to the backend."""
    url = f"{BASE_URL}/api/tasks/complete"
    payload = {
        "device_id": device_id,
        "task_id": task_id,
        "result_data": result_data,
        "duration_seconds": duration_seconds
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to complete task: {e}")
        return False

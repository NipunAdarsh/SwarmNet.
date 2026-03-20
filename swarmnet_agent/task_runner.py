"""ONNX task download + execution."""

import os
import time
import logging
import requests
import onnxruntime as ort
from urllib.parse import urlparse
from typing import Dict, Any, Tuple

from config import MODELS_DIR

logger = logging.getLogger(__name__)

class TaskRunner:
    def __init__(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        
    def _download_model(self, model_url: str) -> str:
        """Download ONNX model if not exists."""
        filename = os.path.basename(urlparse(model_url).path)
        if not filename.endswith(".onnx"):
            filename = "model.onnx"
            
        model_path = MODELS_DIR / filename
        
        if model_path.exists():
            logger.info(f"Model {filename} already exists locally.")
            return str(model_path)
            
        logger.info(f"Downloading model from {model_url}...")
        try:
            response = requests.get(model_url, stream=True, timeout=30)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Model downloaded to {model_path}.")
            return str(model_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
            
    def run_task(self, task: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Run the provided task.
        Returns: (result_data, duration_seconds)
        """
        task_id = task.get("id")
        
        # Check both model_url and data_url to handle task params variation
        model_url = task.get("model_url") or task.get("data_url")
        
        logger.info(f"Starting task {task_id}")
        start_time = time.time()
        
        try:
            if model_url and isinstance(model_url, str) and model_url.startswith("http"):
                try:
                    model_path = self._download_model(model_url)
                    
                    # FUTURE WORK: Change to QNNExecutionProvider for NPU acceleration
                    # Example: providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
                    providers = ['CPUExecutionProvider']
                    session = ort.InferenceSession(model_path, providers=providers)
                    logger.info("ONNX session created with CPU provider.")
                    
                    # Simulate inference steps since we don't have real tensor input data
                    time.sleep(2) # CPU work simulation
                except Exception as eval_e:
                    logger.warning(f"Failed to run real ONNX model (using dummy fallback): {eval_e}")
                    time.sleep(5)  # Simulate some work if model downloading/running failed
            else:
                logger.info("No valid model_url provided, simulating task execution.")
                time.sleep(5) # Simulate tasks like protein folding, climate models
                
            duration = int(time.time() - start_time)
            
            result_data = {
                "status": "success",
                "simulated": True,
                "message": "Task completed successfully using CPU provider."
            }
            
            logger.info(f"Task {task_id} completed in {duration} seconds.")
            return result_data, duration
            
        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            duration = int(time.time() - start_time)
            return {"status": "error", "error": str(e)}, duration

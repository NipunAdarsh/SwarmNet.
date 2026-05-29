# Model loader utilities for SwarmNet

"""This module provides thin wrappers around the existing model loading helpers
defined in ``backend.routers.inference``.  Importing here isolates the heavy
ONNX runtime initialisation from the main router file, making the code easier
to test and to replace with a mock implementation if needed.
"""

from backend.routers.inference import _get_execution_providers, _load_model, _load_imagenet_labels

__all__ = ["_get_execution_providers", "_load_model", "_load_imagenet_labels"]

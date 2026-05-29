# Pre‑processing utilities for SwarmNet inference

"""Thin wrapper around the existing image preprocessing logic in
``backend.routers.inference``.  Keeping this in a dedicated module allows the
pre‑processing step to be unit‑tested independently.
"""

from backend.routers.inference import _preprocess_image

__all__ = ["_preprocess_image"]

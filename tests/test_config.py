import os
import sys
import pytest
from backend.config import config, SecurityConfig

def test_admin_secret_enforced(monkeypatch):
    # Ensure that missing ADMIN_SECRET raises RuntimeError on config load
    monkeypatch.setenv("ADMIN_SECRET", "")
    # Reload config module to re‑evaluate validators
    import importlib
    import backend.config as cfg
    try:
        with pytest.raises(RuntimeError):
            importlib.reload(cfg)
    finally:
        monkeypatch.delenv("ADMIN_SECRET", raising=False)
        importlib.reload(cfg)

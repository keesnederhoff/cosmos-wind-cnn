"""Pytest fixtures for the validation folder.

config.py resolves COSMOS_VALIDATION_* at import time, so provide harmless
defaults during collection. Tests needing a real layout use tmp_path + monkeypatch.
"""
import os
import tempfile
from pathlib import Path

# Set BEFORE config is ever imported (collection-time safety).
_TMP = Path(tempfile.gettempdir()) / "cosmos_validation_test_root"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("COSMOS_VALIDATION_DATA_ROOT", str(_TMP))
os.environ.setdefault("COSMOS_VALIDATION_OUTPUT_ROOT", str(_TMP / "out"))

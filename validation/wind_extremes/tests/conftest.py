"""Pytest setup for wind_extremes: importable modules and harmless env-var defaults.

windx_common resolves its roots from env vars at import time, so provide defaults before
any test module imports it (same pattern as validation/tests/conftest.py).
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP = Path(tempfile.gettempdir()) / "cosmos_windx_test_root"
_TMP.mkdir(parents=True, exist_ok=True)
for _var in ("COSMOS_WINDX_OUTPUT_ROOT", "COSMOS_WINDX_CNN_ROOT", "COSMOS_VALIDATION_DATA_ROOT"):
    os.environ.setdefault(_var, str(_TMP))

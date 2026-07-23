import py_compile
import re
from pathlib import Path

VALIDATION = Path(__file__).resolve().parents[1]

# NOTE: the engine builds STATIONS from NetCDFs at import time (validate_met_models.py
# runs build_pws_stations() at module load) and pulls in xarray/matplotlib. A data-less
# unit test therefore verifies the engine PARSES and no longer hardcodes a path, rather
# than importing it. Real import + run is covered by the Task 6 smoke run.


def test_engine_compiles():
    py_compile.compile(str(VALIDATION / "validate_met_models.py"), doraise=True)


def test_engine_has_no_hardcoded_ldb_or_drive_literals():
    src = (VALIDATION / "validate_met_models.py").read_text(encoding="utf-8")
    # The hardcoded assignment must be gone — LDB_FILE now flows from `from config import *`.
    assert not re.search(r'^\s*LDB_FILE\s*=\s*Path\(', src, re.M), \
        "engine still assigns LDB_FILE directly; it should come from config"
    # No `Path(r"X:\...")` drive-path literals anywhere in the engine.
    assert not re.findall(r'Path\(\s*r?["\'][A-Za-z]:', src), \
        "hardcoded drive-path literal still in engine"
    # LDB_FILE is still referenced (used from config via the wildcard import).
    assert "LDB_FILE" in src

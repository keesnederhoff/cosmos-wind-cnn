import py_compile
import re
from pathlib import Path

VALIDATION = Path(__file__).resolve().parents[1]


def _scripts():
    # run_validation.py ends in a bare V.main(); the analysis scripts execute at module
    # level (top-level prints / CSV reads). Importing them does real work, so we verify
    # they PARSE rather than importing them.
    return [VALIDATION / "run_validation.py"] + sorted((VALIDATION / "analysis").glob("*.py"))


def test_driver_and_analysis_compile():
    for f in _scripts():
        py_compile.compile(str(f), doraise=True)


def test_no_hardcoded_path_literals():
    for f in _scripts():
        src = f.read_text(encoding="utf-8")
        # Path(r"X:\...") literals only — the lone d:\ in make_comparison_slides is a
        # caption string (updated in Step 3), not a Path(), so it is not flagged.
        hits = re.findall(r'Path\(\s*r?["\'][A-Za-z]:', src)
        assert not hits, f"hardcoded drive-path literal in {f.name}: {hits}"

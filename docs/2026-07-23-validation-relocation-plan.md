# SF Bay Met-Validation Relocation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate the SF Bay meteorological product-validation framework from `g:\01_meteorlogical_analysis_sfbay\` into a self-contained, env-var-driven `validation/` folder inside the `cosmos-wind-cnn` repo, so it runs on both Windows and Caldera/HPC.

**Architecture:** Faithful copy of the proven engine + config + driver + analysis scripts. The *only* substantive change is the path layer: `config.py`'s ~15 hardcoded `m:/d:/g:/f:` absolute paths become two env-var roots (`COSMOS_VALIDATION_DATA_ROOT` / `COSMOS_VALIDATION_OUTPUT_ROOT`) resolving to a canonical sub-layout. A `stage_validation_data.py` script assembles the data bundle (obs + product subsets) that both feeds local runs and ships to Caldera. Phases 2 (data→Caldera) and 3 (cleanup G:) are gated on Phase-1 verification.

**Tech Stack:** Python 3.11 (`cosmos_wind_cnn` conda env), xarray, numpy, pandas, matplotlib, scipy, pyproj, netCDF4, python-pptx (one analysis script), pytest. Runs on Windows (dev) and Linux/SLURM (Caldera).

**Source of truth for original code:** `g:\01_meteorlogical_analysis_sfbay\`. Line numbers below refer to the originals read on 2026-07-23.

---

## Canonical data-root layout

All paths resolve under `COSMOS_VALIDATION_DATA_ROOT`. Subdir names chosen to **minimise config edits** — where the original `config.py` already anchored a product on an intermediate variable, the subdir mirrors the original nesting so the `MODELS` entry cascades unchanged.

```
$COSMOS_VALIDATION_DATA_ROOT/
├── obs/               # pws_sfbay_waterfront_{iem,ndbc,cwop_madis}.nc, ERO20_GrizzlyBay_meteorological.nc
├── moorings/          # DMP23MW101met.nc, DMP23MW201met.nc, EMC26MW101met.nc
├── reference/         # station_inventory.{csv,md}, deltabay.ldb
├── era5/              # ERA5_*_1940_2026_UTM.nc  (7 vars)
├── hrrr/              # HRRR_WY2015-WY2026_*.nc  (4 vars)
├── conus404/          # CONUS404_SFbay_4km_*_1979_2021_UTM10.nc  (7 vars)
├── rtma/              # RTMA_grid_2p5km_2*.nc, RTMA_grid_2p5km_precip_2*.nc
├── now23/             # now23_ca_bayarea_box_*.nc
├── sup3rwind/         # sup3rwind_bayarea_box_*.nc
├── ucla_reanalysis/   # era5_reanalysis_1hr_*.nc
├── wrf_calnev/        # wrfout_d02_V1_*_bayarea.nc
├── cnn/               # cnn_conus404.nc, cnn_rtma.nc, cnn_allvars.nc, cnn_windonly.nc  (renamed)
└── aorc/              # AORC_SFbay_800m_<year>.nc
```

`COSMOS_VALIDATION_OUTPUT_ROOT` holds per-era run dirs (was `PROJECT_ROOT/results`).

> Refinement vs. the design doc: the design listed `ucla/` and `wrf_calnev/`; the plan uses `ucla_reanalysis/` and `wrf_calnev/` so the `PROD_DATA_DIR`-anchored `MODELS` entries cascade with zero edits. Same data, fewer diffs.

---

## Task 0: Branch + scaffold

**Files:**
- Create: `d:\Git\cosmos-wind-cnn\validation\` (+ `analysis\`, `reference\`, `slurm\`, `tests\`)
- Create: `validation/tests/conftest.py`

- [ ] **Step 1: Create a feature branch**

Run (in `d:\Git\cosmos-wind-cnn`):
```bash
git checkout -b feat/validation-relocation
```
Expected: `Switched to a new branch 'feat/validation-relocation'`

- [ ] **Step 2: Create the folder skeleton**

```bash
mkdir -p validation/analysis validation/reference validation/slurm validation/tests
```

- [ ] **Step 3: Add a pytest conftest that sets dummy validation env vars before any config import**

`config.py` resolves its roots at import time (fail-fast, matching the repo's `get_data_dir`). Tests must therefore have the env vars set during collection. Point them at a throwaway temp dir; individual tests that need a real layout create it.

Create `validation/tests/conftest.py`:
```python
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
```

- [ ] **Step 4: Commit**

```bash
git add validation/tests/conftest.py
git commit -m "chore: scaffold validation/ folder and test conftest"
```

---

## Task 1: Port `config.py` to env-var roots (TDD)

**Files:**
- Create: `validation/config.py` (copy of original, path layer rewritten)
- Test: `validation/tests/test_config_paths.py`

- [ ] **Step 1: Write the failing test**

Create `validation/tests/test_config_paths.py`:
```python
import importlib
import os
import sys
from pathlib import Path

import pytest


def _fresh_config(data_root, out_root):
    """Import config with roots pointed at the given dirs (fresh each call)."""
    os.environ["COSMOS_VALIDATION_DATA_ROOT"] = str(data_root)
    os.environ["COSMOS_VALIDATION_OUTPUT_ROOT"] = str(out_root)
    sys.modules.pop("config", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # validation/
    return importlib.import_module("config")


def test_roots_resolve_from_env(tmp_path):
    cfg = _fresh_config(tmp_path / "data", tmp_path / "out")
    assert cfg.DATA_ROOT == tmp_path / "data"
    assert cfg.OUTPUT_ROOT == tmp_path / "out"


def test_missing_env_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("COSMOS_VALIDATION_DATA_ROOT", raising=False)
    monkeypatch.delenv("COSMOS_VALIDATION_OUTPUT_ROOT", raising=False)
    sys.modules.pop("config", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with pytest.raises(RuntimeError, match="COSMOS_VALIDATION_DATA_ROOT"):
        importlib.import_module("config")


def test_product_paths_anchor_under_data_root(tmp_path):
    cfg = _fresh_config(tmp_path / "data", tmp_path / "out")
    dr = tmp_path / "data"
    # Cascading anchors:
    assert cfg.MODELS["ERA5"]["u_file"] == dr / "era5" / "ERA5_eastward_wind_1940_2026_UTM.nc"
    assert cfg.MODELS["HRRR"]["u_file"].parent == dr / "hrrr"
    assert cfg.MODELS["CONUS404"]["u_file"].parent == dr / "conus404"
    assert cfg.MODELS["NOW-23"]["data_dir"] == dr / "now23"
    assert cfg.MODELS["Sup3rWind"]["data_dir"] == dr / "sup3rwind"
    assert cfg.MODELS["UCLA"]["data_dir"] == dr / "ucla_reanalysis"
    assert cfg.MODELS["WRF_CalNev"]["data_dir"] == dr / "wrf_calnev"
    # Explicit re-anchors:
    assert cfg.MODELS["RTMA"]["data_dir"] == dr / "rtma"
    assert cfg.MODELS["CNN"]["u_file"] == dr / "cnn" / "cnn_conus404.nc"
    assert cfg.MODELS["CNN-RTMA-20260625"]["u_file"] == dr / "cnn" / "cnn_rtma.nc"
    assert cfg.MODELS["CNN-allvars"]["u_file"] == dr / "cnn" / "cnn_allvars.nc"
    assert cfg.MODELS["CNN-windonly"]["u_file"] == dr / "cnn" / "cnn_windonly.nc"
    assert cfg.MODELS["AORC"]["u_file"].parent == dr / "aorc"
    # Obs / moorings / reference:
    assert cfg.PWS_DIR == dr / "obs"
    assert cfg.LDB_FILE == dr / "reference" / "deltabay.ldb"
    assert cfg.USGS_MOORINGS["WT_MW101"]["file_path"].parent == dr / "moorings"
    assert cfg.USGS_MOORINGS["ERO20_GRZ"]["file_path"].parent == dr / "obs"


def test_no_hardcoded_path_literals_in_source():
    import re
    src = (Path(__file__).resolve().parents[1] / "config.py").read_text(encoding="utf-8")
    # Flag actual path literals like Path(r"m:\...") — NOT drive letters that appear
    # in the RuntimeError help text (e.g. "set VAR=G:\...").
    hits = re.findall(r'Path\(\s*r?["\'][A-Za-z]:', src)
    assert not hits, f"hardcoded drive-path literal(s) still in config.py: {hits}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd d:\Git\cosmos-wind-cnn && python -m pytest validation/tests/test_config_paths.py -v`
Expected: FAIL / collection error (`validation/config.py` does not exist yet).

- [ ] **Step 3: Copy the original config and rewrite the path layer**

Copy the original verbatim first:
```bash
cp "g:/01_meteorlogical_analysis_sfbay/config.py" "d:/Git/cosmos-wind-cnn/validation/config.py"
```

Then apply the edits below to `validation/config.py`.

**Edit 3a — imports + roots.** Replace the original header block (the module docstring lines 1-12 plus `from pathlib import Path` and the `PROJECT_ROOT`/`OUTPUT_ROOT` block, lines 13-19) with:
```python
"""
Centralized configuration for SF Bay meteorological product validation.

Single source of truth for data roots, obs archive, model/product registry,
station scope, run options, plot control and physical constants. The engine
(validate_met_models.py), the driver (run_validation.py) and the analysis
scripts all import from here.

Paths resolve from two env vars so the same code runs on Windows and Caldera:
    COSMOS_VALIDATION_DATA_ROOT   -> canonical data bundle (see docs/…-plan.md)
    COSMOS_VALIDATION_OUTPUT_ROOT -> per-era run outputs
"""
import os
from pathlib import Path


def _root(var: str) -> Path:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"{var} is not set. Point it at the validation data/output base.\n"
            f"  Windows:  set {var}=G:\\03-downscaling_meteo_cnn\\validation\n"
            f"  Caldera:  export {var}=/caldera/.../validation"
        )
    return Path(val)


# ===========================================================================
# Roots (env-var driven — Windows + Caldera/HPC)
# ===========================================================================
DATA_ROOT   = _root("COSMOS_VALIDATION_DATA_ROOT")
OUTPUT_ROOT = _root("COSMOS_VALIDATION_OUTPUT_ROOT")   # per-era run dirs land here
```

**Edit 3b — obs archive anchor.** Replace `PWS_DIR = Path(r"d:\data\meteo\SFBay\data")` (line 40) with:
```python
PWS_DIR      = DATA_ROOT / "obs"
MOORINGS_DIR = DATA_ROOT / "moorings"
REFERENCE_DIR = DATA_ROOT / "reference"
LDB_FILE     = REFERENCE_DIR / "deltabay.ldb"   # land boundary for cartopy spatial maps
```

**Edit 3c — USGS moorings file paths.** In the `USGS_MOORINGS` dict (lines 52-78) replace the four `file_path=` literals:
- `DMP23MW101met.nc`, `DMP23MW201met.nc` → `MOORINGS_DIR / "DMP23MW101met.nc"`, `MOORINGS_DIR / "DMP23MW201met.nc"`
- `EMC26MW101met.nc` → `MOORINGS_DIR / "EMC26MW101met.nc"`
- `ERO20_GrizzlyBay_meteorological.nc` → `PWS_DIR / "ERO20_GrizzlyBay_meteorological.nc"`  (staged under `obs/`)

So the four lines become:
```python
        'file_path': MOORINGS_DIR / "DMP23MW101met.nc",
        'file_path': MOORINGS_DIR / "DMP23MW201met.nc",
        'file_path': MOORINGS_DIR / "EMC26MW101met.nc",
        'file_path': PWS_DIR / "ERO20_GrizzlyBay_meteorological.nc",
```

**Edit 3d — product dir anchors.** Replace the anchor block (lines 86-93, `_CNN_ROOT … CNN_RTMA_FILE`) with:
```python
ERA5_DIR      = DATA_ROOT / "era5"
HRRR_DIR      = DATA_ROOT / "hrrr"
C404_DIR      = DATA_ROOT / "conus404"
RTMA_DIR      = DATA_ROOT / "rtma"
PROD_DIR      = DATA_ROOT                 # NOW-23 -> now23/, Sup3rWind -> sup3rwind/
PROD_DATA_DIR = DATA_ROOT                 # UCLA -> ucla_reanalysis/, WRF_CalNev -> wrf_calnev/
DOWNSCALED_DIR = DATA_ROOT / "conus404_downscaled"
CNN_DIR       = DATA_ROOT / "cnn"
CNN_FILE      = CNN_DIR / "cnn_conus404.nc"
CNN_RTMA_FILE = CNN_DIR / "cnn_rtma.nc"
```
(NOTE: `MODELS` entries for NOW-23/Sup3rWind/UCLA/WRF/ERA5/HRRR/CONUS404/CNN/CNN-RTMA already reference these anchor variables, so they cascade with no further edits.)

**Edit 3e — CONUS404-downscaled literal dirs.** In the two downscaled entries (lines 118, 124) replace `'data_dir': Path(r"m:\wind_downscaling\output\delft3d")` with `'data_dir': DOWNSCALED_DIR` (both). These are dropped from the bundle but the entries stay harmless (skip-clean if absent).

**Edit 3f — RTMA literal dir.** In the `MODELS['RTMA']` block (line 167) replace `'data_dir': Path(r"m:\emeryville_crescent\04_model_runs\meteo\rtma_gee_grid")` with `'data_dir': RTMA_DIR`.

**Edit 3g — CNN-study anchor + AORC.** Replace `_CNN_STUDY_DIR = PROJECT_ROOT / "data"` (line 267) with `_CNN_STUDY_DIR = CNN_DIR`, and change the two study file paths (lines 269-271 and 274-276) so both `u_file`/`v_file` point at the renamed files:
```python
MODELS['CNN-allvars'] = {
    'u_file': CNN_DIR / 'cnn_allvars.nc',
    'v_file': CNN_DIR / 'cnn_allvars.nc',
    'u_var': 'hr_u', 'v_var': 'hr_v', 'single_file': True,
}
MODELS['CNN-windonly'] = {
    'u_file': CNN_DIR / 'cnn_windonly.nc',
    'v_file': CNN_DIR / 'cnn_windonly.nc',
    'u_var': 'hr_u', 'v_var': 'hr_v', 'single_file': True,
}
```
Then update the `_AV_FILE` reference (line 286) to `_AV_FILE = MODELS['CNN-allvars']['u_file']` (already derives from the dict — verify it still reads from the dict, not a literal). Replace `AORC_DIR = PROJECT_ROOT / "data" / "aorc"` (line 306) with `AORC_DIR = DATA_ROOT / "aorc"`.

**Edit 3h — remove `INCLUDE_USGS_MOORINGS` surprises.** Leave `INCLUDE_USGS_MOORINGS` as-is (currently `True`); no change. (Documented caveat only.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd d:\Git\cosmos-wind-cnn && python -m pytest validation/tests/test_config_paths.py -v`
Expected: PASS (4 tests), including `test_no_drive_letter_literals_in_source`.

- [ ] **Step 5: Commit**

```bash
git add validation/config.py validation/tests/test_config_paths.py
git commit -m "feat(validation): port config.py to COSMOS_VALIDATION_* env roots"
```

---

## Task 2: Copy the engine, remove its one hardcoded path

**Files:**
- Create: `validation/validate_met_models.py` (verbatim copy minus 1 line)
- Test: `validation/tests/test_engine_import.py`

- [ ] **Step 1: Write the failing test**

Create `validation/tests/test_engine_import.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd d:\Git\cosmos-wind-cnn && python -m pytest validation/tests/test_engine_import.py -v`
Expected: FAIL (`validate_met_models.py` does not exist yet → compile/read errors).

- [ ] **Step 3: Copy the engine and delete the hardcoded LDB line**

```bash
cp "g:/01_meteorlogical_analysis_sfbay/validate_met_models.py" "d:/Git/cosmos-wind-cnn/validation/validate_met_models.py"
```
Then edit `validation/validate_met_models.py`: delete lines 1924-1925 (the `# Land boundary file` comment and `LDB_FILE = Path(r"f:\Alameda\...\deltabay.ldb")`). `LDB_FILE` now comes from `config` via the existing `from config import *` (line 45). Leave everything else untouched.

Exact deletion — remove:
```python
# Land boundary file
LDB_FILE = Path(r"f:\Alameda\03_modelsetup\_inputs\inputnoah\deltabay.ldb")
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd d:\Git\cosmos-wind-cnn && python -m pytest validation/tests/test_engine_import.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add validation/validate_met_models.py validation/tests/test_engine_import.py
git commit -m "feat(validation): copy engine, drop hardcoded LDB path (now from config)"
```

---

## Task 3: Copy the driver and analysis scripts (already config-driven)

**Files:**
- Create: `validation/run_validation.py`
- Create: `validation/analysis/{rank_products,combined_skill,make_windroses,make_comparison_slides,redraw_comparisons}.py`
- Test: `validation/tests/test_modules_import.py`

- [ ] **Step 1: Write the failing test**

Create `validation/tests/test_modules_import.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd d:\Git\cosmos-wind-cnn && python -m pytest validation/tests/test_modules_import.py -v`
Expected: FAIL (the driver/analysis files don't exist yet → compile/read errors).

- [ ] **Step 3: Copy the driver and the five analysis scripts verbatim**

```bash
cp "g:/01_meteorlogical_analysis_sfbay/run_validation.py"                 "d:/Git/cosmos-wind-cnn/validation/run_validation.py"
cp "g:/01_meteorlogical_analysis_sfbay/analysis/rank_products.py"         "d:/Git/cosmos-wind-cnn/validation/analysis/rank_products.py"
cp "g:/01_meteorlogical_analysis_sfbay/analysis/combined_skill.py"        "d:/Git/cosmos-wind-cnn/validation/analysis/combined_skill.py"
cp "g:/01_meteorlogical_analysis_sfbay/analysis/make_windroses.py"        "d:/Git/cosmos-wind-cnn/validation/analysis/make_windroses.py"
cp "g:/01_meteorlogical_analysis_sfbay/analysis/make_comparison_slides.py" "d:/Git/cosmos-wind-cnn/validation/analysis/make_comparison_slides.py"
cp "g:/01_meteorlogical_analysis_sfbay/analysis/redraw_comparisons.py"    "d:/Git/cosmos-wind-cnn/validation/analysis/redraw_comparisons.py"
```
No code edits: `run_validation.py` reads `config.OUTPUT_ROOT` + the engine module API, and every analysis script already does `import config` / `BASE = config.OUTPUT_ROOT`. The lone `d:\data\meteo\SFBay\` occurrence in `make_comparison_slides.py:127` is inside a text caption string — update it to describe the new bundle location, keeping it a string (not a path):
```python
           "Deliverables staged under $COSMOS_VALIDATION_DATA_ROOT : per-source NetCDFs, "
           "station-inventory table, bay-wide station "
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd d:\Git\cosmos-wind-cnn && python -m pytest validation/tests/test_modules_import.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add validation/run_validation.py validation/analysis validation/tests/test_modules_import.py
git commit -m "feat(validation): copy era driver + analysis scripts"
```

---

## Task 4: Data-staging script (TDD)

**Files:**
- Create: `validation/stage_validation_data.py`
- Test: `validation/tests/test_staging_manifest.py`

The script owns a `MANIFEST`: a list of `(source_dir, glob, dest_subdir, [rename_map])` entries defining exactly what the engine reads. It supports `--dry-run` (print planned copies, touch nothing) and real copy via `robocopy` on Windows / `shutil` fallback elsewhere. The `MANIFEST` is the single definition of the Caldera bundle.

- [ ] **Step 1: Write the failing test**

Create `validation/tests/test_staging_manifest.py`:
```python
import importlib
import sys
from pathlib import Path

VALIDATION = Path(__file__).resolve().parents[1]
CANONICAL = {
    "obs", "moorings", "reference", "era5", "hrrr", "conus404", "rtma",
    "now23", "sup3rwind", "ucla_reanalysis", "wrf_calnev", "cnn", "aorc",
}


def _mod():
    sys.path.insert(0, str(VALIDATION))
    sys.modules.pop("stage_validation_data", None)
    return importlib.import_module("stage_validation_data")


def test_manifest_dest_subdirs_are_canonical():
    m = _mod()
    for entry in m.MANIFEST:
        assert entry.dest in CANONICAL, f"non-canonical dest {entry.dest!r}"


def test_manifest_covers_every_engine_product():
    m = _mod()
    dests = {e.dest for e in m.MANIFEST}
    # every product dir the engine reads must be represented
    for need in ("era5", "hrrr", "conus404", "rtma", "now23", "sup3rwind",
                 "ucla_reanalysis", "wrf_calnev", "cnn", "aorc", "obs", "moorings", "reference"):
        assert need in dests, f"MANIFEST missing {need}"


def test_cnn_entries_have_rename_map():
    m = _mod()
    cnn_entries = [e for e in m.MANIFEST if e.dest == "cnn"]
    assert cnn_entries, "no cnn entries"
    targets = set()
    for e in cnn_entries:
        assert e.rename, f"cnn entry {e.glob} needs a rename map"
        targets.update(e.rename.values())
    assert {"cnn_conus404.nc", "cnn_rtma.nc", "cnn_allvars.nc", "cnn_windonly.nc"} <= targets


def test_dry_run_touches_nothing(tmp_path, monkeypatch, capsys):
    m = _mod()
    dest_root = tmp_path / "bundle"
    m.stage(dest_root, dry_run=True)
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert not dest_root.exists() or not any(dest_root.rglob("*.nc"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd d:\Git\cosmos-wind-cnn && python -m pytest validation/tests/test_staging_manifest.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the staging script**

Create `validation/stage_validation_data.py`:
```python
"""
Assemble the SF Bay validation data bundle from its scattered source drives
into the canonical $COSMOS_VALIDATION_DATA_ROOT layout.

Produces the local bundle AND defines the exact set shipped to Caldera.
The full bundle is ~490 GB (CNN full-record files 29-86 GB each, UCLA 82 GB,
AORC 81 GB), so copies use robocopy (multithreaded, resumable) on Windows.

Preview:              python stage_validation_data.py --dry-run
Full copy (~490 GB): python stage_validation_data.py
Subset (fast):       python stage_validation_data.py --products=era5,rtma,obs,reference

# === CONFIGURATION ===  (source drives; edit if the raw data moves)
"""
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- source roots (Windows raw-data homes; source of truth stays here) ------
SFBAY_OBS   = Path(r"d:\data\meteo\SFBay\data")
EMERYVILLE  = Path(r"m:\emeryville_crescent")
CNN_OUT     = Path(r"G:\03-downscaling_meteo_cnn")
PROJECT_OLD = Path(r"g:\01_meteorlogical_analysis_sfbay")
LDB_SRC     = Path(r"f:\Alameda\03_modelsetup\_inputs\inputnoah\deltabay.ldb")


@dataclass
class Entry:
    src: Path
    glob: str
    dest: str                       # canonical subdir name
    rename: dict = field(default_factory=dict)   # {source_name: dest_name}


MANIFEST = [
    # obs
    Entry(SFBAY_OBS, "pws_sfbay_waterfront_*.nc", "obs"),
    Entry(SFBAY_OBS, "ERO20_GrizzlyBay_meteorological.nc", "obs"),
    # moorings
    Entry(EMERYVILLE / "01_data" / "whales_tale", "DMP23MW*.nc", "moorings"),
    Entry(EMERYVILLE / "01_data" / "emc_data", "EMC26MW101met.nc", "moorings"),
    # reference
    Entry(PROJECT_OLD / "reference", "station_inventory.*", "reference"),
    Entry(LDB_SRC.parent, LDB_SRC.name, "reference"),
    # reanalysis / hi-res products
    Entry(CNN_OUT / "sf_bay_conus404" / "raw_data", "ERA5_*_UTM.nc", "era5"),
    Entry(EMERYVILLE / "04_model_runs" / "meteo", "HRRR_WY2015-WY2026_*.nc", "hrrr"),
    Entry(EMERYVILLE / "03_model_setup" / "meteo", "CONUS404_SFbay_4km_*.nc", "conus404"),
    Entry(EMERYVILLE / "04_model_runs" / "meteo" / "rtma_gee_grid", "RTMA_grid_2p5km_*.nc", "rtma"),
    Entry(EMERYVILLE / "01_data" / "other_meteo_data" / "now23", "now23_ca_bayarea_box_*.nc", "now23"),
    Entry(EMERYVILLE / "01_data" / "other_meteo_data" / "sup3rwind", "sup3rwind_bayarea_box_*.nc", "sup3rwind"),
    Entry(EMERYVILLE / "01_data" / "other_meteo_data" / "data" / "ucla_reanalysis", "era5_reanalysis_1hr_*.nc", "ucla_reanalysis"),
    Entry(EMERYVILLE / "01_data" / "other_meteo_data" / "data" / "wrf_calnev", "wrfout_d02_V1_*_bayarea.nc", "wrf_calnev"),
    Entry(PROJECT_OLD / "data" / "aorc", "AORC_SFbay_800m_*.nc", "aorc"),
    # cnn (rename to distinct names — sources share full_record_ERA5_*.nc)
    Entry(CNN_OUT / "sf_bay_conus404" / "results" / "3679830" / "output_inference",
          "full_record_ERA5_19400101_20270101.nc", "cnn",
          rename={"full_record_ERA5_19400101_20270101.nc": "cnn_conus404.nc"}),
    Entry(CNN_OUT / "sf_bay_rtma" / "results" / "3732177" / "output_inference",
          "full_record_ERA5_19400101_20270101.nc", "cnn",
          rename={"full_record_ERA5_19400101_20270101.nc": "cnn_rtma.nc"}),
    Entry(PROJECT_OLD / "data" / "os_av_bc24_terr_res_s2",
          "full_record_ERA5_20110101_20260101.nc", "cnn",
          rename={"full_record_ERA5_20110101_20260101.nc": "cnn_allvars.nc"}),
    Entry(PROJECT_OLD / "data" / "os_wo_bc24_base_res_s2",
          "full_record_ERA5_20110101_20260101.nc", "cnn",
          rename={"full_record_ERA5_20110101_20260101.nc": "cnn_windonly.nc"}),
]


def _robocopy(src_dir: Path, dst_dir: Path, file_pattern: str) -> None:
    """Bulk copy via robocopy — multithreaded, resumable, network-friendly.
    robocopy exit codes 0-7 are success; 8+ is a real error."""
    cp = subprocess.run(
        ["robocopy", str(src_dir), str(dst_dir), file_pattern,
         "/MT:16", "/Z", "/R:2", "/W:5", "/NFL", "/NDL", "/NP", "/NJH", "/NJS"],
        capture_output=True, text=True,
    )
    if cp.returncode >= 8:
        raise RuntimeError(
            f"robocopy failed ({cp.returncode}) for {src_dir}\\{file_pattern}\n"
            f"{cp.stdout}\n{cp.stderr}"
        )


def stage(dest_root: Path, dry_run: bool = False, products=None) -> None:
    """Stage the bundle into dest_root.

    products=None -> every MANIFEST entry; else an iterable of canonical dest
    names (e.g. ['era5','rtma','obs','reference']) to stage a subset.
    """
    if dry_run:
        print("DRY-RUN — no files will be copied")
    total_files = total_bytes = 0
    for e in MANIFEST:
        if products is not None and e.dest not in products:
            continue
        dst_dir = dest_root / e.dest
        if not e.src.exists():
            print(f"  SKIP (source absent): {e.src}")
            continue
        matches = sorted(e.src.glob(e.glob))
        if not matches:
            print(f"  SKIP (no match): {e.src / e.glob}")
            continue
        total_files += len(matches)
        total_bytes += sum(f.stat().st_size for f in matches)
        if dry_run:
            for f in matches:
                out_name = e.rename.get(f.name, f.name)
                print(f"  DRY-RUN {f}  ->  {dst_dir / out_name}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        if e.rename:
            # single-file entries needing a distinct name (robocopy can't rename)
            for f in matches:
                target = dst_dir / e.rename.get(f.name, f.name)
                shutil.copy2(f, target)
                print(f"  staged {f.name}  ->  {e.dest}/{target.name}")
        elif os.name == "nt":
            _robocopy(e.src, dst_dir, e.glob)
            print(f"  staged {len(matches)} file(s)  ->  {e.dest}/  (robocopy)")
        else:  # Caldera / Linux
            for f in matches:
                shutil.copy2(f, dst_dir / f.name)
            print(f"  staged {len(matches)} file(s)  ->  {e.dest}/")
    print(f"\n{'planned' if dry_run else 'staged'}: {total_files} files, "
          f"{total_bytes / 1024**3:.1f} GB")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    products = None   # default: everything;  --products=era5,rtma,obs,reference for a subset
    for a in sys.argv:
        if a.startswith("--products="):
            products = [p.strip() for p in a.split("=", 1)[1].split(",") if p.strip()]
    root = Path(os.environ.get("COSMOS_VALIDATION_DATA_ROOT",
                               r"G:\03-downscaling_meteo_cnn\validation"))
    sel = "ALL products" if products is None else ", ".join(products)
    print(f"{'DRY-RUN: ' if dry else ''}staging {sel} -> {root}")
    stage(root, dry_run=dry, products=products)
    print("done.")
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd d:\Git\cosmos-wind-cnn && python -m pytest validation/tests/test_staging_manifest.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add validation/stage_validation_data.py validation/tests/test_staging_manifest.py
git commit -m "feat(validation): add data-staging manifest + script (Caldera bundle)"
```

---

## Task 5: README + `.gitignore` + reference note

**Files:**
- Create: `validation/README.md`
- Modify: `d:\Git\cosmos-wind-cnn\.gitignore` (ensure run outputs / staged data never get committed)

- [ ] **Step 1: Write `validation/README.md`**

Create `validation/README.md`:
```markdown
# SF Bay Meteorological Product Validation

Point-observation validation & ranking of gridded wind (and met) products over SF Bay —
which product best reproduces observed winds, and is therefore the most defensible forcing
for the SF Bay Community Model over 1940–present. Relocated from
`g:\01_meteorlogical_analysis_sfbay\` (see `docs/2026-07-23-validation-relocation-design.md`).

## Layout
- `config.py` — single source of truth (products, obs, station scope, run options).
- `validate_met_models.py` — the engine.
- `run_validation.py` — era-aware driver (set `ERA`, run).
- `analysis/` — ranking, combined skill, wind roses, comparison slides.
- `reference/` — station inventory + land boundary.
- `stage_validation_data.py` — build the data bundle from raw sources.

## Environment
Runs in the `cosmos_wind_cnn` conda env (base env fails on the HRRR 2-D grid). On Windows:
```
conda activate cosmos_wind_cnn
set KMP_DUPLICATE_LIB_OK=TRUE
set COSMOS_VALIDATION_DATA_ROOT=G:\03-downscaling_meteo_cnn\validation
set COSMOS_VALIDATION_OUTPUT_ROOT=G:\03-downscaling_meteo_cnn\validation\results
```
On Caldera the SLURM launcher exports the Linux equivalents.

## Run
1. Build the bundle once:  `python stage_validation_data.py`  (preview with `--dry-run`).
2. Edit `ERA` in `run_validation.py`, then:  `python run_validation.py`.
3. Rank / pool:  `python analysis/rank_products.py`, `python analysis/combined_skill.py`.

## Product × era matrix
| Era | Window | Products |
|---|---|---|
| 1 | 1990–2010 | NOW-23, Sup3rWind, ERA5, CONUS404, UCLA, WRF_CalNev, CNN |
| 2 | 2011–2021 | + RTMA, HRRR, CNN-RTMA |
| 3 | 2022–present | RTMA, HRRR, ERA5, CNN, CNN-RTMA, NOW-23 |

## CNN file rename map (staging)
| Product | Bundle file |
|---|---|
| CNN (CONUS404) | `cnn/cnn_conus404.nc` |
| CNN-RTMA | `cnn/cnn_rtma.nc` |
| CNN-allvars | `cnn/cnn_allvars.nc` |
| CNN-windonly | `cnn/cnn_windonly.nc` |

## Caveats
- Anemometer height: IEM/NDBC/CWOP treated at 10 m (log-correction is a no-op); USGS moorings
  kept at measured height, compared directly to 10 m model output. Documented, not silently corrected.
- `CONUS404-downscaled` / `-100m` are wired in `config.MODELS` but excluded from the bundle
  (in no era's product list); they skip-clean if absent.
```

- [ ] **Step 2: Ensure outputs are git-ignored**

Add to `d:\Git\cosmos-wind-cnn\.gitignore` (append if not present):
```
# validation run outputs and any locally-staged data
validation/results/
validation/**/*.nc
validation/**/*.png
validation/**/*.pptx
```

- [ ] **Step 3: Commit**

```bash
git add validation/README.md .gitignore
git commit -m "docs(validation): add README and ignore run outputs"
```

---

## Task 6: Build the bundle, smoke run, parity check (verification — manual)

This task is manual verification against real data, not TDD. It requires the `cosmos_wind_cnn` env and the source drives (`m:`, `d:`, `f:`, `G:`) mounted.

- [ ] **Step 1: Dry-run, then stage the SMOKE SUBSET first (fast verification path)**

The full bundle is ~490 GB / multi-hour. To verify code + parity quickly, stage only the
smoke-test products first, then kick off the full copy in the background (Step 5).

```bash
conda activate cosmos_wind_cnn
set COSMOS_VALIDATION_DATA_ROOT=G:\03-downscaling_meteo_cnn\validation
set COSMOS_VALIDATION_OUTPUT_ROOT=G:\03-downscaling_meteo_cnn\validation\results
python d:\Git\cosmos-wind-cnn\validation\stage_validation_data.py --dry-run
python d:\Git\cosmos-wind-cnn\validation\stage_validation_data.py --products=era5,rtma,obs,reference
```
Expected: DRY-RUN lists every product resolving ≥1 file; then `staged …` lines for era5, rtma,
obs, reference (~50 GB) and a `staged: N files, ~50 GB` summary.

- [ ] **Step 2: Smoke run (minutes)**

Edit `validation/run_validation.py`: set `ERA = '3'`, `ONLY_GROUPS = None`, and temporarily narrow at the top of the config block for the smoke test by exporting overrides — simplest is to edit in place for the smoke run: `MODELS_TO_RUN=['ERA5','RTMA']`, `STATION_GROUPS`→ NDBC only, `VARIABLES=['wind']`, `MAKE_SPATIAL_MAPS=False`. Then:
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
set COSMOS_VALIDATION_OUTPUT_ROOT=G:\03-downscaling_meteo_cnn\validation\results
cd d:\Git\cosmos-wind-cnn\validation
python run_validation.py
```
Expected: preflight path-audit prints ERA5 + RTMA as reachable; per-station scatter/timeseries PNGs + `validation_statistics.csv` written under `…\validation\results\era3_2022-present\`. Revert the smoke-narrowing edit afterward.

- [ ] **Step 3: Parity spot-check**

Compare a handful of ERA5/RTMA NDBC wind stats (bias, RMSE, corr) from the new
`…\validation\results\era3_2022-present\validation_statistics.csv` against the corresponding rows
in the existing `g:\01_meteorlogical_analysis_sfbay\results\` for the same stations/period.
Expected: identical numbers (same inputs, same engine) — proving the path refactor introduced no
regression. Record the compared rows in the commit message.

- [ ] **Step 4: Commit the verification note**

```bash
cd d:\Git\cosmos-wind-cnn
git add -A
git commit -m "test(validation): smoke subset staged; smoke run + parity vs g:\\ results confirmed"
```

- [ ] **Step 5: Launch the full ~490 GB copy in the background**

Only after Steps 3–4 confirm the code + parity are correct, stage the remaining products
(the run is resumable via robocopy `/Z`, so re-running is safe if interrupted):
```bash
python d:\Git\cosmos-wind-cnn\validation\stage_validation_data.py
```
Expected: robocopy `staged …` lines for hrrr, conus404, now23, sup3rwind, ucla_reanalysis,
wrf_calnev, cnn (4 renamed singles), aorc, moorings; final `staged: ~490 GB` summary. This is
the local bundle; Phase 2 selects which of these products ship to Caldera.

**Phase 1 code is complete when Steps 1–4 pass** (the Step 5 bulk copy may still be running).
Stop and report before Phases 2–3.

---

## Task 7 (Phase 2 — GATED on Phase-1 verification + explicit go-ahead): ship bundle to Caldera

- [ ] **Step 1:** `rsync`/`scp` `G:\03-downscaling_meteo_cnn\validation\` → caldera project space (obs + product subsets only; already the curated bundle).
- [ ] **Step 2:** Create `validation/slurm/cpu_caldera_validation.slurm` modeled on `scripts/cpu_rtma_eval.slurm`: export `COSMOS_VALIDATION_DATA_ROOT` / `COSMOS_VALIDATION_OUTPUT_ROOT` at caldera paths, `module load` the env, `cd $SLURM_SUBMIT_DIR/../validation`, `python run_validation.py`.
- [ ] **Step 3:** HPC smoke run (ERA5+RTMA, NDBC, wind) + parity spot-check against the Windows result. Commit the SLURM script.

## Task 8 (Phase 3 — GATED on Phase-2 verification + explicit go-ahead): clean up `g:\01_meteorlogical_analysis_sfbay\`

- [ ] **Step 1:** Confirm the repo + Caldera runs are verified and the parity numbers are captured.
- [ ] **Step 2:** Remove from `g:\01_...`: `config.py`, `validate_met_models.py`, `run_validation.py`, `analysis/`, `reference/`, `docs/`, `__pycache__/`, and the ~15 archived experiment drivers (`run_ndbc.py`, `run_full_parallel.py`, `run_tmp_2025_cnn_rtma.py`, `run_cnn_validation*.py`, `analyze_cnn_run*.py`, `run_aorc_*.py`, `analyze_aorc_*.py`, `run_rtma_vs_aorc*.py`, `analyze_rtma_vs_aorc.py`).
- [ ] **Step 3:** KEEP: `download_aorc.py`, `data\` (raw AORC + CNN-study NetCDFs), `results\`.
- [ ] **Step 4:** Do the deletion in one reviewable move (e.g. move-to-`_removed\` first, delete after a final look), never blind `rm -rf`.

---

## Self-review

**Spec coverage:** §2 phases → Tasks 0–8. §3 layout → Task 0. §4 path refactor → Task 1 (+ engine straggler in Task 2). §5 staging manifest/bundle + inventory → Task 4. §6 curation (move vs archive) → Tasks 2/3 move, Task 8 archive list. §7 verification (smoke + parity + preflight) → Task 6. §8 Caldera → Task 7. §9 cleanup → Task 8. §10 non-goals respected (faithful copy; only path layer changes). §11 decisions all reflected. No gaps.

**Placeholder scan:** no TBD/TODO; every code step shows real code; every command has an expected result.

**Type/name consistency:** `DATA_ROOT`/`OUTPUT_ROOT`/`PWS_DIR`/`MOORINGS_DIR`/`REFERENCE_DIR`/`LDB_FILE`/`CNN_DIR`/`AORC_DIR` used consistently across config + tests + engine test + staging. Canonical subdir set (`ucla_reanalysis`, `wrf_calnev`, etc.) matches between the layout section, Task 1 assertions, and Task 4 `MANIFEST`/`CANONICAL`. CNN bundle filenames (`cnn_conus404.nc`/`cnn_rtma.nc`/`cnn_allvars.nc`/`cnn_windonly.nc`) match across config edits (3d/3g), the config test, the staging rename maps, and the README.

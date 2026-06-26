# Externalize Case-Study Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep only `configs/` + `README.md` per case study in the repo; read raw data from `$COSMOS_DATA_ROOT/<case>/raw_data/` (shared, result-independent) and write all per-run artifacts to `$COSMOS_RESULTS_ROOT/<case>/results/<job>/`, erroring clearly when those env vars are unset.

**Architecture:** The two path helpers in `utils/config.py` (`get_data_dir`, `get_run_dirs`) already centralize storage paths. Change them to (a) require their env var (no in-repo fallback — raise a clear error), (b) use `raw_data` and a `results/<job>` level. Route the few scripts that still build `case_dir/data/...` paths directly through the helpers, remove the repo's placeholder dirs, update the SLURM env values + docs, and add the (currently missing) tests for these helpers including the error paths.

**Tech Stack:** Python 3.11 package `cosmos_wind_cnn` (conda env `cosmos_wind_cnn`), pytest. Bash tool = Git Bash on Windows; env python `C:/Users/keesn/anaconda3/envs/cosmos_wind_cnn/python.exe` for pytest. Decisions (with Kees): two separate roots (`COSMOS_DATA_ROOT`, `COSMOS_RESULTS_ROOT` — both set to `g:\03-downscaling_meteo_cnn` locally, can differ on HPC); error fast when unset; repo + code + docs only (Kees moves/rsyncs data himself).

---

## Target layout

| Kind | Path | Notes |
|------|------|-------|
| Raw inputs (shared) | `$COSMOS_DATA_ROOT/<case>/raw_data/` | result-independent; e.g. `g:\03-downscaling_meteo_cnn\sf_bay_rtma\raw_data\` |
| Per-run artifacts | `$COSMOS_RESULTS_ROOT/<case>/results/<job>/{checkpoint, data_processed, logs, output_inference, output_evaluation}/` | e.g. `g:\03-downscaling_meteo_cnn\sf_bay_rtma\results\3732177\` |
| In repo (tracked) | `case_studies/<case>/{configs/, README.md}` (+ `sf_bay_rtma/stage_data.py`) | no data/results dirs |

## File Structure

| Path | Action |
|------|--------|
| `src/cosmos_wind_cnn/utils/config.py` | **Modify** `get_data_dir` + `get_run_dirs` (require env var, `raw_data`, `results/<job>`) |
| `tests/test_paths.py` | **Create** tests for both helpers incl. error-when-unset |
| `scripts/preprocess_training.py:39`, `scripts/preprocess_inference.py:61`, `scripts/run_inference.py:81` | **Modify** `case_dir/'data'/'raw'` → `get_data_dir(case_dir)` |
| `case_studies/sf_bay_rtma/stage_data.py:14-15` | **Modify** route `ERA5_SRC`/`DEST` through `get_data_dir` |
| `case_studies/*/{data,results,checkpoints,logs,outputs}/.gitkeep` (14 files) | **Delete** placeholder dirs |
| `.gitignore` | **Modify** drop `.gitkeep` exceptions; ignore case-study data/results scaffolding |
| `scripts/{gpu_tallgrass,gpu_tallgrass_rtma,gpu_tallgrass_rtma_fresh,gpu_tallgrass_rtma_infer,cpu_rtma_eval}.slurm` | **Modify** point `COSMOS_*_ROOT` at the project base |
| `README.md`, `case_studies/*/README.md`, `docs/{data_preparation,adding_case_study}.md`, launcher `CLAUDE.md` | **Modify** document external storage + env vars |

---

### Task 0: Branch

- [ ] **Step 1: Create the branch from clean main**
```bash
cd /d/Git/cosmos-wind-cnn
git status --short | grep -v "^??" || echo "(no tracked changes)"
git checkout -b restructure/external-storage
git branch --show-current
```
Expected: on `restructure/external-storage`.

---

### Task 1: Path helpers require env vars; use `raw_data` + `results/<job>` (TDD)

**Files:** Create `tests/test_paths.py`; Modify `src/cosmos_wind_cnn/utils/config.py` (`get_data_dir`, `get_run_dirs`)

- [ ] **Step 1: Write the failing tests** — create `tests/test_paths.py`:
```python
from pathlib import Path
import pytest
from cosmos_wind_cnn.utils.config import get_data_dir, get_run_dirs


def test_get_data_dir_uses_data_root(monkeypatch):
    monkeypatch.setenv('COSMOS_DATA_ROOT', '/storage')
    assert get_data_dir('case_studies/sf_bay_conus404') == \
        Path('/storage') / 'sf_bay_conus404' / 'raw_data'


def test_get_data_dir_errors_when_unset(monkeypatch):
    monkeypatch.delenv('COSMOS_DATA_ROOT', raising=False)
    with pytest.raises(RuntimeError, match='COSMOS_DATA_ROOT'):
        get_data_dir('case_studies/sf_bay_conus404')


def test_get_run_dirs_layout(monkeypatch):
    monkeypatch.setenv('COSMOS_RESULTS_ROOT', '/storage')
    d = get_run_dirs('case_studies/sf_bay_rtma', '3732177')
    base = Path('/storage') / 'sf_bay_rtma' / 'results' / '3732177'
    assert d['run_root'] == base
    assert d['checkpoint'] == base / 'checkpoint'
    assert d['data_processed'] == base / 'data_processed'
    assert d['logs'] == base / 'logs'
    assert d['output_inference'] == base / 'output_inference'
    assert d['output_evaluation'] == base / 'output_evaluation'


def test_get_run_dirs_errors_when_unset(monkeypatch):
    monkeypatch.delenv('COSMOS_RESULTS_ROOT', raising=False)
    with pytest.raises(RuntimeError, match='COSMOS_RESULTS_ROOT'):
        get_run_dirs('case_studies/sf_bay_rtma', '123')
```

- [ ] **Step 2: Run, confirm failure**
```bash
cd /d/Git/cosmos-wind-cnn
PY="C:/Users/keesn/anaconda3/envs/cosmos_wind_cnn/python.exe"
"$PY" -m pytest tests/test_paths.py -q
```
Expected: FAIL — current `get_data_dir` returns `…/raw` (not `…/raw_data`) and doesn't raise when unset; `get_run_dirs` returns `…/<job>` (no `results/` level) and doesn't raise.

- [ ] **Step 3: Update `get_data_dir`** — replace the whole function body so it requires the env var and uses `raw_data`:
```python
def get_data_dir(case_dir):
    """Directory holding a case study's raw NetCDF inputs (shared across runs).

    Read from ``<COSMOS_DATA_ROOT>/<case_name>/raw_data``. Data lives OUTSIDE the
    repo, so COSMOS_DATA_ROOT must be set; raises RuntimeError if it is not.
    """
    case_dir = Path(case_dir)
    root = os.environ.get('COSMOS_DATA_ROOT')
    if not root:
        raise RuntimeError(
            "COSMOS_DATA_ROOT is not set. Point it at your raw-data storage base "
            "(raw inputs are read from <COSMOS_DATA_ROOT>/<case_name>/raw_data/).\n"
            "  Windows:  set COSMOS_DATA_ROOT=G:\\03-downscaling_meteo_cnn\n"
            "  Linux:    export COSMOS_DATA_ROOT=/path/to/storage"
        )
    return Path(root) / case_dir.name / 'raw_data'
```

- [ ] **Step 4: Update `get_run_dirs`** — change the env-var block + docstring. Replace the docstring line `All run artefacts live under  ``<case_dir>/results/<run_name>/``.` with `All run artefacts live under  ``<COSMOS_RESULTS_ROOT>/<case_name>/results/<run_name>/``.`, and replace the body block:
```python
    case_dir = Path(case_dir)
    _results_root = os.environ.get('COSMOS_RESULTS_ROOT')
    if _results_root:
        run_root = Path(_results_root) / case_dir.name / str(run_name)
    else:
        run_root = case_dir / 'results' / str(run_name)
    return {
```
with:
```python
    case_dir = Path(case_dir)
    _results_root = os.environ.get('COSMOS_RESULTS_ROOT')
    if not _results_root:
        raise RuntimeError(
            "COSMOS_RESULTS_ROOT is not set. Point it at your results storage base "
            "(run outputs go to <COSMOS_RESULTS_ROOT>/<case_name>/results/<run_name>/).\n"
            "  Windows:  set COSMOS_RESULTS_ROOT=G:\\03-downscaling_meteo_cnn\n"
            "  Linux:    export COSMOS_RESULTS_ROOT=/path/to/storage"
        )
    run_root = Path(_results_root) / case_dir.name / 'results' / str(run_name)
    return {
```

- [ ] **Step 5: Run tests + full suite**
```bash
cd /d/Git/cosmos-wind-cnn
PY="C:/Users/keesn/anaconda3/envs/cosmos_wind_cnn/python.exe"
"$PY" -m py_compile src/cosmos_wind_cnn/utils/config.py && echo "compile ok"
"$PY" -m pytest tests/test_paths.py -q
"$PY" -m pytest -q
```
Expected: `compile ok`; `tests/test_paths.py` 4 passed; full suite passes (17 total: 13 prior + 4 new).

- [ ] **Step 6: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
git add src/cosmos_wind_cnn/utils/config.py tests/test_paths.py
git commit -m "$(cat <<'EOF'
feat: require COSMOS_DATA_ROOT/RESULTS_ROOT; raw_data + results/<job> layout

get_data_dir -> <COSMOS_DATA_ROOT>/<case>/raw_data; get_run_dirs ->
<COSMOS_RESULTS_ROOT>/<case>/results/<run>/... Both now raise a clear error when
their env var is unset (data/results live outside the repo). Adds tests for both
helpers including the error paths.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Route direct in-repo data paths through `get_data_dir`

**Files:** `scripts/preprocess_training.py:39`, `scripts/preprocess_inference.py:61`, `scripts/run_inference.py:81`, `case_studies/sf_bay_rtma/stage_data.py:14-15`

- [ ] **Step 1: Fix the three scripts**

In each of `scripts/preprocess_training.py`, `scripts/preprocess_inference.py`, `scripts/run_inference.py`, find the line:
```python
    data_dir = case_dir / 'data' / 'raw'
```
and replace with:
```python
    data_dir = get_data_dir(case_dir)
```
Ensure each file imports `get_data_dir`: check `grep -n "from cosmos_wind_cnn.utils.config import" scripts/<file>.py` and add `get_data_dir` to that import list if missing.

- [ ] **Step 2: Fix `stage_data.py`** — replace lines 14-15:
```python
ERA5_SRC = Path(__file__).resolve().parents[1] / "sf_bay_conus404" / "data" / "raw"
DEST = Path(__file__).resolve().parent / "data" / "raw"
```
with:
```python
from cosmos_wind_cnn.utils.config import get_data_dir
CASE_DIR = Path(__file__).resolve().parent
ERA5_SRC = get_data_dir(CASE_DIR.parent / "sf_bay_conus404")
DEST = get_data_dir(CASE_DIR)
```
(Place the `from cosmos_wind_cnn...` import with the other imports near the top; keep `CASE_DIR`/`ERA5_SRC`/`DEST` where the originals were. The later `DEST.mkdir(...)` line stays.)

- [ ] **Step 3: Verify no direct in-repo data paths remain + compile**
```bash
cd /d/Git/cosmos-wind-cnn
PY="C:/Users/keesn/anaconda3/envs/cosmos_wind_cnn/python.exe"
git grep -nE "case_dir ?/ ?'data' ?/ ?'raw'|/ \"data\" / \"raw\"" -- scripts/ case_studies/ && echo "FAIL: direct path remains" || echo "ok: all routed through get_data_dir"
"$PY" -m py_compile scripts/preprocess_training.py scripts/preprocess_inference.py scripts/run_inference.py case_studies/sf_bay_rtma/stage_data.py && echo "compile ok"
"$PY" -m pytest -q
```
Expected: `ok: all routed through get_data_dir`; `compile ok`; tests pass.

- [ ] **Step 4: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
git add scripts/preprocess_training.py scripts/preprocess_inference.py scripts/run_inference.py case_studies/sf_bay_rtma/stage_data.py
git commit -m "$(cat <<'EOF'
refactor: route raw-data paths through get_data_dir (no in-repo data/)

preprocess_training, preprocess_inference, run_inference and sf_bay_rtma
stage_data now resolve raw inputs via get_data_dir (external raw_data) instead
of case_dir/data/raw.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Remove repo placeholder dirs + update `.gitignore`

**Files:** delete 14 `.gitkeep`s under `case_studies/*/{data,results,checkpoints,logs,outputs}/`; modify `.gitignore`

- [ ] **Step 1: Remove the placeholder dirs from git**
```bash
cd /d/Git/cosmos-wind-cnn
git rm -r --quiet \
  case_studies/_template/checkpoints case_studies/_template/data case_studies/_template/logs case_studies/_template/outputs \
  case_studies/puget_sound/checkpoints case_studies/puget_sound/data case_studies/puget_sound/logs case_studies/puget_sound/outputs \
  case_studies/sf_bay_conus404/data case_studies/sf_bay_conus404/results \
  case_studies/sf_bay_rtma/data case_studies/sf_bay_rtma/results
git status --short | grep "^D" | wc -l | xargs echo "deleted entries:"
```
Expected: 14 `.gitkeep` deletions. (These dirs only ever held `.gitkeep`.)

- [ ] **Step 2: Update `.gitignore`**

Replace the existing output/results block:
```gitignore
# Output files (generated)
**/outputs/**
!**/outputs/.gitkeep

# Per-run output trees (results/<run_id>/...) — large, regenerated, never tracked
**/results/**
!**/results/.gitkeep
```
with:
```gitignore
# Case-study data + run outputs live OUTSIDE the repo
# (COSMOS_DATA_ROOT / COSMOS_RESULTS_ROOT). Ignore any locally-created scaffolding.
case_studies/*/data/
case_studies/*/results/
case_studies/*/checkpoints/
case_studies/*/logs/
case_studies/*/outputs/
```
Also delete the now-moot line `!**/.gitkeep` (search for it near the bottom of `.gitignore`).

- [ ] **Step 3: Verify repo now holds only configs + README per case study**
```bash
cd /d/Git/cosmos-wind-cnn
echo "=== tracked under case_studies/ (should be only configs/, README.md, stage_data.py) ==="
git ls-files case_studies/ | sed -E 's#(case_studies/[^/]+)/([^/]+).*#\1/\2#' | sort -u
echo "=== no .gitkeep left tracked ==="
git ls-files case_studies/ | grep "\.gitkeep" && echo "FAIL gitkeep tracked" || echo "ok: no gitkeeps"
```
Expected: each case study shows only `configs/...`, `README.md` (and `sf_bay_rtma/stage_data.py`); `ok: no gitkeeps`.

- [ ] **Step 4: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
git add -A
git commit -m "$(cat <<'EOF'
cleanup: drop in-repo case-study data/results scaffolding

Data and run outputs now live outside the repo. Remove the empty
data/results/checkpoints/logs/outputs placeholder dirs (.gitkeep) from every
case study and ignore any locally-created scaffolding. Repo keeps only
configs/ + README per study.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Update SLURM env values to the project base

**Files:** `scripts/gpu_tallgrass.slurm`, `scripts/gpu_tallgrass_rtma.slurm`, `scripts/gpu_tallgrass_rtma_fresh.slurm`, `scripts/gpu_tallgrass_rtma_infer.slurm`, `scripts/cpu_rtma_eval.slurm`

- [ ] **Step 1: Inspect current values**
```bash
cd /d/Git/cosmos-wind-cnn
git grep -nE "export COSMOS_(DATA|RESULTS)_ROOT" -- scripts/*.slurm
```
Note: the code now appends `<case>/raw_data` and `<case>/results/<job>`, so the env values must be the **project base** (NOT end in `/data` or `/results`, which would double the path).

- [ ] **Step 2: Point both vars at the project base in each file**

In every file above, set both exports to the caldera project base (drop any trailing `/data` or `/results`):
```bash
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn
```
Use the Edit tool per file; preserve surrounding comments. If a file's existing base path differs from the one above, keep that file's base but still strip the trailing `/data` / `/results` so the structure resolves to `<base>/<case>/raw_data` and `<base>/<case>/results/<job>`. Update any nearby comment that describes the old `<case_name>/raw` or `<case_name>/<job>` layout to `<case_name>/raw_data` and `<case_name>/results/<job>`.

- [ ] **Step 3: Verify**
```bash
cd /d/Git/cosmos-wind-cnn
for f in gpu_tallgrass gpu_tallgrass_rtma gpu_tallgrass_rtma_fresh gpu_tallgrass_rtma_infer cpu_rtma_eval; do
  bash -n "scripts/$f.slurm" && echo "ok bash: $f"
done
echo "=== no env value ends in /data or /results (would double the path) ==="
git grep -nE "export COSMOS_(DATA|RESULTS)_ROOT=.*/(data|results)\b" -- scripts/*.slurm && echo "FAIL trailing /data or /results" || echo "ok: bases clean"
```
Expected: `ok bash` for all 5; `ok: bases clean`.

- [ ] **Step 4: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
git add scripts/*.slurm
git commit -m "$(cat <<'EOF'
chore: point SLURM COSMOS_*_ROOT at the project base

The path helpers now append <case>/raw_data and <case>/results/<job>, so the
env vars must be the base (not .../data or .../results) to avoid a doubled path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Documentation

**Files:** `README.md`, `case_studies/*/README.md`, `docs/data_preparation.md`, `docs/adding_case_study.md`, launcher `CLAUDE.md` (`C:\Users\keesn\.claude_projects\cosmos-wind-cnn\CLAUDE.md`, outside the repo — edited in place, not committed)

- [ ] **Step 1: Find storage references to update**
```bash
cd /d/Git/cosmos-wind-cnn
git grep -nE "data/raw|data/processed|results/<run|case_studies/[^ ]*/data|case_studies/[^ ]*/results|raw inputs|raw data|/results/" -- '*.md' | grep -v docs/superpowers | head -40
```

- [ ] **Step 2: Update the docs**

In each doc, replace descriptions of in-repo `case_studies/<case>/data/raw` / `case_studies/<case>/results/<run>` with the external layout, and add the env-var setup. Key edits:
- **`_template/README.md`** and **`docs/adding_case_study.md`** (canonical guides): state that a case study in the repo contains only `configs/` + `README.md`; raw data goes in `$COSMOS_DATA_ROOT/<case>/raw_data/`, run outputs in `$COSMOS_RESULTS_ROOT/<case>/results/<job>/`; set both env vars before running (locally both = your storage drive, e.g. `G:\03-downscaling_meteo_cnn`).
- **`docs/data_preparation.md`**: prepared raw NetCDFs go to `$COSMOS_DATA_ROOT/<case>/raw_data/` (not `case_studies/<case>/data/raw`).
- **case-study READMEs** (`sf_bay_conus404`, `sf_bay_rtma`, `puget_sound`): update any "Run Output Structure" / data-location section to the external layout; show the `set COSMOS_DATA_ROOT=...` / `export ...` step before the example commands.
- **top-level `README.md`**: in the dir-map / quickstart, note data + results are external (env vars); the repo case-study folders hold only `configs/` + `README.md`.
- **launcher `CLAUDE.md`**: update the "Per-run isolation" / repo-layout bullet to say data lives at `$COSMOS_DATA_ROOT/<case>/raw_data` and runs at `$COSMOS_RESULTS_ROOT/<case>/results/<job>`; both env vars are required.

- [ ] **Step 3: Verify**
```bash
cd /d/Git/cosmos-wind-cnn
echo "=== no doc still tells users to put data in case_studies/<case>/data/raw ==="
git grep -nE "case_studies/[a-z_]+/data/raw|case_studies/[a-z_]+/results/" -- '*.md' | grep -v docs/superpowers || echo "ok: docs point to external storage"
```
Expected: `ok: docs point to external storage` (or only legitimate mentions remain — review each).

- [ ] **Step 4: Commit (repo docs only; CLAUDE.md is outside the repo)**
```bash
cd /d/Git/cosmos-wind-cnn
git add README.md case_studies/*/README.md docs/data_preparation.md docs/adding_case_study.md
git commit -m "$(cat <<'EOF'
docs: document external case-study storage (COSMOS_DATA_ROOT/RESULTS_ROOT)

Repo case studies hold only configs/ + README. Raw data lives in
<COSMOS_DATA_ROOT>/<case>/raw_data and run outputs in
<COSMOS_RESULTS_ROOT>/<case>/results/<job>. Document the env-var setup.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Final verification + finalize

- [ ] **Step 1: Full sweep**
```bash
cd /d/Git/cosmos-wind-cnn
PY="C:/Users/keesn/anaconda3/envs/cosmos_wind_cnn/python.exe"
"$PY" -m py_compile scripts/*.py src/cosmos_wind_cnn/**/*.py case_studies/sf_bay_rtma/stage_data.py && echo "all compile"
"$PY" -m pytest -q
echo "=== repo holds only configs + README (+ stage_data) per case study ==="
git ls-files case_studies/ | grep -vE "/configs/|/README.md$|/stage_data.py$" || echo "ok: nothing else tracked"
echo "=== helpers resolve to the target layout (smoke) ==="
COSMOS_DATA_ROOT=/tmp/store COSMOS_RESULTS_ROOT=/tmp/store "$PY" - <<'PYEOF'
from cosmos_wind_cnn.utils.config import get_data_dir, get_run_dirs
print('data:', get_data_dir('case_studies/sf_bay_rtma'))
print('run :', get_run_dirs('case_studies/sf_bay_rtma', '3732177')['run_root'])
PYEOF
echo "=== unset -> clear error ==="
"$PY" - <<'PYEOF'
import os
for v in ('COSMOS_DATA_ROOT','COSMOS_RESULTS_ROOT'): os.environ.pop(v, None)
from cosmos_wind_cnn.utils.config import get_data_dir
try:
    get_data_dir('case_studies/sf_bay_rtma'); print('FAIL: no error')
except RuntimeError as e:
    print('ok error:', str(e).splitlines()[0])
PYEOF
git --no-pager diff --stat main..HEAD | tail -15
```
Expected: `all compile`; tests pass; `ok: nothing else tracked`; the smoke prints `.../sf_bay_rtma/raw_data` and `.../sf_bay_rtma/results/3732177`; the unset case prints `ok error: COSMOS_DATA_ROOT is not set...`.

- [ ] **Step 2: Finalize** — use `superpowers:finishing-a-development-branch` to integrate `restructure/external-storage` (offer merge-to-main-locally / push+PR / keep / discard). Do not push without Kees's go-ahead.

---

## Rollback
All work is on `restructure/external-storage`; `main` is the integration point.
- Undo last commit (keep changes): `git reset --soft HEAD~1`
- Abandon branch: `git checkout main && git branch -D restructure/external-storage`

## Notes / out of scope
- **Data migration is manual** (Kees's choice): move/rsync existing raw inputs to `$COSMOS_DATA_ROOT/<case>/raw_data/` and any results to `$COSMOS_RESULTS_ROOT/<case>/results/<job>/`. The `g:\03-downscaling_meteo_cnn\{sf_bay_conus404,sf_bay_rtma}\` folders already exist.
- Running any pipeline/script now requires both env vars set (fail-fast by design).
- HPC: with both vars pointed at the project base, raw data resolves to `<base>/<case>/raw_data` and runs to `<base>/<case>/results/<job>` — the SLURM log-copy `LOG_DIR` in `gpu_tallgrass.slurm` (already `${COSMOS_RESULTS_ROOT}/$(basename "$CASE_STUDY")/...`) should be reviewed to include the new `results/<job>` level if exact alignment is wanted (minor; the pipeline itself uses `get_run_dirs`).

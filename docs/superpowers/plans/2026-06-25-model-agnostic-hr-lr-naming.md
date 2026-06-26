# Model-Agnostic HR/LR Variable Naming — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the whole workflow model-agnostic by renaming variable keys from source-specific (`conus404_`/`rtma_`/`era5_`) to role-based `hr_` (high-resolution target) / `lr_` (low-resolution input), so CONUS404→ERA5 and RTMA→ERA5 (and future pairings) are configured identically. Actual source identity is preserved as `hr_source`/`lr_source` metadata.

**Architecture:** The codebase is already ~70% agnostic — `variable_pairs` use `high_res`/`low_res` roles, `classify_file_keys`/`var_units_for`/`wind_var_names`/`parse_variable_config` operate on configurable prefixes / suffixes / pair-values. This refactor (1) renames the variable **keys** in all case-study configs, (2) flips two code defaults to `hr_`/`lr_`, (3) replaces hardcoded `'CONUS404'`/`'ERA5'` plot labels and adds `hr_source`/`lr_source` to output NetCDF attributes, and (4) updates docs. Clean break — old gitignored experiment runs are not migrated.

**Tech Stack:** Python 3.11 package `cosmos_wind_cnn` (conda env `cosmos_wind_cnn`), YAML configs, pytest. Bash tool = Git Bash on Windows.

---

## Naming rules (apply everywhere)

| Old key prefix | New | Notes |
|---|---|---|
| `conus404_` (sf_bay, puget_sound, _template) | `hr_` | high-res target |
| `rtma_` (sf_bay_rtma) | `hr_` | high-res target |
| `era5_` (all) | `lr_` | low-res input |

- Variable-type **suffixes are unchanged**: `u, v, air_temp, dew_temp, pressure, solar, thermal, rain`. So `conus404_air_temp`→`hr_air_temp`, `era5_u`→`lr_u`, `rtma_v`→`hr_v`.
- **Pair names** (`wind_u`, `air_temperature`, …) and **role keys** (`high_res`/`low_res`) are already generic — DO NOT change them.
- **PRESERVE provenance:** real filenames (`CONUS404_SFbay_4km_…nc`, `ERA5_…nc`, `RTMA_…nc`) stay verbatim; factual notes ("CONUS404 covers 1979–2021") stay; case-study run-script names (`gpu_tallgrass_rtma.slurm`) stay. Only rename *structural identifiers* (variable keys, prefix defaults, plot labels).
- `hr_source`/`lr_source` values record provenance: sf_bay→`CONUS404`/`ERA5`, sf_bay_rtma→`RTMA`/`ERA5`, puget_sound→`CONUS404`/`ERA5`, _template→placeholder.

## File Structure (what changes)

| Path | Action |
|---|---|
| `src/cosmos_wind_cnn/utils/config.py` | `classify_file_keys` defaults → `'hr_'`/`'lr_'`; update a comment |
| `src/cosmos_wind_cnn/data/preprocessing.py` | prefix defaults → `'hr_'`/`'lr_'`; generalize docstring/comment examples |
| `src/cosmos_wind_cnn/data/regridder.py` | generalize docstring examples |
| `tests/test_config_helpers.py` | update example prefixes + defaults test to `hr_`/`lr_` |
| `case_studies/{sf_bay,sf_bay_rtma,puget_sound,_template}/configs/preprocessing.yaml` | rename `file_dict` + `physical_bounds` keys; drop `target_prefix`/`input_prefix` |
| `…/configs/training.yaml` (×4) | rename `variable_pairs.{high_res,low_res}` values; add `hr_source`/`lr_source` |
| `…/configs/inference_preprocessing.yaml` (×4) | rename `sources:` + `physical_bounds` keys to `lr_*` |
| `src/cosmos_wind_cnn/utils/visualization.py` | parameterize labels (`hr_label`/`lr_label`); drop hardcoded `'CONUS404'`/`'ERA5'` |
| `scripts/run_training_pipeline.py` | write `hr_source`/`lr_source` NetCDF attrs; pass labels to eval plots |
| `README.md`, `case_studies/*/README.md`, `docs/{adding_case_study,data_preparation,model_architecture}.md`, `pyproject.toml` | generalize prose/examples to HR/LR |
| **Out of scope** | `notebooks/*.ipynb` (deferred follow-up) |

---

### Task 0: Branch

- [ ] **Step 1: Create the refactor branch from clean main**
```bash
cd /d/Git/cosmos-wind-cnn
git status --short    # expect clean
git checkout -b refactor/hr-lr-naming
git branch --show-current
```
Expected: on `refactor/hr-lr-naming`, clean tree.

---

### Task 1: Core code defaults + docstring examples + tests

**Files:** `src/cosmos_wind_cnn/utils/config.py`, `src/cosmos_wind_cnn/data/preprocessing.py`, `src/cosmos_wind_cnn/data/regridder.py`, `tests/test_config_helpers.py`

- [ ] **Step 1: `config.py` — flip `classify_file_keys` defaults**

Edit `def classify_file_keys(file_dict, target_prefix: str = 'conus404_', input_prefix: str = 'era5_'):` →
`def classify_file_keys(file_dict, target_prefix: str = 'hr_', input_prefix: str = 'lr_'):`

And update the comment above `_UNIT_BY_SUFFIX` from `# Units keyed by variable-name suffix (prefix-agnostic: works for conus404_*, rtma_*, ...)` →
`# Units keyed by variable-name suffix (prefix-agnostic: works for hr_*, lr_*, ...)`

- [ ] **Step 2: `preprocessing.py` — flip prefix defaults + generalize example text**

Edit lines 31–32:
```python
        self.target_prefix = config.get('target_prefix', 'conus404_')
        self.input_prefix = config.get('input_prefix', 'era5_')
```
→
```python
        self.target_prefix = config.get('target_prefix', 'hr_')
        self.input_prefix = config.get('input_prefix', 'lr_')
```
Then generalize the *example* mentions of CONUS404/ERA5 in comments/docstrings WITHOUT changing logic. Find them with:
```bash
cd /d/Git/cosmos-wind-cnn
grep -nE "conus404|era5|CONUS404|ERA5" src/cosmos_wind_cnn/data/preprocessing.py
```
Replace example variable keys `conus404_x`→`hr_x`, `era5_x`→`lr_x` (e.g. line ~92 "Defaults: conus404_ / era5_" → "Defaults: hr_ / lr_"; "Get 'u' from 'era5_u' or 'conus404_u'" → "Get 'u' from 'lr_u' or 'hr_u'"; "conus404_v: {min...}" → "hr_v: {min...}"; "conus404_air_temp" → "hr_air_temp"). Replace narrative "load all CONUS404 DataArrays" → "load all HR (target) DataArrays" and "interpolate onto CONUS404 grid"/"ERA5 interpolation" → "interpolate onto HR grid"/"LR interpolation". Keep any genuinely factual provenance untouched.

- [ ] **Step 3: `regridder.py` — generalize docstring examples**
```bash
grep -nE "conus404|era5" src/cosmos_wind_cnn/data/regridder.py
```
In the docstring examples: `conus404_ds`→`hr_ds`, `era5_ds`→`lr_ds`, `regridded_era5`→`regridded_lr`, and `{'era5_u': 'u10', 'era5_v': 'v10'}`→`{'lr_u': 'u10', 'lr_v': 'v10'}`. These are usage examples only; the regridder stays grid-agnostic.

- [ ] **Step 4: `tests/test_config_helpers.py` — update to hr_/lr_**

Replace the file body's tests so example keys use `hr_`/`lr_` and the defaults test asserts the new defaults. Use the Edit tool for each:

(a) `test_classify_file_keys_rtma_prefix` → rename to `test_classify_file_keys_explicit_prefix` and use explicit `hr_`/`lr_`:
```python
def test_classify_file_keys_explicit_prefix():
    file_dict = {
        'hr_u': 'a.nc', 'lr_u': 'b.nc',
        'hr_pressure': 'c.nc', 'lr_pressure': 'd.nc',
        'static_terrain': 'e.nc',
    }
    target, inp, other = classify_file_keys(
        file_dict, target_prefix='hr_', input_prefix='lr_'
    )
    assert target == ['hr_u', 'hr_pressure']
    assert inp == ['lr_u', 'lr_pressure']
    assert other == ['static_terrain']
```

(b) `test_classify_file_keys_defaults_conus404` → rename to `test_classify_file_keys_defaults_hr_lr`:
```python
def test_classify_file_keys_defaults_hr_lr():
    file_dict = {'hr_u': 'a.nc', 'lr_u': 'b.nc'}
    target, inp, other = classify_file_keys(file_dict)
    assert target == ['hr_u']
    assert inp == ['lr_u']
    assert other == []
```

(c) `test_var_units_for_rtma_and_conus404` → rename to `test_var_units_for_hr_and_lr`:
```python
def test_var_units_for_hr_and_lr():
    units = var_units_for(['hr_u', 'hr_v', 'hr_air_temp',
                           'hr_dew_temp', 'hr_pressure', 'hr_rain'])
    assert units == {
        'hr_u': 'm s**-1', 'hr_v': 'm s**-1',
        'hr_air_temp': 'K', 'hr_dew_temp': 'K',
        'hr_pressure': 'Pa', 'hr_rain': 'mm hr**-1',
    }
    rad = var_units_for(['lr_solar', 'lr_thermal'])
    assert rad == {'lr_solar': 'W m**-2', 'lr_thermal': 'W m**-2'}
```

(d) `test_var_units_for_skips_unknown`: change `['rtma_visibility']` → `['hr_visibility']`.

(e) `test_wind_var_names_rtma` → rename to `test_wind_var_names_hr_lr`:
```python
def test_wind_var_names_hr_lr():
    variable_pairs = {
        'wind_u': {'high_res': 'hr_u', 'low_res': 'lr_u'},
        'wind_v': {'high_res': 'hr_v', 'low_res': 'lr_v'},
        'pressure': {'high_res': 'hr_pressure', 'low_res': 'lr_pressure'},
    }
    assert wind_var_names(variable_pairs) == ('hr_u', 'hr_v', 'lr_u', 'lr_v')
```

(f) `test_wind_var_names_none_when_absent`: change `'rtma_pressure'`/`'era5_pressure'` → `'hr_pressure'`/`'lr_pressure'`.

- [ ] **Step 5: Verify + commit**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn python -m py_compile src/cosmos_wind_cnn/utils/config.py src/cosmos_wind_cnn/data/preprocessing.py src/cosmos_wind_cnn/data/regridder.py && echo "compile ok"
conda run -n cosmos_wind_cnn pytest -q
# defaults really are hr_/lr_ now:
conda run -n cosmos_wind_cnn python -c "import inspect; from cosmos_wind_cnn.utils.config import classify_file_keys as f; d=inspect.signature(f).parameters; print(d['target_prefix'].default, d['input_prefix'].default)"
```
Expected: `compile ok`; all tests PASS; printed defaults `hr_ lr_`.
```bash
git add src/cosmos_wind_cnn/utils/config.py src/cosmos_wind_cnn/data/preprocessing.py src/cosmos_wind_cnn/data/regridder.py tests/test_config_helpers.py
git commit -m "$(cat <<'EOF'
refactor: default variable-key prefixes to hr_/lr_ (model-agnostic)

classify_file_keys and NetCDFPreprocessor now default to hr_/lr_ instead of
conus404_/era5_. Docstring examples and config-helper tests updated to the
new convention. Logic was already prefix-agnostic; only defaults/examples change.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Rename variable keys in all case-study configs + add provenance

**Files (12):** for each of `sf_bay`, `sf_bay_rtma`, `puget_sound`, `_template`: `configs/preprocessing.yaml`, `configs/training.yaml`, `configs/inference_preprocessing.yaml`.

Apply the **Naming rules** table. Source prefix is `conus404_` for sf_bay/puget_sound/_template and `rtma_` for sf_bay_rtma; `era5_`→`lr_` everywhere.

- [ ] **Step 1: `preprocessing.yaml` (×4)** — In `file_dict` and `physical_bounds`, rename every key: `conus404_*`/`rtma_*`→`hr_*`, `era5_*`→`lr_*`. **Keep filenames and min/max values and provenance comments verbatim.** In `sf_bay_rtma/configs/preprocessing.yaml`, also DELETE the now-redundant lines `target_prefix: 'rtma_'` and `input_prefix: 'era5_'` (the new `hr_`/`lr_` defaults cover them).

- [ ] **Step 2: `training.yaml` (×4)** — In `variable_pairs`, rename each pair's `high_res:` value (`conus404_*`/`rtma_*`→`hr_*`) and `low_res:` value (`era5_*`→`lr_*`). Leave pair names unchanged. Then add two top-level fields near the top (after the title comment, before `variable_pairs`):
  - sf_bay: `hr_source: CONUS404` / `lr_source: ERA5`
  - sf_bay_rtma: `hr_source: RTMA` / `lr_source: ERA5`
  - puget_sound: `hr_source: CONUS404` / `lr_source: ERA5`
  - _template: `hr_source: 'YOUR_HIGH_RES_SOURCE'` / `lr_source: 'YOUR_LOW_RES_SOURCE'`
  Add a one-line comment: `# Provenance labels for output metadata + plots (model-agnostic keys above)`.

- [ ] **Step 3: `inference_preprocessing.yaml` (×4)** — Under `sources:`, rename each key `era5_*`→`lr_*` (these are the LR inputs). If a `physical_bounds:` block exists, rename its `era5_*`→`lr_*` too. Keep filenames/values.

- [ ] **Step 4: Verify — every case study loads and partitions into hr_/lr_, and no structural key prefixes remain**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn python - <<'PY'
from pathlib import Path
from cosmos_wind_cnn.utils.config import load_config, parse_variable_config, classify_file_keys, wind_var_names
for cs in ['sf_bay', 'sf_bay_rtma', 'puget_sound', '_template']:
    base = Path('case_studies') / cs / 'configs'
    tr = load_config(base / 'training.yaml')
    pp = load_config(base / 'preprocessing.yaml')
    iv, ov, wp = parse_variable_config(tr)
    assert all(v.startswith('lr_') for v in iv), (cs, iv)
    assert all(v.startswith('hr_') for v in ov), (cs, ov)
    tk, ik, ok = classify_file_keys(pp['file_dict'])
    assert tk and ik, (cs, tk, ik)
    assert 'hr_source' in tr and 'lr_source' in tr, cs
    print(f"{cs}: in={iv[:2]} out={ov[:2]} hr_source={tr['hr_source']} lr_source={tr['lr_source']} OK")
print("all case studies OK")
PY
# No structural key prefixes left in configs (filenames are uppercase CONUS404_/ERA5_/RTMA_ and remain):
git grep -nE "(conus404|era5|rtma)_[a-z]" -- 'case_studies/**/*.yaml' | grep -vE "\.nc'|\.nc\"" ; echo "<-- expected: EMPTY (only .nc filenames may contain them, which are excluded)"
```
Expected: `all case studies OK`; the grep is EMPTY.

- [ ] **Step 5: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn pytest -q
git add case_studies/*/configs/*.yaml
git commit -m "$(cat <<'EOF'
refactor: rename case-study config variable keys to hr_/lr_

All file_dict / physical_bounds / variable_pairs / sources keys use the
model-agnostic hr_/lr_ scheme. Added hr_source/lr_source provenance labels to
each training.yaml. Filenames and bounds values unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Provenance plumbing — plot labels + output NetCDF attributes

**Files:** `src/cosmos_wind_cnn/utils/visualization.py`, `scripts/run_training_pipeline.py`

- [ ] **Step 1: Parameterize `visualization.py` labels**

In `_plot_wind_sample` and `_plot_scalar_sample`, add keyword params `hr_label='HR', lr_label='LR'` to the signatures, and replace the hardcoded source names in the subplot titles:
- `'Input U (ERA5)'`→`f'Input U ({lr_label})'`, `'Target U (CONUS404)'`→`f'Target U ({hr_label})'`
- `'Input V (ERA5)'`→`f'Input V ({lr_label})'`, `'Target V (CONUS404)'`→`f'Target V ({hr_label})'`
- `'Input Speed (ERA5)'`→`f'Input Speed ({lr_label})'`, `'Target Speed (CONUS404)'`→`f'Target Speed ({hr_label})'`
- `'Input (ERA5)'`→`f'Input ({lr_label})'`, `'Target (CONUS404)'`→`f'Target ({hr_label})'`

Then thread `hr_label`/`lr_label` from the public entry function (the one containing the `_plot_wind_sample`/`_plot_scalar_sample` calls, around lines 40–81) down into both calls — add `hr_label='HR', lr_label='LR'` params to that function and pass them through. Read the file first to get the exact signature.

- [ ] **Step 2: `run_training_pipeline.py` — write provenance attrs + pass labels**

(a) Where the inference output NetCDF global attributes are written (near `nc.crs = ...` / `nc.run_name = ...`), add:
```python
    nc.hr_source = str(train_config.get('hr_source', 'HR'))
    nc.lr_source = str(train_config.get('lr_source', 'LR'))
```
(b) In the evaluation step, where the visualization entry function is called to make sample plots, pass:
```python
        hr_label=train_config.get('hr_source', 'HR'),
        lr_label=train_config.get('lr_source', 'LR'),
```
Read the relevant regions first (`grep -n "hr_source\|create_.*plot\|_plot_\|\.crs\b\|visualization" scripts/run_training_pipeline.py` and inspect the visualization import) to wire the exact call site.

- [ ] **Step 3: Verify + commit**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn python -m py_compile src/cosmos_wind_cnn/utils/visualization.py scripts/run_training_pipeline.py && echo "compile ok"
echo "--- no hardcoded source labels remain in visualization.py ---"
grep -nE "CONUS404|ERA5|RTMA" src/cosmos_wind_cnn/utils/visualization.py || echo "none — good"
echo "--- attrs + labels wired ---"
grep -n "hr_source\|lr_source\|hr_label\|lr_label" scripts/run_training_pipeline.py src/cosmos_wind_cnn/utils/visualization.py | head
conda run -n cosmos_wind_cnn pytest -q
git add src/cosmos_wind_cnn/utils/visualization.py scripts/run_training_pipeline.py
git commit -m "$(cat <<'EOF'
refactor: drive plot labels + output attrs from hr_source/lr_source

visualization sample plots take hr_label/lr_label (default HR/LR) instead of
hardcoded CONUS404/ERA5; inference output NetCDFs gain hr_source/lr_source
global attributes from the training config.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: `compile ok`; `none — good`; grep shows wiring; tests PASS.

---

### Task 4: Documentation

**Files:** `README.md`, `case_studies/{sf_bay,sf_bay_rtma,puget_sound,_template}/README.md`, `docs/adding_case_study.md`, `docs/data_preparation.md`, `docs/model_architecture.md`, `pyproject.toml`

- [ ] **Step 1: Generalize prose + config examples to HR/LR**

For each file: find structural references with
```bash
grep -nE "conus404_[a-z]|era5_[a-z]|rtma_[a-z]" <file>
```
and replace variable-key *examples* with `hr_`/`lr_` (e.g. config snippets showing `conus404_u: ...`→`hr_u: ...`). Generalize narrative that describes the *workflow* (e.g. "downscale ERA5 → CONUS404" → "downscale low-res (LR) → high-res (HR)") while KEEPING factual statements about the specific case studies (e.g. "the sf_bay study uses CONUS404 (1979–2021) as HR and ERA5 as LR"). In `_template/README.md`, document the `hr_`/`lr_` key convention and the `hr_source`/`lr_source` fields as the canonical example. In `pyproject.toml`, the `description` may keep "ERA5 to CONUS404 resolution" as an illustrative example or generalize to "coarse reanalysis to high-resolution" — generalize it.

- [ ] **Step 2: Verify + commit**
```bash
cd /d/Git/cosmos-wind-cnn
echo "--- remaining structural key examples in docs (should be empty) ---"
git grep -nE "(conus404|era5|rtma)_[a-z]" -- '*.md' 'pyproject.toml' | grep -vE "\.nc" ; echo "<-- expected empty"
git add README.md case_studies/*/README.md docs/adding_case_study.md docs/data_preparation.md docs/model_architecture.md pyproject.toml
git commit -m "$(cat <<'EOF'
docs: describe the model-agnostic hr_/lr_ convention

Generalize workflow prose and config examples to HR/LR; keep factual
per-case-study provenance. _template documents the hr_/lr_ + hr_source/lr_source
convention as the canonical example.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: doc grep empty; commit succeeds.

---

### Task 5: Final verification

- [ ] **Step 1: Full sweep**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn python -m py_compile scripts/*.py src/cosmos_wind_cnn/**/*.py && echo "all compile"
conda run -n cosmos_wind_cnn pytest -q
echo "=== remaining lowercase conus404_/era5_/rtma_ STRUCTURAL tokens (should be only .nc filenames) ==="
git grep -nE "(conus404|era5|rtma)_[a-z]" -- . ':!docs/superpowers' ':!notebooks' | grep -vE "\.nc'|\.nc\"|\.nc\b"
echo "<-- review: every remaining hit must be a real filename, not a variable key"
echo "=== all case studies still load + partition hr_/lr_ ==="
conda run -n cosmos_wind_cnn python - <<'PY'
from pathlib import Path
from cosmos_wind_cnn.utils.config import load_config, parse_variable_config
for cs in ['sf_bay','sf_bay_rtma','puget_sound','_template']:
    tr = load_config(Path('case_studies')/cs/'configs'/'training.yaml')
    iv, ov, _ = parse_variable_config(tr)
    assert all(v.startswith('lr_') for v in iv) and all(v.startswith('hr_') for v in ov)
print("configs OK")
PY
git --no-pager diff --stat main..HEAD | tail -20
```
Expected: `all compile`; tests PASS; the structural-token grep shows ONLY `.nc` filenames (uppercase `CONUS404_`/`ERA5_`/`RTMA_`) — no lowercase variable keys; `configs OK`.

- [ ] **Step 2: Finalize** — use `superpowers:finishing-a-development-branch` to integrate `refactor/hr-lr-naming` (offer merge-to-main-locally / push+PR / keep / discard). Do not push without Kees's go-ahead.

---

## Rollback
All work is on `refactor/hr-lr-naming`; `main` is untouched until finalize.
- Undo last commit (keep changes): `git reset --soft HEAD~1`
- Abandon branch: `git checkout main && git branch -D refactor/hr-lr-naming`

## Notes / follow-ups
- **Notebooks** (`notebooks/01_…`, `02_…`, `03_…`) still reference old keys — deferred per decision; update in a separate pass.
- Old gitignored experiment runs under `results/` keep their old-name archived configs and are not migrated (clean break). To re-run one, re-preprocess with the renamed configs.

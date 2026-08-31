# CNN-quantile-v3 validation (IEM+NDBC+USGS, quantile skill, per-station Taylor, faceted bars) — Plan

**Goal:** Register the new `20260827_validation\inference\` record as a validation product, run it through the existing three-era obs track (E1/E2/E3) with IEM+NDBC+USGS only (no CWOP, equal weight 1), and redesign the combined-skill outputs: drop the top-10% image in favor of a quantile-bin CSV, add per-station+pooled-marker Taylor diagrams (per category and combined), and add a faceted per-category × per-variable bar chart.

**Approach:** Register `CNN-quantile-v3` as one new multifile `MODELS` entry in `config.py` (same glob pattern/caveat as the existing `V3_ERA_MODELS`). Point `run_validation.py`'s three-era track at it instead of the three seeds. Run E1/E2/E3 with `VAL_USGS=1` into fresh, dated output directories so the published `_v20260810` results are untouched. Generalize the top-10% stat in `validate_met_models.py` into a 5-bin quantile function that writes CSV rows only (no plots). Rewrite `combined_skill.py`'s `taylor()` to plot per-station markers plus a bold pooled/weighted marker with centered-RMS-difference arcs (porting the working pattern from `rainfall_analysis/run_validation_figures.py`'s `fig09_taylor()`), produced both per-category and combined. Add a new faceted bar chart (per era, 3 subplots: speed/u10/v10, grouped bars by category) alongside the existing flat pooled bar chart. Add a small `skill_by_quantile.py` script that pools the new quantile-bin rows the same way `combine_skill()` pools everything else.

**Depends on:** `D:\Git\cosmos-wind-cnn\validation` package (env `COSMOS_VALIDATION_DATA_ROOT=g:\01_meteorlogical_analysis_sfbay`, `COSMOS_VALIDATION_OUTPUT_ROOT=g:\01_meteorlogical_analysis_sfbay\results\20260827_validation`, `cosmos_wind_cnn` conda env). Inference NetCDFs already staged at `g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\inference\`. No HPC step — everything below runs on the Windows desktop against data already on `G:`.

**Design:** none (this plan folds in the design decisions directly — see "Decisions already made" below).

---

## Decisions already made (do not re-ask)

- New product is registered as a **single** `MODELS` entry `'CNN-quantile-v3'`, not three seeds.
- Station scope: **IEM + NDBC + USGS**, CWOP dropped entirely. `config.STATION_GROUPS` already defaults to `['IEM', 'NDBC']` (CWOP is never loaded by default — see `validate_met_models.py:83`), so no code change is needed to exclude CWOP; USGS is included by setting `VAL_USGS=1` at run time (`config.py:67`, `INCLUDE_USGS_MOORINGS`).
- Category weights become the new default: `{'IEM': 1.0, 'NDBC': 1.0, 'USGS': 1.0}` (was `{'USGS': 2.0, 'NDBC': 1.0, 'IEM': 1.0, 'CWOP': 0.5}`). The `VAL_WEIGHTS` env override stays, so anyone re-running an old era for a like-for-like comparison can still pass the old weights.
- Top-10% image (`combined_skill_speed_top10.png`, `combined_taylor_speed_top10.png`) is removed entirely, including the **per-station** top-10% scatter PNG — replaced by a CSV-only quantile-bin stat: quartiles + tail (`q00-25`, `q25-50`, `q50-75`, `q75-90`, `q90-100` of observed wind speed), **wind speed only** (not u10/v10).
- Taylor diagram style: port `rainfall_analysis/run_validation_figures.py`'s `fig09_taylor()` pattern (per-station small transparent markers, product-specific color+shape, centered-RMS-difference dotted arcs about the (r=1, nstd=1) reference point, dashed r=1 arc, star obs marker, explanatory textbox). Deviation from that reference: the bold "network" marker is the **pooled, sample-size-and-category-weighted** position already computed by `combine_skill()` (consistent with the bar-chart headline number), not a naive per-station median.
- Taylor diagrams are produced **both** per station-category (IEM-only / NDBC-only / USGS-only) **and** combined/pooled — for every variable already in `SKILL_VARS` (speed, u10, v10, temp, pressure, dewpoint, rh, radiation, precip), same as today.
- The reference image for the "more interesting" combined bar chart never arrived twice; per Kees's explicit fallback ("use your judgment"), the new faceted chart design below is final for this plan — flag it for a look once generated, not for re-approval before running.
- The existing flat pooled bar chart (`combined_skill_<key>.png`) is **kept** for every `SKILL_VARS` entry; the new faceted per-category chart is **added** only for speed/u10/v10, as a new file `combined_skill_by_category.png` (one PNG per era, 3 subplots).
- New output directories use the existing `_v20260810`-style dated-suffix convention, dated for this run: `obsE1_2000-2010_v20260828`, `obsE2_2011-2019_v20260828`, `obsE3_2020-2026_v20260828`. Produced via the existing `VAL_OUTDIR_SUFFIX=v20260828` env knob (`run_validation.py:31,125-126`) — no code change needed for this.

## Artifacts touched

| File | Role |
|---|---|
| `D:\Git\cosmos-wind-cnn\validation\config.py` | Register `CNN-quantile-v3` as a new `MODELS`/`MODEL_COLORS` entry |
| `D:\Git\cosmos-wind-cnn\validation\run_validation.py` | Point the three-era track (`_ERA_TR`) at the new product instead of `V3_ERA_MODELS` |
| `D:\Git\cosmos-wind-cnn\validation\validate_met_models.py` | Replace `_validate_top_percentile` with a 5-bin quantile function; update its call site and the record-building loop |
| `D:\Git\cosmos-wind-cnn\validation\analysis\combined_skill.py` | New default weights; drop `speed_top10` from `SKILL_VARS`; new `MODEL_MARKERS`/`model_marker_map`; rewritten `taylor()` (per-station + pooled marker + RMS arcs); per-category Taylor diagrams; new faceted bar chart; guard the script body under `if __name__ == '__main__':` so it becomes safely importable |
| `D:\Git\cosmos-wind-cnn\validation\analysis\skill_by_quantile.py` (new) | Pool the per-station quantile-bin rows into one tidy CSV, reusing `combine_skill()` from `combined_skill.py` |

---

### Task 1: Register `CNN-quantile-v3` in `config.py`

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\config.py` (append after the `V3_ERA_MODELS` block, i.e. after line 456)

- [ ] **Step 1: Add the new MODELS entry**

Open `config.py` and append this block immediately after the `V3_ERA_MODELS` loop (after line 456):

```python
# ---- CNN-quantile-v3: single-arm full-record inference (2026-08-27 request).
# Same multifile layout/caveat as V3_ERA_MODELS above (one arm, not three
# seeds), staged locally rather than on Caldera -- lives directly under
# DATA_ROOT/results/20260827_validation/inference/.
MODELS['CNN-quantile-v3'] = {
    'data_dir': DATA_ROOT / 'results' / '20260827_validation' / 'inference',
    'u_pattern': 'speed_full_record_ERA5_????0101_*.nc',
    'v_pattern': 'speed_full_record_ERA5_????0101_*.nc',
    'u_var': 'hr_u', 'v_var': 'hr_v', 'crs': 'utm10n',
}
MODEL_COLORS['CNN-quantile-v3'] = '#000000'
```

The `????0101` glob is load-bearing (same reasoning as `V3_ERA_MODELS`, `config.py:438-444`): `output_inference/` under `results\20260827_validation\inference\` also holds two overlapping leftover files (`speed_full_record_ERA5_20240314_20250206.nc`, `speed_full_record_ERA5_20250206_20260101.nc`) whose windows overlap the per-year files. Do not widen the glob.

Expected: `config.py` still imports cleanly (no syntax errors) — confirmed in Step 2.

- [ ] **Step 2: Verify the entry loads and resolves real files**

Run (PowerShell, `cosmos_wind_cnn` env active, `COSMOS_VALIDATION_DATA_ROOT` set to `g:\01_meteorlogical_analysis_sfbay`):

```powershell
cd D:\Git\cosmos-wind-cnn\validation
python -c "import config; from pathlib import Path; import glob; c=config.MODELS['CNN-quantile-v3']; d=c['data_dir']; print(d, d.exists()); print(len(glob.glob(str(d / c['u_pattern']))), 'files matched')"
```

Expected: prints the inference directory path with `True`, and a file count matching the number of per-year segments actually present (roughly one per year, ~27 files) — **not** including the two overlapping leftover files (so the count should be 2 less than the total file count in that folder).

---

### Task 2: Point the three-era track at the new product

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\run_validation.py` (lines 91-94)

- [ ] **Step 1: Replace the V3_ERA_MODELS import/usage with the single new product**

Change:

```python
if ERA in _ERA_TR:
    from config import V3_ERA_MODELS as _VE
    tr, outdir, _refs = _ERA_TR[ERA]
    models = list(_VE) + _refs
```

to:

```python
if ERA in _ERA_TR:
    tr, outdir, _refs = _ERA_TR[ERA]
    models = ['CNN-quantile-v3'] + _refs
```

This does **not** remove or alter `config.V3_ERA_MODELS` itself (`config.py:445-456`) — it stays registered for other uses (e.g. `VAL_MODELS=V3-ERAS-s1,...` still works via the existing `VAL_MODELS` override at `run_validation.py:105-107`).

Expected: `run_validation.py` still imports and runs (checked in Task 3).

---

### Task 3: Run E1/E2/E3 with IEM+NDBC+USGS, fresh output dirs

**Files:**
- Check: `D:\Git\cosmos-wind-cnn\validation\run_validation.py`
- Creates: `g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\obsE1_2000-2010_v20260828\`, `...\obsE2_2011-2019_v20260828\`, `...\obsE3_2020-2026_v20260828\`

- [ ] **Step 1: Run Era E1 (2000-2010, ERA5+CONUS404 refs, no RTMA)**

PowerShell, `cosmos_wind_cnn` env active, both `COSMOS_VALIDATION_DATA_ROOT` and `COSMOS_VALIDATION_OUTPUT_ROOT` set:

```powershell
cd D:\Git\cosmos-wind-cnn\validation
$env:VAL_ERA = "E1"
$env:VAL_USGS = "1"
$env:VAL_OUTDIR_SUFFIX = "v20260828"
python run_validation.py
```

Expected: console prints `=== Era E1 [ALL]: 3 models x <N> stations, ('2000-01-01', '2011-01-01') -> ...\obsE1_2000-2010_v20260828 ===` (3 models = `CNN-quantile-v3`, `ERA5`, `CONUS404`), runs to completion without exceptions, and `g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\obsE1_2000-2010_v20260828\validation_statistics.csv` exists afterward.

- [ ] **Step 2: Confirm E1's obs mix is IEM+NDBC+USGS, no CWOP**

```powershell
python -c "import pandas as pd; df = pd.read_csv(r'g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\obsE1_2000-2010_v20260828\validation_statistics.csv'); print(sorted(df['source'].unique()))"
```

Expected: `['IEM', 'NDBC', 'USGS']` (USGS may be absent for E1 specifically since `ERO20_GRZ` only covers 2020-01-22 onward per `config.py:84-93` — if so this prints `['IEM', 'NDBC']` for E1, which is correct and expected, not a bug).

- [ ] **Step 3: Run Era E2 (2011-2019, +RTMA-SFbay ref)**

```powershell
$env:VAL_ERA = "E2"
python run_validation.py
```

Expected: same pattern as Step 1, output lands in `...\obsE2_2011-2019_v20260828\`, models = `CNN-quantile-v3`, `ERA5`, `CONUS404`, `RTMA-SFbay`.

- [ ] **Step 4: Run Era E3 (2020-2026, ERA5+RTMA-SFbay refs)**

```powershell
$env:VAL_ERA = "E3"
python run_validation.py
```

Expected: output lands in `...\obsE3_2020-2026_v20260828\`; USGS stations present (`ERO20_GRZ` covers this window) — verify with the same source-check as Step 2 and confirm `USGS` appears this time.

- [ ] **Step 5: Clear the env overrides**

```powershell
Remove-Item Env:VAL_ERA, Env:VAL_USGS, Env:VAL_OUTDIR_SUFFIX
```

Expected: no error; subsequent runs in the same shell fall back to `run_validation.py`'s defaults.

---

### Task 4: Generalize the top-10% stat into a 5-bin quantile CSV-only function

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\validate_met_models.py` (lines 3017-3066 and the call site at lines 3242-3246)

- [ ] **Step 1: Replace `_validate_top_percentile` with `_validate_quantile_bins`**

Replace the block at `validate_met_models.py:3028-3066` (the `TOP_PERCENTILE_LABEL` constant and the whole `_validate_top_percentile` function) with:

```python
QUANTILE_BINS = [(0, 25, 'q00-25'), (25, 50, 'q25-50'), (50, 75, 'q50-75'),
                 (75, 90, 'q75-90'), (90, 100, 'q90-100')]


def _validate_quantile_bins(model_matched, obs_matched, common_time,
                             var_name, station_id, model_name, output_dir):
    """Per-quantile-bin stats for wind speed, CSV-only (no plots).

    Bins the OBSERVED distribution into quartiles + a tail decile
    (0-25/25-50/50-75/75-90/90-100 pct) and computes calculate_statistics()
    within each bin. Returns a list of (bin_label, stats_dict) pairs, skipping
    any bin with fewer than 10 valid points."""
    mask_valid = ~(np.isnan(model_matched) | np.isnan(obs_matched))
    if mask_valid.sum() < 20:
        return []

    obs_clean = np.where(mask_valid, obs_matched, np.nan)
    edges = np.nanpercentile(obs_clean, [b[0] for b in QUANTILE_BINS] + [100])

    out = []
    for i, (lo, hi, label) in enumerate(QUANTILE_BINS):
        lo_val, hi_val = edges[i], edges[i + 1]
        if hi == 100:
            bin_mask = mask_valid & (obs_matched >= lo_val) & (obs_matched <= hi_val)
        else:
            bin_mask = mask_valid & (obs_matched >= lo_val) & (obs_matched < hi_val)
        n_bin = bin_mask.sum()
        if n_bin < 10:
            print(f"      {label} ({lo}-{hi}%): only {n_bin} points, skipping.")
            continue
        st_bin = calculate_statistics(model_matched[bin_mask], obs_matched[bin_mask])
        if st_bin is None:
            continue
        print(f"      {label} [{lo_val:.2f},{hi_val:.2f}): N={st_bin['n']}  "
              f"RMSE={st_bin['rmse']:.3f}  bias={st_bin['bias']:.3f}  "
              f"R={st_bin['corr']:.3f}  skill={st_bin['skill']:.3f}")
        out.append((label, st_bin))
    return out
```

- [ ] **Step 2: Update the call site inside `validate_variable`**

Replace `validate_met_models.py:3017-3025`:

```python
    # Top-percentile analysis for wind speed (stats always; plot gated inside)
    st_top = None
    if not is_direction and 'Wind Speed' in var_name:
        st_top = _validate_top_percentile(
            model_matched, obs_matched, common_time,
            var_name, station_id, model_name, output_dir,
            percentile=90, make_plots=make_plots)

    return st, st_top
```

with:

```python
    # Quantile-bin analysis for wind speed (CSV-only, no plots)
    st_bins = []
    if not is_direction and 'Wind Speed' in var_name:
        st_bins = _validate_quantile_bins(
            model_matched, obs_matched, common_time,
            var_name, station_id, model_name, output_dir)

    return st, st_bins
```

- [ ] **Step 3: Update the record-building loop to write one row per bin**

Replace `validate_met_models.py:3236-3246`:

```python
                    st, st_top = result if isinstance(result, tuple) else (result, None)
                    if st is not None:
                        all_records.append({
                            'model': model_name, 'station': sid, 'source': grp,
                            'variable': var_label, **st,
                        })
                    if st_top is not None:
                        all_records.append({
                            'model': model_name, 'station': sid, 'source': grp,
                            'variable': f"{var_label} (top 10%)", **st_top,
                        })
```

with:

```python
                    st, st_bins = result if isinstance(result, tuple) else (result, [])
                    if st is not None:
                        all_records.append({
                            'model': model_name, 'station': sid, 'source': grp,
                            'variable': var_label, **st,
                        })
                    for bin_label, st_bin in st_bins:
                        all_records.append({
                            'model': model_name, 'station': sid, 'source': grp,
                            'variable': f"{var_label} ({bin_label})", **st_bin,
                        })
```

- [ ] **Step 4: Check the second call site at `validate_met_models.py:3309-3316` for the same tuple-unpacking pattern**

Read `validate_met_models.py` around line 3300-3320 (this is the CWOP-sample or scalar-variable path — confirm which). If it also unpacks `result` as `(st, st_top)` and appends a `(top 10%)` row, apply the same replacement as Step 3. If that call site only ever passes non-wind-speed variables (so `st_top`/`st_bins` is always `[]`/`None` there), leave it as a defensive no-op update: change `st, st_top = ...` to `st, st_bins = ...` for consistency, and drop the corresponding `if st_top is not None` block if present, replacing it with the same `for bin_label, st_bin in st_bins:` loop for consistency even if it never fires.

Expected: no other reference to `TOP_PERCENTILE_LABEL` or `_validate_top_percentile` remains anywhere in the file — confirm with:

```powershell
python -c "import re; s = open('validate_met_models.py').read(); print('top_percentile' in s.lower())"
```

Expected output: `False`.

- [ ] **Step 5: Confirm `plot_scatter` is no longer called for the old top-10% path**

The old function called `plot_scatter(m_top, o_top, var_top, ...)` (removed line 3063 in the old code) — the new `_validate_quantile_bins` has no plotting call at all. Grep to confirm:

```powershell
python -c "import re; s = open('validate_met_models.py').read(); print(s.count('plot_scatter('))"
```

Expected: same count as before this task started minus 1 (the one removed top-10% call). If you did not record the before-count, instead confirm structurally: open `validate_met_models.py` and check `_validate_quantile_bins` (Step 1's new function body) contains no `plot_scatter` call — it should not.

---

### Task 5: Re-run E1/E2/E3 with the quantile-bin change live

**Files:**
- Check: same three output directories as Task 3

- [ ] **Step 1: Re-run all three eras** (same commands as Task 3 Steps 1, 3, 4 — `VAL_ERA` = `E1`, `E2`, `E3` in turn, `VAL_USGS=1`, `VAL_OUTDIR_SUFFIX=v20260828`)

This overwrites the same three output dirs created in Task 3 with the quantile-bin rows now included. Expected: same success criteria as Task 3, plus the new quantile-bin rows are present:

```powershell
python -c "import pandas as pd; df = pd.read_csv(r'g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\obsE3_2020-2026_v20260828\validation_statistics.csv'); print(sorted(v for v in df['variable'].unique() if 'q' in v and 'Speed' in v))"
```

Expected: `['Wind Speed [m/s] (q00-25)', 'Wind Speed [m/s] (q25-50)', 'Wind Speed [m/s] (q50-75)', 'Wind Speed [m/s] (q75-90)', 'Wind Speed [m/s] (q90-100)']` (some bins may be absent for a given model/station if `n_bin < 10` — that is expected, not an error).

- [ ] **Step 2: Clear env overrides again** (`Remove-Item Env:VAL_ERA, Env:VAL_USGS, Env:VAL_OUTDIR_SUFFIX`)

---

### Task 6: `combined_skill.py` — new default weights, drop top-10%, guard the script body

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\analysis\combined_skill.py`

- [ ] **Step 1: Change the default category weights**

Replace `combined_skill.py:41`:

```python
_DEFAULT_WEIGHTS = {'USGS': 2.0, 'NDBC': 1.0, 'IEM': 1.0, 'CWOP': 0.5}
```

with:

```python
_DEFAULT_WEIGHTS = {'IEM': 1.0, 'NDBC': 1.0, 'USGS': 1.0}
```

Update the comment above it (`combined_skill.py:39-40`, "NOTE the default up-weights USGS to 2.0...") to read:

```python
# NOTE the default weights every category equally and drops CWOP entirely
# (2026-08-28). Pass VAL_WEIGHTS to reproduce an older run's weighting exactly,
# e.g. VAL_WEIGHTS="USGS:2,NDBC:1,IEM:1,CWOP:0.5" for the pre-2026-08-28 default.
```

- [ ] **Step 2: Remove the top-10% entry from SKILL_VARS**

Replace `combined_skill.py:69-73`:

```python
SKILL_VARS = [('Wind Speed [m/s]', 'speed'),
              # Top decile: Murphy skill is almost always negative here (the
              # conditional obs variance is tiny), so read the RMSE column.
              ('Wind Speed [m/s] (top 10%)', 'speed_top10'),
              ('Wind U10 [m/s]', 'u10'),
```

with:

```python
SKILL_VARS = [('Wind Speed [m/s]', 'speed'),
              ('Wind U10 [m/s]', 'u10'),
```

(leave the remaining lines 74-80 — `v10`, `temp`, `pressure`, `dewpoint`, `rh`, `radiation`, `precip` — unchanged.)

- [ ] **Step 3: Guard the module-level script body so the module becomes importable without side effects**

The block from `rows_all = []` (`combined_skill.py:212`) through the final `print("\nDONE.")` (`combined_skill.py:285`) currently executes at import time. Indent that entire block one level and wrap it in `if __name__ == '__main__':`. Concretely, change:

```python
rows_all = []
for label, d in ERA_DIRS.items():
```

to:

```python
if __name__ == '__main__':
    rows_all = []
    for label, d in ERA_DIRS.items():
```

and indent every line from `rows_all = []` through `print("\nDONE.")` by 4 spaces (the whole `for label, d in ERA_DIRS.items():` loop body and the trailing `if rows_all:` block). Do not change the `def` blocks above it (`taylor`, `_bar`, `combine_skill`, etc.) — only the trailing script body.

Expected: running `python combined_skill.py` directly still behaves exactly as before (produces the same console output and files); `python -c "import analysis.combined_skill"` from `D:\Git\cosmos-wind-cnn\validation` now succeeds with **no** side effects (no prints, no file writes) — this is what Task 8's `skill_by_quantile.py` depends on.

- [ ] **Step 4: Verify the module still runs standalone**

```powershell
cd D:\Git\cosmos-wind-cnn\validation
python analysis\combined_skill.py
```

Expected: same console output shape as before this task (era-by-era skill tables), and `combined_skill_speed_top10.png` / `combined_taylor_speed_top10.png` are **no longer written** to any era directory that gets processed (check by re-running against the existing `obsE1_2000-2010_v20260810` directory, which still has the old top10 PNGs from a prior run — confirm no *new* timestamp on those two files after this run, or delete them first and confirm they are not regenerated).

---

### Task 7: `MODEL_MARKERS` + `model_marker_map` in `combined_skill.py`

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\analysis\combined_skill.py`

- [ ] **Step 1: Add a marker dict next to `MODEL_COLORS`**

After `combined_skill.py:83-89` (the `MODEL_COLORS` / `_FALLBACK_COLORS` block), add:

```python
MODEL_MARKERS = {
    'CNN-quantile-v3': 'o', 'ERA5': 's', 'CONUS404': 'D', 'RTMA-SFbay': '^',
    'RTMA': '^', 'AORC': 'v', 'CNN': 'P',
}
_FALLBACK_MARKERS = ['X', '*', 'h', '8', 'p', '<', '>']
```

- [ ] **Step 2: Add `model_marker_map`, mirroring `model_color_map` (`combined_skill.py:93-100`)**

Add immediately after `model_color_map`:

```python
def model_marker_map(models):
    mmap = {}
    for i, m in enumerate(sorted(m for m in set(models) if m not in MODEL_MARKERS)):
        mmap[m] = _FALLBACK_MARKERS[i % len(_FALLBACK_MARKERS)]
    for m in models:
        if m in MODEL_MARKERS:
            mmap[m] = MODEL_MARKERS[m]
    return mmap
```

Expected: `python -c "import sys; sys.path.insert(0,'.'); from analysis.combined_skill import model_marker_map; print(model_marker_map(['ERA5','CNN-quantile-v3','SOME-NEW-MODEL']))"` run from `D:\Git\cosmos-wind-cnn\validation` prints a dict with `'ERA5': 's'`, `'CNN-quantile-v3': 'o'`, and `'SOME-NEW-MODEL': 'X'`.

---

### Task 8: Expose per-category detail from `combine_skill()`

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\analysis\combined_skill.py`

- [ ] **Step 1: Add `corr` to `_cat_stats`'s return dict**

Replace `combined_skill.py:135-137`:

```python
    return dict(mse=mse, cmse=cmse, var=var, bias=bias, rz=rz, stdr=stdr,
                ew1=ew1, ew=ew, ew3=ew3,
                skill_c=(1.0 - mse / var if var > 0 else np.nan), n_sta=len(g))
```

with:

```python
    return dict(mse=mse, cmse=cmse, var=var, bias=bias, rz=rz, stdr=stdr,
                corr=np.tanh(rz), ew1=ew1, ew=ew, ew3=ew3,
                skill_c=(1.0 - mse / var if var > 0 else np.nan), n_sta=len(g))
```

- [ ] **Step 2: Return the per-category dict from `combine_skill()`**

Replace `combined_skill.py:148-159` (the `return dict(...)` in `combine_skill`) so its final line reads:

```python
        cats='+'.join(f"{c}({cats[c]['n_sta']})" for c in sorted(cats)),
        cats_detail=cats)
```

(i.e. add `cats_detail=cats` as a new trailing key — every other existing key stays exactly as-is.)

Expected: `res[m]['cats_detail']` is now a dict keyed by category name (`'IEM'`, `'NDBC'`, `'USGS'`), each value itself a dict with `corr`, `stdr`, `n_sta`, etc. — this is what Task 9's per-category Taylor diagrams read.

---

### Task 9: Rewrite `taylor()` — per-station markers, pooled marker, RMS arcs

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\analysis\combined_skill.py`

- [ ] **Step 1: Replace the `taylor()` function**

Replace the whole current function at `combined_skill.py:185-198`:

```python
def taylor(ax, models_xy, colors):
    for r in [0.5, 1.0, 1.5, 2.0]:
        ax.plot(np.linspace(0, np.pi / 2, 100), [r] * 100, color='0.85', lw=0.6, zorder=0)
    for Rc in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99]:
        ax.plot([np.arccos(Rc)] * 2, [0, 2.2], color='0.85', lw=0.6, zorder=0)
        ax.text(np.arccos(Rc), 2.25, f'{Rc:g}', fontsize=7, color='0.4', ha='center')
    ax.plot(0, 1.0, 'k*', ms=16, zorder=5)
    for m, (R, s) in models_xy.items():
        ax.plot(np.arccos(np.clip(R, -1, 1)), s, 'o', ms=11,
                color=colors.get(m, 'gray'), mec='white', mew=0.6, zorder=4)
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_rmax(2.3); ax.set_rticks([0.5, 1.0, 1.5, 2.0]); ax.set_rlabel_position(95)
    ax.text(np.pi / 4, 2.55, 'correlation', fontsize=9, ha='center', color='0.3')
    ax.set_xlabel('normalized standard deviation', fontsize=9)
```

with:

```python
def taylor(ax, models_xy, colors, markers, station_points=None, rmax=2.3):
    """Taylor diagram: per-station markers (small, transparent) plus one bold
    pooled/weighted marker per model, with centered-RMS-difference arcs about
    the (r=1, nstd=1) reference point.

    models_xy   : {model: (corr, std_ratio)} -- the bold marker position,
                  already pooled/weighted by combine_skill()/combine_dir().
    station_points : optional list of (model, corr, std_ratio) -- one row per
                  raw station observation, drawn as small alpha=0.32 markers
                  underneath the bold marker. None/[] draws no per-station layer.
    """
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    ax.set_rmax(rmax)

    # normalized-std-dev rings (constant radius, centered at origin)
    rticks = [r for r in [0.5, 1.0, 1.5, 2.0] if r <= rmax]
    for r in rticks:
        ax.plot(np.linspace(0, np.pi / 2, 100), [r] * 100, color='0.85', lw=0.6, zorder=0)
    ax.set_rticks(rticks); ax.set_rlabel_position(95)

    # correlation gridlines
    for Rc in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99]:
        ax.plot([np.arccos(Rc)] * 2, [0, rmax], color='0.85', lw=0.6, zorder=0)
        ax.text(np.arccos(Rc), rmax * 1.02, f'{Rc:g}', fontsize=7, color='0.4', ha='center')

    # centered RMS-difference arcs about the (r=1, nstd=1) reference point
    th = np.linspace(0, np.pi / 2, 400)
    for rms in [x for x in (0.5, 1.0, 1.5, 2.0) if x < rmax]:
        xx = 1.0 + rms * np.cos(th)
        yy = rms * np.sin(th)
        rr = np.hypot(xx, yy)
        tt = np.arctan2(yy, xx)
        keep = rr <= rmax
        ax.plot(tt[keep], rr[keep], ':', color='0.72', lw=0.8, zorder=1)
        if keep.any():
            idx = np.where(keep)[0][-1]
            ax.text(tt[idx], rr[idx], f'{rms:g}', color='0.55', fontsize=7,
                    ha='center', va='bottom', bbox=dict(fc='white', ec='none', pad=0.4))

    # dashed reference arc at nstd = 1, and the obs/gauge reference point
    ax.plot(np.linspace(0, np.pi / 2, 200), np.ones(200), '--', color='0.5', lw=0.9, zorder=2)
    ax.plot(0, 1.0, 'k*', ms=16, zorder=8)

    # per-station small markers (underneath the bold marker)
    for m, r, s in (station_points or []):
        ax.plot(np.arccos(np.clip(r, -1, 1)), s, markers.get(m, 'o'),
                color=colors.get(m, 'gray'), ms=3.2, alpha=0.32, mew=0, zorder=4)

    # bold pooled/weighted marker per model
    for m, (R, s) in models_xy.items():
        ax.plot(np.arccos(np.clip(R, -1, 1)), s, markers.get(m, 'o'), ms=11,
                color=colors.get(m, 'gray'), mec='white', mew=1.1, zorder=7)

    ax.text(np.pi / 4, rmax * 1.12, 'correlation', fontsize=9, ha='center', color='0.3')
    ax.set_xlabel('normalized standard deviation', fontsize=9)
```

Expected: no callers exist yet that pass `markers`/`station_points` — this is fixed in Step 2. `python -c "import ast; ast.parse(open('analysis/combined_skill.py').read())"` from `D:\Git\cosmos-wind-cnn\validation` parses without a `SyntaxError`.

---

### Task 10: Wire the new `taylor()` into the main loop — combined scope

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\analysis\combined_skill.py` (inside the `if __name__ == '__main__':` block from Task 6 Step 3)

- [ ] **Step 1: Build station_points and pass markers into the combined Taylor call**

Inside the `for var, key in SKILL_VARS:` loop (originally `combined_skill.py:225-256`, now indented one level under `if __name__ == '__main__':`), the block that builds `colors`/`models` and calls `taylor()` currently reads (original line numbers 244-256):

```python
        colors = model_color_map(list(res)); models = list(res)
        _bar(models, [res[m]['skill'] for m in models], colors,
             'combined Murphy skill (pooled, weighted)',
             f'{label} — combined {var} skill\nweights {WEIGHTS}',
             BASE / d / f'combined_skill_{key}.png')
        fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, polar=True)
        taylor(ax, {m: (res[m]['corr'], res[m]['std_ratio']) for m in models}, colors)
        h = [Line2D([0], [0], marker='*', color='k', ms=13, ls='None', label='Obs (ref)')]
        h += [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[m], ms=10,
                     ls='None', label=m) for m in models]
        ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.32, 1.05), fontsize=9)
        ax.set_title(f'{label} — combined Taylor: {var}\nweights {WEIGHTS}', fontsize=10, pad=24)
        fig.tight_layout(); fig.savefig(BASE / d / f'combined_taylor_{key}.png', dpi=150); plt.close(fig)
```

Replace it with:

```python
        colors = model_color_map(list(res)); markers = model_marker_map(list(res)); models = list(res)
        _bar(models, [res[m]['skill'] for m in models], colors,
             'combined Murphy skill (pooled, weighted)',
             f'{label} — combined {var} skill\nweights {WEIGHTS}',
             BASE / d / f'combined_skill_{key}.png')

        station_points = [(m, r, s) for m, r, s in
                          zip(df['model'], df['corr'], df['model_std'] / df['obs_std'])]

        fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, polar=True)
        taylor(ax, {m: (res[m]['corr'], res[m]['std_ratio']) for m in models}, colors, markers,
               station_points=station_points)
        h = [Line2D([0], [0], marker='*', color='k', ms=13, ls='None', label='Obs (ref)')]
        h += [Line2D([0], [0], marker=markers[m], color='w', markerfacecolor=colors[m], ms=10,
                     ls='None', label=m) for m in models]
        ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.32, 1.05), fontsize=9)
        ax.set_title(f'{label} — combined Taylor: {var}\nweights {WEIGHTS}\n'
                     f'small = per-station, bold = pooled/weighted', fontsize=10, pad=28)
        fig.tight_layout(); fig.savefig(BASE / d / f'combined_taylor_{key}.png', dpi=150); plt.close(fig)
```

`df` here is the already-filtered per-variable DataFrame from earlier in the same loop iteration (`df = raw[raw['variable'] == var]` then the `obs_std > 0.05` / `n >= 50` / finite filter — original lines 226-230, unchanged). It has one row per (model, station), so `zip(df['model'], df['corr'], df['model_std'] / df['obs_std'])` gives exactly the raw per-station Taylor points across all weighted categories.

Expected: after Task 12 (re-running), `combined_taylor_speed.png` etc. show a cloud of small markers plus one bold marker per model, matching the pasted reference style.

---

### Task 11: Add per-category Taylor diagrams

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\analysis\combined_skill.py` (same loop as Task 10, immediately after the combined-Taylor block)

- [ ] **Step 1: Add a per-category Taylor diagram block**

Immediately after the combined-Taylor `fig.savefig(...)` / `plt.close(fig)` line added in Task 10, add:

```python
        for cat in sorted(WEIGHTS):
            cat_models = [m for m in models if cat in res[m]['cats_detail']]
            if not cat_models:
                continue
            cat_points = [(m, r, s) for m, r, s in
                          zip(df.loc[df['source'] == cat, 'model'],
                              df.loc[df['source'] == cat, 'corr'],
                              df.loc[df['source'] == cat, 'model_std'] / df.loc[df['source'] == cat, 'obs_std'])]
            fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, polar=True)
            taylor(ax, {m: (res[m]['cats_detail'][cat]['corr'], res[m]['cats_detail'][cat]['stdr'])
                       for m in cat_models}, colors, markers, station_points=cat_points)
            h = [Line2D([0], [0], marker='*', color='k', ms=13, ls='None', label='Obs (ref)')]
            h += [Line2D([0], [0], marker=markers[m], color='w', markerfacecolor=colors[m], ms=10,
                         ls='None', label=m) for m in cat_models]
            ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.32, 1.05), fontsize=9)
            ax.set_title(f'{label} — {cat} Taylor: {var}\n'
                         f'small = per-station, bold = {cat} pooled', fontsize=10, pad=28)
            fig.tight_layout()
            fig.savefig(BASE / d / f'combined_taylor_{key}_{cat}.png', dpi=150)
            plt.close(fig)
```

Expected (checked in Task 12): for an era with all three categories present, this produces `combined_taylor_speed_IEM.png`, `combined_taylor_speed_NDBC.png`, `combined_taylor_speed_USGS.png` (and the same for `u10`, `v10`, and every other `SKILL_VARS` entry) inside the era output directory, each showing only that category's stations as small markers plus a bold marker at that category's own pooled position (not the cross-category pooled position).

---

### Task 12: Faceted per-category × per-variable bar chart

**Files:**
- Modify: `D:\Git\cosmos-wind-cnn\validation\analysis\combined_skill.py`

- [ ] **Step 1: Add a `_faceted_category_bar` helper**

Add this new function near `_bar()` (after `combined_skill.py:209`, i.e. right after the existing `_bar` function definition, still above the `if __name__ == '__main__':` guard):

```python
_CATEGORY_COLOR = {'IEM': 'tab:blue', 'NDBC': 'tab:orange', 'USGS': 'tab:green', 'CWOP': 'tab:gray'}


def _faceted_category_bar(raw, weights, models, label, fpath):
    """One figure, 3 subplots (speed / u10 / v10): grouped bars per model,
    one bar per category, plus a black diamond at the pooled/weighted skill."""
    facet_vars = [('Wind Speed [m/s]', 'Wind Speed'), ('Wind U10 [m/s]', 'U10'), ('Wind V10 [m/s]', 'V10')]
    cats = sorted(weights)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (var, short) in zip(axes, facet_vars):
        df = raw[raw['variable'] == var]
        df = df[(df['obs_std'] > 0.05) & (df['n'] >= 50)
                & np.isfinite(df['rmse']) & np.isfinite(df['obs_std'])]
        if df.empty:
            ax.set_title(f'{short} (no data)'); continue
        per_cat = {}   # {category: {model: skill}}
        for cat in cats:
            dcat = df[df['source'] == cat]
            per_cat[cat] = {}
            for m, g in dcat.groupby('model'):
                cs = _cat_stats(g)
                per_cat[cat][m] = cs['skill_c']
        pooled = {}
        for m, g in df.groupby('model'):
            cs = combine_skill(g)
            if cs:
                pooled[m] = cs['skill']
        model_order = [m for m in models if m in pooled]
        n_cat = len(cats)
        width = 0.8 / (n_cat + 1)
        x = np.arange(len(model_order))
        for i, cat in enumerate(cats):
            vals = [per_cat[cat].get(m, np.nan) for m in model_order]
            ax.bar(x + (i - n_cat / 2) * width, vals, width,
                   color=_CATEGORY_COLOR.get(cat, 'gray'), edgecolor='k', lw=0.4, label=cat)
        ax.plot(x, [pooled.get(m, np.nan) for m in model_order], 'D', color='black',
                ms=7, zorder=5, label='pooled/weighted')
        ax.axhline(0.0, color='k', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(model_order, rotation=30, ha='right', fontsize=8)
        ax.set_title(short, fontsize=10); ax.grid(axis='y', alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel('Murphy skill (1 = perfect)')
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=len(cats) + 1, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'{label} — skill by category and variable', fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
```

`_cat_stats` and `combine_skill` are the existing module-level functions (`combined_skill.py:120-137`, `140-159`) — this helper calls them directly rather than duplicating the pooling math.

- [ ] **Step 2: Call it once per era, inside the `for label, d in ERA_DIRS.items():` loop**

Immediately after the `for var, key in SKILL_VARS:` loop finishes for a given era (i.e. right before the `# ---- Wind direction (circular) ----` comment, originally `combined_skill.py:258`, now indented under `if __name__ == '__main__':`), add:

```python
    all_models_seen = sorted(raw['model'].unique(),
                             key=lambda m: -raw.loc[raw['model'] == m, 'n'].sum())
    _faceted_category_bar(raw, WEIGHTS, all_models_seen, label,
                          BASE / d / 'combined_skill_by_category.png')
```

Expected (checked in Task 13): every era directory gains one new `combined_skill_by_category.png` with 3 subplots (Wind Speed / U10 / V10), each with grouped bars per model split by IEM/NDBC/USGS plus a black diamond marking the pooled/weighted skill — alongside the untouched flat `combined_skill_speed.png` / `combined_skill_u10.png` / `combined_skill_v10.png`.

---

### Task 13: Re-run `combined_skill.py` and check every new output

**Files:**
- Check: `g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\obsE1_2000-2010_v20260828\`, `...\obsE2_2011-2019_v20260828\`, `...\obsE3_2020-2026_v20260828\`

- [ ] **Step 1: Run combined_skill.py scoped to only the three new era dirs**

```powershell
cd D:\Git\cosmos-wind-cnn\validation
$env:VAL_ERA_DIRS = "Obs E1 2000-2010=obsE1_2000-2010_v20260828,Obs E2 2011-2019=obsE2_2011-2019_v20260828,Obs E3 2020-2026=obsE3_2020-2026_v20260828"
python analysis\combined_skill.py
```

Using `VAL_ERA_DIRS` (already supported, `combined_skill.py:62-67`) instead of editing the hardcoded `ERA_DIRS` dict keeps this run from also reprocessing the five pre-existing `_v20260810`/`__withusgs`/`_usgs` directories with the new weights/plots — those stay exactly as they were unless Kees separately asks for them to be regenerated.

Expected: runs to completion; console prints one skill table per era per `SKILL_VARS` entry (no `speed_top10` table now); no traceback.

- [ ] **Step 2: Confirm the old top-10% files are gone from the new dirs**

```powershell
Get-ChildItem "g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\obsE1_2000-2010_v20260828" -Filter "*top10*"
```

Expected: empty result (no `combined_skill_speed_top10.png` / `combined_taylor_speed_top10.png` — they were never written for these fresh directories).

- [ ] **Step 3: Confirm the new per-station Taylor diagrams exist**

```powershell
Get-ChildItem "g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\obsE3_2020-2026_v20260828" -Filter "combined_taylor_speed*"
```

Expected: `combined_taylor_speed.png`, `combined_taylor_speed_IEM.png`, `combined_taylor_speed_NDBC.png`, `combined_taylor_speed_USGS.png` (assuming USGS has stations in E3 — confirmed in Task 3 Step 4).

- [ ] **Step 4: Confirm the faceted bar chart exists**

```powershell
Test-Path "g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\obsE3_2020-2026_v20260828\combined_skill_by_category.png"
```

Expected: `True`.

- [ ] **Step 5: Open `combined_taylor_speed.png` for E3 and visually confirm it matches the reference style**

Open the file and check: a cloud of small semi-transparent markers per model color, one bold larger marker per model with a white edge, a dashed arc at normalized-std=1, dotted centered-RMS arcs, and a black star at the origin-adjacent reference point (angle 0, radius 1). If the bold `CNN-quantile-v3` marker sits far outside the cloud of its own small markers, that indicates a units/scale bug in the `station_points` computation (Task 10 Step 1) — stop and re-check the `model_std / obs_std` computation against `res[m]['std_ratio']` before proceeding.

- [ ] **Step 6: Clear the env override**

```powershell
Remove-Item Env:VAL_ERA_DIRS
```

---

### Task 14: `skill_by_quantile.py` — pool the new quantile-bin rows

**Files:**
- Create: `D:\Git\cosmos-wind-cnn\validation\analysis\skill_by_quantile.py`

- [ ] **Step 1: Write the script**

```python
"""Pool the per-station wind-speed quantile-bin stats (validate_met_models.py's
_validate_quantile_bins) into one tidy CSV per era, using the same
sample-size-weighted-within-category / WEIGHTS-across-category pooling as
combined_skill.py's combine_skill() -- no plot, per the 2026-08-28 request.

Reads validation_statistics.csv rows shaped "Wind Speed [m/s] (q75-90)" etc.
Run standalone: python analysis\\skill_by_quantile.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from analysis.combined_skill import WEIGHTS, ERA_DIRS, BASE, RANK_DIR, combine_skill

QUANTILE_BIN_LABELS = ['q00-25', 'q25-50', 'q50-75', 'q75-90', 'q90-100']
VAR_BASE = 'Wind Speed [m/s]'


def load_era(d):
    fp = BASE / d / 'validation_statistics.csv'
    if not fp.exists():
        print(f"  (missing: {fp})")
        return None
    raw = pd.read_csv(fp)
    return raw[(~raw['station'].astype(str).str.startswith('CWOP')) & raw['source'].isin(WEIGHTS)]


def main():
    out_rows = []
    for label, d in ERA_DIRS.items():
        raw = load_era(d)
        if raw is None:
            continue
        print(f"=== {label} ({d}) ===")
        for bin_label in QUANTILE_BIN_LABELS:
            var = f"{VAR_BASE} ({bin_label})"
            df = raw[raw['variable'] == var]
            df = df[(df['obs_std'] > 0.05) & (df['n'] >= 10)
                    & np.isfinite(df['rmse']) & np.isfinite(df['obs_std'])]
            if df.empty:
                print(f"  {bin_label}: no rows")
                continue
            for m, g in df.groupby('model'):
                cs = combine_skill(g)
                if cs is None:
                    continue
                print(f"  {bin_label:8s} {m:20s} skill={cs['skill']:.3f}  "
                      f"n_sta={sum(v['n_sta'] for v in cs['cats_detail'].values())}")
                out_rows.append({'era': label, 'quantile_bin': bin_label, 'model': m,
                                  'skill': cs['skill'], 'rmse': cs['rmse'], 'bias': cs['bias'],
                                  'corr': cs['corr'], 'std_ratio': cs['std_ratio'],
                                  'cats': cs['cats']})
    if not out_rows:
        print("No quantile-bin rows found in any era -- nothing written.")
        return
    out_df = pd.DataFrame(out_rows)
    out_dir = BASE / RANK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / 'skill_by_quantile.csv'
    out_df.to_csv(out_fp, index=False)
    print(f"\nWrote {len(out_df)} rows -> {out_fp}")


if __name__ == '__main__':
    main()
```

Expected: file parses (`python -c "import ast; ast.parse(open('analysis/skill_by_quantile.py').read())"` from `D:\Git\cosmos-wind-cnn\validation`, no `SyntaxError`). Note `combine_skill()`'s return dict must actually contain a `'rmse'`/`'bias'` key for this to work as written — verify against the real `combine_skill()` return signature (`combined_skill.py:140-159`, as amended by Task 8) before running; if the pooled RMSE/bias keys are named differently there (e.g. only `mse`/`bias` are returned, not a top-level `rmse`), adjust the `out_rows.append` dict keys to match exactly what `combine_skill()` actually returns rather than guessing — this is the one place in the plan where the exact upstream key names should be double-checked against the live function signature at execution time, since Task 8 only documented the *added* keys (`corr`, `cats_detail`), not a full re-listing of the pre-existing ones.

- [ ] **Step 2: Run it**

```powershell
cd D:\Git\cosmos-wind-cnn\validation
$env:VAL_ERA_DIRS = "Obs E1 2000-2010=obsE1_2000-2010_v20260828,Obs E2 2011-2019=obsE2_2011-2019_v20260828,Obs E3 2020-2026=obsE3_2020-2026_v20260828"
python analysis\skill_by_quantile.py
Remove-Item Env:VAL_ERA_DIRS
```

Expected: console prints one line per (era, quantile bin, model) with a skill value; ends with `Wrote <N> rows -> ...\rankings\skill_by_quantile.csv`.

- [ ] **Step 3: Verify the CSV**

```powershell
python -c "import pandas as pd; df = pd.read_csv(r'g:\01_meteorlogical_analysis_sfbay\results\20260827_validation\rankings\skill_by_quantile.csv'); print(df.shape); print(sorted(df['quantile_bin'].unique())); print(sorted(df['era'].unique())); print(sorted(df['model'].unique()))"
```

Expected: `quantile_bin` contains exactly `['q00-25', 'q25-50', 'q50-75', 'q75-90', 'q90-100']`; `era` contains the three era labels; `model` includes `CNN-quantile-v3`, `ERA5`, `CONUS404` (E1/E2 only), `RTMA-SFbay` (E2/E3 only).

---

## Self-review

**Coverage** — every explicit change request from the original message is covered: IEM/NDBC/USGS-only + weight 1 (Task 1 station scope note + Task 6 Step 1), CWOP fully dropped (Task 6 Step 1, confirmed already excluded upstream in Task 1), top-10% image removed (Task 4, Task 6 Step 2), quantile-skill CSV added (Task 4, Task 14), per-station + pooled-marker Taylor diagrams (Task 9, Task 10), both per-category and combined Taylor (Task 11), faceted combined bar chart per category/variable (Task 12).

**Placeholders** — scanned; none remain. The one soft spot is Task 14 Step 1's note about double-checking `combine_skill()`'s exact pre-existing return keys (`rmse`/`bias` vs `mse`) at execution time — this is flagged explicitly as a verification step, not left silent, because the summary of `combine_skill()`'s return signature captured before this plan was written listed the *keys changed* by Task 8 but not the *complete* return dict, and guessing a wrong key name would fail loudly (KeyError) rather than silently, so it is safe to leave as a documented one-line check rather than re-deriving the whole function here.

**Consistency** — output dir names (`obsE1_2000-2010_v20260828` etc.) match exactly across Tasks 3, 5, 13, 14. `CNN-quantile-v3` spelling is consistent everywhere. `VAL_ERA_DIRS` label strings (`"Obs E1 2000-2010"` etc.) match the format already used by the existing `ERA_DIRS` dict values in `combined_skill.py` (`label: outdir` pairs) — same convention, new dates.

**Verifiability** — every task ends with a concrete check (file existence, printed value, CSV shape/columns) rather than "looks right."

---

## Handoff

Plan saved to `D:\Git\cosmos-wind-cnn\validation\plans\2026-08-28-cnn-quantile-v3-validation.md`. Two ways to run it:

1. **Subagent-driven** — one fresh agent per task, reviewed between tasks. Good here since the tasks are mostly independent code edits (config/run_validation/validate_met_models/combined_skill) followed by mostly-independent execution/verification steps.
2. **Inline** — work through it here, checking in at the marked points. Good since several tasks build directly on each other's code (Task 8's new `cats_detail` key feeds Task 11; Task 9's rewritten `taylor()` feeds Tasks 10-11) and the context of *why* each change was made (ported from `fig09_taylor()`, deliberately deviating on the bold-marker convention, etc.) is easiest to keep straight in one continuous pass.

Which?

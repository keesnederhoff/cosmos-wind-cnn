# Products-Comparison Redo of the CNN-v3 Obs Validation — Plan

**Goal:** Re-run the three-era obs validation (E1 2000-2010, E2 2011-2019, E3 2020-2026)
of `CNN-quantile-v3` with the full wind-product field — RTMA-SFbay, HRRR, UCLA, Sup3rWind,
NOW-23, CONUS404, ERA5 — so the v3 model can be ranked against every product from the
2026-07-12 `20260712_several_models` comparison, not just its three references.

**Approach:** Re-add the four products dropped from `config.py` in the 2026-07-23
self-contained relocation (HRRR, UCLA, NOW-23, Sup3rWind — their loaders and dispatch
branches are all still in `validate_met_models.py`, only the config entries were deleted;
entries recovered from git `d1b927e~1`, repointed at the untouched `m:\` originals). Extend
the per-era reference lists in `run_validation.py`, run the three eras wind-only into a
fresh output root, and re-rank with `analysis\combined_skill.py`.

**Depends on:** drives `g:` (obs archive + ERA5/RTMA/CNN data + UCLA/NOW-23/Sup3rWind
under `modeled_data\other_meteo_data\`, output home) and `m:` (HRRR/CONUS404); conda env
`cosmos_wind_cnn`; no HPC. Repo `d:\Git\cosmos-wind-cnn`.
*(Corrected 2026-08-30: UCLA/NOW-23/Sup3rWind had moved off `m:` into the `g:\01` data
home — Task 1 verification caught it; and the era runs need `VAL_ERA=E1/E2/E3`, not
`1/2/3`, which select the legacy era branch.)*

**Design decisions (made 2026-08-29, veto before running):**
- **Era membership follows product coverage**, same precedent as the existing E-era refs
  ("CONUS404 ends 2021 → excluded from E3 rather than scored on a partial window"):
  - **E1 2000-2010:** CNN-quantile-v3, ERA5, CONUS404, UCLA, NOW-23, Sup3rWind *(partial: 2007-2010 only)*
  - **E2 2011-2019:** CNN-quantile-v3, ERA5, CONUS404, RTMA-SFbay, HRRR *(partial: Oct 2014-2019)*, UCLA, NOW-23, Sup3rWind *(partial: 2011-2013 only)*
  - **E3 2020-2026:** CNN-quantile-v3, ERA5, RTMA-SFbay, HRRR. UCLA (ends 2020), CONUS404
    (ends 2021) and NOW-23 (ends 2022) cover ≤2 of E3's 6.6 years → excluded.
  - Sup3rWind's 2007-2013 record straddles the E1/E2 boundary, so it is partial in both
    eras — kept anyway because it was explicitly requested; its rows must be read as
    "its own overlap", not matched-period (Murphy skill is period-referenced).
- **Wind-only** (`VAL_VARIABLES` default), matching the 20260827_CNN_v3 deliverable and the
  NOW-23/Sup3rWind products (which have no scalars).
- **No spatial maps** (`VAL_SPATIAL` unset): the peak-event maps already exist in
  `20260827_CNN_v3` for the core products; this run is about the ranking.
- **Stations = IEM + NDBC (+ USGS moorings auto in E3)**, weights IEM:1, NDBC:1, USGS:1 —
  the current defaults, matching the run being redone. NOT the old CWOP-included weighting
  of the 20260712 figure (reproduce that later with
  `VAL_WEIGHTS="USGS:2,NDBC:1,IEM:1,CWOP:0.5"` if a like-for-like is wanted).
- **RTMA = `RTMA-SFbay`** (the fm_netcdf 2011-2026 build the v3 run used), not the older
  `RTMA` multifile entry. In the outputs the v3 model is labeled `CNN-quantile-v3` (what
  Kees calls CNN-RTMA-v3).
- **New output root** `g:\03-downscaling_meteo_cnn\validation_results\20260829_products\`
  — nothing in `20260827_CNN_v3\` is touched.

---

## Artifacts

| Artifact | Action | Purpose |
|---|---|---|
| `d:\Git\cosmos-wind-cnn\validation\config.py` | Modify | Re-add HRRR/UCLA/NOW-23/Sup3rWind `MODELS` entries + colors; repoint stale `CNN-quantile-v3` data_dir |
| `d:\Git\cosmos-wind-cnn\validation\run_validation.py` | Modify | Extend `_ERA_TR` per-era reference lists |
| `g:\03-downscaling_meteo_cnn\validation_results\20260829_products\obsE{1,2,3}_*\` | Create (by run) | Per-era stats CSV + per-station figures |
| `g:\03-downscaling_meteo_cnn\validation_results\20260829_products\rankings\` | Create (by run) | `combined_skill_weighted.csv` + per-era `combined_skill_speed.png` etc. |

All Python runs use the same shell setup (PowerShell, from the repo):

```powershell
cd d:\Git\cosmos-wind-cnn\validation
conda activate cosmos_wind_cnn
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:COSMOS_VALIDATION_DATA_ROOT   = "g:\01_meteorlogical_analysis_sfbay"
$env:COSMOS_VALIDATION_OUTPUT_ROOT = "g:\03-downscaling_meteo_cnn\validation_results\20260829_products"
```

`cosmos_wind_cnn` is mandatory — base anaconda's xarray 2022.12 raises
`MissingDimensionsError` on HRRR's 2-D `x`/`y` grid.

---

### Task 1: Re-add the four dropped products to config.py

**Files:**
- Modify: `d:\Git\cosmos-wind-cnn\validation\config.py`

- [ ] **Step 1: Append the product entries.**

Add at the END of `config.py` (after the `CNN-quantile-v3` block):

```python
# ===========================================================================
# 2026-08-29: products-comparison redo -- re-add the four secondary wind
# products dropped in the 2026-07-23 self-contained relocation (entries from
# git d1b927e~1). Like CONUS404's C404_DIR above they live off DATA_ROOT on
# m:\ and are NOT staged for Caldera -- Windows-only until staged.
# Coverage: HRRR Oct2014-2026, UCLA 1980-2020, NOW-23 2000-2022,
# Sup3rWind 2007-2013. Loaders (load_model, load_model_ucla, load_model_box)
# and their dispatch branches never left validate_met_models.py.
# ===========================================================================
_HRRR_METEO  = Path(r"m:\emeryville_crescent\04_model_runs\meteo")
_OTHER_METEO = DATA_ROOT / "modeled_data" / "other_meteo_data"
MODELS['HRRR'] = {
    'u_file': _HRRR_METEO / "HRRR_WY2015-WY2026_u10_eastward_wind.nc",
    'v_file': _HRRR_METEO / "HRRR_WY2015-WY2026_v10_northward_wind.nc",
    'temp_file': _HRRR_METEO / "HRRR_WY2015-WY2026_air_temp.nc",
    'u_var': 'eastward_wind', 'v_var': 'northward_wind',
    'temp_var': 'air_temperature', 'single_file': False,
}
MODELS['UCLA'] = {
    'data_dir': _OTHER_METEO / 'data' / 'ucla_reanalysis',
    'u_pattern': 'era5_reanalysis_1hr_u10_*.nc',
    'v_pattern': 'era5_reanalysis_1hr_v10_*.nc',
    'temp_pattern': 'era5_reanalysis_1hr_t2_*.nc',
    'u_var': 'u10', 'v_var': 'v10', 'temp_var': 't2', 'crs': 'lcc',
}
MODELS['NOW-23'] = {
    'kind': 'box', 'crs': 'unstructured',
    'data_dir': _OTHER_METEO / 'now23', 'pattern': 'now23_ca_bayarea_box_*.nc',
    'speed_var': 'windspeed_10m', 'dir_var': 'winddirection_10m',
}
MODELS['Sup3rWind'] = {
    'kind': 'box', 'crs': 'latlon_2d', 'has_uv': True,
    'data_dir': _OTHER_METEO / 'sup3rwind', 'pattern': 'sup3rwind_bayarea_box_*.nc',
    'u_var': 'u_10m', 'v_var': 'v_10m', 'speed_var': 'windspeed_10m',
}
MODEL_COLORS.update({'HRRR': 'tab:red', 'UCLA': 'tab:purple',
                     'NOW-23': 'tab:pink', 'Sup3rWind': 'tab:olive'})
```

- [ ] **Step 2: Repoint the stale CNN-quantile-v3 data_dir.**

The 20260827 run's home (`g:\01_...\results\20260827_validation\`) was moved wholesale to
`g:\03-downscaling_meteo_cnn\validation_results\20260827_CNN_v3\` — the year-segment
inference files now live under its `inference\` subdir, so the existing DATA_ROOT-relative
entry resolves to an empty path. In the existing `MODELS['CNN-quantile-v3']` block, replace

```python
    'data_dir': DATA_ROOT / 'results' / '20260827_validation' / 'inference',
```

with

```python
    # Results home moved 2026-08-29: g:\01_...\results\20260827_validation ->
    # this validation_results dir. Hardcoded (off DATA_ROOT) like C404_DIR.
    'data_dir': Path(r"g:\03-downscaling_meteo_cnn\validation_results"
                     r"\20260827_CNN_v3\inference"),
```

- [ ] **Step 3: Verify the entries resolve to real files.**

```powershell
python -c "import os; os.environ.setdefault('COSMOS_VALIDATION_DATA_ROOT', r'g:\01_meteorlogical_analysis_sfbay'); os.environ.setdefault('COSMOS_VALIDATION_OUTPUT_ROOT', r'g:\03-downscaling_meteo_cnn\validation_results\20260829_products'); import config; from pathlib import Path; import glob; [print(m, ':', config.MODELS[m].get('u_file', config.MODELS[m].get('data_dir')), '->', (Path(config.MODELS[m]['u_file']).exists() if 'u_file' in config.MODELS[m] else len(glob.glob(str(Path(config.MODELS[m]['data_dir']) / config.MODELS[m].get('pattern', config.MODELS[m].get('u_pattern', '*')))))) ) for m in ['HRRR','UCLA','NOW-23','Sup3rWind','CNN-quantile-v3','RTMA-SFbay','CONUS404','ERA5']]"
```

Expected: `HRRR ... -> True`, `UCLA ... -> 41` (yearly u10 files 1980-2020),
`NOW-23 ... -> 23` (2000-2022), `Sup3rWind ... -> 7` (2007-2013),
`CNN-quantile-v3 ... -> 27` (speed_full_record year segments 2000-2026),
`RTMA-SFbay/CONUS404/ERA5 -> True`. Any `False`/`0` = path typo, stop and fix.

### Task 2: Extend the per-era reference lists in run_validation.py

**Files:**
- Modify: `d:\Git\cosmos-wind-cnn\validation\run_validation.py` (`_ERA_TR`, ~line 86)

- [ ] **Step 1: Replace the `_ERA_TR` dict** with:

```python
_ERA_TR = {
    # 2026-08-29 products-comparison: full reference field per era, each product
    # only in eras it (mostly) covers. Sup3rWind (2007-2013) is PARTIAL in both
    # E1 and E2; HRRR (Oct 2014+) partial in E2. UCLA/CONUS404/NOW-23 end
    # 2020/2021/2022 -> excluded from E3 rather than scored on <=2 of 6.6 yr.
    'E1': (('2000-01-01', '2011-01-01'), 'obsE1_2000-2010',
           ['ERA5', 'CONUS404', 'UCLA', 'NOW-23', 'Sup3rWind']),
    'E2': (('2011-01-01', '2020-01-01'), 'obsE2_2011-2019',
           ['ERA5', 'CONUS404', 'RTMA-SFbay', 'HRRR', 'UCLA', 'NOW-23', 'Sup3rWind']),
    'E3': (('2020-01-01', _E3_END), 'obsE3_2020-2026',
           ['ERA5', 'RTMA-SFbay', 'HRRR']),
}
```

(The driver prepends `CNN-quantile-v3` itself; leave everything else in the file alone.)

- [ ] **Step 2: Confirm the driver builds the expected lists** (dry check, no run):

```powershell
$env:VAL_ERA = "E2"
python -c "import run_validation" 2>&1 | Select-String "Era"
```

Expected: a line like `=== Era 2 [ALL]: 8 models x 4x stations, ('2011-01-01', '2020-01-01') -> ...\20260829_products\obsE2_2011-2019 ===`
followed by the engine starting — kill it with Ctrl+C once the header printed correctly
(or let it roll straight into Task 3's smoke if convenient). 8 models = CNN-quantile-v3 + 7 refs.

### Task 3: Smoke test — one station through all eight E2 products

**Files:**
- Create (throwaway): `g:\03-downscaling_meteo_cnn\validation_results\20260829_products\obsE2_2011-2019_smoke\`

- [ ] **Step 1: Run E2 restricted to AAMC1** (Alameda — inside every product's domain):

```powershell
$env:VAL_ERA = "E2"; $env:VAL_STATIONS = "AAMC1"; $env:VAL_OUTDIR_SUFFIX = "smoke"
python run_validation.py
```

Expected: completes in minutes-not-hours; console shows all 8 products loading without a
skip warning (`WARNING: model '...' not in MODELS config` or `Skipping <model> (could not
load)` = Task 1 failure). RTMA-multifile windowing keeps this fast (29 s/station class).

- [ ] **Step 2: Check the smoke stats.**

```powershell
python -c "import pandas as pd; df = pd.read_csv(r'g:\03-downscaling_meteo_cnn\validation_results\20260829_products\obsE2_2011-2019_smoke\validation_statistics.csv'); print(df[df.variable.str.contains('peed')].groupby('model')[['n','rmse','skill']].first())"
```

Expected: 8 rows (CNN-quantile-v3, CONUS404, ERA5, HRRR, NOW-23, RTMA-SFbay, Sup3rWind,
UCLA), all with n > 10000, RMSE in the 1-3 m/s range. Sup3rWind and HRRR have smaller n
(partial windows ~3 and ~5 yr). Then clear the run knobs:

```powershell
Remove-Item Env:VAL_STATIONS; Remove-Item Env:VAL_OUTDIR_SUFFIX
```

- [ ] **Step 3: Delete the smoke dir** once checked:

```powershell
Remove-Item -Recurse -Force "g:\03-downscaling_meteo_cnn\validation_results\20260829_products\obsE2_2011-2019_smoke"
```

### Task 4: Full E1 run (2000-2010, 6 products)

- [ ] **Step 1:**

```powershell
$env:VAL_ERA = "E1"
python run_validation.py
```

Expected: header `=== Era E1 [ALL]: 6 models x ...`; output to
`...\20260829_products\obsE1_2000-2010\`. UCLA (11 yearly LCC files) and NOW-23 (11 box
files) make this the slow class — expect 1-3 h. No RTMA/HRRR in this era.

- [ ] **Step 2: Verify.**

```powershell
python -c "import pandas as pd; df = pd.read_csv(r'g:\03-downscaling_meteo_cnn\validation_results\20260829_products\obsE1_2000-2010\validation_statistics.csv'); print(sorted(df.model.unique()), len(df))"
```

Expected: `['CNN-quantile-v3', 'CONUS404', 'ERA5', 'NOW-23', 'Sup3rWind', 'UCLA']` and a
few hundred rows. No USGS rows (moorings start 2020).

### Task 5: Full E2 run (2011-2019, 8 products)

- [ ] **Step 1:**

```powershell
$env:VAL_ERA = "E2"
python run_validation.py
```

Expected: 8 models, output to `...\obsE2_2011-2019\`. Longest era (most products); 2-4 h.

- [ ] **Step 2: Verify.**

```powershell
python -c "import pandas as pd; df = pd.read_csv(r'g:\03-downscaling_meteo_cnn\validation_results\20260829_products\obsE2_2011-2019\validation_statistics.csv'); print(sorted(df.model.unique()), len(df))"
```

Expected: all 8 model names from Task 3 Step 2.

### Task 6: Full E3 run (2020-2026, 4 products)

- [ ] **Step 1:**

```powershell
$env:VAL_ERA = "E3"
python run_validation.py
```

Expected: 4 models (CNN-quantile-v3, ERA5, RTMA-SFbay, HRRR), output to
`...\obsE3_2020-2026\`; USGS moorings included automatically (all start after 2020-01-22).
Window end defaults to `VAL_ERA_END=2026-08-11`; RTMA truth stops 2026-07-30 — coverage
difference, not bias (each product matched to obs independently).

- [ ] **Step 2: Verify.**

```powershell
python -c "import pandas as pd; df = pd.read_csv(r'g:\03-downscaling_meteo_cnn\validation_results\20260829_products\obsE3_2020-2026\validation_statistics.csv'); print(sorted(df.model.unique())); print(df.source.unique())"
```

Expected: 4 models; sources include `USGS` alongside `IEM`/`NDBC`.

### Task 7: Combined weighted ranking across the three eras

**Files:**
- Create (by run): `g:\03-downscaling_meteo_cnn\validation_results\20260829_products\rankings\combined_skill_weighted.csv` + per-era `combined_skill_speed.png`, Taylor and direction figures

- [ ] **Step 1: Run combined_skill scoped to the three new era dirs** (same env vars as above still set):

```powershell
$env:VAL_ERA_DIRS = "Obs E1 2000-2010=obsE1_2000-2010,Obs E2 2011-2019=obsE2_2011-2019,Obs E3 2020-2026=obsE3_2020-2026"
python analysis\combined_skill.py
Remove-Item Env:VAL_ERA_DIRS
```

Expected: prints `weights={'IEM': 1.0, 'NDBC': 1.0, 'USGS': 1.0}` per era and
`Wrote ...\20260829_products\rankings\combined_skill_weighted.csv`. Default weights =
current convention; this is deliberately NOT the 20260712 figure's CWOP-weighted pooling.

- [ ] **Step 2: Read the E2 speed ranking and sanity-check ordering.**

```powershell
python -c "import pandas as pd; df = pd.read_csv(r'g:\03-downscaling_meteo_cnn\validation_results\20260829_products\rankings\combined_skill_weighted.csv'); w = df[(df.era.str.contains('E2')) & (df.variable.str.contains('peed'))]; print(w.sort_values('skill', ascending=False).to_string())"
```

(If the CSV's columns differ — inspect with `df.columns` — the check is: E2 wind-speed
pooled skill sorted descending.)

Expected sanity anchors from the 2026-07-12 baseline (era2 2011-2021, different window and
weights, so numbers shift but ordering should be recognizable): RTMA-SFbay on top
(~0.4-0.45), HRRR/UCLA mid-pack (~0.3-0.35), Sup3rWind below them, ERA5/NOW-23/CONUS404
~0.25, and CNN-quantile-v3 somewhere above its ERA5 input. If RTMA is NOT first in E2/E3,
or any product's skill is < -0.5, stop and investigate before reporting (wrong window,
unit or loader issue — see systematic-debugging).

### Task 8: Report + record

- [ ] **Step 1: Write a short summary** `g:\03-downscaling_meteo_cnn\validation_results\20260829_products\SUMMARY.md`: per-era wind-speed ranking table (model, pooled skill, RMSE, n), the partial-coverage caveats (Sup3rWind E1/E2, HRRR E2), which products were excluded from E3 and why, and the one-line answer: where does CNN-quantile-v3 land in the field per era.

- [ ] **Step 2: Commit the code changes** (config.py, run_validation.py, this plan):

```
validation: products-comparison redo -- re-add HRRR/UCLA/NOW-23/Sup3rWind, repoint CNN-quantile-v3, per-era reference field
```

- [ ] **Step 3: Append a dated entry to the CoSMoS vault note** (vault-note skill) with the
outcome and the results path.

---

## Self-review (done 2026-08-29)

- **Coverage:** every requested product appears in at least one era; CNN-quantile-v3 in all
  three; ranking + figures produced by Task 7. WRF_CalNev was in the 20260712 figure but
  NOT in Kees's request list — left out (speed-only latlon_2d product; add later if wanted).
- **Placeholders:** none; all paths verified on disk 2026-08-29 (HRRR files, 23 NOW-23 +
  7 Sup3rWind box files, UCLA yearly files, 27 CNN inference segments).
- **Consistency:** output root `20260829_products` and era dir names identical across
  Tasks 3-7; model names match `config.MODELS` keys exactly (`NOW-23` with hyphen,
  `Sup3rWind` capital W, `RTMA-SFbay`).
- **Verifiability:** each run task ends with a CSV check listing the expected model set.

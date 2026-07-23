# SF Bay Met-Validation → `cosmos-wind-cnn` Relocation — Design

**Date:** 2026-07-23
**Author:** Kees Nederhoff (with Claude)
**Source project:** `g:\01_meteorlogical_analysis_sfbay\`
**Target repo:** `d:\Git\cosmos-wind-cnn\`
**Status:** Approved design → implementation planning

---

## 1. Purpose

Relocate the standalone SF Bay meteorological product validation & ranking framework
(currently local-only at `g:\01_meteorlogical_analysis_sfbay\`) into the version-controlled,
HPC-portable `cosmos-wind-cnn` repository, then move its input data to USGS Caldera and clean
up the origin folder.

This **deliberately reverses** a decision from the original 2026-07-12 validation design, which
declared HPC portability an explicit non-goal ("local Windows analysis... portability is
explicitly a non-goal"). The scientific driver is unchanged — *which gridded wind product best
reproduces observed SF Bay winds, and is therefore the most defensible forcing for the SF Bay
Community Model over 1940–present* — but the validation now needs to (a) live under git alongside
the CNN it evaluates, and (b) run on Caldera where the CNN training/inference already runs.

## 2. Three phases

| Phase | Goal | Deliverable |
|---|---|---|
| **1 — code → repo** *(this plan)* | Curated validation code in the repo, config refactored to env-var roots, data-staging manifest, smoke-tested locally | `d:\Git\cosmos-wind-cnn\validation\` |
| **2 — data → Caldera** | The staged bundle on caldera project space; SLURM launcher; env exports; HPC smoke run | validation runs on Tallgrass/Caldera |
| **3 — clean up G:** | Remove the code that moved; **keep** `download_aorc.py`, `data\` (raw AORC + CNN-study NetCDFs), and `results\` | tidy `g:\01_meteorlogical_analysis_sfbay\` |

Phases 2 and 3 are scoped here but executed after Phase 1 lands and is verified.

## 3. Repo layout (self-contained `validation/` folder)

Kept as plain scripts — **not** an installed subpackage — so the validation stays decoupled from
the `cosmos_wind_cnn` package. Scripts run from inside `validation/` (bare `import config` /
`import validate_met_models`); the SLURM launcher honors this via `cd` + `PYTHONPATH`.

```
d:\Git\cosmos-wind-cnn\validation\
├── README.md                 # purpose, env vars, how-to-run, product×era matrix, caveats
├── config.py                 # SINGLE source of truth — now root-relative
├── validate_met_models.py    # engine (faithful copy; 1 hardcoded path → config)
├── run_validation.py         # consolidated era-aware driver (copied as-is)
├── analysis\
│   ├── rank_products.py
│   ├── combined_skill.py
│   ├── make_windroses.py
│   ├── make_comparison_slides.py
│   └── redraw_comparisons.py
├── reference\
│   ├── station_inventory.csv
│   ├── station_inventory.md
│   └── deltabay.ldb          # land boundary for cartopy spatial maps
├── stage_validation_data.py  # builds the canonical data bundle from scattered sources
└── slurm\
    └── cpu_caldera_validation.slurm   # (phase 2)
```

## 4. Path refactor — the core of Phase 1

`config.py` (345 lines) currently anchors ~15 absolute paths across **four physical drives**:

- `g:\01_meteorlogical_analysis_sfbay` — project root / output / AORC / CNN-study data
- `G:\03-downscaling_meteo_cnn` — ERA5 baseline + CNN/CNN-RTMA inference outputs
- `m:\emeryville_crescent\...` and `m:\wind_downscaling\...` — HRRR, CONUS404, RTMA, NOW-23,
  Sup3rWind, UCLA, WRF_CalNev, USGS moorings, downscaled products
- `d:\data\meteo\SFBay\data` — observation archive + ERO20
- `f:\Alameda\...\deltabay.ldb` — land boundary (spatial maps)

These are replaced by **two dedicated env-var roots**, mirroring the repo's own
`utils/config.get_data_dir()` pattern (raise a clear `RuntimeError` if unset):

```python
import os
from pathlib import Path

DATA_ROOT = Path(os.environ["COSMOS_VALIDATION_DATA_ROOT"])
OUT_ROOT  = Path(os.environ["COSMOS_VALIDATION_OUTPUT_ROOT"])
# Windows:  set COSMOS_VALIDATION_DATA_ROOT=G:\03-downscaling_meteo_cnn\validation
# Caldera:  export COSMOS_VALIDATION_DATA_ROOT=/caldera/.../validation
```

Every product/obs path is re-anchored to a **canonical sub-layout** under `DATA_ROOT`:

```
obs/  moorings/  reference/  era5/  hrrr/  conus404/  rtma/
now23/  sup3rwind/  ucla/  wrf_calnev/  cnn/  aorc/
```

Everything else in `config.py` — the `MODELS` registry structure, variable names, units, `scalars`
sub-dicts, station scope, run options, `MODEL_COLORS`, physical constants — is preserved verbatim.
Only the root anchors change.

**Engine straggler:** `validate_met_models.py:1925` hardcodes
`LDB_FILE = Path(r"f:\Alameda\...\deltabay.ldb")`. This becomes
`config.LDB_FILE = DATA_ROOT / "reference" / "deltabay.ldb"` and the engine imports it. This is the
**only** code change to the 3,348-line engine; the rest is a faithful copy.

**Behavior preserved:** same values, one home. No science change.

## 5. Data-staging manifest = the Caldera bundle

`stage_validation_data.py` copies exactly the files the engine reads — obs + the product subsets —
from their scattered `m:/d:/g:/f:` homes into the canonical `DATA_ROOT` layout, via `robocopy`
(real copies, not junctions). This single artifact does **double duty**: it produces the **local
bundle** (Phase 1) and defines the **exact set that ships to Caldera** (Phase 2), so there is no
separate inventory to maintain. It implements the agreed data split: *obs + product subsets the
engine reads → bundle → Caldera; full raw downloads + download scripts stay on G:.*

### Staged inventory (source → canonical destination)

| Product / asset | Source | Dest |
|---|---|---|
| IEM / NDBC / CWOP archives | `d:\data\meteo\SFBay\data\pws_sfbay_waterfront_*.nc` | `obs/` |
| ERO20 Grizzly Bay | `d:\data\meteo\SFBay\data\ERO20_GrizzlyBay_meteorological.nc` | `obs/` |
| USGS moorings (Whales Tale / EMC) | `m:\emeryville_crescent\01_data\{whales_tale,emc_data}\*.nc` | `moorings/` |
| ERA5 (7 vars) | `G:\03-downscaling_meteo_cnn\sf_bay_conus404\raw_data\ERA5_*_UTM.nc` | `era5/` |
| HRRR (4 vars) | `m:\emeryville_crescent\04_model_runs\meteo\HRRR_WY2015-WY2026_*.nc` | `hrrr/` |
| CONUS404 (7 vars) | `m:\emeryville_crescent\03_model_setup\meteo\CONUS404_SFbay_4km_*.nc` | `conus404/` |
| RTMA (wind + scalars + precip) | `m:\emeryville_crescent\04_model_runs\meteo\rtma_gee_grid\RTMA_grid_2p5km_*.nc` | `rtma/` |
| NOW-23 | `m:\...\other_meteo_data\now23\now23_ca_bayarea_box_*.nc` | `now23/` |
| Sup3rWind | `m:\...\other_meteo_data\sup3rwind\sup3rwind_bayarea_box_*.nc` | `sup3rwind/` |
| UCLA | `m:\...\other_meteo_data\data\ucla_reanalysis\era5_reanalysis_1hr_*.nc` | `ucla/` |
| WRF_CalNev | `m:\...\other_meteo_data\data\wrf_calnev\wrfout_d02_V1_*.nc` | `wrf_calnev/` |
| CNN (CONUS404) | `G:\03-downscaling_meteo_cnn\sf_bay_conus404\results\3679830\output_inference\full_record_ERA5_19400101_20270101.nc` | `cnn/` *(renamed)* |
| CNN-RTMA | `G:\03-downscaling_meteo_cnn\sf_bay_rtma\results\3732177\output_inference\full_record_ERA5_19400101_20270101.nc` | `cnn/` *(renamed)* |
| CNN-allvars / CNN-windonly | `g:\01_...\data\os_{av,wo}_*_s2\full_record_ERA5_20110101_20260101.nc` | `cnn/` *(renamed)* |
| AORC (per-year) | `g:\01_...\data\aorc\AORC_SFbay_800m_<year>.nc` | `aorc/` |
| Land boundary | `f:\Alameda\03_modelsetup\_inputs\inputnoah\deltabay.ldb` | `reference/` |
| Station inventory | `g:\01_...\reference\station_inventory.{csv,md}` | `reference/` |

**CNN filename collision:** the four CNN products share `full_record_ERA5_*.nc` filenames. Staging
renames them to distinct names under `cnn/` (e.g. `cnn_conus404.nc`, `cnn_rtma.nc`,
`cnn_allvars.nc`, `cnn_windonly.nc`) and `config.py`'s CNN entries point at the renamed files.

**Excluded (per decision):** `CONUS404-downscaled` and `CONUS404-downscaled-100m`
(`m:\wind_downscaling\...`) are wired in `MODELS` but appear in **no era's** product list — dropped
from the bundle; their config entries stay (harmless, skip-clean if absent).

### Bundle location (Windows)
`G:\03-downscaling_meteo_cnn\validation\` — a **new** directory *outside* the folder that Phase 3
cleans, so cleanup of `g:\01_...` stays clean. Staging duplicates only the SF-Bay *subsets* on G:
(not the global raw downloads), which is acceptable.

## 6. Curated vs. archived

**Move (curated):** `config.py`, `validate_met_models.py`, `run_validation.py`, `analysis/` (5
scripts), `reference/`, plus the new `stage_validation_data.py`.

**Archive (stay on G:, not in repo):** the ~15 experiment-specific drivers —
`run_ndbc.py`, `run_full_parallel.py`, `run_tmp_2025_cnn_rtma.py`, `run_cnn_validation.py`,
`run_cnn_validation_allvars.py`, `analyze_cnn_run.py`, `analyze_cnn_run_allvars.py`,
`run_aorc_validation.py`, `analyze_aorc_run.py`, `run_aorc_fullrecord_validation.py`,
`analyze_aorc_fullrecord.py`, `run_rtma_vs_aorc_matched.py`, `analyze_rtma_vs_aorc.py`.

Their science is not lost: every product they exercised is preserved in the `config.MODELS`
registry and the `run_validation.py` era matrix. `download_aorc.py` is **kept on G:** (it is the
download script, not validation logic).

## 7. Verification

1. **Repo-local smoke run** against the staged bundle: `ERA=3`,
   `MODELS_TO_RUN=['ERA5','RTMA']`, `STATION_GROUPS=['NDBC']`, `VARIABLES=['wind']`, spatial maps
   off — exercises loaders, interpolation, temporal matching, stats, figures, and CSV write
   end-to-end in minutes.
2. **Parity spot-check:** compare one ranking/skill table produced from the repo against the
   existing `g:\01_...\results\` numbers for the same inputs. Same inputs → same numbers proves the
   path refactor introduced zero regression.
3. **Missing-data preflight:** the engine's product×(exists?, time-span) audit confirms which
   products are reachable from the bundle before a full multi-era run.

## 8. Phase 2 (Caldera) outline

- `rsync`/`scp` the `validation/` bundle from `G:\03-downscaling_meteo_cnn\validation\` to caldera
  project space.
- `slurm/cpu_caldera_validation.slurm`: exports `COSMOS_VALIDATION_DATA_ROOT` /
  `COSMOS_VALIDATION_OUTPUT_ROOT` at caldera paths, `cd`s into `validation/`, runs
  `run_validation.py`. Modeled on the repo's existing `cpu_rtma_eval.slurm`.
- HPC smoke run + parity spot-check against the local result.

## 9. Phase 3 (cleanup) outline

Within `g:\01_meteorlogical_analysis_sfbay\`:
- **Remove:** `config.py`, `validate_met_models.py`, `run_validation.py`, `analysis/`,
  `reference/`, `docs/`, `__pycache__/`, and the ~15 archived experiment drivers (once confirmed
  captured / no longer needed).
- **Keep:** `download_aorc.py`, `data\` (raw AORC + CNN-study NetCDFs), `results\` (existing
  figures/CSVs, incl. the parity-reference numbers).

## 10. Non-goals

- No re-download of model/obs data; staging copies existing subsets only.
- No modular rewrite of the 3,348-line engine — faithful copy plus the single `LDB_FILE` fix.
- No science change: height-correction convention, matching rules, stats, and weights are
  carried verbatim (the documented anemometer-height caveat remains).
- No integration into the `cosmos_wind_cnn` package namespace — validation stays a standalone
  `validation/` folder.

## 11. Resolved decisions

1. **Placement:** self-contained `validation/` folder (not an installed subpackage).
2. **Paths:** dedicated `COSMOS_VALIDATION_DATA_ROOT` / `COSMOS_VALIDATION_OUTPUT_ROOT`.
3. **Curation:** move core + consolidated driver + analysis; archive the ~15 experiment drivers.
4. **Data split:** obs + engine-read product subsets → bundle → Caldera; raw downloads + download
   scripts stay on G:.
5. **Bundle location:** `G:\03-downscaling_meteo_cnn\validation\` (outside the cleaned folder).
6. **Staging method:** `robocopy` real copies.
7. **Downscaled products:** dropped from bundle, config entries retained.
8. **Git:** spec committed to the repo; a commit message will be *suggested* (not auto-committed),
   per the local git-workflow preference.

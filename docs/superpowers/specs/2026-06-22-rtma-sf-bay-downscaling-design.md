# Design: RTMA-based SF Bay downscaling (`sf_bay_rtma`)

**Date:** 2026-06-22
**Status:** Approved (design), pending implementation plan
**Author:** Kees Nederhoff (with Claude)

## 1. Motivation

The SF Bay case study currently downscales ERA5 (~31 km) to **CONUS404 4 km** as
the high-resolution training target. CONUS404 is a regional reanalysis/downscaling
product covering 1979–2021. We want to switch the SF Bay target to **RTMA
(Real-Time Mesoscale Analysis) at 2.5 km**, an observation-constrained NOAA
analysis. RTMA is a finer grid and is closer to observed truth, at the cost of a
shorter record (2011–2026) and no radiation fields.

This work adds a **new, self-contained case study `sf_bay_rtma`** alongside the
existing CONUS404 `sf_bay`, so the two targets can be compared directly. It makes
three small, backward-compatible generalizations to shared code so the pipeline is
no longer hardcoded to the `conus404_`/`era5_` naming.

## 2. RTMA data assessment (verified 2026-06-22)

All 11 RTMA SF Bay files live in `m:\emeryville_crescent\03_model_setup\meteo\`,
on a common grid **162 (x) × 123 (y) @ 2.5 km, UTM Zone 10N**, hourly.

**NaN / data quality:** A full per-timestep scan found **zero NaN values** in every
file — no fill-value artifacts, no partial- or all-NaN timesteps. (This is cleaner
than CONUS404, whose northward-wind file carried a ~1e35 fill value that the
`physical_bounds` mask caught.)

**The real issue is missing hours (gaps in the time axis)**, not NaN pixels:

| Variable group | Period | Missing hours | Notes |
|---|---|---|---|
| u, v, air_temp, dew_temp, pressure | 2011–2026 | 1,316 / 135,547 (~1.0%) | 904 of them in 2013; 2011–2015 hold nearly all |
| precipitation | 2011–2026 | 1,362 (~1.0%) | a few more than the others |
| wind_gust | 2016–2026 | 40 (~0.05%) | very clean |
| cloud_cover, specific_humidity, visibility | 2017–2026 | 40 (~0.05%) | very clean |
| surface_height | static | — | terrain, no time dim |

**Available variables vs. the current 8-pair setup:** RTMA has u, v, air temperature,
dew point, MSL pressure, and precipitation — but **no shortwave/longwave radiation**.
It additionally has cloud cover, specific humidity, visibility, wind gust, and static
terrain (not used in this design).

## 3. Scope (decisions)

1. **Layout:** new `case_studies/sf_bay_rtma/` alongside `sf_bay` (CONUS404 untouched).
2. **Targets:** the **6 core variables** with clean ERA5 input pairs, over the full
   2011–2026 record. Radiation is dropped (no RTMA equivalent).
3. **Data staging:** copy the needed RTMA target files and the ERA5 input files into
   `case_studies/sf_bay_rtma/data/raw/` (matches the existing layout and the HPC rsync
   workflow).

## 4. Variable mapping (6 pairs)

Target keys use a new `rtma_` prefix; input keys keep `era5_`.

| Output pair | Target key → file (var) | Input key → file (var) | Units | Bounds (min/max) |
|---|---|---|---|---|
| wind_u | `rtma_u` → `RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10.nc` (`eastward_wind`) | `era5_u` → `ERA5_eastward_wind_1940_2026_UTM.nc` | m/s | −100 / 100 |
| wind_v | `rtma_v` → `RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10.nc` (`northward_wind`) | `era5_v` → `ERA5_northward_wind_1940_2026_UTM.nc` | m/s | −100 / 100 |
| air_temperature | `rtma_air_temp` → `RTMA_SFbay_2p5km_air_temperature_2011_2026_UTM10.nc` (`air_temperature`) | `era5_air_temp` → `ERA5_air_temperature_1940_2026_UTM.nc` | K | 220 / 330 |
| dew_point_temperature | `rtma_dew_temp` → `RTMA_SFbay_2p5km_dew_point_temperature_2011_2026_UTM10.nc` (`dew_point_temperature`) | `era5_dew_temp` → `ERA5_dew_point_temperature_1940_2026_UTM.nc` | K | 200 / 315 |
| pressure | `rtma_pressure` → `RTMA_SFbay_2p5km_air_pressure_fixed_height_2011_2026_UTM10.nc` (`air_pressure_fixed_height`, MSLP) | `era5_pressure` → `ERA5_air_pressure_fixed_height_1940_2026_UTM.nc` | Pa | 85000 / 110000 |
| rainfall | `rtma_rain` → `RTMA_SFbay_2p5km_precipitation_2011_2026_UTM10.nc` (`precipitation`) | `era5_rain` → `ERA5_precipitation_1940_2026_UTM.nc` | mm/hr | 0 / 200 |

Notes:
- RTMA pressure is **MSL pressure**, which matches the ERA5 `msl`-derived input the
  current setup already uses — physically consistent.
- `_identify_variable` in `preprocessing.py` resolves each `rtma_*` key correctly
  because every RTMA file holds exactly one data variable (pattern match for
  u/v/pressure/rain; single-variable fallback for the temps).
- **Caveat:** RTMA precipitation skill is known to be limited for extremes; `rtma_rain`
  should be validated against gauges before it is trusted for compound-flood forcing.
  Radiation can be re-added later as ERA5 *input-only* channels if needed.

## 5. Period, split, gap handling

- **Overlap:** 2011-01-01 → 2026-06-18 (ERA5 ∩ RTMA).
- **Split:** chronological 0.70 / 0.15 / 0.15 → train ≈ 2011–2021, val ≈ 2021–2024,
  test ≈ 2024–2026.
- **Gap handling ("the NaN fix"):** add an opt-in `regular_time_grid: true` flag to
  preprocessing. When set, each variable is reindexed onto a complete hourly axis with
  NaN fill before splitting. The **existing** `_get_valid_indices` NaN-window-dropping
  (in `dataset.py` and the inference sliding window) then correctly excludes the ~1% of
  sequence windows that straddle a gap — without this, a window indexed by array
  position could silently mix non-consecutive hours. CONUS404 `sf_bay` keeps its current
  behavior (flag defaults off). No new NaN-handling code or loss changes.

## 6. Code changes (shared, backward-compatible)

1. **`src/cosmos_wind_cnn/data/preprocessing.py`**
   - Read `target_prefix` (default `'conus404_'`) and `input_prefix` (default `'era5_'`)
     from config; use them in `load_and_align_datasets` instead of the literal
     `'conus404_'` / `'era5_'` prefix checks. The "no target keys" guard message updates
     accordingly.
   - Add the optional `regular_time_grid` hourly reindex (NaN fill) applied to all
     variables after loading, gated by config (default off).
2. **`scripts/run_training_pipeline.py`**
   - `step_evaluate_grid_points`: derive the wind target/input variable names from the
     parsed config (`output_vars` / `input_vars`) rather than the literals
     `conus404_u`/`conus404_v`/`era5_u`/`era5_v`, so evaluation works for any prefix.
   - `step_inference`: make the `VAR_UNITS` lookup suffix-based (keyed on the variable
     suffix such as `_u`, `_pressure`) so output attributes are written for any prefix.
3. No changes to `models/`, `training/losses.py`, `training/metrics.py`,
   `training/trainer.py`, `data/dataset.py`, or `data/regridder.py`.

These changes are verified to leave the existing CONUS404 `sf_bay` path behavior
identical (defaults reproduce current literals; `regular_time_grid` off).

## 7. New case-study files

```
case_studies/sf_bay_rtma/
├── configs/
│   ├── preprocessing.yaml          # rtma_/era5_ file_dict, bounds, target_prefix, regular_time_grid
│   ├── training.yaml               # 6 variable_pairs (rtma_* high_res, era5_* low_res), crs EPSG:32610
│   └── inference_preprocessing.yaml# era5_* sources, bounds (unchanged ERA5 inputs)
├── data/raw/                       # staged: 6 RTMA target files + 6 ERA5 input files
├── results/.gitkeep
└── README.md                       # RTMA 2.5 km, 2011–2026, variable table, caveats
```

Config specifics:
- `preprocessing.yaml`: `start_date: '2011-01-01'`, `end_date: '2026-06-18'`,
  `target_prefix: 'rtma_'`, `regular_time_grid: true`, split 0.70/0.15/0.15,
  `physical_bounds` per the table above (both `rtma_*` and `era5_*` keys).
- `training.yaml`: same architecture/hyperparameters as `sf_bay` (base_channels 16,
  sequence_length 6, forecast_horizon 0, loss alpha/beta/gamma 1.0/0.5/0.3), only the
  6 `variable_pairs` changed.
- `inference_preprocessing.yaml`: the 6 ERA5 `sources` and their bounds (the model still
  takes ERA5 as input at inference; targets are RTMA only during training).

## 8. Verification

1. **Dry-run preprocess on a short slice** (e.g. one month in 2018 via temporary
   `start/end_date`): confirm ERA5→RTMA-grid alignment, the `regular_time_grid` reindex,
   gap-window drop counts, `target_grid_reference.nc` (162×123), and normalization stats.
2. **Shape check:** processed `train/val/test.nc` carry 6 variables on 162×123.
3. **CPU smoke-train** a few epochs to confirm the model builds with 6 in / 6 out
   channels and the loss runs.
4. **Inference + evaluation end-to-end** on a short period to confirm the generalized
   `rtma_`/`era5_` names flow through `step_inference` and `step_evaluate_grid_points`.
5. Full training run on Tallgrass (4× V100 DDP) once the dry run passes.

## 9. Known limitations (documented, not blocking)

- **Shorter record:** the model learns the ERA5→RTMA relationship from 2011–2026 only.
  Full hindcast inference (1940–2026) therefore extrapolates to a regime the training
  set never saw; results before ~2011 should be treated as lower-confidence.
- **No radiation targets** (RTMA limitation).
- **RTMA precipitation** reliability for extremes is uncertain (see §4).

## 10. Out of scope

Cloud cover / specific humidity / visibility / wind gust targets (would shorten the
record to 2016–2017+), radiation as input-only channels, CMIP6 input, and any change to
the model architecture or loss.

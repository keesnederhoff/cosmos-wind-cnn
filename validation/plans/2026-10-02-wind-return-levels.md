# SF Bay Wind Return Levels (RP1 / RP10 / RP100) — Plan

**Goal:** Produce maps and station tables of the 1-, 10- and 100-year 10 m wind speed for ERA5, the RTMA analysis, CNN-RTMA v3 raw and CNN-RTMA v3 bias-corrected. Fit the observations the same way, and score every product against the observations with a scatter plot and skill table.

**Approach:**
- Use one estimator everywhere: declustered POT/GPD at λ = 3 peaks/yr, with peaks separated by at least 48 h.
- Run it identically on every gridded cell, every station observation series and every product series extracted at a station.
- Stage 1 streams the four products one water year at a time into cached daily-maximum cubes plus hourly series at the station cells. The two CNN files are 220 GB each, so this is the only slow part.
- Stages 2–4 (obs, EVA, figures) run from the caches in minutes.
- Return levels at the stations are compared in two ways:
  - **matched**: obs and product on the same water years and the same valid-hour mask.
  - **full**: each source's own record.
- Maps come from the full record and from a common WY2012–WY2025 window, so RTMA can be compared like for like.

**Depends on:**
- The `g:` drive (all inputs and outputs) and `m:` (one file copied once in Task 1).
- Conda env `acfcd_rain` (xarray, netCDF4, scipy, pyproj, matplotlib and the installed `rainlib`), plus `cmocean`, which Task 1 installs.
- Roughly 12 GB free on `g:` for the caches.
- About 4–5 h of unattended desktop time for Stage 1.

**Design decisions (agreed with Kees 2026-10-02):**

| Item | Decision |
|---|---|
| Products | ERA5 (25 km, 1940–2025), RTMA analysis (2.5 km, 2011–2026), CNN-raw P50, CNN-BC P50 |
| CNN quantile | P50 only (τ = 0.5, index 1). Upper τ levels are per-hour conditional levels, so they are **not** used as EVA input |
| Estimator | POT/GPD, λ = 3 /yr, greedy declustering with min separation 48 h, on daily maxima of hourly speed. GPD by L-moments (`mode="free"`); the exponential tail (`mode="exp"`) is kept as a reference column. 1000-draw nonparametric bootstrap, 90 % CI |
| Year | Water year, `WY_START_MONTH = 10`; a WY counts if ≥ 80 % of its days are valid; a day is valid if ≥ 12 of 24 hours are valid |
| Record floor | Every series with ≥ 6 valid WY is fitted and all RPs are printed (the CI carries the uncertainty); < 6 WY → row kept with `status="dropped_<6yr"` and no numbers |
| Obs networks | IEM (20 ASOS) + NDBC/C-MAN + USGS moorings. **Note:** all four USGS moorings span 2.7–9.5 months, so they end up `dropped_<6yr` (listed, not fitted) |
| Obs time base | Hourly mean of all readings in [h − 30 min, h + 30 min), stamped at h (centred on the model's instantaneous hour) |
| Direction | Omnidirectional only |
| Map figures | One figure per RP: 1 × 4 panels (ERA5, RTMA, CNN-raw, CNN-BC), `pcolormesh` on each product's native grid, obs as circles with a white edge in the same colormap and limits |
| Compute | Desktop, cached and resumable stages |

**Known facts that shape the code (verified 2026-10-02):**
- CNN files: `wind_speed(time, quantile, y, x)` with time in hours since 1940-01-01, 759,188 h, from 1940-01-01 to 2026-08-10. `quantile = [0.1316, 0.5, 0.8158, 0.9211, 0.9737]` and P50 is index 1. Chunks are `(24, 1, 123, 162)`, so read one quantile and whole days. The grid is 123 × 162 at 2.5 km in UTM10N, with x from 318750 to 721250 and y from 4015750 to 4320750 (cell centres).
- The raw P50 in the new 1940–2026 file is **not** bit-identical to the series scored in `validation_results\20260827_CNN_v3` (Jan 2010: mean +0.11 m/s, max |Δ| 3.2 m/s). This plan re-scores the files actually used.
- BC = `np.interp(P50, cnn_q, rtma_q)` per cell (`d:\Git\cosmos-wind-cnn\scripts\bias_correct.py:50`). It uses 200 levels fitted on 7,891 h (2024-03-14..2025-02-06). `np.interp` **clamps** above `cnn_q[-1]`, so every raw value above the fit-window CNN maximum maps to the same fit-window RTMA maximum `rtma_q[-1]`. The top of the BC tail is therefore a flat cap set by single hours. In the Jan 2010 test window this inflated one cell from 16.2 to 27.4 m/s. Task 10 quantifies the effect.
- ERA5: `ERA5_{eastward,northward}_wind_1940_2025_UTM.nc` with variables `eastward_wind` and `northward_wind`. Grid is 21 × 17 at 25 km in UTM10, time in hours since 1900-01-01, running 1940-01-01 to 2025-12-31 23:00, chunked `(1, 17, 21)`.
- RTMA: `RTMA_SFbay_2p5km_wind_speed_2011_2026_UTM10.nc` with variable `wind_speed`, same 123 × 162 grid as the CNN, 2011-01-01 to 2026-07-30 18:00, chunked `(168, 123, 162)`.
- Obs:
  - `pws_sfbay_waterfront_iem.nc`: 20 stations, `wind_speed(station, time)`, `station_id`, `latitude`, `longitude`. It contains spurious 55–75 m/s values.
  - `pws_sfbay_waterfront_ndbc.nc`: 33 stations, 16 of them empty.
  - USGS mooring files: `DMP23MW101met.nc`, `DMP23MW201met.nc`, `EMC26MW101met.nc`, `ERO20_GrizzlyBay_meteorological.nc`, all with variable `wind_speed` and time in minutes since 1970.
  - The CWOP archive is not used (it fails to open with HDF error −101).
- Station records with ≥ 80 % hourly coverage per year:
  - Long: OAK 97 yr, SFO 95, SUU 82, NUQ 76, SJC 43, 46026 34.
  - Mid-length: STS, APC, LVK, VCB, HWD and CCR at 28–32 yr, and 46012 at 27.
  - NDBC Bay stations: 9–20 yr.
  - PAO and RHV have 0 qualifying years.
- Water years: the full window is WY1941–WY2025 for ERA5 and the CNN (85 WY), and WY2012–WY2025 for RTMA (14 WY). The common window is WY2012–WY2025.

---

## Artifacts

All new files go under `g:\03-downscaling_meteo_cnn\wind_extremes\`:

| File | Responsibility |
|---|---|
| `windx_common.py` | `# === CONFIGURATION ===` block (all paths, windows and parameters), product readers, obs loaders + QC + hourly mean, water-year and validity helpers, nearest-cell lookup, the `pot_fit()` wrapper around `rainlib.extremes` |
| `s1_extract_products.py` | Streams each product per WY into `cache\dailymax_<product>.nc` and `cache\stations_hourly_<product>.nc`; resumable via `cache\tmp_<product>\WY####.npz` |
| `s2_prepare_obs.py` | Loads obs, applies QC, builds hourly means, writes `cache\obs_hourly.nc`, `results\station_table.csv`, `results\qc_top_peaks.csv` |
| `s3_fit_return_levels.py` | POT/GPD on grids (full + common window) → `results\rl_map_<product>_<window>.nc`; on stations (full + matched) → `results\station_rl.csv`; Pettitt check → `results\homogeneity.csv` |
| `s4_figures.py` | Map per RP (×2 windows), station scatter + `results\skill_station_rl.csv`, BC-tail diagnostic |
| `skill_utils.py` | Copied unchanged from `m:\emeryville_crescent\05_post_processing\skill_utils.py` (single source of truth for metrics; copied so the folder is self-contained) |
| `figstyle.py` | Copied unchanged from `C:\Users\keesn\.claude\skills\figure-style\scripts\figstyle.py` |
| `tests\test_windx.py` | pytest unit tests for the helpers (Task 3) |
| `README.md` | Run order, inputs, outputs, decisions table above |

Product keys used everywhere (file names, CSV columns, figure panels, in this order): `era5`, `rtma`, `cnn_raw`, `cnn_bc`. Window keys: `full`, `common`. Station mode keys: `full`, `matched`.

---

### Task 1: Folder, environment, copied modules

**Files:**
- Create: `g:\03-downscaling_meteo_cnn\wind_extremes\` with subfolders `cache\`, `results\`, `figures\`, `tests\`
- Create: `wind_extremes\skill_utils.py`, `wind_extremes\figstyle.py` (copies)

- [ ] **Step 1: Create the folders**

```powershell
New-Item -ItemType Directory -Force g:\03-downscaling_meteo_cnn\wind_extremes\cache, g:\03-downscaling_meteo_cnn\wind_extremes\results, g:\03-downscaling_meteo_cnn\wind_extremes\figures, g:\03-downscaling_meteo_cnn\wind_extremes\tests
```

Expected: four directories listed, no error.

- [ ] **Step 2: Copy the shared modules**

```powershell
Copy-Item m:\emeryville_crescent\05_post_processing\skill_utils.py g:\03-downscaling_meteo_cnn\wind_extremes\
Copy-Item C:\Users\keesn\.claude\skills\figure-style\scripts\figstyle.py g:\03-downscaling_meteo_cnn\wind_extremes\
```

Expected: both files present. Check that `skill_utils.py` defines `compute_skill_scores(obs, model, min_n=10)`.

- [ ] **Step 3: Add cmocean to the env**

```powershell
conda activate acfcd_rain
pip install cmocean
python -c "import cmocean, rainlib.extremes as rx, netCDF4, pyproj; print('ok', cmocean.__version__, rx.decluster_peaks.__name__)"
```

Expected: `ok <version> decluster_peaks`. Ignore the GDAL `gdalvrt.xsd` warning.

---

### Task 2: `windx_common.py`

**Files:**
- Create: `wind_extremes\windx_common.py`

- [ ] **Step 1: Write the CONFIGURATION block** with exactly these names and values:

```python
# === CONFIGURATION ===
from pathlib import Path
ROOT       = Path(r"g:\03-downscaling_meteo_cnn\wind_extremes")
CACHE_DIR  = ROOT / "cache"
RESULT_DIR = ROOT / "results"
FIG_DIR    = ROOT / "figures"

CNN_DIR    = Path(r"g:\03-downscaling_meteo_cnn\sf_bay_rtma\results")
CNN_RAW    = CNN_DIR / "CNN_RTMA_v3_r1b_do010_s3_speed_quantiles_raw_19400101_20260810.nc"
CNN_BC     = CNN_DIR / "CNN_RTMA_v3_r1b_do010_s3_speed_quantiles_BC_19400101_20260810.nc"
BC_MAP     = CNN_DIR / "BC_map_r1b_do010_s3_valfit.nc"
CNN_Q_INDEX = 1                     # quantile axis index of tau = 0.5 (P50)

MET_ROOT   = Path(r"g:\01_meteorlogical_analysis_sfbay")
ERA5_U     = MET_ROOT / "modeled_data" / "era5" / "fm_netcdf" / "ERA5_eastward_wind_1940_2025_UTM.nc"
ERA5_V     = MET_ROOT / "modeled_data" / "era5" / "fm_netcdf" / "ERA5_northward_wind_1940_2025_UTM.nc"
RTMA_WS    = MET_ROOT / "modeled_data" / "rtma" / "fm_netcdf" / "RTMA_SFbay_2p5km_wind_speed_2011_2026_UTM10.nc"
OBS_DIR    = MET_ROOT / "observed_data"
IEM_FILE   = OBS_DIR / "pws_sfbay_waterfront_iem.nc"
NDBC_FILE  = OBS_DIR / "pws_sfbay_waterfront_ndbc.nc"
USGS_FILES = {                      # id: (file, lat, lon, anemometer height m)
    "WT_MW101":  (OBS_DIR / "DMP23MW101met.nc", 37.576294, -122.208054, 1.2),
    "WT_MW201":  (OBS_DIR / "DMP23MW201met.nc", 37.576397, -122.208038, 1.2),
    "EMC_MW101": (OBS_DIR / "EMC26MW101met.nc", 37.840395, -122.342093, 1.2),
    "ERO20_GRZ": (OBS_DIR / "ERO20_GrizzlyBay_meteorological.nc", 38.11725, -122.039833, 4.93),
}
LDB_FILE   = MET_ROOT / "reference" / "deltabay.ldb"   # coastline, UTM10 metres

PRODUCTS   = ["era5", "rtma", "cnn_raw", "cnn_bc"]     # fixed order everywhere
PRODUCT_LABEL = {"era5": "ERA5", "rtma": "RTMA", "cnn_raw": "CNN-RTMA raw", "cnn_bc": "CNN-RTMA BC"}

WY_START_MONTH   = 10
FULL_WY          = {"era5": (1941, 2025), "rtma": (2012, 2025), "cnn_raw": (1941, 2025), "cnn_bc": (1941, 2025)}
COMMON_WY        = (2012, 2025)
OBS_MAX_WY       = 2025
MIN_HOURS_PER_DAY = 12              # valid hours for a valid day
MIN_DAY_FRAC_WY   = 0.80            # valid-day fraction for a valid water year
MIN_YEARS        = 6                # below this: status dropped_<6yr, no fit

POT_LAMBDA       = 3.0              # peaks per year
SEP_HOURS        = 48.0             # declustering min separation
GPD_MODE         = "free"           # L-moment GPD; "exp" written as reference
RETURN_PERIODS   = [1, 10, 100]
N_BOOT           = 1000
CI_ALPHA         = 0.10
RNG_SEED         = 20261002

# Obs QC
WS_MAX_RAW       = 40.0             # m/s, readings above -> NaN (engine uses 50; 55-75 m/s spikes exist)
SPIKE_FACTOR     = 3.0              # hourly value > 3x mean of both neighbours AND > SPIKE_MIN -> NaN
SPIKE_MIN        = 15.0
REVIEW_RATIO     = 2.5              # top peak flagged for review if obs > 2.5 x ERA5 at that hour and obs > 20 m/s
QC_EXCLUDE: dict[str, list[str]] = {}   # station_id -> list of 'YYYY-MM-DD HH' hours set to NaN after review (Task 7)
Z0_WATER         = 2e-4             # m, log-law roughness for the USGS height correction
```

- [ ] **Step 2: Write the helpers.** Use these exact names and behaviour, because the later tasks call them:
  - `water_year(times: pd.DatetimeIndex) -> np.ndarray`: WY = year + 1 if month ≥ `WY_START_MONTH`, else year.
  - `daily_max(hourly: np.ndarray, times: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]`:
    - Input is `hourly` with shape (T, ...) on an exact hourly UTC axis.
    - Returns `(dmax (D, ...), days, nvalid (D, ...) uint8)`.
    - `dmax` is NaN where `nvalid < MIN_HOURS_PER_DAY`.
    - Callers must pass whole UTC days; raise `ValueError` otherwise.
  - `valid_water_years(days, dmax_1d_or_nd) -> dict[int, np.ndarray|bool]`: True for a WY whose valid-day fraction is ≥ `MIN_DAY_FRAC_WY`, where the denominator is the number of calendar days in that WY.
  - `pot_fit(daily: pd.Series, n_years: int, rng) -> dict`:
    - Calls `rainlib.extremes.decluster_peaks(daily, SEP_HOURS, max_peaks=round(POT_LAMBDA*n_years)+1)`, then `pot_sample`, `gpd_fit_variant(excess, GPD_MODE)`, `gpd_quantile` and `pot_bootstrap_quantiles(..., mode=GPD_MODE, rng=rng, n_boot=N_BOOT)`.
    - Also returns the `"exp"` point quantiles.
    - Keys: `threshold, sigma, xi, rate, n_peaks, n_years, rl_1, rl_10, rl_100, lo_1, lo_10, lo_100, hi_1, hi_10, hi_100, rl_exp_1, rl_exp_10, rl_exp_100, method`. `method = "POT-GPD-Lmom lam3 sep48h dailymax"`.
    - If `n_years < MIN_YEARS`, return `{"status": "dropped_<6yr", "n_years": n_years}` and fit nothing.
  - `nearest_cell(x_grid_1d, y_grid_1d, x, y) -> tuple[int, int, float]`: returns `(iy, ix, distance_m)`.
  - `latlon_to_utm10(lat, lon)`: `pyproj.Transformer.from_crs(4326, 32610, always_xy=True)`.
  - `read_ldb(path) -> list[np.ndarray]`: parses the Tekal blocks (a name line, then an `nrow ncol` line, then rows), splits polylines at `-999` rows, and returns arrays of shape (n, 2).
- [ ] **Step 3: Import check**

```powershell
cd g:\03-downscaling_meteo_cnn\wind_extremes
python -c "import windx_common as w; print(w.PRODUCTS, w.FULL_WY['rtma'], len(w.read_ldb(w.LDB_FILE)))"
```

Expected: `['era5', 'rtma', 'cnn_raw', 'cnn_bc'] (2012, 2025) <N>` with N > 1 polylines and no exception.

---

### Task 3: Unit tests for the helpers

**Files:**
- Create: `wind_extremes\tests\test_windx.py`

- [ ] **Step 1: Write these tests:**
  1. `daily_max` returns the known peak of a synthetic 48-h series, and returns NaN for a day with 11 valid hours.
  2. `daily_max` raises `ValueError` when the input starts at 05:00.
  3. `water_year`: 2010-09-30 → 2010 and 2010-10-01 → 2011.
  4. `valid_water_years`: a WY with 79 % valid days is False and one with 80 % is True.
  5. `pot_fit` on a synthetic record **(corrected from the original draft, which was
     statistically wrong: background noise at the original density could set the POT
     threshold instead of the storm population)**:
     - Daily series: 200 WY of background noise ~ U(0, 5), plus 1000 storms placed ≥ 3 days
       apart (storm rate 5/yr, well above the λ = 3 fit rate, so every one of the ~600
       peaks `pot_fit` selects is a storm, never background noise).
     - Storm peaks: u = 10 + GPD(σ = 2, ξ = −0.1), fixed seed.
     - `pot_fit` is called with `n_years=200` at λ = 3, so it picks the top 600 peaks.
     - Analytic RP100 (at the true storm rate, 5/yr): `10 + (2/−0.1)·((5·100)^−0.1 − 1)`.
     - Requirements: `|xi_hat − (−0.1)| < 0.1` and `rl_100` within 5 % of that analytic value.
     - Implemented result (seed 12345/999): `xi_hat = -0.140`, analytic RP100 = 19.26 m/s,
       fitted `rl_100` = 18.84 m/s (2.2 % difference).
  6. `pot_fit` monotonicity: `threshold < rl_1 < rl_10 < rl_100` and `lo_100 < rl_100 < hi_100`.
  7. `pot_fit` with `n_years = 5` returns `status == "dropped_<6yr"`.
  8. `nearest_cell` returns the exact index for a point placed on a known cell centre, with distance 0.
  9. Hourly-mean centring (a `hourly_mean_centred` helper from Task 6, imported from `windx_common`): readings at 00:53 and 01:05 both land on 01:00, and 01:31 lands on 02:00.
  10. BC clamp detector `bc_clamped(raw, cnn_top)`: returns True where `raw > cnn_top`.

  Put `hourly_mean_centred(times, values) -> pd.Series` and `bc_clamped` in `windx_common.py` now, so the tests can import them.
- [ ] **Step 2: Run**

```powershell
cd g:\03-downscaling_meteo_cnn\wind_extremes
python -m pytest tests -q
```

Expected: `10 passed`.

---

### Task 4: Stage 1, product extraction

**Files:**
- Create: `wind_extremes\s1_extract_products.py`

- [ ] **Step 1: Build the station list first.** Stage 1 needs to know the station cells. The script calls a `station_locations()` function in `windx_common.py`, which returns a DataFrame with columns `station_id, network, lat, lon, x, y`. It covers every IEM station and every NDBC station with any finite `wind_speed`, plus the 4 USGS moorings. Coordinates come from the archive `latitude`/`longitude` (IEM/NDBC) and from `USGS_FILES` (USGS). Write this helper into `windx_common.py`.
- [ ] **Step 2: Write the reader generator** `iter_product_wy(product) -> (wy, times, field[T, ny, nx], x1d, y1d)`. It yields one water year at a time, from 00:00 UTC on 1 Oct to 23:00 on 30 Sep, over `FULL_WY[product]` plus the edges needed (the WY before the first full one is not needed).
  - `cnn_raw` / `cnn_bc`: `netCDF4.Dataset(..)["wind_speed"][i0:i1, CNN_Q_INDEX, :, :]`. Indices come from the hours-since-1940 time axis, and `filled(np.nan)`.
  - `rtma`: read `wind_speed[i0:i1]` the same way.
  - `era5`: `hypot(u, v)` from the two files, sliced by index.
  - Hours absent from a file become NaN rows, so every WY has exactly 8760/8784 rows.
- [ ] **Step 3: Write the per-WY loop.** For each WY:
  - Compute `daily_max` for the whole grid.
  - Extract the hourly series at each station's `nearest_cell`, computed once per product grid.
  - Save `cache\tmp_<product>\WY####.npz` with arrays `dmax, nvalid, days, st_hourly, times`.
  - Skip a WY whose npz already exists, and print `"<product> WY#### done in <s> s"`.
  - After the last WY, concatenate and write:
    - `cache\dailymax_<product>.nc`: dims `(day, y, x)`, vars `dmax` (float32, zlib 4, chunks `(366, ny, nx)`) and `nvalid` (uint8), plus coords `x, y` and attributes `product, wy_range, min_hours_per_day`.
    - `cache\stations_hourly_<product>.nc`: dims `(station, time)`, vars `ws`, `iy`, `ix` and `dist_m`.
  - Delete `tmp_<product>` only after the NetCDF reopens and has the expected `day` length.
- [ ] **Step 4: Smoke run on one WY.** Set `SMOKE_WY = 2010` (a module-level variable in `s1_extract_products.py`, `None` for production) and `PRODUCTS_TO_RUN = ["cnn_raw"]`.

```powershell
python s1_extract_products.py
```

Expected: `cnn_raw WY2010 done in <~60-120> s`, and one npz.
- [ ] **Step 5: Verify the smoke output against the source.** This is the gate: if either check fails, stop and debug. Write `wind_extremes\check_s1_smoke.py` that does two things:
  - For station `OAK`, compare the cached hourly series for 2010-01-10..2010-01-24 against `CNN_RAW["wind_speed"][:, 1, iy, ix]` read directly.
  - For one day (2010-01-18), compare `dmax` against `np.nanmax` of the 24 hours.

Expected: `max|diff| = 0.000` for both.
- [ ] **Step 6: Production run (unattended, ~4–5 h).** Set `SMOKE_WY = None` and `PRODUCTS_TO_RUN = PRODUCTS`, then start it in a separate terminal:

```powershell
conda activate acfcd_rain
cd g:\03-downscaling_meteo_cnn\wind_extremes
python s1_extract_products.py *> cache\s1_log.txt
```

Expected: four `dailymax_*.nc` files and four `stations_hourly_*.nc` files.
- [ ] **Step 7: Check the caches.** Write `check_s1_full.py`, which prints per product: the number of days, the WY range, the fraction of NaN days, and the number of valid WY under `valid_water_years`.

Expected: `era5`/`cnn_raw`/`cnn_bc` have 85 valid WY and `rtma` has 14. The CNN NaN-day fraction should be < 0.5 % (the annual 7-h gust seam must not invalidate days at `MIN_HOURS_PER_DAY = 12`).

---

### Task 5: Stage 1 resume check

- [ ] **Step 1:** Delete `cache\tmp_rtma\` if it still exists, and delete `cache\dailymax_rtma.nc`. Rerun with `PRODUCTS_TO_RUN = ["rtma"]`, interrupt it after two WYs, then rerun.

Expected: the second run prints the remaining WYs only. The final file is byte-identical in `dmax` to the first run's (`np.array_equal` on reload).

---

### Task 6: Stage 2, observations

**Files:**
- Create: `wind_extremes\s2_prepare_obs.py`

- [ ] **Step 1: Load the readings.**
  - **IEM/NDBC**: read per station from the archives. There is no height correction: the archives are 10 m by convention, as in `validate_met_models.load_pws_archive`.
  - **USGS**: read `wind_speed`, convert time from minutes since 1970, and apply a log-law correction from z to 10 m with `Z0_WATER`.
- [ ] **Step 2: QC per reading.** Set readings outside 0 ≤ ws ≤ `WS_MAX_RAW` to NaN. Print the count removed per station.
- [ ] **Step 3: Hourly mean.** Apply `hourly_mean_centred`. Then the spike test: an hourly value is set to NaN when it is > `SPIKE_FACTOR` × the mean of the hours before and after **and** > `SPIKE_MIN`. Then apply `QC_EXCLUDE`.
- [ ] **Step 4: Validity.** Compute `daily_max` and `valid_water_years` per station, keeping only WY ≤ `OBS_MAX_WY`.
- [ ] **Step 5: Write `cache\obs_hourly.nc`** with dims `(station, time)` on one hourly axis from 1914-05-01 to 2026-06-30, var `ws`.
- [ ] **Step 6: Write `results\station_table.csv`.** Columns: `station_id, network, lat, lon, x, y, n_raw_removed, n_spike_removed, n_valid_wy, first_wy, last_wy, status`. `status` is `ok` or `dropped_<6yr`.

Expected:
- OAK about 97 valid WY, SFO about 95, SUU about 82, NUQ about 76.
- PAO and RHV `dropped_<6yr`.
- All 4 USGS rows `dropped_<6yr`.
- About 16 NDBC rows present.
- [ ] **Step 7: Top-peak review table.** For every `ok` station, decluster its daily maxima (`SEP_HOURS`) and take the 10 largest peaks. Write `results\qc_top_peaks.csv` with columns `station_id, rank, time, obs, era5, cnn_raw, rtma, ratio_obs_era5, review_flag`. Product values come from the Stage 1 station caches at that hour, NaN where a product doesn't cover the hour. `review_flag = obs > REVIEW_RATIO*era5 and obs > 20`.

---

### Task 7: Obs QC review (check-in with Kees)

- [ ] **Step 1:** Print `qc_top_peaks.csv` rows with `review_flag` True. For each one, plot the ±3 days of obs, ERA5 and CNN-raw (`figures\qc_review\<station>_<time>.png`, 150 dpi).
- [ ] **Step 2:** Show the list and the plots to Kees, and record his decision per peak in `QC_EXCLUDE` in `windx_common.py`.
- [ ] **Step 3:** Rerun `python s2_prepare_obs.py`.

Expected: no `review_flag` rows remain except the ones Kees kept on purpose. List those in `README.md` under "QC decisions".
- [ ] **Step 4: UTC sanity check.** Compute mean hourly obs speed by hour of day at OAK and SFO for Jun–Aug 2015–2024, and the same for `cnn_raw` at their cells.

Expected: both peak within ±1 h of each other, around 22–01 UTC (the afternoon sea breeze). A 7–8 h offset would mean the obs are in local time. In that case stop and fix the loader.

---

### Task 8: Stage 3, return levels

**Files:**
- Create: `wind_extremes\s3_fit_return_levels.py`

- [ ] **Step 1: Grid fits.** For each product and window (`full` = `FULL_WY[product]`, `common` = `COMMON_WY`):
  - Load `dailymax_<product>.nc`, restricted to the window's WYs, and keep only valid WYs per cell.
  - Run `pot_fit` per cell. Use `n_years` = the number of valid WY of that cell, and set NaN days of invalid WYs to NaN before declustering. The rng per cell is `np.random.default_rng([RNG_SEED, PRODUCTS.index(p), win_idx, iy, ix])`.
  - Write `results\rl_map_<product>_<window>.nc`: vars `rl_1, rl_10, rl_100, lo_*, hi_*, rl_exp_*, threshold, sigma, xi, n_years, n_peaks` on `(y, x)`, attributes `method` and `wy_range`.
  - Print progress every 1000 cells.
- [ ] **Step 2: Station fits, mode `full`.**
  - **obs**: `pot_fit` on the station's own valid WY (≤ 2025).
  - **products**: read the product's `full` map at the station's `(iy, ix)`, so the scatter and the maps are the same numbers.
- [ ] **Step 3: Station fits, mode `matched`.** For each `ok` station and each product:
  - Hourly mask = obs-valid hours ∩ product-valid hours.
  - WYs = obs-valid WY ∩ product `FULL_WY`.
  - Apply the mask to both series, recompute `daily_max` and `valid_water_years` on the masked series, and keep the WYs valid in both.
  - Run `pot_fit` on obs and on the product with the same `n_years`.

  Each product gets its own obs fit, because RTMA overlaps only WY2012–2025.
- [ ] **Step 4: Write `results\station_rl.csv`** in long format. Columns: `station_id, network, mode, product, source (obs|model), n_years, wy_first, wy_last, n_peaks, threshold, sigma, xi, rl_1, lo_1, hi_1, rl_10, lo_10, hi_10, rl_100, lo_100, hi_100, rl_exp_1, rl_exp_10, rl_exp_100, method, status`.
- [ ] **Step 5: Homogeneity.** For obs stations with ≥ 20 valid WY:
  - Run `rainlib.trends.pettitt_test` on the WY maximum of the hourly mean.
  - Also fit `pot_fit` on the post-break sub-period, using WY ≥ 1997 (ASOS) when no break is significant at p < 0.05.
  - Write `results\homogeneity.csv` with columns `station_id, n_years, break_wy, p_value, rl_100_full, rl_100_post, pct_diff`.
- [ ] **Step 6: Sanity checks**, printed and asserted at the end of the script:
  - `rl_1 < rl_10 < rl_100` in every fitted cell and row (violations = 0).
  - `n_years`: 85 for `era5`/`cnn_*` full, 14 for `rtma` full, 14 for all `common`.
  - No all-NaN map.

```powershell
python s3_fit_return_levels.py *> results\s3_log.txt
```

Expected: 8 `rl_map_*.nc` files, `station_rl.csv`, `homogeneity.csv`, and the log ends with `checks passed`. Runtime is about 10 min per product per window.

---

### Task 9: Stage 4, maps (one figure per RP and window)

**Files:**
- Create: `wind_extremes\s4_figures.py` (map section)

- [ ] **Step 1: Layout.**
  - Use `apply_style()` from `figstyle`.
  - For each RP in (1, 10, 100) and window in (`full`, `common`): `fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharex=True, sharey=True, constrained_layout=True)` with panel order `PRODUCTS`.
  - Draw `pcolormesh` of `rl_<RP>` on each product's native cell edges (centre ± half-spacing: 1250 m for RTMA and CNN, 12500 m for ERA5), with `cmap=cmocean.cm.speed`, `rasterized=True`.
- [ ] **Step 2: Colour limits.** One `vmin/vmax` per figure: the 2nd and 98th percentile of the pooled `rtma`/`cnn_raw`/`cnn_bc` values plus the obs return levels, rounded outward to 1 m/s. Use one shared colorbar labelled `RP<RP> wind speed (m s$^{-1}$)`.
- [ ] **Step 3: Observations.**
  - Plot `scatter` of obs `rl_<RP>` from `station_rl.csv`, mode `full`, `source=obs`, `status=ok`.
  - Style: same cmap and limits, `s=60`, `edgecolors="white"`, `linewidths=1.2`, `zorder=5`.
  - Draw them on every panel, so each product is judged against the same dots.
- [ ] **Step 4: Chrome.**
  - Coastline from `read_ldb(LDB_FILE)` as 0.4 pt grey lines.
  - Extent: the CNN grid edges (316.25–723.75 km E, 4013.25–4323.25 km N).
  - `km_axes`; `add_scalebar(ax, 50000)` and `add_north_arrow` on panel (a) only.
  - Panel labels `(a) ERA5` etc. inside the top-left corner.
  - No suptitle.
  - For the `common` window, add the text `WY2012–2025` to the top-right of each panel.
- [ ] **Step 5: Save** as `figures\map_rl<RP>_<window>.png` at 300 dpi via `save_figure(..., purpose="report")`, 6 files.

Expected: 6 PNGs. In each figure, ERA5 visibly lacks the coastal and gap structure (Golden Gate, Carquinez) that RTMA and the CNN resolve, and the circles are readable on top.

---

### Task 10: Stage 4, station scatter, skill table, BC-tail diagnostic

**Files:**
- Modify: `wind_extremes\s4_figures.py` (scatter + diagnostic sections)

- [ ] **Step 1: Scatter, one figure per mode** (`full`, `matched`).
  - Layout: `plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)` for RP1, RP10 and RP100.
  - x = obs `rl_<RP>`, y = product `rl_<RP>`.
  - Colours: `era5` grey `#7f7f7f`, `rtma` blue `#1f77b4`, `cnn_raw` orange `#ff7f0e`, `cnn_bc` black `#000000`.
  - Markers: IEM `o`, NDBC `s`.
  - Thin 90 % CI whiskers in both directions (alpha 0.4), a 1:1 line, and ±20 % dashed lines.
  - Equal, shared axis limits per panel.
  - Legend (products + networks) in panel (a) only.
  - In `matched` mode, every product point uses its own matched obs value, so x differs slightly by product. That is intended and goes in the caption note in `README.md`.
- [ ] **Step 2: Skill table.**
  - For each mode × RP × product: `compute_skill_scores(obs, model, min_n=5)` from `skill_utils` over the `ok` stations that have both values.
  - Also compute `median_ratio = median(model/obs)` and `ci_overlap_frac`, the fraction of stations whose obs and model 90 % CIs overlap.
  - Write `results\skill_station_rl.csv` with columns `mode, rp, product, n, bias, rmse, mae, sci, skill, r2, median_ratio, ci_overlap_frac`.
  - Annotate each scatter panel with a 4-line text box: `<label>: bias ±x.x, RMSE x.x, skill 0.xx`.
- [ ] **Step 3: BC-tail diagnostic** `figures\bc_tail_diagnostic.png`, 1 × 3 panels on the CNN grid:
  - (a) `rl_100(cnn_bc) − rl_100(cnn_raw)`, `RdBu_r`, symmetric limits from `diverging_limits`.
  - (b) The fraction of each cell's `cnn_raw` POT peaks (full window) that exceed `BC_MAP.cnn_q[-1]`, i.e. peaks that sit on the flat `rtma_q[-1]` cap after BC (`bc_clamped`). Sequential `viridis`, 0–1.
  - (c) `BC_MAP.rtma_q[-1]` (the fit-window RTMA maximum that acts as the cap), `cmocean.cm.speed`.
  - Print the domain median and the 95th percentile of (a) and (b).
- [ ] **Step 4: Run**

```powershell
python s4_figures.py
```

Expected: `scatter_rl_full.png`, `scatter_rl_matched.png`, `bc_tail_diagnostic.png` and `skill_station_rl.csv`, with 24 rows (2 modes × 3 RPs × 4 products), none empty.

---

### Task 11: README and interpretation (check-in with Kees)

- [ ] **Step 1:** Write `wind_extremes\README.md`: run order (`s1` → `s2` → `s3` → `s4`), the decisions table from this plan, inputs, outputs, the QC decisions, and the matched-mode note.
- [ ] **Step 2: Interpretation**, at the end of `README.md`, in prose:
  - Which products are biased low or high at RP1/RP10/RP100, by how much, and whether the error is spatially organised (Bay interior vs coast vs inland valleys).
  - Whether BC improves or degrades RP100 against obs, and how much of `cnn_bc`'s RP100 sits on the clamp (Task 10 Step 3b).
  - The homogeneity result for the long ASOS records.
  - The single most promising next step. One example: refit the BC map on the full 2020–2025 RTMA with linear tail extrapolation instead of `np.interp` clamping, if the clamp fraction is material.
- [ ] **Step 3:** Hand Kees the six maps, the two scatters and the skill CSV.

---

## Self-review notes

- **Coverage:**
  - RP1/10/100 maps → Tasks 8–9.
  - Obs return levels → Tasks 6–8.
  - ERA5/RTMA/CNN raw/BC → Tasks 4 and 8.
  - Scatter + skill → Task 10.
  - pcolormesh + white-edged circles → Task 9.
  - POT/GPD, matched + full → Tasks 2 and 8.
- **Deliberate departure from the existing validation engine:** obs are hourly means, centred on the hour, not nearest readings within ±1 h (Kees's choice). The engine's 50 m/s cap is tightened to 40 m/s plus a spike test and manual review, because a single spike becomes the RP100 here.
- **Approximation stated:** declustering runs on daily maxima, not hourly peaks, so a 48 h separation means peaks on non-adjacent days. The rule is identical for obs and every product, which is what keeps the comparison fair.
- **USGS moorings** cannot carry a return period (max 9.5 months). They are listed with their record length and `status="dropped_<6yr"`, not silently omitted.

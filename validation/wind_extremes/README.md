# SF Bay wind return levels (RP1 / RP10 / RP100)

The 1-, 10- and 100-year 10 m wind speed for ERA5, the RTMA analysis, CNN-RTMA v4 raw (P50) and CNN-RTMA v4 bias-corrected (P50), plus the v4 upper quantile levels (tau 0.89 and 0.95), fitted the same way at 37 observation stations and scored against them.

Plan: `validation/plans/2026-10-02-wind-return-levels.md`. Built 2026-10-02. Moved into the repo with `validation/plans/2026-10-03-wind-extremes-to-repo.md`. Rerun on CNN-RTMA v4 and WY2026 on 2026-11-02 (Phase 6 of `g:\03-downscaling_meteo_cnn\plans\2026-10-31-v4-cleanup-and-redo.md`).

**v4 inputs.** `windx_common.py` points at CNN-RTMA v4 (`sf_bay_rtma\results\v1940_product`), ERA5 and RTMA through WY2026 and the refreshed IEM/NDBC files; the statistics are unchanged. CNN-BC and its tail diagnostic are removed (rejected), and `cnn_q82`/`cnn_q92` are now `cnn_q89`/`cnn_q95` (native tau 0.8922 and 0.9510). The v3 results are archived at `g:\03-downscaling_meteo_cnn\archive\v3\wind_extremes\`; sections below that describe v3 numbers are labelled v3.

## Environment

Runs in the `cosmos_wind_cnn` conda env, which has no `cmocean` and no `rainlib`. The return-level code those need is in `evt.py`. Paths come from three environment variables, set before any stage runs (`windx_common.py` raises with the set-command if one is missing):

```
conda activate cosmos_wind_cnn
set KMP_DUPLICATE_LIB_OK=TRUE
set COSMOS_VALIDATION_DATA_ROOT=g:\01_meteorlogical_analysis_sfbay
set COSMOS_WINDX_CNN_ROOT=g:\03-downscaling_meteo_cnn
set COSMOS_WINDX_OUTPUT_ROOT=g:\03-downscaling_meteo_cnn\wind_extremes
```

- `COSMOS_VALIDATION_DATA_ROOT` is the validation bundle (`modeled_data\`, `observed_data\`, `reference\`), the same variable `validation/config.py` reads.
- `COSMOS_WINDX_CNN_ROOT` holds `sf_bay_rtma\results` (the v1940 product) and `archive\v3` (read by `v3_vs_v4_rl.py`).
- `COSMOS_WINDX_OUTPUT_ROOT` receives `cache\`, `results\` and `figures\`. Nothing generated is written inside the repo. All file names below that start with `cache`, `results` or `figures` are relative to this root.

This pipeline runs on the desktop only. The Caldera validation bundle (`validation/assemble_bundle.sh`) does not provide `modeled_data/cnn_inputs/sf_bay_rtma/raw_data`, so a Caldera run would need that path added first.

`evt.py` is a vendored copy of 16 objects from the private ACFCD `rainlib` (`extremes.py`, `trends.py`), taken at repo commit 902ae61. Do not edit it; re-copy if rainlib changes. `tests/test_evt_vendored.py` checks it against `rainlib` and skips when `rainlib` is not importable (run it in the `acfcd_rain` env).

## Run

From this folder, in order:

```
python s1_extract_products.py   # 220 GB CNN files -> daily-max cubes + station series (hours; resumable)
python s2_prepare_obs.py        # obs QC, hourly means, station table, top-peak review table (~35 s)
python s3_fit_return_levels.py  # POT/GPD on grids (full + common) and stations (full + matched) (~15 min from cached grids)
python s4_figures.py            # 6 maps, 2 scatters, skill table (~3 min)
python s5_obs_correction.py     # observation-anchored correction of the CNN raw maps
python s6_station_eva.py        # per-station return-level plots, obs vs every flavour
python v3_vs_v4_rl.py           # v3 vs v4 table: refits cnn_raw/cnn_ba on WY1941-2025 into results\v4_wy1941_2025 (~6 min)
python -m pytest tests -q       # unit tests of the helpers
```

s1 skips any product with a `finalize OK` line in `cache\s1_log.txt`. It reads the 128 GB P50 files and one level of the 328 GB quantile file (its chunks are one level wide, so only that level is read). v4 run, 2026-11-01/02: cnn_raw 38 min, ERA5 and RTMA 9 min together, cnn_ba 17 min, cnn_q89 and cnn_q95 about 30 min each. s2 takes 40 s, s3 about 3 min per product on 8 workers, s4 2 min, s5 1 min, s6 2 min. `checks\compare_to_golden.py` compares a fresh run with the saved v3 snapshot of results (rtol 1e-9). That snapshot is tied to the v3 inputs and to commit `5e2edc8` (tag `wind-extremes-v3`): from the v4 input change on, a run reads different files and a different record, so the comparison no longer applies to the default configuration. To re-verify the engine against the golden, check out that tag and run s2-s6 with the v3 inputs.

`windx_common.py` holds every path and parameter in its CONFIGURATION block.
- `s1` skips water years already in `cache\tmp_<product>\`.
- `s3` skips grid maps that already exist (`OVERWRITE_EXISTING`) but always refits stations (`OVERWRITE_STATIONS`).
- After a clean `s1` run the `cache\tmp_*` folders (about 6.5 GB) can be deleted by hand. The script no longer deletes them itself.

## Inputs

| Source | File | Notes |
|---|---|---|
| CNN-RTMA v4 raw / BA (P50) | `sf_bay_rtma\results\v1940_product\CNN_RTMA_s3_P50_uvs_{raw,BA}_19400101_20261001.nc` | variable `wind_speed`, (time, y, x), no quantile axis; time starts 1940-01-01 07:00 UTC |
| CNN-RTMA v4 quantiles | `...\v1940_product\CNN_RTMA_s3_speed_quantiles_raw_19400101_20261001.nc` | (time, quantile, y, x); tau selected by value: 0.8922 (`cnn_q89`) and 0.9510 (`cnn_q95`), no interpolation |
| BA factor map | `...\v1940_product\BA_factor_map_s3_E3q90.nc` | constant factor per cell; BA return levels = F x raw |
| ERA5 | `modeled_data\era5\fm_netcdf\ERA5_{eastward,northward}_wind_1940_2026_UTM.nc` (under `COSMOS_VALIDATION_DATA_ROOT`) | 25 km; real data to 2026-10-01 13:00 UTC |
| RTMA | `modeled_data\cnn_inputs\sf_bay_rtma\raw_data\RTMA_SFbay_2p5km_{eastward,northward}_wind_2011_2026_UTM10_earthrel.nc` (under `COSMOS_VALIDATION_DATA_ROOT`) | 2.5 km; speed = hypot(u, v); record to 2026-10-04 18:00 UTC with gaps |
| Obs | `...\observed_data\pws_sfbay_waterfront_{iem,ndbc}_to20261001.nc`, 4 USGS mooring files | CWOP not used (file fails to open) |

## Method (decisions agreed 2026-10-02)

| Item | Choice |
|---|---|
| Estimator | POT/GPD at λ = 3 peaks/yr. Greedy declustering with peaks ≥ 48 h apart, run on daily maxima of hourly speed. GPD by L-moments; the exponential-tail value is kept as `rl_exp_*`. 1000-draw nonparametric bootstrap, 90 % CI. Code: `rainlib.extremes` (acfcd_rain) |
| Year | Water year from October. A day is valid with ≥ 12 h, a WY with ≥ 80 % valid days. Series with < 6 valid WY are listed as `dropped_<6yr`, not fitted |
| Windows | Maps: `full` (ERA5/CNN WY1941–2026, RTMA WY2012–2026) and `common` (WY2012–2026). Stations: `full` (each source's own record; the product value is read from its full map) and `matched` (obs and product on the same WYs and the same valid-hour mask; each product has its own obs fit) |
| Obs time base | Hourly mean of readings in [h − 30 min, h + 30 min). For ASOS this is usually a single 2-min report |
| Direction | Omnidirectional |

### QC decisions (reviewed with Kees 2026-10-02)

Readings above 40 m/s are removed first. On the hourly series two tests then run:
- **Spike test:** an hour above 15 m/s that is more than 3× the mean of its neighbours is removed.
- **Persistence test:** an hour above 15 m/s is removed if its higher neighbour (nearest finite hour within 2 h each side) is below 0.6× its value.

The persistence test removes the IEM values sitting at exactly 20.00 and 25.00 m/s between neighbours near 10 m/s (a doubling artifact in the archive), and one-hour spikes in calm weather such as SFO 1941-09-18 and SQL 1998-06-29. Four two-hour bursts surrounded by calm were removed by hand (`QC_EXCLUDE`): HWD 1989-09-23 21–22 UTC, SQL 2017-04-18 17–18, RHV 2024-12-15 19–20 and LVK 1997-06-06 16–17. All remaining flagged top peaks are sustained storms with high neighbouring hours, and were kept:
- OAK and NUQ 1951–59
- SUU 1940s
- SQL 2025-02-13
- pxoc1/ppxc1 2023-03-21

`n_spike_removed` in `station_table.csv` counts both tests. PAO and RHV report only about 14 h per day (nightly gap). They are kept, but their full-record return levels are biased low; `median_hours_per_valid_day` in `station_rl.csv` exposes this. Matched mode applies the same gap to the models, so their skill rows remain fair.

## Outputs

- `results\rl_map_<product>_<window>.nc`: rl_1/10/100 with 90 % CI, rl_exp_*, threshold, sigma, xi, n_years, n_peaks, and fit_code (0 ok, 1 dropped, 2 insufficient peaks, 3 GPD fit error).
- `results\station_rl.csv`: long format, one row per station × mode × product × source.
- `results\skill_station_rl.csv`: n, bias, RMSE, MAE, SCI, Willmott skill, r², median model/obs ratio and CI-overlap fraction per mode × RP × product, from `skill_utils.compute_skill_scores`.
- `results\station_table.csv`, `qc_top_peaks.csv`, `homogeneity.csv`, `obs_correction_loo.csv`, `rl_map_cnn_obscorr_{full,common}.nc` (s5).
- `results\v3_vs_v4_rl.csv` and `results\v4_wy1941_2025\` (WY1941-2025 refit of cnn_raw and cnn_ba), from `v3_vs_v4_rl.py`.
- `figures\map_rl{1,10,100}_{full,common}.png`: 1 × 4 panels with obs circles in the same colour scale.
- `figures\scatter_rl_{full,matched}.png`: 1 × 3 panels (RP1/10/100) with 90 % CI whiskers, the 1:1 line and ±20 %. In matched mode each product's point uses its own matched obs value, so x differs slightly between products.
- v3 only: `figures\bc_tail_diagnostic.png` (BC − raw RP100, fraction of peaks on the BC clamp, the RTMA cap). Removed in v4 with the BC product.

### Deviations from the plan

- RTMA has 15 valid WY (2012-2026) in v4, because the local archive was rebuilt from the v4 `raw_data`; v3 had 13 (WY2015 was 30 % missing). The median share of finite hours at the stations is 0.99 or better in every WY except WY2013 (0.90). WY2011 starts partway through and is outside `FULL_WY`.
- The L-moment GPD fit failed in no cell and no station in v4 (fit_code 0 everywhere, every product and window). The v3 CNN-BC failures went with CNN-BC.
- The monotonicity check uses a 1e-3 m/s tolerance. v4 has no violation at that tolerance and no collapsed cell (RP100 - RP1 < 0.05 m/s).
- `s5_obs_correction.py` hard-coded 36 stations; it now reads `EXPECTED_N_STATIONS = 37`. The 37th is DWA, which reaches 6 valid WY only with WY2026.

## Results (v4, 2026-11-02)

Matched mode, 37 stations (IEM 20, NDBC 17). Five stations have no fit: the four USGS moorings and sfxc1 have no valid water year. Median model/obs ratio in brackets; bias in m/s. Source: `results\skill_station_rl.csv`. Skill is Willmott d across stations. In matched mode each product is scored on its own obs window (RTMA WY2012-2026, ERA5 and CNN up to the full record), so the RTMA rows and the CNN rows are not scored against identical obs.

| | RP1 bias | RP10 bias | RP100 bias | Willmott skill RP1 / RP10 / RP100 | RMSE RP1 / RP10 / RP100 |
|---|---|---|---|---|---|
| ERA5 (86 WY) | -4.2 (0.68) | -5.4 (0.65) | -7.0 (0.62) | 0.43 / 0.41 / 0.39 | 4.7 / 5.9 / 7.7 |
| RTMA (15 WY) | -1.8 (0.84) | -2.1 (0.87) | -2.5 (0.86) | 0.69 / 0.64 / 0.56 | 2.6 / 3.2 / 4.2 |
| CNN-RTMA v4 raw (P50) | -3.4 (0.74) | -4.4 (0.72) | -5.3 (0.71) | 0.51 / 0.47 / 0.46 | 3.9 / 5.0 / 6.2 |
| CNN-RTMA v4 BA | -1.9 (0.85) | -2.7 (0.83) | -3.3 (0.83) | 0.76 / 0.68 / 0.59 | 2.3 / 3.2 / 4.4 |
| CNN-RTMA v4 tau 0.89 | -1.7 (0.86) | -2.3 (0.84) | -2.9 (0.84) | 0.66 / 0.59 / 0.52 | 2.5 / 3.3 / 4.4 |
| CNN-RTMA v4 tau 0.95 | -0.8 (0.92) | -1.1 (0.91) | -1.4 (0.89) | 0.73 / 0.65 / 0.50 | 2.0 / 2.6 / 3.8 |

Record lengths: ERA5 and all CNN products WY1941-2026 (86 WY, WY2026 ends 2026-09-30; the CNN and ERA5 data run to 2026-10-01 13 UTC), RTMA WY2012-2026. Every product still underestimates the observed return levels at every return period. RTMA is closest among the unadjusted products (about 0.86x) and 84 % of its RP100 confidence intervals overlap the observed ones; raw CNN P50 has 57 %, ERA5 19 %.

**The raw CNN P50 still sits barely above ERA5 at the extremes.** It is about 0.72x the observations against ERA5's 0.62-0.68x, so the v3 reading holds: a conditional median regresses the peaks. The v4 raw product is a little better than v3's (table below).

**BA is a constant factor per cell.** BA return levels equal F x raw (checks below). F runs from 1.00 to 1.62 and is floored at 1.0 by construction, so 24 of 19,926 cells keep BA = raw. BA is within 0.01-0.04 of RTMA's median ratio at RP1 and RP10; at RP100 its skill (0.59) is about RTMA's (0.56) with a larger RMSE (4.4 against 4.2). It is fitted in-sample at most scoring stations, so the s5 leave-one-out numbers are the stricter test.

**Upper quantile levels.** The tau 0.95 level is the least biased product here (ratio 0.89-0.92, RMSE 2.0 / 2.6 / 3.8 m/s), but its skill drops with return period and at RP100 it is no better than BA. As in v3, picking tau by fit is itself a correction and was never validated against stations; treat tau 0.89 and 0.95 as sensitivity runs, not products.

**The error is spatially organised.** Median matched RP10 ratio, IEM against NDBC:

| Product | IEM | NDBC |
|---|---|---|
| RTMA | 0.79 | 0.93 |
| CNN raw | 0.68 | 0.75 |
| CNN BA | 0.83 | 0.84 |
| CNN tau 0.95 | 0.88 | 0.96 |
| ERA5 | 0.65 | 0.66 |

Part of the airport (IEM) deficit is representativeness: an ASOS hourly value is one 2-min average at a point, so its annual maximum is gustier than an hourly 2.5 km grid value.

**Observation-anchored correction (`s5_obs_correction.py`, v4).** Same method as v3, fitted on the 37 matched cnn_raw stations (20 land, 17 water). Mean leave-one-station-out RMSE over RP1/10/100 (`results\obs_correction_loo.csv`):

| Variant | Mean LOO RMSE (m/s) |
|---|---|
| none | 5.02 |
| global | 3.06 |
| land/water | 2.90 |
| IDW | 2.94 |
| land/water + 20 km kernel | 2.74 |
| land/water + GP | **2.64** |

The winner changed from v3 (kernel variant, 2.87) to the GP variant. Its LOO bias is +0.06 / +0.11 / +0.40 m/s, RMSE 1.69 / 2.26 / 3.96 m/s and skill 0.82 / 0.71 / 0.51 at RP1/10/100. Land / water factors: RP1 1.47 / 1.25, RP10 1.47 / 1.34, RP100 1.46 / 1.35. At RP100 the GP has almost no spatial signal (kernel amplitude 0.01, at its lower bound) and the map is the land/water step. The corrected RP100 map exceeds 30 m/s in 4,492 cells, almost all of them offshore (4,491 ocean cells, 1 land cell, spanning x 319-599 km; maximum 34.3 m/s), where no station constrains the factor (`figures\map_obscorr.png`); treat the offshore corrected RP100 as an extrapolation of the factor. Caveats as in v3: the land factor rests on ASOS airports only, and the factor steps at the shoreline.

**Homogeneity.** Pettitt finds a significant break (p < 0.05) at 9 of the 16 records with >= 20 WY (v3: 9 of 16). Post-break RP100 against the full record: OAK -9.7 %, NUQ -12.0 %, STS -18.9 %, LVK -18.7 %, 46026 -11.1 %. SQL (+71 %), SUU (+22 %) and CCR (+15 %) come from short post-break sub-periods with unstable free-shape fits and are not trends. Observed RP100 at the long airport records carries about +-10 % from inhomogeneity alone.

### Checks run on the v4 results

- **BA = F x raw.** Against `BA_factor_map_s3_E3q90.nc`, the maximum relative difference over all cells is 1.6e-7 (RP1), 1.9e-7 (RP10) and 9.2e-7 (RP100) in the full window, and 1.7e-7 / 2.2e-7 / 1.2e-6 in the common window; no cell exceeds 3e-6. In the 24 cells with F = 1.0 (the floor), BA = raw exactly.
- **RP1 <= RP10 <= RP100** within 1e-3 m/s: 0 violations in all 12 maps (6 products x 2 windows) and in 733 ok station rows.
- **Failed fits:** none. fit_code is 0 in every cell of every map (357 cells for ERA5, 19,926 for the others) and no station fit failed.
- **N of WY:** ERA5 and the four CNN products have 86 WY (WY1941-2026) in the full window in every cell; RTMA has 15 (WY2012-2026); the common window has 15 everywhere.
- **Stations per WY** (valid obs WY, all / IEM / NDBC): WY1941 2/2/0, WY1981 10/9/1, WY2001 16/14/2, WY2012 25/17/8, WY2020 33/16/17, WY2021 35/18/17, WY2022 35/19/16, WY2023 35/20/15, WY2024 35/20/15 (35 is the maximum, reached in WY2021-2024), WY2025 34/20/14, WY2026 20/20/0. All 20 IEM stations reach WY2026 and no NDBC station does. The NDBC file runs to 2026-09-30, but its WY2026 is gappy: aamc1, for example, has data only in Oct-Dec 2025, May-Jun 2026 and part of Sep 2026, and nothing in Jan-Apr or Jul-Aug. No NDBC station reaches the 80 % valid-day threshold. This is the archive, not the pipeline.
- **No jump across the ERA5-driven eras.** The ratio is the median over the 37 ok stations of the CNN raw annual-maximum hourly speed (WY1941-2026, at the station cell) divided by the same median for ERA5. Averaged over 5-WY blocks (WY1941-45, 1946-50, ... 2021-25) it is 1.08-1.16 (lowest 1951-55, highest 2011-15); per WY through 2024 the mean is 1.12 and the SD 0.05. WY2026 is 1.02 (-2.1 SD) and WY2025 1.09. Only WY1956 (1.01) and WY1983 (1.00) are lower than WY2026, so the drop is at the low end but not unique. With all 42 cache stations, including the five dropped from the fits, WY2026 is 1.05 (-1.6 SD). Computed from `cache\stations_hourly_{cnn_raw,era5}.nc`.
- **RTMA far-field tails.** The same far-south cell (x 611 km, y 4038 km, outside the station zoom) is the RTMA maximum in v3 and v4 (RP100 36.3 and 37.6 m/s). RTMA RP100 exceeds 30 m/s in 386 cells (v3: 137), 192 cells have xi > 0.3 and 321 have xi < -0.5, from a 15-WY free-shape fit. Treat the RTMA map away from the stations as unconstrained. In the full window no CNN cell exceeds 30 m/s (raw maximum 25.4, BA 28.1). In the common window BA has 3 cells above 30 (maximum 30.4), cnn_q89 has 5 (maximum 30.7) and cnn_q95 has 97 (maximum 33.0); cnn_raw stays below 30 (maximum 27.5). Counts from `rl_100` in `results\rl_map_*_common.nc`.

### v3 vs v4 (CNN raw and BA, WY1941-2025 refit)

`results\v3_vs_v4_rl.csv`, from `v3_vs_v4_rl.py`. v4 is refitted on WY1941-2025 to match the v3 window with the s3 settings unchanged. Scores use the 36 stations that are ok in both versions (matched mode, each version's own matched obs). The v3 side is read from `archive\v3\wind_extremes\results\`. Grid statistics are over all 19,926 cells.

| Product | RP | grid median v3 -> v4 | grid range v3 -> v4 | bias v3 -> v4 | RMSE v3 -> v4 | skill v3 -> v4 | median ratio v3 -> v4 |
|---|---|---|---|---|---|---|---|
| raw | 1 | 15.80 -> 15.99 | 4.6-18.7 -> 4.4-19.1 | -3.76 -> -3.50 | 4.25 -> 3.95 | 0.48 -> 0.51 | 0.70 -> 0.73 |
| raw | 10 | 18.12 -> 18.52 | 6.2-21.5 -> 6.3-21.8 | -4.79 -> -4.49 | 5.29 -> 5.00 | 0.45 -> 0.47 | 0.69 -> 0.72 |
| raw | 100 | 19.75 -> 20.46 | 7.0-26.0 -> 7.1-25.5 | -6.02 -> -5.58 | 6.90 -> 6.63 | 0.43 -> 0.44 | 0.69 -> 0.71 |
| BA | 1 | 17.43 -> 17.74 | 5.8-20.6 -> 5.2-21.1 | -2.15 -> -1.97 | 2.60 -> 2.36 | 0.74 -> 0.76 | 0.85 -> 0.84 |
| BA | 10 | 20.04 -> 20.54 | 7.8-23.6 -> 7.4-24.2 | -2.86 -> -2.69 | 3.47 -> 3.19 | 0.66 -> 0.67 | 0.84 -> 0.84 |
| BA | 100 | 21.86 -> 22.71 | 8.8-28.6 -> 8.4-28.3 | -3.76 -> -3.51 | 5.06 -> 4.83 | 0.55 -> 0.54 | 0.80 -> 0.83 |

At the same window v4 is a modest improvement: the grid median rises 0.2-0.9 m/s, the bias shrinks by 0.2-0.4 m/s and the RMSE by 0.2-0.3 m/s. Skill changes by +0.01 to +0.03 except BA at RP100 (-0.003). The WY1941-2026 values in the results table above differ from this refit because of WY2026 and the 37th station. The matched obs side also differs slightly between the versions (cnn_raw, 36 common stations, v4 refit against v3: -4.6 % to +1.1 %, median 0), so part of the v3 -> v4 delta is the obs side, not the product.

## v3 record (WY1941-2025, CNN-RTMA v3; superseded by the v4 results above)

The sections below are the v3 results and stay as the record of that run. CNN-BC (`cnn_bc`) and the `cnn_q82`/`cnn_q92` levels exist only in v3.

### Results and interpretation (v3)

v3, matched mode, 36 stations; median model/obs ratio in brackets.

| | RP1 bias | RP10 bias | RP100 bias | Willmott skill RP1 / RP10 / RP100 |
|---|---|---|---|---|
| ERA5 | −4.2 (0.68) | −5.5 (0.66) | −7.3 (0.63) | 0.44 / 0.41 / 0.38 |
| RTMA (13 WY) | −1.7 (0.85) | −2.2 (0.87) | −3.5 (0.85) | 0.70 / 0.63 / 0.51 |
| CNN-RTMA raw | −3.8 (0.70) | −4.8 (0.69) | −6.0 (0.69) | 0.48 / 0.45 / 0.43 |
| CNN-RTMA BC | −2.3 (0.80) | −4.6 (0.70) | −7.4 (0.62) | 0.61 / 0.45 / 0.39 |

Bias is in m/s.

**Every product underestimates the station return levels at every return period, and RTMA is clearly closest.** ERA5's deficit grows with return period. RTMA stays at about 0.85× the observations and puts 86 % of its RP100 confidence intervals in overlap with the observed ones.

**The raw CNN P50 does not carry its hourly skill into the extremes.** Its return levels sit only slightly above ERA5's, about 0.69× the observations at every return period, even though the model was trained to emulate RTMA and beats ERA5 by about 0.2 in hourly station skill. This is the expected behaviour of a conditional median: P50 regresses each hour toward the middle of its predictive distribution, and the peaks are exactly where that costs most. The tail shapes agree (median ξ around −0.03 for the CNN against −0.15 to 0.01 for the observations), so the problem is the level of the peaks, not the shape of the tail. That matches the "location, not shape" error found in the v3 validation.

**The bias-corrected field is not usable above RP1.** The correction maps each cell through `np.interp`, which clamps every raw value above the 11-month fit window's CNN maximum to that window's single RTMA maximum. As a result:
- In a typical cell (the median), 60 % of the POT peaks sit on that cap; offshore it is 100 % (panel b).
- In 52 % of cells, RP100 − RP1 < 0.05 m/s.

So the correction raises RP1 to near RTMA's level (bias −2.3) but flattens the tail, and its RP100 ends up worse than the uncorrected field's (median BC − raw −2.5 m/s). Fitting the correction on hourly speeds while extremes live above the fit range is the root cause. More data alone won't fix it; the tail handling has to change.

**The error is spatially organised.** The airport stations (IEM, mostly inland and valley sites) are underestimated more than the Bay and offshore stations (NDBC). Median matched RP10 ratio, IEM against NDBC:

| Product | IEM | NDBC |
|---|---|---|
| RTMA | 0.77 | 0.93 |
| CNN raw | 0.68 | 0.75 |
| ERA5 | 0.65 | 0.66 |

Part of this is representativeness, not model error. An ASOS hourly value is one 2-min average at a single point, so its annual maximum is inherently gustier than an hourly 2.5 km grid value.

**The long ASOS records are not homogeneous.** Pettitt finds a significant break (p < 0.05) at 9 of 16 records with ≥ 20 WY. Where the post-break sub-period is long, its RP100 is 6–17 % lower than the full record (OAK −10 %, NUQ −11 %, STS −17 %, 46026 −11 %). That points to instrument and exposure changes rather than a climate signal. The large positive changes at SQL, SUU, RHV, LVK and CCR come from short post-break sub-periods with unstable free-shape fits and should not be read as trends. For the long airport records, inhomogeneity alone puts about ±10 % on the observed RP100.

### Follow-up 2026-10-02: why BC fails, the upper τ levels, and an observation-anchored correction

**Why the BC fails.** The quantile map was fitted on 7,891 hours (Mar 2024–Feb 2025), so its top level is that window's single largest hour. In 79 % of cells a typical year's CNN maximum already exceeds the window's CNN maximum. Everything above that point is mapped to one RTMA value, e.g. 46026: 15.2 → 17.8 m/s gives BC RP1 = RP10 = RP100 = 17.8. At SJC the window's RTMA maximum (8.7) is below its CNN maximum (9.2), so BC lowers the peaks there.

**Upper τ levels** (`cnn_q82`, `cnn_q92`; added to `s1`/`s3`, scatter `figures\scatter_rl_matched_quantiles.png`). Matched mode, median model/obs ratio, with RMSE in m/s in brackets (nothing is fitted here, so it is directly comparable to the LOO RMSE below):

| | RP1 | RP10 | RP100 |
|---|---|---|---|
| P50 (cnn_raw) | 0.70 (4.3) | 0.69 (5.3) | 0.69 (6.9) |
| τ0.82 | 0.78 (3.2) | 0.79 (4.0) | 0.80 (5.5) |
| τ0.92 | 0.86 (2.5) | 0.87 (3.1) | 0.87 (4.7) |
| RTMA | 0.85 (2.6) | 0.87 (3.3) | 0.85 (5.2) |

τ0.92 lands on RTMA's level, not on the observations, so the upper τ series is not a substitute for a correction. Choosing τ by fit is itself a correction, and these levels were never validated against stations.

**Observation-anchored correction** (`s5_obs_correction.py`). This scales the CNN-raw P50 return-level maps by a factor fitted on obs/CNN ratios at the 36 stations in matched mode, working in log space. Six variants were scored by leave-one-station-out (`results\obs_correction_loo.csv`). Mean LOO RMSE over RP1/10/100:

| Variant | Mean LOO RMSE (m/s) |
|---|---|
| none | 5.48 |
| global | 3.21 |
| land/water | 2.96 |
| IDW | 3.21 |
| land/water + 20 km kernel (V4) | **2.87** |
| land/water + GP | 2.90 |

The winner, V4, has LOO bias +0.1 to +0.2 m/s and skill 0.79/0.71/0.51 at RP1/10/100. Fitted factors (land / water):

| | Land | Water |
|---|---|---|
| RP1 | 1.51 | 1.23 |
| RP10 | 1.48 | 1.33 |
| RP100 | 1.45 | 1.38 |

Outputs are `results\rl_map_cnn_obscorr_{full,common}.nc` (with `factor_*` fields; lo/hi exclude correction uncertainty), `figures\map_obscorr.png` and `figures\obs_correction_loo.png`.

Caveats on the correction:
- The land factor comes only from ASOS airports, and part of it is the single 2-min sample against an hourly field. It is also applied to hills and valleys with no station.
- The land/water split follows the RTMA terrain mask (surface height = 0), so the factor steps at the shoreline.
- RP100 gains least: LOO RMSE goes from 6.9 to 3.9 m/s, but skill only from 0.43 to 0.51.

### Follow-up 2026-10-03: USGS BA product (`cnn_ba`)

The BA product is `cnn_fullrecord\CNN_RTMA_v3_r1b_do010_s3_P50_uvs_BA_19400101_20260810.nc`. It is the raw P50 multiplied by a constant factor F per cell, `BA_factor_map_r1b_do010_s3_E3q90.nc`. F is obs/CNN at the hourly P90, fitted at 39 stations over 2020-01 to 2026-08 and interpolated by a Gaussian RBF that is exact at the stations (σ 5 km). The class constants are land 1.257, bay 1.118 and ocean 1.101, and F ranges from 0.61 to 1.77. The new `_P50_uvs_raw_` file matches the old quantile P50 to about 1e-6.

Because F is constant in time, the BA return levels are exactly F × the raw return levels. We checked this to a relative difference of ≤3e-6. BA has replaced CNN-BC as panel (d) in the maps and the main scatters. `cnn_bc` is still in the skill table.

| Matched, 36 stations | RP1 | RP10 | RP100 |
|---|---|---|---|
| BA bias (m/s) | −2.15 | −2.86 | −3.76 |
| BA RMSE (m/s) | 2.60 | 3.47 | 5.06 |
| BA Willmott skill | 0.74 | 0.66 | 0.55 |
| BA median ratio to obs | 0.85 | 0.84 | 0.80 |
| s5 V4 (LOO) bias / RMSE | +0.15 / 2.06 | +0.22 / 2.63 | +0.11 / 3.92 |

BA is a large improvement on raw and on CNN-BC, and it roughly matches RTMA. It still runs 15–20 % low at the return levels, for two reasons. First, it is anchored at the hourly P90, and observed peaks scale up more than P90 does. Second, it is fitted in-sample at most of the scoring stations, so the s5 numbers (which are LOO) are the stricter test and it still wins. The station-exact RBF puts bull's-eyes in the map where F < 1, at TIBC1, PXOC1 and RCMC1 near Richmond and the Golden Gate.

### Recommended next step

Do not use CNN-RTMA BC for RP10 and above. Two ways to get a usable high-resolution 85-year return-level field:

1. **Return-level delta (cheap, available now).** Scale the CNN-raw full-record maps by RTMA / CNN-raw over the common WY2012–2025 window, separately for each RP: `RL_full = RL_cnn_raw_full × RL_rtma_common / RL_cnn_raw_common`. This keeps the CNN's 85-year sampling of storms and RTMA's level and texture. Its weakness is that RTMA has only 13 years, so smooth the ratio field (for example a 5 × 5-cell median) before applying it.
2. **Refit the bias correction with tail extrapolation.** Use the full 2020–2025 RTMA overlap instead of 11 months, and above the top level apply a constant ratio `rtma_q[-1]/cnn_q[-1]` instead of clamping. Then rerun `s1` → `s4` for `cnn_bc`.

Option 1 can be built from the existing `rl_map_*` files in a few lines and scored with the same `s4` scatter.

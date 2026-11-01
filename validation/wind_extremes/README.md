# SF Bay wind return levels (RP1 / RP10 / RP100)

The 1-, 10- and 100-year 10 m wind speed for ERA5, the RTMA analysis, CNN-RTMA v3 raw (P50) and CNN-RTMA v3 bias-corrected (P50), fitted the same way at 36 observation stations and scored against them.

Plan: `validation/plans/2026-10-02-wind-return-levels.md`. Built 2026-10-02. Moved into the repo with `validation/plans/2026-10-03-wind-extremes-to-repo.md`.

**v4 inputs (2026-11-01).** `windx_common.py` now points at CNN-RTMA v4 (`sf_bay_rtma\results\v1940_product`), ERA5 and RTMA through WY2026 and the refreshed IEM/NDBC files; the statistics are unchanged. CNN-BC and its tail diagnostic are removed (rejected), and `cnn_q82`/`cnn_q92` are now `cnn_q89`/`cnn_q95` (native tau 0.8922 and 0.9510). The Results sections below are the v3 results (WY1941-2025) and stay as the record of that run until the v4 results replace them.

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
- `COSMOS_WINDX_CNN_ROOT` holds `sf_bay_rtma\results` and `cnn_fullrecord`.
- `COSMOS_WINDX_OUTPUT_ROOT` receives `cache\`, `results\` and `figures\`. Nothing generated is written inside the repo. All file names below that start with `cache`, `results` or `figures` are relative to this root.

On Linux (Caldera) use `export` with the equivalent paths.

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
python -m pytest tests -q       # unit tests of the helpers
```

s1 skips any product with a `finalize OK` line in `cache\s1_log.txt`. It needs the 127 GB CNN files and takes hours. s2 to s6 take about 25 min together. `checks\compare_to_golden.py` compares a fresh run with the saved v3 snapshot of results (rtol 1e-9). That snapshot is tied to the v3 inputs and to commit `5e2edc8` (tag `wind-extremes-v3`): from the v4 input change on, a run reads different files and a different record, so the comparison no longer applies to the default configuration. To re-verify the engine against the golden, check out that tag and run s2-s6 with the v3 inputs.

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
| RTMA | `sf_bay_rtma\raw_data\RTMA_SFbay_2p5km_{eastward,northward}_wind_2011_2026_UTM10_earthrel.nc` (under `COSMOS_WINDX_CNN_ROOT`) | 2.5 km; speed = hypot(u, v); record to 2026-10-04 18:00 UTC with gaps |
| Obs | `...\observed_data\pws_sfbay_waterfront_{iem,ndbc}_to20261001.nc`, 4 USGS mooring files | CWOP not used (file fails to open) |

## Method (decisions agreed 2026-10-02)

| Item | Choice |
|---|---|
| Estimator | POT/GPD at λ = 3 peaks/yr. Greedy declustering with peaks ≥ 48 h apart, run on daily maxima of hourly speed. GPD by L-moments; the exponential-tail value is kept as `rl_exp_*`. 1000-draw nonparametric bootstrap, 90 % CI. Code: `rainlib.extremes` (acfcd_rain) |
| Year | Water year from October. A day is valid with ≥ 12 h, a WY with ≥ 80 % valid days. Series with < 6 valid WY are listed as `dropped_<6yr`, not fitted |
| Windows | Maps: `full` (ERA5/CNN WY1941–2025, RTMA WY2012–2025) and `common` (WY2012–2025). Stations: `full` (each source's own record; the product value is read from its full map) and `matched` (obs and product on the same WYs and the same valid-hour mask; each product has its own obs fit) |
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
- `results\station_table.csv`, `qc_top_peaks.csv`, `homogeneity.csv`.
- `figures\map_rl{1,10,100}_{full,common}.png`: 1 × 4 panels with obs circles in the same colour scale.
- `figures\scatter_rl_{full,matched}.png`: 1 × 3 panels (RP1/10/100) with 90 % CI whiskers, the 1:1 line and ±20 %. In matched mode each product's point uses its own matched obs value, so x differs slightly between products.
- v3 only: `figures\bc_tail_diagnostic.png` (BC − raw RP100, fraction of peaks on the BC clamp, the RTMA cap). Removed in v4 with the BC product.

### Deviations from the plan

- RTMA has 13 valid WY, not 14. WY2015 is 30 % missing in the local RTMA archive, so it is dropped.
- In 19 full-window and 716 common-window CNN-BC cells the L-moment GPD fit fails, because the excesses are all tied at the clamp (fit_code 3, NaN on the maps).
- The monotonicity check uses a 1e-3 m/s tolerance. CNN-BC cells with ξ ≪ 0 have RP1 ≈ RP100 to float precision. Collapsed cells are counted separately in `s3_log.txt`.

## Results and interpretation

Matched mode, 36 stations; median model/obs ratio in brackets.

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

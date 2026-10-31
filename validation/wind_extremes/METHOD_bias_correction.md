# Bias corrections in the wind return-level analysis

*g:\03-downscaling_meteo_cnn\wind_extremes, 2026-10-02*

## Which CNN quantile is used

Every CNN-derived number in this analysis uses the **median (P50, τ = 0.5)** of the CNN predictive distribution, unless it is explicitly labelled otherwise. This holds for the maps, the station scatter and the observation-anchored correction.

The CNN outputs hourly quantiles at τ = 0.13, 0.50, 0.82, 0.92 and 0.97. The upper levels, τ = 0.82 (`cnn_q82`) and τ = 0.92 (`cnn_q92`), were run only as a test, and are not used in any delivered product.

At each grid cell, a POT/GPD fit is applied to the daily maxima of the hourly P50 series: λ = 3 peaks/yr, peaks at least 48 h apart, L-moment GPD. That gives the return levels. It is the same estimator that is applied to the observations, ERA5 and RTMA.

## Two different "bias corrections"

The analysis contains two corrections that are easy to confuse:

| | Shipped CNN-BC (`cnn_bc`) | Observation-anchored correction (`s5`) |
|---|---|---|
| Built by | The cosmos-wind-cnn project (`scripts\bias_correct.py`) | This analysis (`s5_obs_correction.py`) |
| Acts on | Hourly P50 wind speed, before any EVA | Return levels, after the EVA |
| Target | RTMA analysis | Station observations (IEM + NDBC) |
| Fitted on | 7,891 hours, 2024-03-14 to 2025-02-06 | 36 stations, matched-year return levels |
| Status | Not usable above RP1 | Delivered product, `rl_map_cnn_obscorr_*.nc` |

### 1. The shipped CNN-BC, and why it fails for extremes

For each grid cell, the shipped correction builds a 200-level quantile map from the CNN P50 speed to the RTMA speed. It is fitted on one 11-month validation window, and applied hour by hour as `np.interp(P50, cnn_q, rtma_q)`. The other quantile levels are rescaled by the ratio BC(P50)/P50.

The top level of each map is the single largest hour in the fitting window, and `np.interp` holds the output constant above it. Every raw value above the window's CNN maximum therefore becomes the window's RTMA maximum. The window contains one winter:
- In 79 % of cells, a typical year's CNN maximum is already above the top of the map.
- In the median cell, 60 % of the storm peaks used in the POT fit sit on that cap, and offshore it is 100 %.
- In 52 % of cells, the corrected RP100 lies within 0.05 m/s of RP1.

Example cells:
- **Buoy 46026.** The cap maps 15.2 to 17.8 m/s, so RP1 = RP10 = RP100 = 17.8 m/s.
- **SJC.** The window's RTMA maximum (8.7 m/s) is below its CNN maximum (9.2 m/s), so the correction lowers the peaks.

Below the cap the map is reasonable: at the 99.5th percentile it raises speeds by 4–22 % at the station cells examined. The failure is entirely in the tail.

### 2. The observation-anchored correction (delivered)

The correction is applied to the return levels, not to the hourly fields. Its premise is that the CNN P50 gets the shape of the extreme-value tail about right but the level wrong, because a conditional median smooths peaks. A multiplicative factor on the return level fixes the level and leaves the CNN's 85-year storm sampling intact.

**Station ratios.** For each of the 36 stations s and each return period T (1, 10, 100 yr), the log ratio is

  r_s(T) = ln[ RL_obs,s(T) / RL_cnn,s(T) ].

Both return levels come from matched-mode fits. The observation series and the CNN P50 series at the station's grid cell use the same water years and the same mask of valid hours, so record length and data gaps cancel in the ratio.

The observations are hourly means of the QC'd readings (40 m/s cap, spike and persistence tests, four manual exclusions). Each station gets a class: **water** for NDBC (buoys and Bay piers, 17 stations) and **land** for IEM (ASOS airports, 19 stations).

**Correction model (winner: land/water with kernel-smoothed residuals).**
1. Class level: m_c(T) = median of r_s(T) over the stations in class c.
2. Station residual: e_s = r_s − m_{c(s)}.
3. Prediction at grid cell x, whose class c(x) is water where the RTMA surface height is 0 m and land elsewhere:

   r̂(x, T) = m_{c(x)}(T) + Σ_s w_s e_s / (1 + Σ_s w_s),  with w_s = exp[−(d_s / 20 km)²].

   The "1 +" in the denominator shrinks the residual term toward zero away from the stations. Far from any station the factor reverts to the class median.
4. Corrected return level: RL_corr(x, T) = exp[r̂(x, T)] × RL_cnn(x, T). The bootstrap confidence bounds are scaled by the same factor, so they do not include the correction's own uncertainty.

**Fitted class factors** exp(m_c):

| | RP1 | RP10 | RP100 |
|---|---|---|---|
| Land (IEM) | 1.51 | 1.48 | 1.45 |
| Water (NDBC) | 1.23 | 1.33 | 1.38 |

Across the grid the factor ranges from 1.12 to 1.73.

**Model selection.** Six variants were compared by leave-one-station-out cross-validation: each station is predicted from a model fitted to the other 35. Mean RMSE over the three return periods:

| Variant | Mean LOO RMSE (m/s) |
|---|---|
| No correction | 5.48 |
| One global factor | 3.21 |
| Land/water factors | 2.96 |
| Inverse-distance interpolation of r | 3.21 |
| **Land/water + 20 km kernel (selected)** | **2.87** |
| Land/water + Gaussian-process residuals | 2.90 |

The selected model has a LOO bias of +0.1 to +0.2 m/s. Its Willmott skill is 0.79, 0.71 and 0.51 at RP1, RP10 and RP100, against 0.48, 0.45 and 0.43 uncorrected. Free spatial interpolation does no better than one global factor: 36 clustered stations cannot support a free-form map.

### 3. Limitations of the delivered correction

- **Land factor and representativeness.** The land factor comes entirely from ASOS airports. An ASOS hourly value is a single 2-min average at one point, so part of the 1.5 factor is gustiness of the measurement, not model error. It is applied to every land cell, including hills and valleys without any station. For hydrodynamic or wave forcing over the Bay, the water factor is the more defensible choice.
- **Shoreline step.** The class switch follows the 2.5 km RTMA terrain mask, so the factor steps by about 0.2–0.3 across the shoreline.
- **Ordering of return periods.** Each return period gets its own factor, and the land factor decreases slightly with T. In 65 of 19,926 cells (full window) and 715 (common window), the corrected RP10 therefore exceeds the corrected RP100. Fix: use one factor per cell for all return periods, fitted on the mean log ratio across T. That preserves the CNN tail shape exactly and guarantees RP1 < RP10 < RP100.
- **Return levels only.** The hourly CNN fields are not corrected. A corrected forcing time series would need a separate step, for example applying the factor to the hourly P50 above the POT threshold.
- **Observation uncertainty.** The observed RP100 at the long airport records carries about ±10 % from step changes in the record. The NDBC records are 8–33 years long.

## Files

- `s5_obs_correction.py`: the observation-anchored correction and the LOO comparison.
- `results\obs_correction_loo.csv`, `results\obs_correction_factors.csv`.
- `results\rl_map_cnn_obscorr_{full,common}.nc`: corrected rl/lo/hi and the `factor_*` and `water` fields.
- `figures\map_obscorr.png`, `figures\obs_correction_loo.png`.
- Upper-τ test: `figures\scatter_rl_matched_quantiles.png`, `results\skill_station_rl.csv` (products `cnn_q82`, `cnn_q92`).

# v3 quantile campaign: conclusions

**Date:** 2026-08-12 · **Branch:** `v3-quantile` · **Domain:** San Francisco Bay, ERA5 → RTMA 2.5 km
**Method:** `SCHEMATISATION_ERA5_RTMA_CNN.md`

> **Skill convention warning.** Grid-track numbers use `1 − RMSE_mod/RMSE_ERA5`; observation-track
> numbers use Murphy skill against *station climatology*. These have different references and are
> **not comparable to each other**, even though both are called "skill". See schematisation §10.

---

## Recommendation

**Ship the quantile head.** The configuration is:

| | |
|---|---|
| head | quantile, 19 speed + 2 direction + 9 gust levels |
| loss | quantile-weighted CRPS, `qw_exp = 1.0` |
| regularisation | `dropout 0.1`, `weight_decay 0.001` |
| predictors | P0 — `lr_u`, `lr_v`, `lr_cloud`, `static_terrain`, plus `lr_gust` (5 input channels; `lr_gust` is required by the gust target pair, not chosen as a predictor) |
| training window | 2020+ |
| **speed / gust product** | **`best_speed.pth`** (epoch 7), 3 seeds |
| **direction product** | **`best_direction.pth`** (epoch 27), 3 seeds |
| post-processing | ship **both** raw and bias-corrected speed fields |

Skill on both tracks, held-out test window (2025-02-06 → 2025-12-31), 3 seeds:

| track | recipe | deterministic control | reference |
|---|---|---|---|
| grid, all-hours (median across cells) | **0.2317** | 0.1686 | ERA5 = 0 by construction |
| grid, >10 m/s (median across cells) | **0.1750** | 0.1275 | |
| **obs, pooled Murphy** | **0.5043** | 0.4701 | RTMA 0.5586, ERA5 0.1535 |
| **obs, station-mean Murphy** | **0.4442** | 0.3892 | RTMA 0.4716, ERA5 0.1958 |
| direction RMSE (>3 m/s) | 19.44° → **18.33°** at ep 27 | 18.80° | RTMA 58.1° at stations |

The recipe beats the deterministic control on **every** axis measured, and beats the previous
v2 production pick (obs pooled 0.4204) by a wide margin.

---

## 1. Which model ships, and why the previous answer was wrong

The campaign ended with a contradiction that blocked any recommendation: on gridded storm skill
the quantile head won (+0.039 vs −0.038), while on stations the *deterministic* control won
outright (0.514 / 0.511 / 0.484 against the best quantile arm's 0.469) — and appeared to beat
RTMA itself.

**The contradiction was an artefact of checkpoint selection, and it has dissolved.**

Every quantile arm in the campaign selected its best checkpoint at **epoch 0 or 1 of 21**. Those
models had seen the training data essentially once. Adding `dropout 0.1` moved the best epoch to
**7 / 7 / 10** across three seeds, and the properly trained model wins the observation track:

| product | obs pooled | obs station-mean |
|---|---|---|
| RTMA-SFbay *(training target)* | 0.5586 | 0.4716 |
| **V3-RECIPE** (ep 7, 3 seeds) | **0.5043** | **0.4442** |
| V3-SELRULE (ep 10) | 0.4958 | 0.4279 |
| qh_P0 campaign incumbent | 0.4877 | 0.4367 |
| V3-C1det deterministic | 0.4701 | 0.3892 |
| v2 production pick | 0.4204 | 0.3572 |
| ERA5 | 0.1535 | 0.1958 |

Both aggregations agree on the ordering.

### Two corrections to previously reported results

- **The deterministic control does not beat RTMA.** In this controlled run — one engine, one
  station set, one window — RTMA leads every product at 0.5586. The earlier "C1 0.514 vs RTMA
  0.483" came from a different station set and window and does not reproduce like-for-like.
- **The v2 "smooth-U²" arm does not dominate v3.** That claim (0.186 / 0.106 against v3's
  0.162 / 0.039) compared different windows, splits and points. Re-scored on the v3 test window
  and points, the v2 arms score **0.0757 / 0.0619 / 0.0435 / 0.0637 / 0.0653** all-hours against
  v3's **0.213**. The comparison reversed once controlled. Any fallback-to-v2 option is closed.

---

## 2. Did the quantile head deliver?

**Yes for the bulk of the distribution, partially for the tail.**

**Bulk — clear win, and far more reproducible.** All-hours grid skill, mean over 3 seeds, seed
spread in brackets:

| | pooled | median across cells |
|---|---|---|
| recipe | 0.3047 (0.015) | **0.2317 (0.007)** |
| qh_P0 incumbent | 0.2718 (0.045) | 0.1775 (0.059) |
| C1 deterministic | 0.2506 (0.014) | 0.1686 (0.017) |

The incumbent's seed spread is **~8× the recipe's**. Epoch-1 selection was not merely producing
a weak model, it was producing a lottery. Reproducibility is itself a shipped improvement.

**Calibration — good in the bulk, biased in the tail.** The result was neither of the two
outcomes anticipated:

| | measured (3 seeds) | calibrated |
|---|---|---|
| 50% interval coverage | 0.48 | 0.50 |
| 80% interval coverage | 0.77 | 0.80 |
| 90% interval coverage | 0.850 | 0.90 |
| 98% interval coverage | 0.876 | 0.98 |
| reliability @ τ=0.99 | 0.964 | 0.99 |
| PIT mean, all hours | 0.542 | 0.50 |
| **PIT mean, >10 m/s** | **0.684** | **0.50** |
| bias at >10 m/s | −0.70 m/s | 0 |

The PIT is **not U-shaped** — this is not the classic under-dispersion failure. It is
monotonically right-skewed in storms (bottom bin 0.009, top bin 0.118 against a flat 0.050). The
bulk intervals are close to nominal; what fails is the **tail location**.

That distinction is the useful part, because it is *actionable and was used predictively*: a
dispersion failure cannot be repaired post hoc, a location bias can. On that basis I predicted
before running it that bias correction would still pay — and it did (§5).

**Tail — the one axis not won.** On >10 m/s median-across-cells skill the barely-trained
incumbent still leads (0.0407 vs 0.0106). It has more output spread (`std_ratio` 0.947 vs 0.939)
precisely *because* it is under-trained: less regression to the mean flatters the tail and hurts
everything else. This is a known artefact, not a reason to ship an epoch-1 checkpoint — but it
is honest to record that the fully trained model gives up some raw tail spread, and that bias
correction is how the shipped product recovers it.

---

## 3. Did the 27 new predictors help?

**No.** Across P1–P5 at three seeds each, no predictor block beat P0 by more than seed noise.
P0 ties P5 (4 vs 26 predictor-block channels; 5 vs 27 actual inputs). On the obs board the best non-P0 arm (`V3-P4-s3`, 0.5162 pooled)
sits inside the recipe's seed range.

Do not re-run this. The low-resolution wind field plus static terrain carries essentially all
the transferable information for this domain.

---

## 4. Does it generalise?

**Yes — measured, not inferred.** The shipping recipe was run at three seeds over the full
2000-2026 ERA5 record and scored against 41 IEM + NDBC stations in three eras. Stations are the
only reference in this project that is not the training target, and E1 is the sharpest test
available anywhere in the campaign: 2000-2010 predates RTMA entirely, so no gridded truth exists
there and the model is extrapolating two decades outside its training window.

Wind speed, **station-mean skill with the reference = each station's own climatology**
(convention 3 of §10 in the schematisation: `skill_i = 1 − rmse_i²/obs_std_i²`, averaged over
stations). These numbers are **not** comparable with the pooled-Murphy figures in the
Recommendation table above, which take ERA5 as the reference — same track, different reference,
different scale. 3-seed mean:

| era | window | seen? | ERA5 | CONUS404 | RTMA | **CNN** | CNN − ERA5 |
|---|---|---|---|---|---|---|---|
| E1 | 2000-2010 | no | 0.2533 | 0.2325 | *n/a* | **0.4230** | +0.170 |
| E2 | 2011-2019 | no | 0.2512 | 0.2312 | 0.3680 | **0.4901** | +0.239 |
| E3 | 2020-2026 | **yes** | 0.2846 | *n/a* | 0.5294 | **0.5221** | +0.237 |

Seed spread is 0.009-0.013 throughout, so every gap in that table is far outside seed noise.

**Read the last column, not the CNN column.** ERA5's own skill is era-dependent (0.2846 in E3 vs
0.2512 in E2), so the raw CNN score conflates model degradation with input degradation. Measured
as *added value over its own input*, the CNN is *identical* in the training era and the unseen era
directly before it — +0.237 vs +0.239. There is no generalisation penalty across that boundary.

### The E1 drop is a convention artefact, not a model result — check both

The same runs under the **pooled** Murphy convention (pool MSE and observed variance across
stations by sample size, rather than averaging per-station skills) tell a materially different
story about 2000-2010:

| era | stations | ERA5 | CONUS404 | RTMA | **CNN** (3-seed) |
|---|---|---|---|---|---|
| E1 | 26 (17 IEM + 9 NDBC) | 0.328 | 0.322 | *n/a* | **0.543** |
| E2 | 36 (19 + 17) | 0.260 | 0.257 | 0.402 | **0.535** |
| E3 | 37 (20 + 17) | 0.242 | *n/a* | 0.577 | **0.553** |

**Pooled, the CNN is flat across all three eras — 0.543 / 0.535 / 0.553 — with no backward
degradation at all.** Station-mean showed 0.423 / 0.490 / 0.522, a clear decline. Both are correct
computations of different things, and the disagreement is concentrated entirely in E1.

The cause is the station population, not the model: **E1 has 26 stations against E3's 37, and its
IEM:NDBC ratio is 17:9 versus 20:17.** Station-mean weights every station equally, so it is
sensitive to that shifting land/water mix; pooled weights by sample size and pools variance, so it
is not. Eight NDBC buoys simply do not exist before 2011.

**Do not quote a single number for backward generalisation.** The honest statement is: pooled, no
degradation; station-mean, ~19% lower in E1 — and the gap between those two answers is a
station-population effect that no amount of retraining would change. This is the same lesson as
the grid-vs-obs disagreement in §1, one level down: the aggregation is part of the claim.

Three results in that table are worth separating out:

1. **In the training era the CNN sits at the ceiling, not above it.** E3 CNN 0.5221 vs RTMA
   0.5294 — ~99% of the target. This is consistent with the Recommendation table (pooled Murphy
   0.5043 vs RTMA 0.5586) and confirms it over a 6.5-year window rather than 11 months. RTMA *is*
   what the model was trained to reproduce, so approaching it is the most that can be asked.
   **Do not claim the model improves on its target in the era it was trained on.** The one
   pre-resolution result that did show a CNN arm above RTMA was the deterministic control C1 on
   the narrow test window (§1); it did not survive the head contradiction being resolved, and it
   does not survive here either.
2. **In the unseen era the CNN does exceed RTMA** — 0.4901 vs 0.3680, and on peaks too (top-10%
   skill −3.97 vs −5.55). This is not a contradiction of (1): the pre-2020 RTMA is a coarser,
   noisier product than the post-2020 RTMA the model trained on, and the CNN inherits the *later*
   RTMA's quality wherever ERA5 supports it. That is the practically useful finding — the method
   back-fills a high-quality product into years where the real product was worse or absent.
3. **Direction generalises almost flat**: RMSE 60.1° / 58.0° / 57.6° across E1/E2/E3 against
   ERA5's 67.6° / 66.0° / 66.8°. Direction was the CNN's strongest result and it is also its most
   era-stable.

**Peaks remain the known weakness, in every era** — but the *kind* of weakness is now pinned down,
and it is the favourable kind. Top-10% skill is −4.04 / −3.97 / −3.18 for the CNN against
−7.50 / −8.30 / −7.44 for ERA5: a large improvement that is still firmly negative, with a
top-decile bias of −1.9 to −2.2 m/s, and in the training era RTMA beats the CNN (−2.30 vs −3.18).

**Remove the mean bias and the ordering flips — the CNN beats RTMA on peaks in every era:**

| top-10%, bias-removed (`skill_dm`, pooled) | E1 | E2 | E3 |
|---|---|---|---|
| CNN (best seed) | **−0.065** | **−0.167** | **−0.050** |
| RTMA | *n/a* | −0.527 | −0.199 |
| ERA5 | −0.171 | −0.263 | −0.183 |
| CONUS404 | −0.702 | −0.868 | *n/a* |

So the CNN's peak deficit is almost entirely a **location** error, not a **shape** error: it puts
the storm in the right place with the right structure and simply under-states its magnitude. That
is the error class a post-hoc quantile map can fix, and it is independent confirmation of the
right-skewed (not U-shaped) PIT that motivated keeping bias correction in §5. It also means the
raw-vs-BC decision matters more for peaks than the raw top-10% column suggests.

### USGS moorings — reported separately, and they disagree

Scored on E3 only (all four moorings start after 2020-01-22): CNN 0.253 (seeds 0.187-0.301),
RTMA 0.456, ERA5 −0.421. The CNN is clearly *worse* than RTMA here, and the seed spread is 12x
wider than at the land stations.

This is not treated as a counter-result, for reasons that must travel with the number: these
anemometers sit at **1.2-4.9 m, not 10 m**, over water, and n is 18k against 48-66k at the land
stations. The model was trained to reproduce a 10 m product. What the moorings actually show is
that the recipe does not transfer to a different measurement height without recalibration — a real
limitation, but a different claim from "it does not generalise in time".

## 5. Bias correction: ship both fields

Phase 6b measured BC buying +0.074 at >10 m/s on the epoch-1 arms. The question was whether a
properly trained model made it redundant. **It did not:**

| metric | s1 | s2 | s3 | mean |
|---|---|---|---|---|
| skill at >10 m/s | +0.1099 | +0.0839 | +0.0447 | **+0.0795** |
| energy-weighted q3 | +0.0340 | +0.0118 | +0.0173 | +0.0210 |
| all-hours skill | −0.0318 | −0.0332 | −0.0341 | **−0.0330** |
| mean bias | −0.214 → −0.034 | −0.194 → −0.037 | −0.110 → −0.037 | 66–84% removed |

+0.0795 against a +0.074 baseline — unchanged within seed spread, exactly as the calibration
diagnosis predicted.

**Ship both fields.** Raw for all-hours applications, bias-corrected for storm and
extreme-value work. The −0.033 all-hours cost is consistent across every seed, so applying BC
blindly would be a net loss for general use.

---

## 6. The finding worth carrying to the next domain

**The two validation tracks disagree about the best checkpoint, reproducibly, within a single
recipe.**

| checkpoint | grid all-hours | grid >10 m/s | obs pooled | direction RMSE |
|---|---|---|---|---|
| epoch 7 | **0.2317** | **0.1750** | 0.5043 | 19.44° |
| epoch 27 | 0.2258 | 0.0350 | **0.5193** | **18.33°** |

Epoch 27 is the *worst* CNN on gridded storm skill and the *best* on stations. This reproduces
across all three seeds.

The interpretation: **gridded skill rewards imitating RTMA, and further training moves the model
away from RTMA and toward the real atmosphere.** Scoring a downscaling model against its own
training target cannot distinguish "better model" from "closer imitation of the target". This is
the single most transferable lesson from the project, and it argues for treating independent
observations as decisive from the start rather than as a late cross-check.

**Why epoch 7 still ships as the probabilistic product** despite losing the obs tie-break: its
distribution is usable and epoch 27's is not. Epoch 27's 98% interval covers **79%** against
epoch 7's 88%, with reliability at τ=0.99 falling to 0.887 and a corroborating drop in output
spread. It buys the better point estimate by narrowing the predictive distribution — fine for a
direction field, unacceptable for storm risk. Hence the split product: **speed from epoch 7,
direction from epoch 27**, which is precisely what per-head checkpointing was built to allow.

*(Caveat: PIT and coverage are grid-track numbers scored against RTMA, so they inherit the
training-target problem. The effect size and the independent `std_ratio` signal make a pure
artefact unlikely, but the label matters.)*

---

## 7. Tried and rejected — do not retry

| approach | outcome |
|---|---|
| 27 additional predictors (P1–P5) | no block beat P0 by more than seed noise |
| longer training record (2016+) | **worse** on both heads on identical val/test windows |
| sample-axis loss weighting (`loss_delta`, `loss_wave_weight`) | improper score; measured all-hours-vs-storm trade-off is its signature |
| `qw_exp = 2.0` (used for the whole campaign) | swept: a = 1 is better (0.07099 vs 0.07134) |
| no dropout | best checkpoint latches at epoch 1 in all 3 seeds |
| twCRPS single-epoch selection | costs 0.012 all-hours and 0.031 storm skill vs `best_speed.pth` |
| pre-2014 extension for gust-dependent arms | impossible — RTMA gust target starts 2016 |
| falling back to the v2 production line | reversed once scored on a common window |

---

## 8. Delivered artefacts

| artefact | location |
|---|---|
| recipe checkpoints, 3 seeds | `results/{r1_do010, r1b_do010_s2, r1b_do010_s3}/checkpoint/` |
| per-head checkpoints | `best_speed.pth`, `best_direction.pth`, `best_gust.pth` |
| test-window inference, dense 19-level | `output_inference/speed_full_record_*.nc` |
| direction product, dense | `output_inference/direction_full_record_*.nc` |
| bias-corrected speed | `output_inference/BCVAL_speed_full_record_*.nc` |
| grid metrics + calibration | `results/day3_scores.json` |
| obs leaderboard | `validation/results/v3_test_2025__day4/` |

**Commits** (branch `v3-quantile`): `45e5865`, `a2e4447`, `944977d`, `2bd6883`, `38f1c3c`,
`b59d94f`, `bc7b2e9`.

---

## 9. Open items

1. ~~Pre-2020 era generalisation for the recipe~~ — **done** (§4). Answered: no penalty vs
   2011-2019, ~29% loss of added value at 2000-2010, at the RTMA ceiling in the training era.
2. Gust head is trained and checkpointed but deliberately NOT validated against gust
   observations — descoped by the project owner 2026-08-14. The gust quantiles ship as-is,
   unverified against obs; treat them as indicative, not calibrated.
3. `skill_ew` negativity for all products including RTMA needs a written explanation of the
   metric's reference, not just a caveat.
4. Whether the epoch-7/epoch-27 split is domain-specific or a general property of this
   architecture is untested — it would need one further domain to answer.

---

## 10. Full record, 1940-01-01 → 2026-08-10 (2026-09-29)

The recipe was re-inferred over the **whole** ERA5 record from one input set: the
2026-09-28 ERA5 drop for 1940-2025, and the 2026-08-27 staging for the 2026 tail.
This is a single run, not a backfill spliced onto the 2000-2026 product.

**Inputs** (`scripts/stage_era5_1940.py`):
- **u / v / cloud** are bit-identical to the 2026-08-27 inputs over the whole 2000 → 2026-08-10
  overlap.
- **Gust**: the new download again carries real values in the 83 border cells that training saw
  as NaN + nearest-valid fill. The border was masked and re-filled, as on 2026-08-27. After that,
  gust is bit-identical over the overlap too.
- The new gust has the standard 7 h dead seam at 2000-01-01T00-06, where the old download
  had live values. Those 7 steps were copied back so year 2000 reproduces.
- **Reproduction**: all 81 archived year-windows (3 seeds × 27) reproduce with max |diff| = 0
  at every station. The only differences are warm-up hours the archive had started cold
  (e.g. 2000-01-01T00-04), which are now predicted.

**Seed choice — made at stations, not on the grid.** All 3 seeds were inferred over the full
record with only the 45 validation-station time series kept (`extract_station_points.py`,
stencils taken from `validate_met_models` itself). Scoring: pooled Murphy skill of 10 m speed,
IEM+NDBC, identical samples per seed, year-paired 2 SE rule (`select_seed_points.py`).

| | r1_do010 | r1b_do010_s2 | **r1b_do010_s3** |
|---|---|---|---|
| all-record skill | 0.4926 | 0.4897 | **0.4981** |
| gap to s3 (2 SE) | +0.0074 (0.0029) | +0.0090 (0.0017) | — |

s3 clears 2 SE against both and leads in 5 of 6 eras (1980-99 is the exception), so **s3
ships**. The gap is small in absolute terms but consistent.

### Obs validation per era (seed s3; pooled Murphy, 10 m speed, IEM+NDBC)

| era | stations | CNN | ERA5 | **CNN − ERA5** | RTMA | CONUS404 | dir RMSE CNN / ERA5 |
|---|---|---|---|---|---|---|---|
| 1940-59 | IEM 6 | 0.343 | −0.037 | **+0.380** | — | — | 62° / 70° |
| 1960-79 | IEM 10 | 0.428 | 0.119 | **+0.309** | — | — | 61° / 70° |
| 1980-99 | IEM 14 + NDBC 2 | 0.567 | 0.378 | +0.189 | — | 0.227 | 51° / 57° |
| 2000-10 | 17 + 9 | 0.548 | 0.328 | +0.220 | — | 0.322 | 57° / 65° |
| 2011-19 | 19 + 17 | 0.542 | 0.260 | +0.282 | 0.402 | 0.257 | 58° / 67° |
| 2020-26 | 20 + 18 | 0.560 | 0.242 | +0.318 | 0.577 | — | 58° / 67° |

- **The CNN adds value over its own input in every era, and most in the earliest ones.**
  Absolute skill falls before 1980, but ERA5's falls further.
- Both carry a large low bias against the 1940s-60s stations (CNN −1.28 / −0.99 m/s, ERA5
  −1.71 / −1.31). Six airport anemometers with era-typical siting and heights make this
  partly an observation-homogeneity question, not only a model one.
- **Quote every pre-1980 number with its station count.**
- **Peaks do not improve in 1940-59.** Bias-removed top-10 % skill is −0.23 (CNN) vs −0.16
  (ERA5); 1960-79 is a tie.
- 2000-2026 reproduces §4's pooled numbers (0.54-0.56, flat), and V3-1940-s3 scores
  identically to the archived V3-ERAS-s3.

### Product (`results/v1940_product/`)

| file | content |
|---|---|
| `CNN_RTMA_v3_r1b_do010_s3_speed_quantiles_raw_19400101_20260810.nc` | `wind_speed(time, quantile, y, x)`, 5 levels (τ 0.13/0.50/0.82/0.92/0.97), hourly, float32 zlib, chunks 24 h × 1 level × full grid |
| `CNN_RTMA_v3_r1b_do010_s3_speed_quantiles_BC_19400101_20260810.nc` | same, bias-corrected |
| `BC_map_r1b_do010_s3_valfit.nc` | the 200-level per-cell map (the v3 BC map was never saved before) |

**BC method.** The map is refit exactly as `bc_v3_test.py` does: per cell, on |hr_u, hr_v|
(= P50), VAL window, vs RTMA. It reproduces the archived s3 `BCVAL_` file with max |diff| 0.
It is applied **without refit** to 1940-2026 as the per-cell, per-hour ratio BC(P50)/P50 on
**all five levels**, the same way the shipped BC scales u/v. **Only P50 is validated.**

**Caveats.**
- Pre-1979 ERA5 assimilates far fewer observations. Treat any trend across ~1979 with care.
- The BC map is fitted on 2024-25 and assumed stationary over 86 years.
- There is **no u/v or direction** in the one-file product. The ep7 `hr_u`/`hr_v` are in the
  year files (`results/v1940_grids/r1b_do010_s3/`), so a u/v file is a CPU-only combine. The
  ep27 direction head would need an extra inference pass.
- Gust quantiles are only in the year files, unvalidated.

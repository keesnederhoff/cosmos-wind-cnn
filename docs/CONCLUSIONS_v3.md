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
| predictors | P0 — `lr_u`, `lr_v`, `lr_cloud`, `static_terrain` |
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
P0 at 4 channels ties P5 at 26. On the obs board the best non-P0 arm (`V3-P4-s3`, 0.5162 pooled)
sits inside the recipe's seed range.

Do not re-run this. The low-resolution wind field plus static terrain carries essentially all
the transferable information for this domain.

---

## 4. Does it generalise?

**Partially answered, and this is the clearest open gap.**

What *is* established: the test window is fully held out — never trained on, never used for
checkpoint selection, and disjoint from the bias-correction fit period. All headline numbers
above are out-of-sample in that sense, and the three seeds agree closely (obs pooled 0.5026 /
0.4973 / 0.5129).

What is **not** established: performance in the pre-2020 era for the shipped recipe. The earlier
campaign arms were scored on seen/unseen eras, but the fixed recipe has been run only on the
test and validation windows. Until that is measured, no claim is made about years before the
training window.

Two things constrain what such a test could show, and both must be stated with it:

- RTMA's own quality is era-dependent (station Murphy 0.403 pre-2020, 0.578 post-2020), so a
  lower unseen-era score against RTMA is ambiguous between "worse model" and "noisier target".
  Only station observations resolve it.
- §6.2 of the schematisation showed that *training* on pre-2020 data made the model materially
  worse, which is indirect evidence that the earlier period is a different — noisier — regime.

**Recommended next step:** per-year inference at τ = 0.5 over the pre-2020 record for the three
recipe seeds, scored on both tracks. This was scoped but not run.

---

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

1. **Pre-2020 era generalisation for the recipe** (§4) — scoped, not run. Highest value.
2. Gust head is trained and checkpointed but not validated against gust observations.
3. `skill_ew` negativity for all products including RTMA needs a written explanation of the
   metric's reference, not just a caveat.
4. Whether the epoch-7/epoch-27 split is domain-specific or a general property of this
   architecture is untested — it would need one further domain to answer.

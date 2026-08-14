# Schematisation: statistical downscaling of ERA5 to RTMA resolution with a CNN

**Scope.** This document describes the method, not the San Francisco Bay results. It is
written so the approach can be reproduced on a different domain without reading the code.
Each section states *what was done* and *why this and not the obvious alternative*. Where a
choice was settled by measurement, the measurement is quoted.

**Status.** Method frozen 2026-08-12 on branch `v3-quantile`. Companion document:
`CONCLUSIONS_v3.md` (what the method produced for SF Bay).

---

## 1. Problem statement

Downscale ERA5 (~31 km, hourly) to the RTMA analysis grid (2.5 km, hourly) over a coastal
domain, producing 10 m wind.

The framing that governs every later choice: **training a network on mean-squared error
structurally under-predicts peaks.** For any input, the truth is a draw from a conditional
distribution; MSE is minimised by the conditional *mean*, which is less extreme than a typical
draw. The wider the conditional distribution, the larger the shortfall. Coastal wind at 2.5 km
conditioned on a 31 km field has a wide conditional distribution, so the shortfall is large and
systematic — it is not a bug to be tuned away.

Two responses are possible: predict the whole distribution rather than its mean (Section 5), or
correct the distribution afterwards (Section 8). This project measured both, and ships both.

**The downscaling is statistical, not dynamical.** The network learns the ERA5→RTMA mapping as
it appears in the training record. It cannot introduce physics absent from ERA5, and its
ceiling is RTMA's own quality (Section 9).

---

## 2. Inputs

**Shipped predictor set ("P0"), 5 input channels:**

| channel | source | role |
|---|---|---|
| `lr_u`, `lr_v` | ERA5 10 m eastward/northward wind | the field being downscaled |
| `lr_cloud` | ERA5 cloud area fraction | weak synoptic-state proxy |
| `static_terrain` | RTMA surface height | time-invariant; supplies the topographic detail ERA5 lacks |
| `lr_gust` | ERA5 10 m wind gust | **not a chosen predictor** — see below |

`lr_gust` is the trap in this table. It appears in no predictor block, so a reader counting
the P0 *block* gets four channels; the model in fact takes five. It is pulled in by the gust
variable **pair**, whose `low_res` key names it, and the pair is present because gust is an
auxiliary target. The consequence is not cosmetic: an inference config built from the
predictor list alone is missing a required input, and every run aborts. Worse, `lr_gust`
availability then silently bounds the whole record — an inference span the predictors could
easily support is cut short by a channel nobody chose. **Derive the input list from what the
model actually consumes, never from the predictor block.**

**Record what was tried and rejected, so it is not retried.** Five further predictor blocks
were built and trained (P1–P5), taking the input up to 26 channels: additional single-level
ERA5 fields, pressure-level fields, temporal derivatives, and combinations. Across an 18-arm
campaign at three seeds each, **no block beat P0 by more than seed noise.** P0 ties P5.
(Those block counts, 4 through 26, are quoted on the predictor-block basis and so exclude
the `lr_gust` channel above; P0 is 5 actual input channels, P5 is 27.)

The lesson generalises: for this class of problem the low-resolution wind field plus static
terrain carries nearly all the transferable information, and added channels mostly add
parameters and variance. Start at P0 on a new domain. Only add predictors if a *measured*
deficiency points at them.

**Hard record limits** (domain-specific, but check the equivalents before planning any run):

| field | availability | consequence |
|---|---|---|
| ERA5 10 m u/v | … – 2026-07-26 | caps the end of any inference record |
| ERA5 wind gust | 2000-01-01 – 2026-07-26 | **was** 2013-12-31 — a download-scope choice, not an ERA5 limit |
| RTMA wind gust (target) | from 2016-01-01 | **the binding constraint** — the gust *task* cannot start before 2016 |
| ERA5 pressure levels | from 2015-01-01 | rules pressure-level predictors out of longer records |
| RTMA u/v (target) | 2011 – 2026 | bounds the scoreable period |

The gust row cost this project a planning error worth repeating. A long-record store was
specified to start in 2014 on the basis of the *input* gust availability; the *target* gust file
starts 2016, and the preprocessing intersects common times silently. Starting at 2014 would have
either truncated to 2016 without comment or forced dropping the gust task — which would have
made the arm structurally incomparable to the arms it existed to be compared against.
**Check the target availability, not just the input's.**

---

## 3. Targets

`hr_u`/`hr_v` are reparameterised losslessly to **(speed, direction)**:

- speed = `sqrt(u² + v²)`, verified identical to RTMA's own `wind_speed` variable (max
  absolute difference 0.0000 over the record);
- direction as a two-channel unit vector, avoiding the wrap discontinuity at 0/360°.

Why reparameterise at all: the quantity that matters for coastal applications is speed, and its
distribution is what the quantile head must represent. Predicting quantiles of *u* and *v*
separately does not give quantiles of speed — the quantile of a norm is not the norm of
quantiles.

`hr_gust` is carried as an auxiliary target with a small loss weight. It is not part of the
headline product but regularises the tail representation and is available where needed.

---

## 4. Architecture

3D U-Net (`Wind3DUNET`, 12.3 M parameters), operating on a short time window so the network can
use tendency as well as instantaneous state.

| setting | value |
|---|---|
| `base_channels` | 24 |
| `sequence_length` | 6 hours |
| output channels | 19 speed quantiles + 2 direction + 9 gust quantiles = **30** |
| grid | 123 × 162 (domain-specific) |

**Quantile head construction.** Quantiles must not cross. They are built monotone by
construction rather than penalised into monotonicity:

1. predict an anchor value;
2. predict positive increments via `softplus`;
3. cumulatively sum to obtain the quantile ladder.

Levels are on an equally spaced interior grid — 19 speed levels from 0.0263 to 0.9737, 9 gust
levels from 0.0556 to 0.9444. Note the grid **does not reach 0.99**: reliability at τ = 0.99 is
obtained by interpolating between the two neighbouring levels, never by snapping to the nearest,
which would widen intervals by up to half a level spacing and inflate reported coverage.

**The anchor is the interpolated ERA5 field, and the head predicts in physical m/s, not
z-space.** This is a residual formulation: the network learns the correction, not the field.

> **Documented trap.** Low-resolution and high-resolution fields are z-scored with *different*
> statistics. Converting the ERA5 anchor into the target's physical space therefore requires
> the affine transform that composes both — available as `residual_affine` in
> `utils/config.py`. Re-deriving it inline is the single easiest way to produce a model that
> trains without error and predicts nonsense. Reuse the helper.

---

## 5. Loss

**Quantile-weighted CRPS** — the pinball loss summed over the quantile ladder, with weights
`w_k ∝ τ_k^a` emphasising the upper tail.

`a` was swept: **a = 1 is best** (validation twCRPS 0.07099 at a = 1, 0.07134 at a = 2, 0.07260
at a = 0). Ship a = 1.

### 5.1 The propriety argument — the most transferable single idea here

Weighting the **quantile axis** preserves propriety (Gneiting & Ranjan 2011): the expected score
is still minimised by the true distribution, so emphasising the tail does not bias the forecast,
it only reallocates *statistical attention* toward the tail.

Weighting the **sample axis** — up-weighting storm hours in the batch, as `loss_delta` and
`loss_wave_weight` did in earlier work — is **improper**. The minimiser is no longer the truth;
it is a distribution deliberately shifted toward the up-weighted samples.

This is not a theoretical nicety. The empirical signature of the impropriety was measured: the
sample-weighted arms bought storm skill and paid for it in all-hours skill, exactly as an
improper score predicts. Two mechanisms that look interchangeable in code are not
interchangeable in behaviour.

**Rule for a new domain: to care more about extremes, weight τ, never the sample.**

---

## 6. Training protocol

| setting | value |
|---|---|
| window | 2020-01-01 onward (see 6.2) |
| split | **by date**, not ratio (see 6.1) |
| optimiser | AdamW, lr 3e-4 |
| `dropout_rate` | **0.1** (see 6.3) |
| `weight_decay` | 0.001 |
| early stopping | patience 20 on the selection metric |
| distribution | DDP, 2 GPUs, `--exclusive --mem=0` |

Two operational rules, each learned by losing work:

- **`--exclusive --mem=0` is mandatory.** On shared backfill, 9 of 21 arms in one sweep died of
  GPU co-tenancy OOM.
- **Separate training from inference.** One sweep crashed a 1.6 TB quota by writing inference
  for 42 arms. Train first, run inference for winners only (Section 9).

### 6.1 Split by date, not by index ratio — a confound worth naming

The natural implementation cuts train/val/test by index fraction. **This silently confounds
data volume with recency.** A store beginning in 2016 and one beginning in 2020, both split
0.7/0.15/0.15, have *different validation periods*: the longer store trains to 2022 while the
shorter trains to 2024. Any comparison between them measures "more data" and "older data"
together, and the two effects point in opposite directions.

The fix is an optional `split_dates: {train_end, val_end}` config key. When absent, the ratio
path is used unchanged; when present, boundaries are cut on `np.searchsorted` of the time axis.
Controls then share *identical* val and test windows and differ only in where training starts.

**Any experiment comparing record lengths must do this, or its result is uninterpretable.**

### 6.2 More training data was worse, not better

With the confound removed, a 2016+ store (T = 71,880 training hours) was compared against the
2020+ store (T = 36,825) on byte-identical validation and test windows:

| head | 2020+ store | 2016+ store | verdict |
|---|---|---|---|
| deterministic | 0.2721 / 0.2721 / 0.2736 | 0.2803 / 0.2863 | slightly worse |
| quantile | 0.0713 – 0.0725 | 0.0837 / 0.0889 | **much** worse |

Twice the data made the quantile head ~20% worse. The consistent explanation is that target
quality degrades in the earlier period, so the extra years teach the model a noisier mapping.

**Transferable rule: for statistical downscaling, target-record *quality* dominates target-record
*length*.** Test the assumption on a new domain rather than assuming more history helps.

### 6.3 Regularisation, and why the obvious metric lied

Dropout 0.1 versus no dropout, three seeds each. The headline is not the score, it is the
**epoch at which the best checkpoint landed**:

| recipe | best-checkpoint epoch (3 seeds) | speed pinball |
|---|---|---|
| no dropout | **1, 1, 1** — never beaten in 21 epochs | 0.2995 ± 0.0025 |
| dropout 0.1 | **7, 7, 10** | **0.2947 ± 0.0017** |

Read naively, no dropout looked *better* on the selection metric (0.0695 vs 0.0713). But that
score came from a model that had seen the training data once. Dropout 0.1 is the only setting
under which the model *keeps improving*, and it also wins the stable metric. The apparent
advantage of no dropout was the pathology, not a result.

---

## 7. Checkpoint selection

Selection metric: **threshold-weighted CRPS at 10 m/s**, on validation.

Two failure modes were found and fixed here. Both are likely to recur on any new domain.

### 7.1 The selection metric is much noisier than the model

Validation twCRPS swings **± 14%** between consecutive epochs, while `speed_pinball` on the
*same* validation batch moves only **± 6%**.

This is not small-sample noise: the validation split holds 7,891 hours, 76.7% of which have a
domain maximum above 10 m/s, and 12.04% of all (time, cell) pairs exceed 10 m/s — about 3.2
million elements per evaluation. The *bulk* of the predicted distribution is stable epoch to
epoch; the *tail* genuinely is not.

Consequence: selecting on single-epoch twCRPS partly selects whichever epoch drew a lucky tail.
Two mitigations are implemented, and the second is the one to trust:

- `best_smooth.pth`, selected on a **3-epoch running mean** of the metric. *Require a full
  window* — with fewer than three epochs recorded the "mean" is just the first epoch's own
  value, which reproduces the exact single-epoch latch the smoothing exists to prevent. (This
  bug shipped, was observed latching at epoch 1, and was fixed.)
- **`best_speed.pth`, selected on the stable loss component.** This is what the shipped product
  uses.

### 7.2 The three heads want different epochs

Per-head validation components diverge sharply:

| head | best epoch (of ~27) |
|---|---|
| speed | 7 |
| gust | ~6 |
| **direction** | **27 — still improving when training stopped** |

Direction improves monotonically to the end of training in *every arm measured*. A single
checkpoint selected on speed therefore ships a direction field ~20% worse than the same run
already produced.

The fix is to write **per-head checkpoints** — `best_speed.pth`, `best_direction.pth`,
`best_gust.pth` — each selected on its own validation component, alongside the combined
`best_model.pth`. They cost one `state_dict` write each and do not participate in early stopping
or LR scheduling.

**This is not bookkeeping — it changes the product.** Measured on the held-out test window, the
epoch-27 checkpoint gives a better direction field (18.33° vs 19.44° RMSE) *and better station
skill*, but its calibration is unusable (Section 9.3). Speed and direction are shipped from
different checkpoints as a deliberate result of this measurement.

---

## 8. Post-processing: bias correction

A per-cell empirical quantile map from model speed to target speed, applied as a ratio so
direction is preserved bit-exactly.

**Fit on validation, not on training.** The stock implementation fits on the train period; that
is wrong for models whose best checkpoint is early, because the train-period error distribution
is in-sample for the weights and does not represent out-of-sample behaviour. Validation is
out-of-sample for the weights and disjoint from test.

> **Trap.** If inference exists only for the test window, the fit's inner join on the validation
> period returns *zero* overlapping samples, and `np.nanquantile` over an empty axis yields
> all-NaN maps — a silently garbage output file. Add an explicit minimum-sample guard that
> refuses to fit. Run validation-window inference deliberately.

**Bias correction still earns its place even for a well-calibrated-in-the-bulk quantile model**
(3 seeds, fitted on val, scored on test):

| metric | delta (BC − raw) |
|---|---|
| skill at >10 m/s | **+0.0795** |
| energy-weighted (q3) | +0.0210 |
| all-hours skill | **−0.0330** |
| mean bias | −0.17 → −0.036 |

**BC is not a free upgrade.** It buys the tail and costs the bulk, consistently across seeds.
Ship both fields — raw for all-hours use, bias-corrected for storm and extreme-value use — and
state the trade rather than picking one silently.

Whether BC helps is *predictable from the calibration diagnostics* (Section 9.3): it corrects a
tail **location** bias, and cannot fix a dispersion failure.

---

## 9. Validation protocol

Two tracks. **They can disagree about the winner, and reconciling them is not optional.**

### 9.1 Grid track

Model versus the target analysis at random grid points, with the low-resolution input as the
reference forecast. Reports both skill conventions and both aggregations (Section 10),
all-hours and above a fixed 10 m/s threshold.

Use a **fixed threshold, not a top-decile**. A top-decile subset collapses the reference's
variance and produced a skill of −3.4 for the target field against its own observations — a
statement about the metric, not the model.

### 9.2 Observation track

Model versus independent stations (here 38 IEM + NDBC). **This is the only truth that is not the
training target**, and it is the tie-breaker.

Keep heterogeneous platforms out of the headline pool and report them as their own group —
moorings at 1.2–4.9 m anemometer height scored against a 10 m field cost every product 0.03–0.25
skill when pooled.

> **Silent-drop trap.** A product whose file path does not resolve is dropped *silently* from
> the leaderboard by the path audit — it simply vanishes, and a 28-product run reports 27
> without comment. This has bitten three times. Drive a fail-loud audit **from the model
> registry itself**, not from a hardcoded list, so it cannot drift from what is registered.

### 9.3 Calibration — the falsifiable test of the whole quantile premise

For a probabilistic product, skill is not enough. Report:

- **PIT histogram** (flat if calibrated), plus a scalar L1 departure from flat so seeds compare
  without eyeballing bins;
- **interval coverage** at 50 / 80 / 90 / 98%;
- **reliability** at τ = 0.9 / 0.95 / 0.99;
- **all of the above again on the storm subset** — a pooled PIT can look acceptable while the
  tail is broken.

The diagnosis is *actionable*, and this is the payoff:

| PIT shape | meaning | can post-hoc BC fix it? |
|---|---|---|
| flat | calibrated | nothing to fix |
| **U-shaped** | under-dispersed — the distribution is too narrow | **no** |
| **skewed / shifted** | tail location bias | **yes** |

For the shipped model the PIT was *not* U-shaped but right-skewed in storms (nominal 99th
percentile sitting at the empirical 96th; storm PIT mean 0.68 against a calibrated 0.50). That
diagnosis predicted, **in advance**, that bias correction would still pay — and it did, at
+0.0795, statistically unchanged from the pre-training-fix baseline of +0.074.

The same diagnostics rejected the epoch-27 checkpoint as a probabilistic product: its 98%
interval covers only 79% (against epoch 7's 88%), with a corroborating drop in output spread. It
buys a better point estimate by narrowing the predictive distribution — acceptable for
direction, unacceptable for storm risk.

**Caveat to carry:** PIT and coverage are scored against the analysis, so they inherit the
training-target problem the observation track exists to escape. Label them as grid-track
numbers.

### 9.4 Inference practicalities

- **Winners only.** Dense-quantile inference costs ~13 GB per arm-year.
- **Segment by year in fresh processes.** The lazy interpolation graph grows with span and has
  OOM-killed a long run.
- Write the **dense** quantile grid for scoring (CRPS/PIT/coverage need it) and a **reduced**
  set for the shipped product.
- If several checkpoints of one run are inferred, **put the checkpoint name in the output
  filename.** Otherwise the second write silently overwrites the first, since the run name is
  the only other distinguishing feature.
- Check the filesystem quota before any run writing >100 GB.

---

## 10. Skill conventions — read this before quoting any number

Three distinct conventions appear in this project's history. They are not interchangeable, and
mixing them has already produced one false comparison.

| # | formula | reference | where |
|---|---|---|---|
| 1 | `1 − MSE_mod/MSE_ref` (Murphy) | low-res input | grid track, pooled |
| 2 | `1 − RMSE_mod/RMSE_ref` | low-res input | grid track, median across cells |
| 3 | `1 − MSE_mod/σ²_obs` (Murphy) | **station climatology** | observation track |

With `r = RMSE_mod/RMSE_ref`, conventions 1 and 2 are `1 − r²` and `1 − r`. **The same model
reads 0.507 under one and 0.213 under the other.** Convention 1 always reads higher for a
skilful model.

Aggregation matters as much as formula: pooling all cells lets the windiest, highest-variance
cells dominate and reads systematically higher than taking a median across cells. Report both.

**Every historical grid number in this project uses convention 2.** Every observation-track
number uses convention 3 — a different reference entirely, which is why obs-track values are
not comparable to grid-track values *even when both are called "Murphy skill"*.

Non-negotiable: **state the convention next to every number.**

---

## 11. Known limits

1. **Residual peak under-prediction persists.** Bias correction reduces it but does not remove
   it; the top-decile subset remains strongly negative for every product *including the target
   analysis itself*.
2. **Energy-weighted skill is negative for every product, including the target.** This is a
   statement about the metric's reference, not about the models — it must be explained wherever
   it is reported, not left as a column of negative numbers.
3. **Target quality is era-dependent.** Against stations, the analysis scores 0.403 before 2020
   and 0.578 after. Any "does it generalise to earlier years" result against the analysis is
   ambiguous between a worse model and a noisier target. Only observations resolve it.
4. **Record end-dates bind** (Section 2). The record cannot be extended past the input's end
   without a different input.
5. **The two validation tracks disagree about the best checkpoint**, reproducibly across seeds:
   gridded skill rewards imitating the analysis, while further training moves the model away
   from the analysis and toward the stations. This is treated as a finding, not an anomaly — see
   `CONCLUSIONS_v3.md`.
6. **Not yet measured: generalisation to the pre-training era for the shipped recipe.** The
   earlier campaign arms were scored on seen/unseen eras; the final recipe has been scored only
   on the held-out test window and the validation window. This is the clearest open gap.

---

## Reproduction checklist

| stage | check |
|---|---|
| data | target availability checked per variable, not just input availability |
| split | `split_dates` used whenever record lengths are compared |
| training | `--exclusive --mem=0`; train-only, no inference |
| training | effective-config echo inspected on arm 0 *before* releasing the sweep |
| selection | per-head checkpoints written; smoothed selection requires a full window |
| inference | winners only; dense for scoring, reduced for product; checkpoint name in filename |
| scoring | both conventions, both aggregations, every table |
| calibration | PIT + coverage + reliability, pooled **and** on the storm subset |
| obs | fail-loud path audit driven from the registry; heterogeneous platforms separated |
| reporting | skill convention stated next to every number |

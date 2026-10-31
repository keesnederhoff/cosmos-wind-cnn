"""Per-station extreme-value plots: observed vs every model flavour.

One figure per station. The x-axis is return period (log). The y-axis is
hourly-mean wind speed (log) in m/s, with mph on the right and the Beaufort classes
alongside. Each source is drawn as:
- its declustered daily-max peaks (lambda = 3/yr, 48-h separation, as s3) at Weibull
  plotting positions T_m = (k + 1) / (m * rate), as markers;
- its return-level curve for the fit FIT_MAIN, as a line. The observations also get a
  90 % bootstrap band.

Fits:
- "regional" (default plot): POT at REGIONAL_LAMBDA = 2/yr. The GPD shape is ONE pooled value
  per source, the L-moment shape of the mean-normalised excesses of all 36 station series of
  that source (rainlib.extremes.pooled_gpd_shape). Threshold and scale stay at-site
  (sigma = mean excess * (1 - xi)). The band is a two-level bootstrap: stations are resampled
  for the shape and excesses for the scale, so it includes the shape uncertainty.
  Why: the at-site L-moment shape at lambda = 3 is unstable at a handful of stations.
  SFO has xi = -0.43, from tied whole-knot ASOS peaks at the threshold. CCR has -0.91, which puts
  RP100 below the record maximum. APC has +0.53, which gives RP100 = 29 m/s against a 20 m/s maximum.
  Across all stations the tail fit is otherwise equal (2026-10-08 diagnostic: median tail
  RMSE 0.42 vs 0.53 m/s, median bias -0.08 vs -0.28 m/s, for peaks with T >= 3 yr).
- "pipeline": windx_common.pot_fit exactly as s3 full mode (lambda = 3, at-site shape, same seeds).
  Its parameters and CIs are asserted equal to results/station_rl.csv. The observed curve
  from it is drawn dashed, for reference.

Outputs:
- figures/station_eva/station_eva_<station>.png
- figures/station_eva_table.csv: per station x source x fit, threshold, n_peaks,
  GPD shape and scale, RP10/50/100 with 90 % CI, n_years, method.

Run order: after s3 (needs cache/obs_hourly.nc, cache/stations_hourly_<product>.nc,
results/station_table.csv, results/station_rl.csv).
"""

# === CONFIGURATION ===
ONLY_STATIONS: "list[str] | None" = None  # e.g. ["SFO"] for the example; None = every station with obs status ok
FLAVOURS = ["era5", "rtma", "cnn_raw", "cnn_ba"]   # model sources overlaid on the obs (cnn_bc dropped 2026-10-08: clamped tail)
SOURCE_COLORS = {"obs": "#000000", "era5": "#7f7f7f", "rtma": "#1f77b4", "cnn_raw": "#ff7f0e",
                 "cnn_bc": "#9467bd", "cnn_ba": "#2ca02c"}   # as s4, except cnn_bc (black there) -> purple
FIT_MAIN = "regional"                     # "regional" or "pipeline": the fit drawn solid for every source
REGIONAL_LAMBDA = 2.0                     # peaks/yr of the regional fit (pipeline uses w.POT_LAMBDA = 3)
SHOW_PIPELINE_OBS = True                  # also draw the s3 obs fit, dashed
TABLE_RPS = [10, 50, 100]
T_MIN, T_MAX, N_T = 0.3, 200.0, 80        # x-axis range (yr); curves start just above T = 1/rate
PARAM_TOL = 1e-5                          # relative tolerance of the station_rl.csv cross-check
MS_TO_MPH = 2.2369363
# WMO Beaufort lower limits (m/s, 10-min mean at 10 m). Our values are hourly means, which run slightly lower.
BEAUFORT = [(5, "fresh breeze", 8.0), (6, "strong breeze", 10.8), (7, "near gale", 13.9), (8, "gale", 17.2),
            (9, "strong gale", 20.8), (10, "storm", 24.5), (11, "violent storm", 28.5), (12, "hurricane force", 32.7)]
Y_TICKS = [6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36]
MPH_TICKS = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]

# === IMPORTS ===
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import evt as rx

import windx_common as w
import s3_fit_return_levels as s3f   # side-effect-free import: cache loaders
from figstyle import apply_style, save_figure

OUT_DIR = w.FIG_DIR / "station_eva"
TABLE_PATH = w.FIG_DIR / "station_eva_table.csv"
INVENTORY = w.MET_ROOT / "reference" / "station_inventory.csv"
LABELS = {"obs": "Observed", **w.PRODUCT_LABEL}
SOURCES = ["obs"] + FLAVOURS


def curve_T(rate: float) -> np.ndarray:
    """Return-period grid from just above 1/rate (where the GPD level is defined) to T_MAX, incl. table RPs."""
    return np.unique(np.concatenate([[1.0], np.asarray(TABLE_RPS, float),
                                     np.logspace(np.log10(1.05 / rate), np.log10(T_MAX), N_T)]))


# === SERIES (same masks as s3 full mode) ===

def _masked(dmax, days, wy_lo: int, wy_hi: int):
    valid = w.valid_water_years(days, dmax)
    years_ok = sorted(y for y, v in valid.items() if wy_lo <= y <= wy_hi and bool(v))
    keep = np.isin(w.water_year(days), years_ok)
    return pd.Series(np.where(keep, dmax, np.nan), index=days), years_ok


def obs_series(obs_cache, station_index_obs: int):
    """Obs daily-max series, valid WYs <= OBS_MAX_WY (s3f.compute_obs_full_results)."""
    _sid, _net, obs_times, obs_ws = obs_cache
    dmax, days, _nvalid = w.daily_max(obs_ws[station_index_obs, :], obs_times)
    return _masked(dmax, days, -10**9, w.OBS_MAX_WY)


def model_series(prod, j: int, product: str):
    """Model daily-max series at the station cell over FULL_WY (s3f._fit_block_core)."""
    _sid, times, ws, _iy, _ix, _dist = prod
    dmax, days, _nvalid = w.daily_max(ws[j, :], times)
    return _masked(dmax, days, *w.FULL_WY[product])


def peaks_desc(series: pd.Series, n_years: int) -> np.ndarray:
    """The k+1 largest declustered peaks at lambda = 3, descending (windx_common.pot_fit)."""
    k = int(round(w.POT_LAMBDA * n_years))
    pk = rx.decluster_peaks(series[np.isfinite(series.to_numpy(float))], w.SEP_HOURS, max_peaks=k + 1)
    return np.sort(pk.to_numpy(dtype=float))[::-1]


# === FITS ===

def fit_pipeline(series: pd.Series, n_years: int, seed: list[int]) -> dict:
    """windx_common.pot_fit + curve + band on an identical rng stream.

    pot_bootstrap_quantiles draws its resamples before it touches T, so a fresh
    generator with the s3 seed gives the very same draws: the band at T = 1/10/100
    equals the station_rl.csv CI exactly.
    """
    try:
        res = w.pot_fit(series, n_years, np.random.default_rng(seed), return_peaks=True)
    except ValueError as e:   # cnn_bc clamp can zero the L-scale (see s3f._safe_pot_fit)
        return {"status": "fit_error", "n_years": n_years, "error": str(e)}
    if res["status"] != "ok":
        return res
    T = curve_T(res["rate"])
    res["T"] = T
    res["curve"] = np.atleast_1d(rx.gpd_quantile(res["threshold"], res["sigma"], res["xi"], res["rate"], T))
    boot = rx.pot_bootstrap_quantiles(res["excess"], res["threshold"], res["rate"], T, mode=w.GPD_MODE,
                                      rng=np.random.default_rng(seed), n_boot=w.N_BOOT)
    res["band_lo"] = np.nanpercentile(boot, 100.0 * w.CI_ALPHA / 2.0, axis=0)
    res["band_hi"] = np.nanpercentile(boot, 100.0 * (1.0 - w.CI_ALPHA / 2.0), axis=0)
    return res


def pooled_shape(excesses: list[np.ndarray], rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Pooled L-moment GPD shape of mean-normalised excesses, plus n_boot station-resampled draws."""
    xi = rx.pooled_gpd_shape(excesses)
    n = len(excesses)
    draws = np.empty(w.N_BOOT)
    for b in range(w.N_BOOT):
        pick = rng.integers(0, n, size=n)
        draws[b] = rx.pooled_gpd_shape([excesses[i] for i in pick])
    return xi, draws


def fit_regional(pk: np.ndarray, n_years: int, xi0: float, xi_draws: np.ndarray, seed: list[int]) -> dict:
    """At-site threshold + scale, pooled shape; two-level bootstrap band (shape draws x excess resamples)."""
    if n_years < w.MIN_YEARS:
        return {"status": "dropped_<6yr", "n_years": n_years}
    u, excess, rate, k = rx.pot_sample(pk, n_years, REGIONAL_LAMBDA)
    sigma, xi = rx.gpd_fit_variant(excess, "fixed", xi0)
    T = curve_T(rate)
    rng = np.random.default_rng(seed)
    l1 = rng.choice(excess, size=(w.N_BOOT, excess.size), replace=True).mean(axis=1)
    q = rx.gpd_quantile_rows(u, l1 * (1.0 - xi_draws), xi_draws, rate, T)
    q = np.where(np.isfinite(q) & (q > 0.0), q, np.nan)
    return {"status": "ok", "threshold": u, "sigma": sigma, "xi": xi, "rate": rate, "n_peaks": k,
            "n_years": n_years, "T": T,
            "curve": np.atleast_1d(rx.gpd_quantile(u, sigma, xi, rate, T)),
            "band_lo": np.nanpercentile(q, 100.0 * w.CI_ALPHA / 2.0, axis=0),
            "band_hi": np.nanpercentile(q, 100.0 * (1.0 - w.CI_ALPHA / 2.0), axis=0)}


def check_against_station_rl(sid: str, source: str, res: dict, station_rl: pd.DataFrame) -> None:
    """The pipeline refit must reproduce s3's full-mode row (parameters, RLs and CIs)."""
    rows = station_rl[(station_rl["station_id"] == sid) & (station_rl["mode"] == "full")
                      & (station_rl["product"] == source)
                      & (station_rl["source"] == ("obs" if source == "obs" else "model"))]
    if rows.empty or rows.iloc[0]["status"] != "ok":
        return
    ref = rows.iloc[0]
    for key in ["threshold", "sigma", "xi", "rl_1", "rl_10", "rl_100", "lo_10", "hi_10", "lo_100", "hi_100"]:
        a, b = float(res[key]), float(ref[key])
        if not np.isclose(a, b, rtol=PARAM_TOL, atol=1e-6):
            raise AssertionError(f"{sid}/{source}: {key} {a:.6g} != station_rl.csv {b:.6g}")
    if int(res["n_peaks"]) != int(ref["n_peaks"]):
        raise AssertionError(f"{sid}/{source}: n_peaks {res['n_peaks']} != {ref['n_peaks']}")


def rp_note(n_years: float) -> str:
    """Record-length support per the EVA convention (values are still printed, as in s3)."""
    if n_years < 20:
        return "RP50/100 extrapolated well beyond record (<20 WY)"
    if n_years <= 30:
        return "RP100 extrapolated (20-30 WY)"
    return ""


# === PLOT ===

def _add_beaufort_and_mph(ax, ylo: float, yhi: float) -> None:
    """Alternating Beaufort bands in the axes, Beaufort labels and mph on two right-hand axes."""
    edges = [b[2] for b in BEAUFORT]
    centres, labels = [], []
    for i, (num, name, lo) in enumerate(BEAUFORT):
        hi = edges[i + 1] if i + 1 < len(edges) else 60.0
        if hi <= ylo or lo >= yhi:
            continue
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#eef1f5", lw=0, zorder=0)
        if ylo < lo < yhi:
            ax.axhline(lo, color="0.75", lw=0.5, zorder=0)
        c = np.sqrt(max(lo, ylo) * min(hi, yhi))   # geometric centre on a log axis
        centres.append(c)
        labels.append(f"Bft {num}  {name}")

    mph = ax.secondary_yaxis("right", functions=(lambda v: v * MS_TO_MPH, lambda v: v / MS_TO_MPH))
    mph.set_yticks([t for t in MPH_TICKS if ylo * MS_TO_MPH <= t <= yhi * MS_TO_MPH])
    mph.yaxis.set_major_formatter(mticker.ScalarFormatter())
    mph.yaxis.set_minor_locator(mticker.NullLocator())
    mph.set_ylabel("mph")

    bft = ax.secondary_yaxis(1.075)
    bft.set_yticks(centres)
    bft.set_yticklabels(labels, fontsize=9, color="0.35")
    bft.yaxis.set_minor_locator(mticker.NullLocator())
    bft.tick_params(length=0)
    bft.spines["right"].set_visible(False)


def plot_station(sid: str, title: str, fits: dict, peaks: dict, xi_pool: dict) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(13.0, 8.0), constrained_layout=True)
    handles = []
    ok_main = {}
    for source in SOURCES:
        if source not in fits:
            continue
        res = fits[source][FIT_MAIN]
        color = SOURCE_COLORS[source]
        is_obs = source == "obs"
        pk = peaks[source]
        if pk is not None and pk.size > 1:
            k = pk.size - 1                        # plotted peaks = the lambda = 3 sample (k largest)
            T_pp = (k + 1.0) / (np.arange(1, k + 1) * (k / fits[source]["n_years"]))
            # no marker edge: ASOS whole-knot reporting gives runs of tied peaks that edges turn into tubes
            ax.scatter(T_pp, pk[:k], s=26 if is_obs else 12, color=color, edgecolors="none",
                       alpha=0.9 if is_obs else 0.5, zorder=4 if is_obs else 3)
        if res.get("status") != "ok":
            h, = ax.plot([], [], color=color, lw=2.0, label=f"{LABELS[source]}: no fit ({res.get('status')})")
            handles.append(h)
            continue
        ok_main[source] = res
        if is_obs:
            ax.fill_between(res["T"], res["band_lo"], res["band_hi"], color="0.72", alpha=0.45, lw=0, zorder=1)
        shape_tag = " pooled" if FIT_MAIN == "regional" else ""
        label = (f"{LABELS[source]} ({res['n_years']:.0f} WY): u={res['threshold']:.1f}, n={res['n_peaks']}, "
                 f"ξ={res['xi']:+.2f}{shape_tag}, σ={res['sigma']:.2f}")
        h, = ax.plot(res["T"], res["curve"], color=color, lw=2.6 if is_obs else 1.8,
                     zorder=5 if is_obs else 4, label=label)
        handles.append(h)

    if SHOW_PIPELINE_OBS and FIT_MAIN != "pipeline" and fits["obs"]["pipeline"].get("status") == "ok":
        p = fits["obs"]["pipeline"]
        h, = ax.plot(p["T"], p["curve"], color="black", lw=1.4, ls="--", zorder=5,
                     label=f"Observed, s3 fit (λ=3, at-site ξ={p['xi']:+.2f}): RP100 {p['rl_100']:.1f} m s$^{{-1}}$")
        handles.append(h)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    for t in (1, 10, 100):
        ax.axvline(t, color="0.6", lw=0.6, ls=":", zorder=0)

    vals = [np.nanmin(pk[:-1]) for pk in peaks.values() if pk is not None and pk.size > 1]
    tops = [np.nanmax(pk) for pk in peaks.values() if pk is not None and pk.size > 1]
    tops += [np.nanmax(r["curve"]) for r in ok_main.values()]
    tops += [np.nanmax(fits["obs"][FIT_MAIN]["band_hi"])] if "obs" in ok_main else []
    ylo, yhi = 0.93 * min(vals), 1.06 * max(tops)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks([t for t in Y_TICKS if ylo <= t <= yhi])
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    _add_beaufort_and_mph(ax, ylo, yhi)

    ax.set_xlabel("Return period (yr)")
    ax.set_ylabel("Hourly-mean wind speed (m s$^{-1}$)")
    ax.set_title(title, loc="left", fontweight="bold")

    band = plt.Rectangle((0, 0), 1, 1, color="0.72", alpha=0.45, lw=0, label="Observed 90 % bootstrap band")
    if FIT_MAIN == "regional":
        head = (f"POT/GPD λ={REGIONAL_LAMBDA:g}/yr, one pooled shape per source (36 stations), 48-h declustering; "
                f"band includes shape uncertainty; markers: λ=3 peaks at Weibull positions")
    else:
        head = "POT/GPD λ=3/yr (s3), 48-h declustering; markers: peaks at Weibull positions"
    fig.legend(handles=handles + [band], loc="outside lower center", ncol=2, fontsize=9,
               frameon=False, title=head, title_fontsize=9)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = save_figure(fig, f"station_eva_{sid}", outdir=OUT_DIR, purpose="report")
    plt.close(fig)
    return Path(written[0])


# === MAIN ===

def _row(sid, name, network, source, fit_name, res, wy, xi_pool_val=None) -> dict:
    row = {"station_id": sid, "name": name, "network": network, "source": source, "label": LABELS[source],
           "fit": fit_name, "mode": "full", "wy_first": wy[0], "wy_last": wy[1],
           "n_years": res.get("n_years"), "status": res.get("status")}
    if res.get("status") == "ok":
        row.update({"threshold": res["threshold"], "n_peaks": res["n_peaks"], "rate_per_yr": res["rate"],
                    "gpd_shape_xi": res["xi"], "gpd_scale_sigma": res["sigma"]})
        for t in TABLE_RPS:
            m = int(np.flatnonzero(res["T"] == t)[0])
            row.update({f"rl_{t}": res["curve"][m], f"lo_{t}": res["band_lo"][m], f"hi_{t}": res["band_hi"][m]})
    if fit_name == "regional":
        method = f"POT-GPD lam{REGIONAL_LAMBDA:g} sep48h dailymax, pooled L-moment shape per source (36 stn), at-site u/sigma"
    else:
        method = "POT-GPD-Lmom lam3 sep48h dailymax (s3)"
    row.update({"method": method, "plotting_position": "Weibull (k+1)/(m*rate), lam3 peaks",
                "rp_note": rp_note(float(res.get("n_years") or 0))})
    return row


def main() -> None:
    station_table = pd.read_csv(w.RESULT_DIR / "station_table.csv", dtype={"station_id": str})
    station_rl = pd.read_csv(w.RESULT_DIR / "station_rl.csv", dtype={"station_id": str})
    inventory = pd.read_csv(INVENTORY, dtype={"station_id": str})
    names = {str(s).upper(): n for s, n in zip(inventory["station_id"], inventory["name"])}

    obs_cache = s3f.load_obs_cache()
    sid_to_i_obs = {s: i for i, s in enumerate(obs_cache[0])}
    prods = {p: s3f.load_stations_hourly(p) for p in FLAVOURS}
    sid_to_j = {p: {s: j for j, s in enumerate(prods[p][0])} for p in FLAVOURS}

    # pass 1: series + lambda=3 peaks for every ok station and source (the pooled shape needs all 36)
    stations = [(i, st) for i, st in enumerate(station_table.itertuples()) if st.status == "ok"]
    data: dict = {}
    for i, st in stations:
        sid = st.station_id
        data[sid] = {}
        series, years = obs_series(obs_cache, sid_to_i_obs[sid])
        data[sid]["obs"] = (series, years, [w.RNG_SEED, 99, 0, i])
        for p in FLAVOURS:
            j = sid_to_j[p].get(sid)
            if j is None:
                continue
            iy, ix = int(prods[p][3][j]), int(prods[p][4][j])
            series, years = model_series(prods[p], j, p)
            data[sid][p] = (series, years, [w.RNG_SEED, w.ALL_PRODUCTS.index(p), 0, iy, ix])
    peaks = {sid: {src: (peaks_desc(s, len(y)) if len(y) >= w.MIN_YEARS else None)
                   for src, (s, y, _seed) in d.items()} for sid, d in data.items()}

    xi_pool, xi_draws = {}, {}
    for si, src in enumerate(SOURCES):
        exc = []
        for sid, d in data.items():
            pk = peaks[sid].get(src)
            if pk is not None:
                exc.append(rx.pot_sample(pk, len(d[src][1]), REGIONAL_LAMBDA).excess)
        xi_pool[src], xi_draws[src] = pooled_shape(exc, np.random.default_rng([w.RNG_SEED, 600, si]))
        print(f"  pooled shape {src:8s}: xi = {xi_pool[src]:+.3f}  "
              f"(90 % {np.percentile(xi_draws[src], 5):+.3f} .. {np.percentile(xi_draws[src], 95):+.3f}, {len(exc)} stations)")

    # pass 2: fits, check, plot
    rows, written = [], []
    for i, st in stations:
        sid = st.station_id
        if ONLY_STATIONS is not None and sid not in ONLY_STATIONS:
            continue
        name = names.get(sid.upper(), "")
        fits = {}
        for src, (series, years, seed) in data[sid].items():
            n = len(years)
            pipe = fit_pipeline(series, n, seed)
            if pipe.get("status") == "ok":
                check_against_station_rl(sid, src, pipe, station_rl)
            reg = (fit_regional(peaks[sid][src], n, xi_pool[src], xi_draws[src], seed + [2])
                   if peaks[sid][src] is not None else {"status": "dropped_<6yr", "n_years": n})
            fits[src] = {"pipeline": pipe, "regional": reg, "n_years": n}
            wy = (years[0], years[-1]) if years else (np.nan, np.nan)
            rows.append(_row(sid, name, st.network, src, "regional", reg, wy))
            rows.append(_row(sid, name, st.network, src, "pipeline", pipe, wy))
        title = f"{sid}" + (f" — {name}" if name else "") + f" ({st.network})"
        written.append(plot_station(sid, title, fits, peaks[sid], xi_pool))
        print(f"  {sid}: {written[-1].name}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_PATH, index=False)
    print(f"wrote {len(written)} figures to {OUT_DIR}")
    print(f"wrote {TABLE_PATH} ({len(table)} rows); pipeline fits match station_rl.csv")


if __name__ == "__main__":
    main()

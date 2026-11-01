"""Stage 4: figures and skill tables for the wind return-level comparison.

Task 9 (maps): one PNG per return period (1, 10, 100) x window (full, common),
1x4 panels in MAP_PRODUCTS order (era5, rtma, cnn_raw, cnn_ba). Panel positions are
fixed -- a product whose results/rl_map_<product>_<window>.nc does not exist
yet gets a blank panel with "not yet available" instead of reflowing the grid.

Task 10 (scatter + skill):
  - Station scatter (1x3, one per RP) for mode in (full, matched), with a skill
    text box per product and results/skill_station_rl.csv.
  (The v3 BC-tail diagnostic was removed for v4: CNN-BC is rejected. The v3 figure is archived.)

Run order: after s3_fit_return_levels.py. Resumable in the trivial sense that
it only reads what exists; nothing here is slow enough to need checkpointing.
"""

# === CONFIGURATION ===
PRODUCT_COLORS = {"era5": "#7f7f7f", "rtma": "#1f77b4", "cnn_raw": "#ff7f0e", "cnn_ba": "#2ca02c"}
NETWORK_MARKERS = {"IEM": "o", "NDBC": "s"}
MAP_EXTENT = (316250.0, 723750.0, 4013250.0, 4323250.0)   # xmin, xmax, ymin, ymax (m, UTM10)
# RP maps zoom on the stations (station x 510-621 km, y 4132-4271 km, plus ~12 km margin)
ZOOM_EXTENT = (498000.0, 634000.0, 4108000.0, 4283000.0)
ZOOM_SCALEBAR_M = 20000.0
MAP_CMAP = "turbo"
MAP_VLIM = (5.0, 30.0)               # m/s, one fixed colour range for every RP map (comparable across figures)
MAP_LEVEL_STEP = 1.0                 # m/s, discrete colour bins on the RP maps
MIN_SKILL_N = 5

# Extra CNN-quantile scatter (cnn_q89/cnn_q95). Kept separate from
# PRODUCT_COLORS/PRODUCTS so the original 4-product maps and scatters never
# reflow; this is a standalone 1x3 figure of its own.
QUANT_SCATTER_PRODUCTS = ["rtma", "cnn_raw", "cnn_q89", "cnn_q95"]
QUANT_COLORS = {"rtma": "#1f77b4", "cnn_raw": "#ff7f0e", "cnn_q89": "#d62728", "cnn_q95": "#9467bd"}

# === IMPORTS ===
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import netCDF4
import numpy as np
import pandas as pd

import windx_common as w
import s3_fit_return_levels as s3f   # side-effect-free import: reuse its CF-time decoder / mask filler
from figstyle import apply_style, km_axes, add_scalebar, add_north_arrow, save_figure
from skill_utils import compute_skill_scores

# cnn_raw is labelled "P50" in the quantile figure's legend/text box, to make
# clear it is one predictive-distribution quantile among the three plotted.
QUANT_LABELS = {**w.PRODUCT_LABEL, "cnn_raw": "CNN-RTMA P50"}


# === SMALL HELPERS ===

def _filled(x) -> np.ndarray:
    return s3f._filled(x)


def cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges from 1-D centre coordinates on a regular grid."""
    centers = np.asarray(centers, dtype=float)
    step = float(np.median(np.diff(centers)))
    return np.concatenate([centers - step / 2.0, [centers[-1] + step / 2.0]])


def _read_map_rp(product: str, window: str, rp: int):
    """(x, y, rl_<rp> array) for a product/window map, or None if it doesn't exist yet."""
    path = w.RESULT_DIR / f"rl_map_{product}_{window}.nc"
    if not path.exists():
        return None
    with netCDF4.Dataset(path) as ds:
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        arr = _filled(ds.variables[f"rl_{rp}"][:, :])
    return x, y, arr


def _draw_coastline(ax, coast: list) -> None:
    for poly in coast:
        ax.plot(poly[:, 0], poly[:, 1], color="0.35", lw=0.4, zorder=3)


def _finish_map_panel(ax, letter: str, title: str, window: str, extent=MAP_EXTENT) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    km_axes(ax)
    ax.text(0.02, 0.97, f"({letter}) {title}", transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", zorder=11, color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75))
    if window == "common":
        ax.text(0.98, 0.97, f"WY{w.COMMON_WY[0]}–{w.COMMON_WY[1]}", transform=ax.transAxes, ha="right", va="top",
                fontsize=10, zorder=11)


# === TASK 9: MAPS ===

def _map_color_limits(rp: int, window: str, station_rl: pd.DataFrame) -> tuple[float, float]:
    """2nd-98th pct of pooled rtma/cnn_raw/cnn_ba map values inside ZOOM_EXTENT, plus
    obs RLs, rounded outward to whole m/s."""
    pooled = []
    for product in ("rtma", "cnn_raw", "cnn_ba"):
        got = _read_map_rp(product, window, rp)
        if got is None:
            continue
        x, y, arr = got
        inside = ((x[None, :] >= ZOOM_EXTENT[0]) & (x[None, :] <= ZOOM_EXTENT[1])
                  & (y[:, None] >= ZOOM_EXTENT[2]) & (y[:, None] <= ZOOM_EXTENT[3]))
        finite = arr[inside & np.isfinite(arr)]
        if finite.size:
            pooled.append(finite)

    obs = station_rl[(station_rl["mode"] == "full") & (station_rl["source"] == "obs")
                      & (station_rl["status"] == "ok")][f"rl_{rp}"].dropna().to_numpy(dtype=float)
    if obs.size:
        pooled.append(obs)

    if not pooled:
        return 0.0, 1.0
    pooled = np.concatenate(pooled)
    lo = np.percentile(pooled, 2.0)
    hi = np.percentile(pooled, 98.0)
    return float(np.floor(lo)), float(np.ceil(hi))


def _obs_points_full(rp: int, station_table: pd.DataFrame, station_rl: pd.DataFrame) -> pd.DataFrame:
    obs = station_rl[(station_rl["mode"] == "full") & (station_rl["source"] == "obs")
                      & (station_rl["status"] == "ok")][["station_id", f"rl_{rp}"]]
    obs = obs.rename(columns={f"rl_{rp}": "rl"})
    merged = obs.merge(station_table[["station_id", "x", "y"]], on="station_id", how="inner")
    return merged.dropna(subset=["rl", "x", "y"])


def plot_rp_maps(rp: int, window: str, station_table: pd.DataFrame, station_rl: pd.DataFrame,
                  coast: list) -> Path:
    apply_style()
    vmin, vmax = MAP_VLIM
    # discrete bins make small differences between products readable
    levels = np.arange(vmin, vmax + MAP_LEVEL_STEP / 2, MAP_LEVEL_STEP)
    cmap = plt.get_cmap(MAP_CMAP)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend="both")
    obs_pts = _obs_points_full(rp, station_table, station_rl)

    fig, axes = plt.subplots(1, 4, figsize=(17, 6.0), sharex=True, sharey=True, constrained_layout=True)
    mappable = None

    for i, product in enumerate(w.MAP_PRODUCTS):
        ax = axes[i]
        letter = chr(ord("a") + i)
        got = _read_map_rp(product, window, rp)
        if got is not None:
            x, y, arr = got
            xe, ye = cell_edges(x), cell_edges(y)
            mesh = ax.pcolormesh(xe, ye, arr, cmap=cmap, norm=norm, rasterized=True, zorder=2)
            mappable = mesh
        else:
            ax.text(0.5, 0.5, "not yet available", transform=ax.transAxes, ha="center", va="center",
                    fontsize=13, color="0.45", style="italic", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.75))

        _draw_coastline(ax, coast)
        if not obs_pts.empty:
            ax.scatter(obs_pts["x"], obs_pts["y"], c=obs_pts["rl"], cmap=cmap, norm=norm,
                       s=60, edgecolors="white", linewidths=1.2, zorder=5)
        _finish_map_panel(ax, letter, w.PRODUCT_LABEL[product], window, extent=ZOOM_EXTENT)
        if i == 0:
            add_scalebar(ax, ZOOM_SCALEBAR_M)
            add_north_arrow(ax)

    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, shrink=0.85, pad=0.012, extend="both",
                          ticks=np.arange(MAP_VLIM[0], MAP_VLIM[1] + 0.1, 5.0))
        cb.set_label(f"RP{rp} wind speed (m s$^{{-1}}$)")

    written = save_figure(fig, f"map_rl{rp}_{window}", outdir=w.FIG_DIR, purpose="report")
    plt.close(fig)
    return written[0]


# === TASK 10 STEP 1-2: SCATTER + SKILL ===

def _slice_rl(df: pd.DataFrame, rp: int) -> pd.DataFrame:
    return df[["station_id", "network", f"rl_{rp}", f"lo_{rp}", f"hi_{rp}"]].rename(
        columns={f"rl_{rp}": "rl", f"lo_{rp}": "lo", f"hi_{rp}": "hi"})


def pair_stations(station_rl: pd.DataFrame, mode: str, rp: int, product: str) -> "pd.DataFrame | None":
    """x = obs RL, y = product RL, both 'ok', paired by station_id.

    full mode: obs is the station's own full-record fit (product == 'obs');
    model is the product's full-window map value at the station cell.
    matched mode: both obs and model come from that product's matched fit, so
    the obs value can differ slightly between products (noted in README).
    """
    if mode == "full":
        obs_rows = station_rl[(station_rl["mode"] == "full") & (station_rl["source"] == "obs")
                               & (station_rl["status"] == "ok")]
    else:
        obs_rows = station_rl[(station_rl["mode"] == "matched") & (station_rl["product"] == product)
                               & (station_rl["source"] == "obs") & (station_rl["status"] == "ok")]
    if obs_rows.empty:
        return None
    obs = _slice_rl(obs_rows, rp).rename(columns={"rl": "obs_rl", "lo": "obs_lo", "hi": "obs_hi"})

    model_rows = station_rl[(station_rl["mode"] == mode) & (station_rl["product"] == product)
                             & (station_rl["source"] == "model") & (station_rl["status"] == "ok")]
    if model_rows.empty:
        return None
    model = _slice_rl(model_rows, rp).rename(columns={"rl": "model_rl", "lo": "model_lo", "hi": "model_hi"})

    merged = obs.merge(model[["station_id", "model_rl", "model_lo", "model_hi"]], on="station_id", how="inner")
    merged = merged.dropna(subset=["obs_rl", "model_rl"])
    return merged if not merged.empty else None


def _legend_handles(products_present: list, colors: dict, labels: dict) -> list:
    handles = [Line2D([], [], marker="o", linestyle="None", color=colors[p],
                       markerfacecolor=colors[p], markeredgecolor=colors[p],
                       markersize=8, label=labels[p])
               for p in products_present]
    for net, marker in NETWORK_MARKERS.items():
        handles.append(Line2D([], [], marker=marker, linestyle="None", color="black",
                               markerfacecolor="none", markeredgecolor="black",
                               markersize=8, label=net))
    return handles


def _skill_row_from_df(mode: str, rp: int, product: str, df: pd.DataFrame) -> "dict | None":
    """Skill dict (one skill_station_rl.csv row) from an already-paired obs/model
    frame (see pair_stations). Shared by the 4-PRODUCT scatter, the extra-product
    skill-only pass, and the quantile scatter, so the numbers are identical
    wherever the same (mode, rp, product) triple is computed."""
    sk = compute_skill_scores(df["obs_rl"].to_numpy(dtype=float),
                               df["model_rl"].to_numpy(dtype=float),
                               min_n=MIN_SKILL_N)
    if not sk:
        return None
    median_ratio = float(np.median(df["model_rl"] / df["obs_rl"]))
    overlap = (df["obs_lo"] <= df["model_hi"]) & (df["model_lo"] <= df["obs_hi"])
    ci_overlap_frac = float(overlap.mean())
    return {
        "mode": mode, "rp": rp, "product": product, "n": sk["n"], "bias": sk["bias"],
        "rmse": sk["rmse"], "mae": sk["mae"], "sci": sk["sci"], "skill": sk["skill"],
        "r2": sk["r2"], "median_ratio": median_ratio, "ci_overlap_frac": ci_overlap_frac,
    }


def compute_skill_rows_only(mode: str, station_rl: pd.DataFrame, products: list) -> list:
    """Skill rows for `products` at every RP, mode, with no plotting. Used to add
    EXTRA_PRODUCTS (cnn_q89, cnn_q95) to skill_station_rl.csv without touching
    the original 4-PRODUCT scatter figures."""
    rows = []
    for rp in w.RETURN_PERIODS:
        for product in products:
            df = pair_stations(station_rl, mode, rp, product)
            if df is None:
                continue
            row = _skill_row_from_df(mode, rp, product, df)
            if row is not None:
                rows.append(row)
    return rows


def _scatter_and_skill_figure(mode: str, station_rl: pd.DataFrame, products: list, colors: dict,
                               labels: dict, fig_name: str) -> list:
    """1x3 (RP1/10/100) scatter of obs vs model RL for `products`, with a skill
    text box per product (upper left) and a figure-level legend below. Shared by
    plot_scatter_and_skill (the original 4-PRODUCT figures) and
    plot_quantile_scatter (the rtma/cnn_raw/cnn_q89/cnn_q95 figure)."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0), constrained_layout=True)
    skill_rows = []

    for j, rp in enumerate(w.RETURN_PERIODS):
        ax = axes[j]
        letter = chr(ord("a") + j)
        text_lines = []
        pooled_vals = []
        products_present = []

        for product in products:
            df = pair_stations(station_rl, mode, rp, product)
            if df is None:
                continue
            products_present.append(product)
            color = colors[product]

            for net, marker in NETWORK_MARKERS.items():
                sub = df[df["network"] == net]
                if sub.empty:
                    continue
                ax.scatter(sub["obs_rl"], sub["model_rl"], marker=marker, s=42,
                           facecolors=color, edgecolors="black", linewidths=0.3, zorder=4)

            xerr = np.vstack([(df["obs_rl"] - df["obs_lo"]).clip(lower=0),
                               (df["obs_hi"] - df["obs_rl"]).clip(lower=0)])
            yerr = np.vstack([(df["model_rl"] - df["model_lo"]).clip(lower=0),
                               (df["model_hi"] - df["model_rl"]).clip(lower=0)])
            ax.errorbar(df["obs_rl"], df["model_rl"], xerr=xerr, yerr=yerr, fmt="none",
                        ecolor=color, alpha=0.4, elinewidth=1.0, zorder=3)

            pooled_vals.append(df[["obs_rl", "obs_lo", "obs_hi", "model_rl", "model_lo", "model_hi"]]
                                .to_numpy(dtype=float).ravel())

            row = _skill_row_from_df(mode, rp, product, df)
            if row is None:
                continue
            skill_rows.append(row)
            text_lines.append(f"{labels[product]}: bias {row['bias']:+.1f}, "
                               f"RMSE {row['rmse']:.1f}, skill {row['skill']:.2f}")

        if pooled_vals:
            allv = np.concatenate(pooled_vals)
            allv = allv[np.isfinite(allv)]
            lo, hi = float(np.min(allv)), float(np.max(allv))
        else:
            lo, hi = 0.0, 30.0
        pad = (hi - lo) * 0.08 if hi > lo else 1.0
        lims = (max(lo - pad, 0.0), hi + pad)

        ax.plot(lims, lims, color="black", lw=1.0, zorder=1)
        ax.plot(lims, [v * 1.2 for v in lims], color="black", lw=0.8, ls="--", alpha=0.7, zorder=1)
        ax.plot(lims, [v * 0.8 for v in lims], color="black", lw=0.8, ls="--", alpha=0.7, zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Observed RP{rp} wind speed (m s$^{{-1}}$)")
        ax.set_ylabel(f"Model RP{rp} wind speed (m s$^{{-1}}$)")
        ax.text(0.02, 0.97, f"({letter}) RP{rp}", transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold", zorder=11)
        if text_lines:
            # every product sits below 1:1, so the upper-left triangle is empty
            ax.text(0.02, 0.90, "\n".join(text_lines), transform=ax.transAxes, ha="left", va="top",
                    fontsize=8.5, family="monospace", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.6", alpha=0.85))
        if j == 0:
            # one figure-level legend below the panels keeps all three panels free for data
            fig.legend(handles=_legend_handles(products_present, colors, labels), loc="outside lower center",
                       ncol=len(products_present) + len(NETWORK_MARKERS), fontsize=10, frameon=False)

    save_figure(fig, fig_name, outdir=w.FIG_DIR, purpose="report")
    plt.close(fig)
    return skill_rows


def plot_scatter_and_skill(mode: str, station_rl: pd.DataFrame) -> list:
    return _scatter_and_skill_figure(mode, station_rl, w.MAP_PRODUCTS, PRODUCT_COLORS, w.PRODUCT_LABEL,
                                      fig_name=f"scatter_rl_{mode}")


def plot_quantile_scatter(station_rl: pd.DataFrame) -> list:
    """Matched-mode-only scatter of rtma / cnn_raw (P50) / cnn_q89 / cnn_q95
    against their own matched obs values. A standalone figure: its skill numbers
    duplicate rows already written by plot_scatter_and_skill/compute_skill_rows_only,
    so its return value is for logging only and is not added to skill_station_rl.csv."""
    return _scatter_and_skill_figure("matched", station_rl, QUANT_SCATTER_PRODUCTS, QUANT_COLORS, QUANT_LABELS,
                                      fig_name="scatter_rl_matched_quantiles")


# === MAIN ===

def main() -> None:
    w.FIG_DIR.mkdir(parents=True, exist_ok=True)
    station_table = pd.read_csv(w.RESULT_DIR / "station_table.csv")
    station_rl = pd.read_csv(w.RESULT_DIR / "station_rl.csv")
    coast = w.read_ldb(w.LDB_FILE)

    print("=== Task 9: maps ===")
    for window in ("full", "common"):
        for rp in w.RETURN_PERIODS:
            path = plot_rp_maps(rp, window, station_table, station_rl, coast)
            print(f"  wrote {path}")

    print("\n=== Task 10 step 1-2: scatter + skill ===")
    all_skill_rows = []
    for mode in ("full", "matched"):
        rows = plot_scatter_and_skill(mode, station_rl)
        all_skill_rows.extend(rows)
        print(f"  wrote {w.FIG_DIR / f'scatter_rl_{mode}.png'} ({len(rows)} skill rows)")

        others = [p for p in w.ALL_PRODUCTS if p not in w.MAP_PRODUCTS]
        extra_rows = compute_skill_rows_only(mode, station_rl, others)
        all_skill_rows.extend(extra_rows)
        print(f"  {mode}: +{len(extra_rows)} skill rows for {others}")

    skill_df = pd.DataFrame(all_skill_rows, columns=["mode", "rp", "product", "n", "bias", "rmse", "mae",
                                                      "sci", "skill", "r2", "median_ratio", "ci_overlap_frac"])
    skill_path = w.RESULT_DIR / "skill_station_rl.csv"
    skill_df.to_csv(skill_path, index=False)
    print(f"  wrote {skill_path} ({len(skill_df)} rows)")

    print("\n=== Quantile scatter (rtma / cnn_raw P50 / cnn_q89 / cnn_q95, matched) ===")
    quant_rows = plot_quantile_scatter(station_rl)
    print(f"  wrote {w.FIG_DIR / 'scatter_rl_matched_quantiles.png'} ({len(quant_rows)} skill rows, "
          f"not added to skill_station_rl.csv -- duplicates rows above)")

    print("\ns4_figures: done")


if __name__ == "__main__":
    main()

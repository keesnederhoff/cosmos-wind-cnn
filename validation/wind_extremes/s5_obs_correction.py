"""Stage 5: observation-anchored correction of the CNN-RTMA raw (P50) return-level maps.

Builds a station-level correction factor f = exp(r), r = ln(obs_rl / cnn_rl), from the
36 matched-mode cnn_raw stations in results/station_rl.csv, picks the correction model
("variant") by leave-one-station-out (LOO) cross-validation, and applies the winning
variant's factor field (fit on all 36 stations) to the cnn_raw full- and common-window
return-level maps as the last step of the pipeline.

Reads only from windx_common.py, s4_figures.py, figstyle.py and skill_utils.py -- none
of those files are modified here (another agent is editing windx_common/s1/s3/s4
concurrently).

Run order: after s4_figures.py (reads results/station_rl.csv, results/station_table.csv
and the rl_map_cnn_raw_*.nc maps that s3 produces).
"""

# === CONFIGURATION ===
from pathlib import Path

GRID_TOL_M = 0.5                       # x/y coordinate match tolerance, map vs. water-mask grid

VARIANTS = ["V0_none", "V1_global", "V2_landwater", "V3_idw", "V4_lw_kernel", "V5_lw_gp"]
SIMPLICITY = {v: i for i, v in enumerate(VARIANTS)}   # tie-break order for the winner

MIN_CLASS_N = 3                        # land/water median falls back to the global median below this
IDW_POWER = 2.0
IDW_EPS_M = 1.0                        # distance floor, guards against division by zero
KERNEL_L_M = 20000.0                   # Gaussian-kernel decay length, V4
GP_CONST = (0.01, (1e-4, 1.0))
GP_RBF_LENGTH = (20000.0, (2000.0, 200000.0))
GP_NOISE = (0.01, (1e-5, 1.0))
GP_N_RESTARTS = 3
GP_RANDOM_STATE = 0

MIN_SKILL_N = 5                        # compute_skill_scores min_n
WINNER_MARGIN = 0.02                   # within 2% of the lowest mean LOO RMSE -> tie, pick simplest

ZOOM_EXTENT = (498000.0, 634000.0, 4108000.0, 4283000.0)   # xmin, xmax, ymin, ymax (m, UTM10)
ZOOM_SCALEBAR_M = 20000.0
MAP_CMAP = "turbo"
MAP_VLIM = (5.0, 30.0)               # m/s, one fixed colour range for every RP map (comparable across figures)
MAP_LEVEL_STEP = 1.0                   # m/s, discrete bins on the corrected RL maps

# === IMPORTS ===
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import netCDF4
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

import windx_common as w

# Defined here, not in the configuration block above: it needs w.MET_ROOT (an env-var root) from the import.
STATIC_WATER_FILE = w.MET_ROOT / "modeled_data" / "rtma" / "fm_netcdf" / "RTMA_SFbay_2p5km_surface_height_static_UTM10.nc"

from s4_figures import cell_edges, _draw_coastline, NETWORK_MARKERS
from figstyle import apply_style, km_axes, add_scalebar, add_north_arrow, save_figure
from skill_utils import compute_skill_scores

RETURN_PERIODS = w.RETURN_PERIODS


# === DATA LOADING ===

def build_station_frame(station_table: pd.DataFrame, station_rl: pd.DataFrame) -> pd.DataFrame:
    """One row per station: x, y, network, class, obs/cnn RL and r = ln(obs/cnn) per RP."""
    sub = station_rl[(station_rl["mode"] == "matched") & (station_rl["product"] == "cnn_raw")
                      & (station_rl["status"] == "ok")]

    obs_cols = {f"rl_{rp}": f"obs_rl_{rp}" for rp in RETURN_PERIODS}
    obs = sub[sub["source"] == "obs"][["station_id", "network"] + list(obs_cols)].rename(columns=obs_cols)

    mod_cols = {f"rl_{rp}": f"cnn_rl_{rp}" for rp in RETURN_PERIODS}
    mod = sub[sub["source"] == "model"][["station_id"] + list(mod_cols)].rename(columns=mod_cols)

    df = obs.merge(mod, on="station_id", how="inner")
    df = df.merge(station_table[["station_id", "x", "y"]], on="station_id", how="left")
    df["class"] = np.where(df["network"] == "NDBC", "water", "land")

    for rp in RETURN_PERIODS:
        df[f"r_{rp}"] = np.log(df[f"obs_rl_{rp}"].to_numpy(dtype=float)
                                / df[f"cnn_rl_{rp}"].to_numpy(dtype=float))

    df = df.reset_index(drop=True)
    assert df["x"].notna().all() and df["y"].notna().all(), "station missing x/y in station_table.csv"
    return df


def load_water_mask(x_map: np.ndarray, y_map: np.ndarray) -> np.ndarray:
    """Boolean (ny, nx) water mask, verified on the same grid as the CNN maps."""
    with netCDF4.Dataset(STATIC_WATER_FILE) as ds:
        x_wm = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y_wm = np.asarray(ds.variables["y"][:], dtype=np.float64)
        sh = np.ma.filled(ds.variables["surface_height"][:, :], np.nan).astype(np.float64)

    dx = float(np.max(np.abs(x_wm - x_map)))
    dy = float(np.max(np.abs(y_wm - y_map)))
    if dx > GRID_TOL_M or dy > GRID_TOL_M:
        raise ValueError(f"water-mask grid does not match the CNN map grid (max dx={dx} m, dy={dy} m)")
    return sh == 0.0


def read_rl_map(path: Path) -> dict:
    """x, y and every rl_*/lo_*/hi_* array (float64) from an rl_map_*.nc file."""
    out = {}
    with netCDF4.Dataset(path) as ds:
        out["x"] = np.asarray(ds.variables["x"][:], dtype=np.float64)
        out["y"] = np.asarray(ds.variables["y"][:], dtype=np.float64)
        for rp in RETURN_PERIODS:
            for prefix in ("rl", "lo", "hi"):
                key = f"{prefix}_{rp}"
                out[key] = np.ma.filled(ds.variables[key][:, :], np.nan).astype(np.float64)
        out["attrs"] = {name: ds.getncattr(name) for name in ds.ncattrs()}
    return out


# === CORRECTION MODELS ===

def class_medians(train_r: np.ndarray, train_class: np.ndarray, min_n: int = MIN_CLASS_N) -> tuple[dict, float]:
    """Per-class median r, falling back to the global median below min_n training stations."""
    global_med = float(np.median(train_r))
    meds = {}
    for cls in ("land", "water"):
        mask = train_class == cls
        meds[cls] = float(np.median(train_r[mask])) if mask.sum() >= min_n else global_med
    return meds, global_med


def idw_predict(tr_x, tr_y, tr_r, qx, qy, power=IDW_POWER, eps=IDW_EPS_M) -> np.ndarray:
    d = np.sqrt((qx[:, None] - tr_x[None, :]) ** 2 + (qy[:, None] - tr_y[None, :]) ** 2)
    d = np.maximum(d, eps)
    wts = 1.0 / d ** power
    return (wts * tr_r[None, :]).sum(axis=1) / wts.sum(axis=1)


def kernel_residual_predict(tr_x, tr_y, e, qx, qy, L=KERNEL_L_M) -> np.ndarray:
    d = np.sqrt((qx[:, None] - tr_x[None, :]) ** 2 + (qy[:, None] - tr_y[None, :]) ** 2)
    wts = np.exp(-(d / L) ** 2)
    return (wts * e[None, :]).sum(axis=1) / (1.0 + wts.sum(axis=1))


def fit_gp(tr_x, tr_y, e) -> GaussianProcessRegressor:
    kernel = (ConstantKernel(GP_CONST[0], GP_CONST[1]) * RBF(GP_RBF_LENGTH[0], GP_RBF_LENGTH[1])
              + WhiteKernel(GP_NOISE[0], GP_NOISE[1]))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                   n_restarts_optimizer=GP_N_RESTARTS, random_state=GP_RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(np.column_stack([tr_x, tr_y]), e)
    return gp


def fit_variant(variant: str, train: pd.DataFrame, rp: int) -> dict:
    """Fit `variant` on `train` stations for return period `rp`; returns a predict-ready dict."""
    tr_x = train["x"].to_numpy(dtype=float)
    tr_y = train["y"].to_numpy(dtype=float)
    tr_r = train[f"r_{rp}"].to_numpy(dtype=float)
    tr_class = train["class"].to_numpy()
    fit = {"variant": variant}

    if variant == "V0_none":
        pass
    elif variant == "V1_global":
        fit["global_med"] = float(np.median(tr_r))
    elif variant == "V2_landwater":
        fit["meds"], _ = class_medians(tr_r, tr_class)
    elif variant == "V3_idw":
        fit.update(tr_x=tr_x, tr_y=tr_y, tr_r=tr_r)
    elif variant == "V4_lw_kernel":
        meds, _ = class_medians(tr_r, tr_class)
        e = tr_r - np.array([meds[c] for c in tr_class])
        fit.update(meds=meds, tr_x=tr_x, tr_y=tr_y, e=e)
    elif variant == "V5_lw_gp":
        meds, _ = class_medians(tr_r, tr_class)
        e = tr_r - np.array([meds[c] for c in tr_class])
        gp = fit_gp(tr_x, tr_y, e)
        fit.update(meds=meds, gp=gp, kernel=gp.kernel_)
    else:
        raise ValueError(variant)
    return fit


def apply_fit(fit: dict, qx: np.ndarray, qy: np.ndarray, qclass: np.ndarray) -> np.ndarray:
    """Predicted r at query points/cells; qx/qy/qclass share any common shape."""
    variant = fit["variant"]
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    qclass = np.asarray(qclass)
    shape = qx.shape

    if variant == "V0_none":
        return np.zeros(shape, dtype=float)
    if variant == "V1_global":
        return np.full(shape, fit["global_med"], dtype=float)
    if variant == "V2_landwater":
        meds = fit["meds"]
        return np.where(qclass == "water", meds["water"], meds["land"]).astype(float)
    if variant == "V3_idw":
        return idw_predict(fit["tr_x"], fit["tr_y"], fit["tr_r"], qx.ravel(), qy.ravel()).reshape(shape)
    if variant == "V4_lw_kernel":
        meds = fit["meds"]
        resid = kernel_residual_predict(fit["tr_x"], fit["tr_y"], fit["e"], qx.ravel(), qy.ravel()).reshape(shape)
        base = np.where(qclass == "water", meds["water"], meds["land"]).astype(float)
        return base + resid
    if variant == "V5_lw_gp":
        meds = fit["meds"]
        resid = fit["gp"].predict(np.column_stack([qx.ravel(), qy.ravel()])).reshape(shape)
        base = np.where(qclass == "water", meds["water"], meds["land"]).astype(float)
        return base + resid
    raise ValueError(variant)


# === LOO CROSS-VALIDATION ===

def run_loo(station_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """LOO-predicted corrected RL per variant x rp x station; returns (score table, raw predictions)."""
    records = []
    pred_store: dict = {v: {} for v in VARIANTS}
    n = len(station_df)

    for variant in VARIANTS:
        for rp in RETURN_PERIODS:
            obs_arr = np.empty(n)
            corr_arr = np.empty(n)
            cls_arr = np.empty(n, dtype=object)
            net_arr = np.empty(n, dtype=object)
            sid_arr = np.empty(n, dtype=object)

            for i in range(n):
                train = station_df.drop(index=i)
                row = station_df.iloc[i]
                fit = fit_variant(variant, train, rp)
                r_pred = apply_fit(fit, np.array([row["x"]]), np.array([row["y"]]), np.array([row["class"]]))[0]
                obs_arr[i] = row[f"obs_rl_{rp}"]
                corr_arr[i] = row[f"cnn_rl_{rp}"] * np.exp(r_pred)
                cls_arr[i] = row["class"]
                net_arr[i] = row["network"]
                sid_arr[i] = row["station_id"]

            pred_store[variant][rp] = dict(obs=obs_arr, corrected=corr_arr, network=net_arr,
                                            class_=cls_arr, station_id=sid_arr)

            sk = compute_skill_scores(obs_arr, corr_arr, min_n=MIN_SKILL_N)
            land = cls_arr == "land"
            water = cls_arr == "water"
            sk_land = compute_skill_scores(obs_arr[land], corr_arr[land], min_n=MIN_SKILL_N)
            sk_water = compute_skill_scores(obs_arr[water], corr_arr[water], min_n=MIN_SKILL_N)

            records.append({
                "variant": variant, "rp": rp, "n": int(n),
                "bias": sk.get("bias", np.nan), "rmse": sk.get("rmse", np.nan),
                "mae": sk.get("mae", np.nan), "sci": sk.get("sci", np.nan),
                "skill": sk.get("skill", np.nan), "r2": sk.get("r2", np.nan),
                "median_ratio": float(np.median(corr_arr / obs_arr)),
                "rmse_land": sk_land.get("rmse", np.nan), "n_land": int(land.sum()),
                "rmse_water": sk_water.get("rmse", np.nan), "n_water": int(water.sum()),
            })

    loo_df = pd.DataFrame(records, columns=["variant", "rp", "n", "bias", "rmse", "mae", "sci", "skill",
                                             "r2", "median_ratio", "rmse_land", "n_land", "rmse_water", "n_water"])
    return loo_df, pred_store


def pick_winner(loo_df: pd.DataFrame) -> tuple[str, pd.Series]:
    mean_rmse = loo_df.groupby("variant")["rmse"].mean().reindex(VARIANTS)
    min_rmse = float(mean_rmse.min())
    candidates = [v for v in VARIANTS if mean_rmse[v] <= min_rmse * (1.0 + WINNER_MARGIN)]
    winner = sorted(candidates, key=lambda v: SIMPLICITY[v])[0]
    return winner, mean_rmse


# === FINAL FIT (all 36 stations) AND FACTOR TABLE ===

def build_factor_table(station_df: pd.DataFrame) -> pd.DataFrame:
    """Per variant x rp: exp(median r) global/land/water, and the GP kernel string for V5."""
    rows = []
    for rp in RETURN_PERIODS:
        tr_r = station_df[f"r_{rp}"].to_numpy(dtype=float)
        tr_class = station_df["class"].to_numpy()
        meds, global_med = class_medians(tr_r, tr_class)
        gp_fit = fit_variant("V5_lw_gp", station_df, rp)   # always fit, so every rp has a kernel string
        for variant in VARIANTS:
            if variant == "V0_none":
                f_global = f_land = f_water = 1.0
            else:
                f_global, f_land, f_water = (float(np.exp(global_med)), float(np.exp(meds["land"])),
                                              float(np.exp(meds["water"])))
            rows.append({
                "variant": variant, "rp": rp,
                "factor_global": f_global, "factor_land": f_land, "factor_water": f_water,
                "gp_kernel": str(gp_fit["kernel"]) if variant == "V5_lw_gp" else "",
            })
    return pd.DataFrame(rows)


# === MAP FIGURES: HELPERS ===

def _zoom_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((x[None, :] >= ZOOM_EXTENT[0]) & (x[None, :] <= ZOOM_EXTENT[1])
            & (y[:, None] >= ZOOM_EXTENT[2]) & (y[:, None] <= ZOOM_EXTENT[3]))


def _obs_full_points(rp: int, station_table: pd.DataFrame, station_rl: pd.DataFrame) -> pd.DataFrame:
    obs = station_rl[(station_rl["mode"] == "full") & (station_rl["source"] == "obs")
                      & (station_rl["status"] == "ok")][["station_id", f"rl_{rp}"]].rename(
        columns={f"rl_{rp}": "rl"})
    merged = obs.merge(station_table[["station_id", "x", "y"]], on="station_id", how="inner")
    return merged.dropna(subset=["rl", "x", "y"])


def _round_limits(values: np.ndarray) -> tuple[float, float]:
    lo = float(np.percentile(values, 2.0))
    hi = float(np.percentile(values, 98.0))
    return float(np.floor(lo)), float(np.ceil(hi))


# === FIGURE 1: LOO SCATTER ===

def plot_loo_figure(pred_store: dict, loo_df: pd.DataFrame, winner: str) -> Path:
    apply_style()
    display = ["V0_none", winner]
    if winner != "V1_global":
        display.append("V1_global")

    colors = {"V0_none": "0.6", "V1_global": "#ff7f0e"}
    colors[winner] = "black"
    labels = {"V0_none": "V0 (raw CNN)", "V1_global": "V1 (global)"}
    labels[winner] = f"{winner} (winner)"

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.3), constrained_layout=True)

    for j, rp in enumerate(RETURN_PERIODS):
        ax = axes[j]
        letter = chr(ord("a") + j)
        text_lines = []
        pooled = []

        for variant in display:
            d = pred_store[variant][rp]
            color = colors[variant]
            for net, marker in NETWORK_MARKERS.items():
                m = d["network"] == net
                if m.any():
                    ax.scatter(d["obs"][m], d["corrected"][m], marker=marker, s=42,
                               facecolors=color, edgecolors="black", linewidths=0.3, zorder=4)
            pooled.append(d["obs"])
            pooled.append(d["corrected"])

            row = loo_df[(loo_df["variant"] == variant) & (loo_df["rp"] == rp)].iloc[0]
            text_lines.append(f"{labels[variant]}: bias {row['bias']:+.1f}, "
                               f"RMSE {row['rmse']:.1f}, skill {row['skill']:.2f}")

        allv = np.concatenate(pooled)
        allv = allv[np.isfinite(allv)]
        lo, hi = float(np.min(allv)), float(np.max(allv))
        pad = (hi - lo) * 0.08 if hi > lo else 1.0
        lims = (max(lo - pad, 0.0), hi + pad)

        ax.plot(lims, lims, color="black", lw=1.0, zorder=1)
        ax.plot(lims, [v * 1.2 for v in lims], color="black", lw=0.8, ls="--", alpha=0.7, zorder=1)
        ax.plot(lims, [v * 0.8 for v in lims], color="black", lw=0.8, ls="--", alpha=0.7, zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Observed RP{rp} wind speed (m s$^{{-1}}$)")
        ax.set_ylabel(f"LOO-predicted RP{rp} wind speed (m s$^{{-1}}$)")
        ax.text(0.02, 0.97, f"({letter}) RP{rp}", transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold", zorder=11)
        ax.text(0.02, 0.90, "\n".join(text_lines), transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, family="monospace", zorder=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.6", alpha=0.85))

    handles = [Line2D([], [], marker="o", linestyle="None", color=colors[v],
                       markerfacecolor=colors[v], markeredgecolor="black", markersize=8, label=labels[v])
               for v in display]
    for net, marker in NETWORK_MARKERS.items():
        handles.append(Line2D([], [], marker=marker, linestyle="None", color="black",
                               markerfacecolor="none", markeredgecolor="black", markersize=8, label=net))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=10, frameon=False)

    written = save_figure(fig, "obs_correction_loo", outdir=w.FIG_DIR, purpose="report")
    plt.close(fig)
    return written[0]


# === FIGURE 2: CORRECTED + FACTOR MAPS ===

def plot_map_figure(winner: str, x: np.ndarray, y: np.ndarray, corrected_full: dict,
                     factor_fields: dict, station_df: pd.DataFrame, station_table: pd.DataFrame,
                     station_rl: pd.DataFrame, coast: list) -> Path:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), constrained_layout=True)
    xe, ye = cell_edges(x), cell_edges(y)
    cmap = plt.get_cmap(MAP_CMAP)
    inside = _zoom_mask(x, y)

    # --- top row: corrected RL, own colour limits per RP ---
    for j, rp in enumerate(RETURN_PERIODS):
        ax = axes[0, j]
        letter = chr(ord("a") + j)
        arr = corrected_full[f"rl_{rp}"]
        obs_full = _obs_full_points(rp, station_table, station_rl)

        pooled = arr[inside & np.isfinite(arr)]
        if not obs_full.empty:
            pooled = np.concatenate([pooled, obs_full["rl"].to_numpy(dtype=float)])
        vmin, vmax = MAP_VLIM
        levels = np.arange(vmin, vmax + MAP_LEVEL_STEP / 2.0, MAP_LEVEL_STEP)
        norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend="both")

        mesh = ax.pcolormesh(xe, ye, arr, cmap=cmap, norm=norm, rasterized=True, zorder=2)
        _draw_coastline(ax, coast)
        if not obs_full.empty:
            ax.scatter(obs_full["x"], obs_full["y"], c=obs_full["rl"], cmap=cmap, norm=norm,
                       s=60, edgecolors="white", linewidths=1.2, zorder=5)

        ax.set_xlim(ZOOM_EXTENT[0], ZOOM_EXTENT[1])
        ax.set_ylim(ZOOM_EXTENT[2], ZOOM_EXTENT[3])
        ax.set_aspect("equal")
        km_axes(ax)
        ax.text(0.02, 0.97, f"({letter}) RP{rp} corrected ({winner})", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold", zorder=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75))
        cb = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02, extend="both",
                          ticks=np.arange(MAP_VLIM[0], MAP_VLIM[1] + 0.1, 5.0))
        cb.set_label(f"RP{rp} wind speed (m s$^{{-1}}$)")
        if j == 0:
            add_scalebar(ax, ZOOM_SCALEBAR_M)
            add_north_arrow(ax)

    # --- bottom row: factor field, one shared diverging norm centred on 1 ---
    zoom_vals = np.concatenate([factor_fields[rp][inside & np.isfinite(factor_fields[rp])]
                                 for rp in RETURN_PERIODS])
    lo_p, hi_p = float(np.percentile(zoom_vals, 2.0)), float(np.percentile(zoom_vals, 98.0))
    dev = max(abs(1.0 - lo_p), abs(hi_p - 1.0))
    norm2 = mpl.colors.Normalize(vmin=1.0 - dev, vmax=1.0 + dev)

    mesh2 = None
    for j, rp in enumerate(RETURN_PERIODS):
        ax = axes[1, j]
        letter = chr(ord("a") + 3 + j)
        arr = factor_fields[rp]
        mesh2 = ax.pcolormesh(xe, ye, arr, cmap="RdBu_r", norm=norm2, rasterized=True, zorder=2)
        _draw_coastline(ax, coast)
        station_ratio = np.exp(station_df[f"r_{rp}"].to_numpy(dtype=float))
        ax.scatter(station_df["x"], station_df["y"], c=station_ratio, cmap="RdBu_r", norm=norm2,
                   s=60, edgecolors="white", linewidths=1.2, zorder=5)

        ax.set_xlim(ZOOM_EXTENT[0], ZOOM_EXTENT[1])
        ax.set_ylim(ZOOM_EXTENT[2], ZOOM_EXTENT[3])
        ax.set_aspect("equal")
        km_axes(ax)
        ax.text(0.02, 0.97, f"({letter}) RP{rp} factor", transform=ax.transAxes, ha="left", va="top",
                fontsize=11, fontweight="bold", zorder=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75))

    cb2 = fig.colorbar(mesh2, ax=axes[1, :], shrink=0.85, pad=0.02)
    cb2.set_label("Correction factor, obs / CNN-raw (station points: station's own ratio)")

    written = save_figure(fig, "map_obscorr", outdir=w.FIG_DIR, purpose="report")
    plt.close(fig)
    return written[0]


# === NETCDF OUTPUT ===

def write_corrected_map(path: Path, x: np.ndarray, y: np.ndarray, corrected: dict, factor_fields: dict,
                         water_mask: np.ndarray, winner: str, n_stations: int, source_attrs: dict) -> None:
    ny, nx = len(y), len(x)
    if path.exists():
        path.unlink()
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        yv = ds.createVariable("y", "f8", ("y",)); yv[:] = y
        xv = ds.createVariable("x", "f8", ("x",)); xv[:] = x

        for rp in RETURN_PERIODS:
            for prefix in ("rl", "lo", "hi"):
                key = f"{prefix}_{rp}"
                v = ds.createVariable(key, "f4", ("y", "x"), zlib=True, complevel=4,
                                       fill_value=np.float32(np.nan))
                v[:, :] = corrected[key].astype(np.float32)
            fv = ds.createVariable(f"factor_{rp}", "f4", ("y", "x"), zlib=True, complevel=4,
                                    fill_value=np.float32(np.nan))
            fv[:, :] = factor_fields[rp].astype(np.float32)

        wv = ds.createVariable("water", "u1", ("y", "x"), zlib=True, complevel=4)
        wv[:, :] = water_mask.astype(np.uint8)
        wv.legend = "1=water (ocean/Bay, NDBC-class), 0=land (IEM-class)"

        ds.method = source_attrs.get("method", "")
        ds.wy_range = source_attrs.get("wy_range", "")
        ds.product = "cnn_obscorr"
        ds.window = source_attrs.get("window", "")
        ds.variant = winner
        ds.n_stations = n_stations
        ds.note = "lo/hi exclude correction uncertainty"


# === MAIN ===

def main() -> None:
    w.FIG_DIR.mkdir(parents=True, exist_ok=True)
    station_table = pd.read_csv(w.RESULT_DIR / "station_table.csv")
    station_rl = pd.read_csv(w.RESULT_DIR / "station_rl.csv")
    coast = w.read_ldb(w.LDB_FILE)

    station_df = build_station_frame(station_table, station_rl)
    n_stations = len(station_df)
    print(f"stations: {n_stations} (expected 36); land={int((station_df['class'] == 'land').sum())}, "
          f"water={int((station_df['class'] == 'water').sum())}")
    assert n_stations == 36, f"expected 36 matched cnn_raw stations, got {n_stations}"

    print("\n=== LOO cross-validation ===")
    loo_df, pred_store = run_loo(station_df)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(loo_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    loo_path = w.RESULT_DIR / "obs_correction_loo.csv"
    loo_df.to_csv(loo_path, index=False)
    print(f"wrote {loo_path}")

    winner, mean_rmse = pick_winner(loo_df)
    print("\nmean LOO RMSE by variant (averaged over RP1/10/100):")
    min_rmse = float(mean_rmse.min())
    for v in VARIANTS:
        pct = 100.0 * (mean_rmse[v] / min_rmse - 1.0)
        flag = " <- WINNER" if v == winner else ""
        print(f"  {v:16s} {mean_rmse[v]:.4f}  (+{pct:5.2f}% vs. lowest){flag}")
    print(f"winner: {winner}, margin {100.0 * (mean_rmse[winner] / min_rmse - 1.0):.2f}% "
          f"above the lowest mean RMSE ({min_rmse:.4f} m/s)")

    print("\n=== Final fit (all 36 stations) and factor table ===")
    factor_table = build_factor_table(station_df)
    factor_path = w.RESULT_DIR / "obs_correction_factors.csv"
    factor_table.to_csv(factor_path, index=False)
    print(f"wrote {factor_path}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(factor_table.drop(columns=["gp_kernel"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    gp_rows = factor_table[factor_table["variant"] == "V5_lw_gp"][["rp", "gp_kernel"]]
    print("\nGP kernel per RP (V5_lw_gp, fit on all 36 stations):")
    for _, r in gp_rows.iterrows():
        print(f"  RP{int(r['rp'])}: {r['gp_kernel']}")

    print(f"\n=== Applying winner ({winner}) to the cnn_raw maps ===")
    full_map = read_rl_map(w.RESULT_DIR / "rl_map_cnn_raw_full.nc")
    common_map = read_rl_map(w.RESULT_DIR / "rl_map_cnn_raw_common.nc")
    x, y = full_map["x"], full_map["y"]
    assert np.allclose(x, common_map["x"]) and np.allclose(y, common_map["y"]), "full/common grids differ"

    water_mask = load_water_mask(x, y)
    class_grid = np.where(water_mask, "water", "land")
    X, Y = np.meshgrid(x, y)   # shape (ny, nx), matches rl_* array layout

    factor_fields = {}
    for rp in RETURN_PERIODS:
        fit = fit_variant(winner, station_df, rp)
        r_field = apply_fit(fit, X, Y, class_grid)
        factor_fields[rp] = np.exp(r_field)

    inside = _zoom_mask(x, y)
    print("domain and zoom median factor, winner, split land/water:")
    for rp in RETURN_PERIODS:
        f = factor_fields[rp]
        dom_land = float(np.median(f[water_mask == False]))
        dom_water = float(np.median(f[water_mask]))
        zoom_land = float(np.median(f[inside & (water_mask == False)]))
        zoom_water = float(np.median(f[inside & water_mask]))
        print(f"  RP{rp}: domain land={dom_land:.4f} water={dom_water:.4f} | "
              f"zoom land={zoom_land:.4f} water={zoom_water:.4f}")

    outputs = {}
    for window, src in (("full", full_map), ("common", common_map)):
        corrected = {}
        for rp in RETURN_PERIODS:
            f = factor_fields[rp]
            corrected[f"rl_{rp}"] = src[f"rl_{rp}"] * f
            corrected[f"lo_{rp}"] = src[f"lo_{rp}"] * f
            corrected[f"hi_{rp}"] = src[f"hi_{rp}"] * f
        outputs[window] = corrected
        out_path = w.RESULT_DIR / f"rl_map_cnn_obscorr_{window}.nc"
        write_corrected_map(out_path, x, y, corrected, factor_fields, water_mask, winner, n_stations, src["attrs"])
        print(f"wrote {out_path}")

    print("\n=== Figures ===")
    fig1 = plot_loo_figure(pred_store, loo_df, winner)
    print(f"wrote {fig1}")
    fig2 = plot_map_figure(winner, x, y, outputs["full"], factor_fields, station_df, station_table,
                            station_rl, coast)
    print(f"wrote {fig2}")

    print("\ns5_obs_correction: done")


if __name__ == "__main__":
    main()

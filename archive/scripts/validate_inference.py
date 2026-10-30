"""
Validation script for cosmos-wind-cnn full-record inference output.

Compares the downscaled model predictions against:
  Path 1 – N random spatial grid points: skill(model vs CONUS404) vs skill(ERA5 vs CONUS404)
  Path 2 – Station point-validation:
              Sources: NDBC sf_bay_winds, Whale's Tale met moorings
              Datasets: ERA5 (input), CONUS404 (truth), Model inference, Observations

Usage:
    python scripts/validate_inference.py [--n-points 50] [--seed 42] [--run-id 3663482]

Outputs (under results/<run_id>/output_evaluation/):
    path1/   skill_map_wind_speed.png, spatial_metrics.csv
    path2/   <station_name>/timeseries_<var>.png, metrics.json
    summary.json, path2_metrics_table.csv, validation.log
"""

import os
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
try:
    from tqdm import tqdm
except ImportError:
    # Minimal no-op fallback so the script still runs without tqdm installed
    def tqdm(iterable=None, *args, **kwargs):  # noqa: E302
        return iterable if iterable is not None else range(0)

# suppress pyproj / cartopy FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Windows OpenMP fix
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── optional: cartopy for geo-aware skill map ─────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


# ─────────────────────────────────────────────────────────────────────────────
# Paths & config
# ─────────────────────────────────────────────────────────────────────────────

from cosmos_wind_cnn.utils.config import get_run_dirs

CASE_STUDY_DIR  = Path(r"d:\Git\cosmos-wind-cnn\case_studies\sf_bay_conus404")
NDBC_DIR        = Path(r"d:\data\NDBC\sf_bay_winds")
WHALES_TALE_DIR = Path(r"m:\emeryville_crescent\01_data\whales_tale")

WHALES_TALE_FILES = [
    WHALES_TALE_DIR / "DMP23MW101met.nc",
    WHALES_TALE_DIR / "DMP23MW201met.nc",
]

CRS_UTM10N = "EPSG:32610"
CRS_WGS84  = "EPSG:4326"

# Known NDBC station positions (lat, lon) — not stored in the .nc files
NDBC_STATION_LATLON = {
    "46026": (37.759, -122.833),   # San Francisco (18NM West)
    "46012": (37.356, -122.881),   # Half Moon Bay (24NM SSW of San Francisco)
    "46013": (38.242, -123.301),   # Bodega Bay
    "46237": (37.786, -122.636),   # San Francisco Bar
    "AAMC1": (37.772, -122.300),   # Alameda, CA (NOT Alcatraz)
    "FTPC1": (37.806, -122.466),   # Fort Point, San Francisco
    "PCOC1": (38.056, -122.039),   # Port Chicago
    "PXSC1": (37.803, -122.397),   # Pier 17, San Francisco Bay (NOT Pillar Point)
    "RTYC1": (37.507, -122.212),   # Redwood City (NOT Richmond)
    "TIBC1": (37.892, -122.447),   # Tiburon Pier
}

# Variable mapping: (obs_var, model_var, era5_var, conus_truth_var, label, unit)
# obs_var=None means derive from u/v
STATION_VAR_MAP = [
    ("u10_ms",           "hr_u",        "lr_u",        "hr_u",        "u_wind",    "m/s"),
    ("v10_ms",           "hr_v",        "lr_v",        "hr_v",        "v_wind",    "m/s"),
    ("wind_speed_ms",    None,          None,          None,          "wind_speed","m/s"),
    ("air_temperature_k","hr_air_temp", "lr_air_temp", "hr_air_temp", "air_temp",  "K"),
]

# Whale's Tale vars (different naming)
WHALES_TALE_VAR_MAP = [
    ("u10_ms",  "hr_u",        "lr_u",        "hr_u",        "u_wind",    "m/s"),
    ("v10_ms",  "hr_v",        "lr_v",        "hr_v",        "v_wind",    "m/s"),
    ("wind_speed", None,       None,          None,          "wind_speed","m/s"),
    ("air_temperature_k","hr_air_temp", "lr_air_temp", "hr_air_temp", "air_temp",  "K"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rmse(a, b):
    diff = a - b
    return float(np.sqrt(np.nanmean(diff ** 2)))

def mae(a, b):
    return float(np.nanmean(np.abs(a - b)))

def bias(a, b):
    return float(np.nanmean(a - b))

def ubrmse(a, b):
    """Unbiased RMSE: RMSE with mean bias removed."""
    _rmse = rmse(a, b)
    _bias = bias(a, b)
    return float(np.sqrt(max(0, _rmse**2 - _bias**2)))

def scatter_index(a, b):
    """Scatter Index (SI): ubRMSE normalised by mean of observations."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    obs_mean = np.nanmean(b[mask])
    if obs_mean == 0:
        return np.nan
    return float(ubrmse(a, b) / abs(obs_mean))

def pearson_r(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])

def skill_score(rmse_model, rmse_ref):
    """SS = 1 - RMSE_model / RMSE_ref  (1=perfect, 0=same as ref, <0=worse)"""
    if rmse_ref == 0:
        return np.nan
    return float(1.0 - rmse_model / rmse_ref)

def murphy_skill(pred, obs):
    """Murphy Skill Score: 1 - MSE / Var(obs).  (1=perfect, 0=climatology, <0=worse)"""
    mask = ~(np.isnan(pred) | np.isnan(obs))
    if mask.sum() < 3:
        return np.nan
    mse = np.nanmean((pred[mask] - obs[mask]) ** 2)
    var_obs = np.nanvar(obs[mask])
    if var_obs == 0:
        return np.nan
    return float(1.0 - mse / var_obs)

def compute_metrics(pred, obs, label):
    pred = np.asarray(pred, dtype=float)
    obs  = np.asarray(obs,  dtype=float)
    n_valid = int((~np.isnan(pred - obs)).sum())
    return {
        "label":       label,
        "n":           n_valid,
        "rmse":        rmse(pred, obs),
        "ubrmse":      ubrmse(pred, obs),
        "mae":         mae(pred, obs),
        "bias":        bias(pred, obs),
        "si":          scatter_index(pred, obs),
        "skill":       murphy_skill(pred, obs),
        "correlation": pearson_r(pred, obs),
        "skill_vs_era5": None,   # filled later
    }


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_transformers():
    if not HAS_PYPROJ:
        raise ImportError("pyproj is required for coordinate transformations. "
                          "Install with: pip install pyproj")
    to_utm   = Transformer.from_crs(CRS_WGS84, CRS_UTM10N, always_xy=True)
    to_wgs84 = Transformer.from_crs(CRS_UTM10N, CRS_WGS84, always_xy=True)
    return to_utm, to_wgs84


def build_grid_tree(ds):
    """Build KDTree over the flattened 2-D (y, x) inference grid."""
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)   # shape (ny, nx)
    pts  = np.column_stack([xx.ravel(), yy.ravel()])  # (ny*nx, 2)
    tree = cKDTree(pts)
    return tree, int(len(ds.x)), int(len(ds.y))


def nearest_grid_cell(tree, nx, x_utm, y_utm):
    """Return (iy, ix, dist_m) for the nearest cell to the given UTM point."""
    dist, flat_idx = tree.query([x_utm, y_utm])
    iy, ix = divmod(int(flat_idx), nx)
    return iy, ix, float(dist)


def in_domain(ds, x_utm, y_utm):
    xmin, xmax = float(ds.x.min()), float(ds.x.max())
    ymin, ymax = float(ds.y.min()), float(ds.y.max())
    return (xmin <= x_utm <= xmax) and (ymin <= y_utm <= ymax)


# ─────────────────────────────────────────────────────────────────────────────
# Wind helpers
# ─────────────────────────────────────────────────────────────────────────────

def uv_to_speed(u, v):
    return np.sqrt(np.asarray(u, float) ** 2 + np.asarray(v, float) ** 2)

def met_wind_to_uv(speed, dir_from_deg):
    """Meteorological convention: direction the wind blows FROM, clockwise from N."""
    rad = np.deg2rad(np.asarray(dir_from_deg, float))
    u = -np.asarray(speed, float) * np.sin(rad)
    v = -np.asarray(speed, float) * np.cos(rad)
    return u, v


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_inference(run_dirs):
    """Load inference output — tries all .nc files in output_inference dir."""
    inf_dir = run_dirs['output_inference']
    # Try full_record first, then glob for any .nc
    fpath = inf_dir / "full_record.nc"
    if not fpath.exists():
        nc_files = sorted(inf_dir.glob("*.nc"))
        if nc_files:
            fpath = nc_files[0]
        else:
            raise FileNotFoundError(f"No inference output found in {inf_dir}")
    logging.info(f"Loading inference: {fpath}")
    return xr.open_dataset(fpath, chunks={"time": 500})


def load_processed(run_dirs):
    """Lazily open and concatenate train/val/test splits, sorted by time."""
    processed_dir = run_dirs['data_processed']
    splits = []
    for name in ("train", "val", "test"):
        p = processed_dir / f"{name}.nc"
        if p.exists():
            splits.append(xr.open_dataset(p, chunks="auto"))
        else:
            logging.warning(f"Processed split not found: {p}")
    if not splits:
        raise FileNotFoundError(f"No processed data found in {processed_dir}")
    ds = xr.concat(splits, dim="time").sortby("time")
    logging.info(f"Processed data: {len(ds.time)} timesteps  "
                 f"({pd.Timestamp(ds.time.values[0]).date()} – "
                 f"{pd.Timestamp(ds.time.values[-1]).date()})")
    return ds


def load_ndbc_station(fpath):
    """
    Load one NDBC station .nc file.
    Time dim is 'datetime'; lat/lon are looked up from NDBC_STATION_LATLON.
    Returns dict or None if station_id not in lookup.
    """
    ds = xr.open_dataset(fpath)

    # Rename datetime → time
    if "datetime" in ds.dims:
        ds = ds.rename({"datetime": "time"})

    # Station ID from filename  e.g. station_46012_2000_2025.nc → "46012"
    station_id = fpath.stem.split("_")[1]

    if station_id not in NDBC_STATION_LATLON:
        logging.warning(f"  No lat/lon entry for NDBC station {station_id}, skipping")
        ds.close()
        return None

    lat, lon = NDBC_STATION_LATLON[station_id]

    # Derive wind speed from u/v if not present
    if "wind_speed_ms" not in ds and "u10_ms" in ds and "v10_ms" in ds:
        ds["wind_speed_ms"] = uv_to_speed(ds["u10_ms"].values, ds["v10_ms"].values)

    return {
        "name":   f"NDBC_{station_id}",
        "source": "NDBC",
        "lat":    lat,
        "lon":    lon,
        "ds":     ds,
        "var_map": STATION_VAR_MAP,
    }


def load_whales_tale_station(fpath):
    """
    Load a Whale's Tale mooring .nc file.
    Lat/lon stored as variables; winds given as speed + direction_from.
    Returns dict.
    """
    ds = xr.open_dataset(fpath)

    lat = float(ds["lat"].mean())
    lon = float(ds["lon"].mean())

    # Convert speed + direction_from → u/v
    if "wind_speed" in ds and "wind_dir_from" in ds:
        u, v = met_wind_to_uv(ds["wind_speed"].values, ds["wind_dir_from"].values)
        ds["u10_ms"]      = xr.DataArray(u, dims=["time"], coords={"time": ds.time})
        ds["v10_ms"]      = xr.DataArray(v, dims=["time"], coords={"time": ds.time})

    return {
        "name":   fpath.stem,
        "source": "WhalesTale",
        "lat":    lat,
        "lon":    lon,
        "ds":     ds,
        "var_map": WHALES_TALE_VAR_MAP,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Temporal alignment
# ─────────────────────────────────────────────────────────────────────────────

def find_overlap(times_a, times_b, tolerance_hours=1):
    """
    Return the set of timestamps in times_a that have a match in times_b
    within ±tolerance_hours.  Returns a DatetimeIndex (possibly empty).
    """
    a = pd.DatetimeIndex(times_a).sort_values()
    b = pd.DatetimeIndex(times_b).sort_values()
    tol = pd.Timedelta(hours=tolerance_hours)

    # Normalize timezone info so intersection/comparison doesn't fail
    if a.tz is not None:
        a = a.tz_localize(None)
    if b.tz is not None:
        b = b.tz_localize(None)

    # Exact intersection first (fast path)
    exact = a.intersection(b)
    if len(exact) > 0:
        return exact

    # Nearest-match within tolerance
    matched = []
    for t in a:
        idx = b.get_indexer([t], method="nearest")[0]
        if idx >= 0 and abs(b[idx] - t) <= tol:
            matched.append(t)
    return pd.DatetimeIndex(matched)


def select_at_times(da, times_target, ref_times, tolerance_hours=1):
    """
    Extract values from DataArray da (with ref_times) at times_target.
    Returns numpy array aligned to times_target (NaN where no match).
    Vectorised: uses get_indexer over the full array at once.
    """
    tol  = pd.Timedelta(hours=tolerance_hours)
    ref  = pd.DatetimeIndex(ref_times)
    tgt  = pd.DatetimeIndex(times_target)

    # Strip timezone info if mixed (both become naive UTC) to avoid comparison errors
    if ref.tz is not None:
        ref = ref.tz_localize(None)
    if tgt.tz is not None:
        tgt = tgt.tz_localize(None)

    # Nearest index for every target timestamp (returns -1 if ref is empty)
    idxs = ref.get_indexer(tgt, method="nearest")

    # Mask out entries that are too far away or have no match
    clipped      = np.clip(idxs, 0, len(ref) - 1)
    matched_ts   = ref[clipped]
    valid        = (np.abs(matched_ts - tgt) <= tol) & (idxs >= 0)

    out          = np.full(len(tgt), np.nan)
    vals         = da.values  # load once
    out[valid]   = vals[clipped[valid]].astype(float)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(times, series_dict, var_label, unit, station_name, out_path):
    """
    series_dict: {label: array}  — obs must be keyed "Observations"
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    style = {
        "Observations": dict(color="black", lw=1.5, zorder=5),
        "ERA5":         dict(color="steelblue", lw=0.9, ls="--", alpha=0.85),
        "CONUS404":     dict(color="forestgreen", lw=0.9, ls="--", alpha=0.85),
        "Model":        dict(color="crimson", lw=1.0, alpha=0.95, zorder=4),
    }
    for lbl, arr in series_dict.items():
        kw = style.get(lbl, dict(lw=0.9, alpha=0.8))
        ax.plot(times, arr, label=lbl, **kw)

    ax.set_title(f"{station_name}  |  {var_label}  [{unit}]", fontsize=10)
    ax.set_ylabel(f"{var_label} [{unit}]")
    ax.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(obs, pred, label, var_label, unit, station_name, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    mask = ~(np.isnan(obs) | np.isnan(pred))
    ax.scatter(obs[mask], pred[mask], s=4, alpha=0.3, color="steelblue")
    lo = min(np.nanmin(obs), np.nanmin(pred))
    hi = max(np.nanmax(obs), np.nanmax(pred))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel(f"Observed {var_label} [{unit}]")
    ax.set_ylabel(f"{label} {var_label} [{unit}]")
    ax.set_title(f"{station_name} – {label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_skill(ds_inf, skill_grid, var_label, out_path, to_wgs84=None):
    """
    skill_grid shape (ny, nx).  If cartopy+pyproj available, plot on a map.
    """
    if HAS_CARTOPY and to_wgs84 is not None:
        xs = ds_inf.x.values
        ys = ds_inf.y.values
        xx, yy = np.meshgrid(xs, ys)
        lons, lats = to_wgs84.transform(xx, yy)

        fig, ax = plt.subplots(1, 1, figsize=(9, 7),
                               subplot_kw={"projection": ccrs.PlateCarree()})
        ax.add_feature(cfeature.LAND,      facecolor="lightgrey")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4)
        sc = ax.pcolormesh(lons, lats, skill_grid,
                           cmap="RdYlGn", vmin=-0.5, vmax=1.0,
                           transform=ccrs.PlateCarree())
        plt.colorbar(sc, ax=ax, label="Skill score vs ERA5", shrink=0.8)
        ax.set_title(f"Model spatial skill – {var_label}\n(1=perfect, 0=same as ERA5)")
        ax.set_extent([lons.min()-0.1, lons.max()+0.1,
                       lats.min()-0.1, lats.max()+0.1], crs=ccrs.PlateCarree())
    else:
        # Plain imshow fallback
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(skill_grid, origin="lower", aspect="auto",
                       cmap="RdYlGn", vmin=-0.5, vmax=1.0)
        plt.colorbar(im, ax=ax, label="Skill score vs ERA5")
        ax.set_title(f"Model spatial skill – {var_label}")
        ax.set_xlabel("x index"); ax.set_ylabel("y index")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_skill_boxplot(records, output_dir):
    """Box-and-whisker of skill scores by dataset for Path 1."""
    df = pd.DataFrame(records)
    if df.empty or "skill_vs_era5" not in df:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    df_model = df[df["label"] == "model_vs_conus"]["skill_vs_era5"].dropna()
    ax.boxplot([df_model.values], labels=["Model vs CONUS404\n(skill vs ERA5 baseline)"])
    ax.axhline(0, color="steelblue", ls="--", lw=0.8, label="ERA5 baseline (SS=0)")
    ax.axhline(1, color="forestgreen", ls="--", lw=0.8, label="Perfect (SS=1)")
    ax.set_ylabel("Skill Score")
    ax.set_title("Path 1 – Wind Speed Skill Score Distribution\n(N random grid points)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "path1_skill_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PATH 1 – Random spatial points
# ─────────────────────────────────────────────────────────────────────────────

def run_path1(inference_ds, processed_ds, n_points, seed, output_dir):
    """
    Compare ERA5 vs CONUS404 vs model at n_points random grid cells.
    Metrics: wind speed RMSE/skill, u/v individually.
    """
    logging.info(f"=== Path 1: {n_points} random spatial points ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    ny, nx = len(inference_ds.y), len(inference_ds.x)
    iys = rng.integers(0, ny, n_points)
    ixs = rng.integers(0, nx, n_points)

    # Align time axes once
    inf_time  = pd.DatetimeIndex(inference_ds.time.values)
    proc_time = pd.DatetimeIndex(processed_ds.time.values)
    common    = inf_time.intersection(proc_time)

    if len(common) < 100:
        logging.warning(f"Path 1: only {len(common)} common timesteps between "
                        f"inference and processed data – skipping Path 1.")
        return []

    inf_idx  = inf_time.get_indexer(common)
    proc_idx = proc_time.get_indexer(common)
    logging.info(f"  Temporal overlap: {len(common)} timesteps  "
                 f"({common[0].date()} – {common[-1].date()})")

    # Pre-load u/v at all points lazily, then compute
    # (We do this point-by-point to avoid huge memory footprint)
    all_metrics  = []
    skill_map_ws = np.full((ny, nx), np.nan)
    skill_map_u  = np.full((ny, nx), np.nan)
    skill_map_v  = np.full((ny, nx), np.nan)

    pbar = tqdm(total=n_points, desc="Path 1 – spatial points",
                unit="pt", dynamic_ncols=True, leave=True)
    running_ss = []

    for pt, (iy, ix) in enumerate(zip(iys, ixs)):
        # -- Model --
        try:
            mod_u = inference_ds["hr_u"].isel(y=int(iy), x=int(ix)).values[inf_idx].astype(float)
            mod_v = inference_ds["hr_v"].isel(y=int(iy), x=int(ix)).values[inf_idx].astype(float)
        except Exception as e:
            logging.debug(f"  pt{pt}: inference isel failed – {e}")
            continue

        # -- HR truth --
        if "hr_u" not in processed_ds or "hr_v" not in processed_ds:
            logging.warning("  processed_ds missing hr_u/v, skipping Path 1")
            break
        tru_u = processed_ds["hr_u"].isel(y=int(iy), x=int(ix)).values[proc_idx].astype(float)
        tru_v = processed_ds["hr_v"].isel(y=int(iy), x=int(ix)).values[proc_idx].astype(float)

        # -- LR (ERA5) --
        if "lr_u" not in processed_ds or "lr_v" not in processed_ds:
            logging.warning("  processed_ds missing lr_u/v, skipping Path 1")
            break
        e5_u = processed_ds["lr_u"].isel(y=int(iy), x=int(ix)).values[proc_idx].astype(float)
        e5_v = processed_ds["lr_v"].isel(y=int(iy), x=int(ix)).values[proc_idx].astype(float)

        # Wind speed
        mod_ws = uv_to_speed(mod_u, mod_v)
        tru_ws = uv_to_speed(tru_u, tru_v)
        e5_ws  = uv_to_speed(e5_u, e5_v)

        m_mod_ws  = compute_metrics(mod_ws, tru_ws, "model_vs_conus")
        m_era5_ws = compute_metrics(e5_ws,  tru_ws, "era5_vs_conus")
        ss_ws = skill_score(m_mod_ws["rmse"], m_era5_ws["rmse"])
        m_mod_ws["skill_vs_era5"]  = ss_ws
        m_era5_ws["skill_vs_era5"] = 0.0

        m_mod_u  = compute_metrics(mod_u, tru_u, "model_u")
        m_era5_u = compute_metrics(e5_u,  tru_u, "era5_u")
        m_mod_u["skill_vs_era5"]  = skill_score(m_mod_u["rmse"], m_era5_u["rmse"])
        m_era5_u["skill_vs_era5"] = 0.0

        m_mod_v  = compute_metrics(mod_v, tru_v, "model_v")
        m_era5_v = compute_metrics(e5_v,  tru_v, "era5_v")
        m_mod_v["skill_vs_era5"]  = skill_score(m_mod_v["rmse"], m_era5_v["rmse"])
        m_era5_v["skill_vs_era5"] = 0.0

        for m in [m_mod_ws, m_era5_ws, m_mod_u, m_era5_u, m_mod_v, m_era5_v]:
            m["iy"] = int(iy); m["ix"] = int(ix); m["pt_idx"] = pt
        all_metrics.extend([m_mod_ws, m_era5_ws, m_mod_u, m_era5_u, m_mod_v, m_era5_v])

        skill_map_ws[iy, ix] = ss_ws
        skill_map_u[iy, ix]  = m_mod_u["skill_vs_era5"]
        skill_map_v[iy, ix]  = m_mod_v["skill_vs_era5"]

        running_ss.append(ss_ws)
        pbar.update(1)
        pbar.set_postfix({"SS_ws": f"{np.nanmean(running_ss):.3f}",
                          "last":  f"{ss_ws:.3f}"})

    pbar.close()

    if not all_metrics:
        logging.warning("Path 1: no metrics computed.")
        return []

    # Save CSV
    df = pd.DataFrame(all_metrics)
    df.to_csv(output_dir / "spatial_metrics.csv", index=False)
    logging.info(f"  Saved {len(df)} metric rows to {output_dir / 'spatial_metrics.csv'}")

    # Skill maps
    to_wgs84 = get_transformers()[1] if HAS_PYPROJ else None
    for grid, vname in [(skill_map_ws, "wind_speed"),
                        (skill_map_u,  "u_wind"),
                        (skill_map_v,  "v_wind")]:
        if not np.all(np.isnan(grid)):
            plot_spatial_skill(inference_ds, grid, vname,
                               output_dir / f"skill_map_{vname}.png",
                               to_wgs84=to_wgs84)

    plot_skill_boxplot(all_metrics, output_dir)

    med_ss = float(np.nanmedian([m["skill_vs_era5"] for m in all_metrics
                                  if m["label"] == "model_vs_conus"]))
    logging.info(f"  Median wind speed skill score vs ERA5: {med_ss:.3f}")
    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# PATH 2 – single station
# ─────────────────────────────────────────────────────────────────────────────

def run_path2_station(station_info, inference_ds, processed_ds,
                      tree, nx, to_utm, output_dir):
    """
    Validate one station against ERA5 / CONUS404 / model.
    Returns list of metric dicts (empty if spatial/temporal overlap fails).
    """
    name   = station_info["name"]
    lat    = station_info["lat"]
    lon    = station_info["lon"]
    obs_ds = station_info["ds"]
    vmap   = station_info["var_map"]

    logging.info(f"  Station: {name}  ({lat:.3f}°N, {lon:.3f}°E)")

    # ── Spatial check ──────────────────────────────────────────────────────
    x_utm, y_utm = to_utm.transform(lon, lat)
    if not in_domain(inference_ds, x_utm, y_utm):
        logging.warning(f"    {name}: outside model domain — skipping")
        return []

    iy, ix, dist_m = nearest_grid_cell(tree, nx, x_utm, y_utm)
    logging.info(f"    -> grid cell iy={iy} ix={ix}  dist={dist_m/1000:.1f} km")

    # ── Temporal overlap ──────────────────────────────────────────────────
    obs_time = pd.DatetimeIndex(obs_ds.time.values)
    inf_time = pd.DatetimeIndex(inference_ds.time.values)

    overlap_inf = find_overlap(obs_time, inf_time)
    if len(overlap_inf) < 10:
        logging.warning(f"    {name}: <10 timesteps overlap with inference "
                        f"(obs: {obs_time[0].date()}–{obs_time[-1].date()}, "
                        f"inference: {inf_time[0].date()}–{inf_time[-1].date()}) — skipping")
        return []

    proc_time      = pd.DatetimeIndex(processed_ds.time.values)
    overlap_proc   = find_overlap(overlap_inf, proc_time)
    has_proc       = len(overlap_proc) >= 10

    logging.info(f"    Overlap w/ inference: {len(overlap_inf)} pts | "
                 f"w/ processed: {len(overlap_proc)} pts")

    # Working time axis: use proc overlap if available, else inference only
    work_times = overlap_proc if has_proc else overlap_inf

    station_out = output_dir / name
    station_out.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for obs_var, model_var, era5_var, conus_var, var_label, unit in tqdm(
            vmap, desc=f"  {name} – variables", unit="var",
            dynamic_ncols=True, leave=False):
        # ── Observations ──────────────────────────────────────────────────
        if obs_var not in obs_ds:
            continue
        obs_arr = select_at_times(obs_ds[obs_var], work_times, obs_time)

        # Derive wind speed from obs u/v if obs_var not found but u/v exist
        if np.all(np.isnan(obs_arr)) and obs_var == "wind_speed_ms":
            if "u10_ms" in obs_ds and "v10_ms" in obs_ds:
                obs_u = select_at_times(obs_ds["u10_ms"], work_times, obs_time)
                obs_v = select_at_times(obs_ds["v10_ms"], work_times, obs_time)
                obs_arr = uv_to_speed(obs_u, obs_v)
            else:
                continue

        n_valid_obs = int(np.sum(~np.isnan(obs_arr)))
        if n_valid_obs < 10:
            n_total = len(obs_arr)
            raw_vals = obs_ds[obs_var].values if obs_var in obs_ds else np.array([])
            n_raw_valid = int(np.sum(~np.isnan(raw_vals)))
            logging.warning(
                f"    {var_label}: skipping — {n_valid_obs}/{n_total} valid after "
                f"time-alignment  (raw obs has {n_raw_valid}/{len(raw_vals)} valid; "
                f"raw range: [{np.nanmin(raw_vals):.3g}, {np.nanmax(raw_vals):.3g}] "
                f"if any)" if n_raw_valid > 0 else
                f"    {var_label}: skipping — {n_valid_obs}/{n_total} valid after "
                f"time-alignment  (raw obs is entirely NaN — variable likely not "
                f"measured at this station)"
            )
            continue

        # ── Model inference ───────────────────────────────────────────────
        if model_var and model_var in inference_ds:
            mod_ts = inference_ds[model_var].isel(y=iy, x=ix)
            mod_arr = select_at_times(mod_ts, work_times, inf_time)
        elif var_label == "wind_speed":
            # Derive from model u/v
            if "hr_u" in inference_ds and "hr_v" in inference_ds:
                mu = select_at_times(inference_ds["hr_u"].isel(y=iy, x=ix),
                                     work_times, inf_time)
                mv = select_at_times(inference_ds["hr_v"].isel(y=iy, x=ix),
                                     work_times, inf_time)
                mod_arr = uv_to_speed(mu, mv)
            else:
                continue
        else:
            continue

        # ── LR (ERA5) & HR truth (from processed data) ───────────────────
        era5_arr  = None
        conus_arr = None

        if has_proc:
            if era5_var and era5_var in processed_ds:
                era5_arr = select_at_times(
                    processed_ds[era5_var].isel(y=iy, x=ix), work_times, proc_time)
            elif var_label == "wind_speed":
                if "lr_u" in processed_ds and "lr_v" in processed_ds:
                    eu = select_at_times(processed_ds["lr_u"].isel(y=iy, x=ix),
                                         work_times, proc_time)
                    ev = select_at_times(processed_ds["lr_v"].isel(y=iy, x=ix),
                                         work_times, proc_time)
                    era5_arr = uv_to_speed(eu, ev)

            if conus_var and conus_var in processed_ds:
                conus_arr = select_at_times(
                    processed_ds[conus_var].isel(y=iy, x=ix), work_times, proc_time)
            elif var_label == "wind_speed":
                if "hr_u" in processed_ds and "hr_v" in processed_ds:
                    cu = select_at_times(processed_ds["hr_u"].isel(y=iy, x=ix),
                                         work_times, proc_time)
                    cv = select_at_times(processed_ds["hr_v"].isel(y=iy, x=ix),
                                         work_times, proc_time)
                    conus_arr = uv_to_speed(cu, cv)

        # ── Compute metrics ───────────────────────────────────────────────
        row_mod = compute_metrics(mod_arr, obs_arr, "Model")
        rows    = [row_mod]

        era5_rmse = None
        if era5_arr is not None:
            row_e5  = compute_metrics(era5_arr,  obs_arr, "ERA5")
            era5_rmse = row_e5["rmse"]
            rows.append(row_e5)
        if conus_arr is not None:
            row_con = compute_metrics(conus_arr, obs_arr, "CONUS404")
            rows.append(row_con)

        # Fill skill score vs ERA5
        for r in rows:
            if era5_rmse is not None and era5_rmse > 0:
                r["skill_vs_era5"] = skill_score(r["rmse"], era5_rmse)
            r["variable"] = var_label
            r["station"]  = name
            r["unit"]     = unit

        all_metrics.extend(rows)
        era5_str = f"{era5_rmse:.3f}" if era5_rmse is not None else "n/a"
        ss_str   = f"{row_mod['skill_vs_era5']:.3f}" if row_mod["skill_vs_era5"] is not None else "n/a"
        logging.info(f"    {var_label:12s}  n={row_mod['n']}  "
                     f"RMSE_model={row_mod['rmse']:.3f}  "
                     f"RMSE_era5={era5_str}  "
                     f"SS={ss_str}")

        # ── Plots ─────────────────────────────────────────────────────────
        series = {"Observations": obs_arr, "Model": mod_arr}
        if era5_arr  is not None: series["ERA5"]     = era5_arr
        if conus_arr is not None: series["CONUS404"] = conus_arr

        plot_timeseries(work_times, series, var_label, unit, name,
                        station_out / f"timeseries_{var_label}.png")
        plot_scatter(obs_arr, mod_arr, "Model", var_label, unit, name,
                     station_out / f"scatter_model_{var_label}.png")

    # Save per-station metrics CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        cols = ["station", "variable", "label", "n", "rmse", "ubrmse", "mae", "bias",
                "si", "skill", "correlation", "skill_vs_era5", "unit"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].round(4)
        df.to_csv(station_out / "metrics.csv", index=False)

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# PATH 2 – all stations
# ─────────────────────────────────────────────────────────────────────────────

def run_path2(inference_ds, processed_ds, output_dir):
    logging.info("=== Path 2: Station observation comparison ===")

    if not HAS_PYPROJ:
        logging.error("pyproj not available — cannot run Path 2. Install with: pip install pyproj")
        return []

    to_utm, _ = get_transformers()
    tree, nx, ny = build_grid_tree(inference_ds)
    all_metrics  = []

    # ── NDBC stations ─────────────────────────────────────────────────────
    ndbc_files = sorted(NDBC_DIR.glob("station_*.nc")) if NDBC_DIR.exists() else []
    logging.info(f"  Found {len(ndbc_files)} NDBC station files")

    for fpath in tqdm(ndbc_files, desc="Path 2 – NDBC stations",
                      unit="station", dynamic_ncols=True, leave=True):
        try:
            info = load_ndbc_station(fpath)
            if info is None:
                continue
            m = run_path2_station(info, inference_ds, processed_ds,
                                  tree, nx, to_utm,
                                  output_dir / "ndbc")
            all_metrics.extend(m)
            info["ds"].close()
        except Exception as e:
            logging.error(f"  NDBC {fpath.name}: {e}", exc_info=True)

    # ── Whale's Tale moorings ─────────────────────────────────────────────
    wt_existing = [f for f in WHALES_TALE_FILES if f.exists()]
    for fpath in tqdm(wt_existing, desc="Path 2 – Whale's Tale",
                      unit="station", dynamic_ncols=True, leave=True):
        try:
            info = load_whales_tale_station(fpath)
            m = run_path2_station(info, inference_ds, processed_ds,
                                  tree, nx, to_utm,
                                  output_dir / "whales_tale")
            all_metrics.extend(m)
            info["ds"].close()
        except Exception as e:
            logging.error(f"  Whale's Tale {fpath.name}: {e}", exc_info=True)

    for fpath in WHALES_TALE_FILES:
        if not fpath.exists():
            logging.warning(f"  Whale's Tale file not found: {fpath} — skipping")

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(p1_metrics, p2_metrics, output_dir, run_id, run_dirs):
    summary = {
        "generated":  datetime.utcnow().isoformat() + "Z",
        "run_id":     str(run_id),
        "inference":  str(run_dirs['output_inference']),
        "path1": {
            "n_metric_records": len(p1_metrics),
            "median_skill_wind_speed": float(np.nanmedian(
                [m["skill_vs_era5"] for m in p1_metrics if m["label"] == "model_vs_conus"]
            )) if p1_metrics else None,
        },
        "path2": {"n_metric_records": len(p2_metrics)},
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if p2_metrics:
        df = pd.DataFrame(p2_metrics)
        cols = ["station", "variable", "label", "n", "rmse", "ubrmse", "mae", "bias",
                "si", "skill", "correlation", "skill_vs_era5", "unit"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].round(4)
        csv_path = output_dir / "path2_metrics_table.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved Path 2 metric table to {csv_path}")

        # ── Combined NDBC summary table (Model only, wind_speed) ─────────
        ndbc_model = df[(df["station"].str.startswith("NDBC_")) & (df["label"] == "Model")]
        if not ndbc_model.empty:
            # Pivot: one row per station, columns for each variable's metrics
            pivot_rows = []
            for station in ndbc_model["station"].unique():
                row = {"station": station}
                for var in ndbc_model["variable"].unique():
                    subset = ndbc_model[(ndbc_model["station"] == station) &
                                        (ndbc_model["variable"] == var)]
                    if not subset.empty:
                        s = subset.iloc[0]
                        row[f"{var}_n"] = int(s["n"])
                        row[f"{var}_rmse"] = s["rmse"]
                        row[f"{var}_ubrmse"] = s["ubrmse"]
                        row[f"{var}_bias"] = s["bias"]
                        row[f"{var}_si"] = s["si"]
                        row[f"{var}_skill"] = s["skill"]
                        row[f"{var}_r"] = s["correlation"]
                pivot_rows.append(row)

            ndbc_summary = pd.DataFrame(pivot_rows).round(4)
            ndbc_csv_path = output_dir / "ndbc_combined_summary.csv"
            ndbc_summary.to_csv(ndbc_csv_path, index=False)
            logging.info(f"Saved NDBC combined summary to {ndbc_csv_path}")

            # Print to terminal
            print("\n" + "="*100)
            print("NDBC STATIONS – Combined Model Validation Summary")
            print("="*100)
            print(ndbc_summary.to_string(index=False))

        # Print full table
        print("\n" + "="*80)
        print("PATH 2 – All Station Metrics")
        print("="*80)
        print(df.to_string(index=False))

    logging.info(f"Summary written to {output_dir / 'summary.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Validate cosmos-wind-cnn inference output")
    p.add_argument("--run-id",   default="3663482",
                   help="Run ID (subdirectory under results/)")
    p.add_argument("--n-points", type=int, default=50,
                   help="Number of random grid points for Path 1 (default: 50)")
    p.add_argument("--seed",     type=int, default=42,
                   help="Random seed for spatial point sampling")
    p.add_argument("--skip-path1", action="store_true",
                   help="Skip Path 1 (spatial comparison)")
    p.add_argument("--skip-path2", action="store_true",
                   help="Skip Path 2 (station comparison)")
    return p.parse_args()


def main():
    args = parse_args()

    run_dirs = get_run_dirs(CASE_STUDY_DIR, args.run_id)
    output_dir = run_dirs['output_evaluation'] / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "validation.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"cosmos-wind-cnn inference validation  |  run_id={args.run_id}")
    logging.info(f"Output directory: {output_dir}")

    # Load datasets
    inference_ds = load_inference(run_dirs)
    processed_ds = load_processed(run_dirs)

    p1_metrics = []
    p2_metrics = []

    #if not args.skip_path1:
    #    p1_metrics = run_path1(
    #        inference_ds, processed_ds,
    #        n_points=args.n_points,
    #        seed=args.seed,
    #        output_dir=output_dir / "path1",
    #    )

    if not args.skip_path2:
        p2_metrics = run_path2(
            inference_ds, processed_ds,
            output_dir=output_dir / "path2",
        )

    write_summary(p1_metrics, p2_metrics, output_dir, args.run_id, run_dirs)

    inference_ds.close()
    processed_ds.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()

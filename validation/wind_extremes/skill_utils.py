"""
Shared skill-score utilities for all SFBay validation scripts.

Metrics
-------
All functions return a dict with the following keys (unless otherwise noted):

  n              – number of valid pairs
  bias           – mean(model - obs)
  rmse           – root-mean-square error
  mae            – mean absolute error
  sci            – Scatter Index  = RMSE / max(RMS(obs), |mean(obs)|)
  nrmse          – Normalised RMSE = RMSE / (obs_p75 - obs_p25)  (inter-quartile range)
  relative_bias  – bias / max(RMS(obs), |mean(obs)|)
  skill          – Willmott skill score = 1 - sum((m-o)²) / sum((|m-ō|+|o-ō|)²)
  r2             – coefficient of determination  (bonus; NaN if n < 2)
  obs_mean       – mean of observations
  obs_std        – std of observations
  model_mean     – mean of model
  model_std      – std of model

Empty dict {} is returned when there are fewer than min_n valid pairs.
"""

import os
import numpy as np


# ── core function ──────────────────────────────────────────────────────────────

def compute_skill_scores(obs, model, min_n: int = 10) -> dict:
    """
    Compute unified validation skill metrics.

    Parameters
    ----------
    obs, model : array-like
        Observed and modelled values (NaN-safe; must be the same length).
    min_n : int
        Minimum number of valid pairs required; returns {} otherwise.

    Returns
    -------
    dict  – see module docstring for key descriptions.
    """
    obs   = np.asarray(obs,   dtype=float)
    model = np.asarray(model, dtype=float)

    valid = ~(np.isnan(obs) | np.isnan(model))
    o = obs[valid]
    m = model[valid]

    if len(o) < min_n:
        return {}

    obs_mean  = float(np.mean(o))
    bias      = float(np.mean(m - o))
    rmse      = float(np.sqrt(np.mean((m - o) ** 2)))
    mae       = float(np.mean(np.abs(m - o)))

    # IQR-based normalisation: robust to outliers (which would inflate the
    # min-max range and silently shrink NRMSE).  Switched from max-min on
    # 2026-05-22.
    obs_iqr = float(np.percentile(o, 75) - np.percentile(o, 25))
    nrmse   = rmse / obs_iqr if obs_iqr > 0 else np.nan

    obs_rms   = float(np.sqrt(np.mean(o ** 2)))
    sci_denom = max(obs_rms, abs(obs_mean))
    sci           = rmse / sci_denom if sci_denom > 0 else np.nan
    relative_bias = bias / sci_denom if sci_denom > 0 else np.nan

    # Willmott skill score
    num   = float(np.sum((m - o) ** 2))
    den   = float(np.sum((np.abs(m - obs_mean) + np.abs(o - obs_mean)) ** 2))
    skill = float(1.0 - num / den) if den > 0 else np.nan

    # R² (bonus metric; kept for backward compatibility with T/S script)
    try:
        from sklearn.metrics import r2_score as _r2
        r2 = float(_r2(o, m)) if len(o) >= 2 else np.nan
    except ImportError:
        r2 = np.nan

    return {
        'n':             int(valid.sum()),
        'bias':          bias,
        'rmse':          rmse,
        'mae':           mae,
        'sci':           sci,
        'nrmse':         nrmse,
        'relative_bias': relative_bias,
        'skill':         skill,
        'r2':            r2,
        'obs_mean':      obs_mean,
        'obs_std':       float(np.std(o)),
        'model_mean':    float(np.mean(m)),
        'model_std':     float(np.std(m)),
    }


# ── shapefile writer ───────────────────────────────────────────────────────────

def write_skill_shapefile(records, output_path,
                          lat_col: str = 'lat',
                          lon_col: str = 'lon',
                          crs: str = 'EPSG:4326') -> None:
    """
    Write station skill scores to a point shapefile.

    Parameters
    ----------
    records : list of dict
        Each dict must contain *lat_col* and *lon_col* entries plus any
        number of attribute columns (station_id, station_name, skill scores
        etc.).  Rows with missing/NaN coordinates are silently dropped.
    output_path : str or Path
        Destination .shp path.  The directory is created if it does not exist.
        The '.shp' extension is appended automatically if omitted.
    lat_col, lon_col : str
        Keys in *records* dicts that hold the latitude / longitude values.
    crs : str
        Coordinate reference system string understood by pyproj / geopandas
        (default: EPSG:4326 – geographic WGS-84).

    Notes
    -----
    Shapefile field names are limited to 10 characters.  Long column names are
    silently truncated (with a numeric suffix to keep them unique).  A mapping
    of original → truncated names is printed to stdout.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("  WARNING: geopandas not available – shapefile output skipped.")
        print("           Install with: conda install geopandas")
        return

    import pandas as pd

    if not records:
        print("  No records supplied – shapefile skipped.")
        return

    df = pd.DataFrame(records)

    # Drop rows with unusable coordinates
    coord_ok = df[lat_col].notna() & df[lon_col].notna()
    df = df[coord_ok].copy()
    if df.empty:
        print("  No rows with valid coordinates – shapefile skipped.")
        return

    # ── truncate column names to ≤ 10 chars (shapefile limit) ──────────────
    rename_map = {}
    seen = set()
    truncated = []
    for col in df.columns:
        if col in (lat_col, lon_col, 'geometry'):
            continue
        if len(col) <= 10:
            rename_map[col] = col
            seen.add(col)
        else:
            base  = col[:8]
            short = col[:10]
            if short not in seen:
                rename_map[col] = short
                seen.add(short)
            else:
                i = 2
                candidate = f"{base}_{i}"
                while candidate in seen:
                    i += 1
                    candidate = f"{base}_{i}"
                rename_map[col] = candidate
                seen.add(candidate)
            if rename_map[col] != col:
                truncated.append(f"  {col!r:35s} -> {rename_map[col]!r}")

    if truncated:
        print("  Shapefile column name truncations:")
        for t in truncated:
            print(t)

    df = df.rename(columns=rename_map)

    # ── build GeoDataFrame ─────────────────────────────────────────────────
    geometry = [Point(lon, lat)
                for lon, lat in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(
        df.drop(columns=[lat_col, lon_col], errors='ignore'),
        geometry=geometry,
        crs=crs,
    )

    # ── write file ─────────────────────────────────────────────────────────
    output_path = str(output_path)
    if not output_path.endswith('.shp'):
        output_path += '.shp'
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    gdf.to_file(output_path)
    print(f"  Shapefile saved: {output_path}  ({len(gdf)} features)")

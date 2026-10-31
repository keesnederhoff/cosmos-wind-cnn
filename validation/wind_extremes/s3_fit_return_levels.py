"""Stage 3: POT/GPD return levels on grids and at stations.

Grid fits (Task 8 Step 1): per product x window (full, common), fit every cell
of dailymax_<product>.nc with windx_common.pot_fit, in parallel over row
blocks, and write results/rl_map_<product>_<window>.nc.

Station fits (Steps 2-4): "full" mode reads obs from its own record and model
values from the product's own full-window map (so scatter == map numbers);
"matched" mode refits obs and model on the hourly-masked, WY-intersected
common record. Each product's station results are written to a per-product
part file results/station_rl_part_<product>.csv; a merge step at the end of
every run combines whatever part files exist into results/station_rl.csv, so
the CSV grows as more products' Stage-1 caches finish.

Homogeneity (Step 5) is obs-only and does not depend on any product cache; it
runs on every invocation and writes results/homogeneity.csv.

Resumable: a product/window whose output file already exists is skipped
unless OVERWRITE_EXISTING is True.
"""

# === CONFIGURATION ===
PRODUCTS_TO_FIT = ["era5", "rtma", "cnn_raw", "cnn_bc", "cnn_q82", "cnn_q92", "cnn_ba"]   # == windx_common.ALL_PRODUCTS; subset to run this call
WINDOWS = ["full", "common"]
OVERWRITE_EXISTING = False          # True re-fits / re-writes outputs that already exist on disk
OVERWRITE_STATIONS = True           # station fits are cheap: always refresh (obs QC changes don't touch the grids)
MONO_TOL = 1e-3                     # m/s, float tolerance for the rl_1 <= rl_10 <= rl_100 check
DEGENERATE_SPREAD = 0.05            # m/s, rl_100 - rl_1 below this = collapsed tail (reported, not failed)
TIME_ONLY_PRODUCTS: list[str] = []  # if non-empty: only run the 20x20 grid-fit timing harness for these products, then exit

# === IMPORTS ===
import concurrent.futures as cf
import math
import os
import time

import netCDF4
import numpy as np
import pandas as pd
import evt as rt

import windx_common as w

N_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))

GRID_OUT_KEYS = ["rl_1", "rl_10", "rl_100", "lo_1", "lo_10", "lo_100", "hi_1", "hi_10", "hi_100",
                 "rl_exp_1", "rl_exp_10", "rl_exp_100", "threshold", "sigma", "xi", "n_years", "n_peaks"]
FIT_CODE_MAP = {"ok": 0, "dropped_<6yr": 1, "insufficient_peaks": 2, "fit_error": 3}
CODE_TO_STATUS = {v: k for k, v in FIT_CODE_MAP.items()}


def _safe_pot_fit(series: pd.Series, n_years, rng) -> dict:
    """windx_common.pot_fit, but never crashes the run. cnn_bc's flat
    bias-correction clamp (Known facts in the plan) can put many declustered
    peaks at the exact same capped value, giving the GPD excesses a zero
    L-scale; rainlib.extremes.gpd_fit_lmom_excess raises ValueError on that.
    A handful of such cells is an expected, reportable property of cnn_bc's
    tail, not a bug -- so it is caught here and recorded as status
    'fit_error' (NaN outputs, like 'insufficient_peaks') rather than
    propagated out of the worker and killing the whole grid fit."""
    try:
        return w.pot_fit(series, n_years, rng)
    except ValueError as e:
        return {"status": "fit_error", "n_years": n_years, "error": str(e)}

STATION_CSV_COLUMNS = [
    "station_id", "network", "mode", "product", "source", "n_years", "wy_first", "wy_last", "n_peaks",
    "threshold", "sigma", "xi",
    "rl_1", "lo_1", "hi_1", "rl_10", "lo_10", "hi_10", "rl_100", "lo_100", "hi_100",
    "rl_exp_1", "rl_exp_10", "rl_exp_100", "method", "status",
    "median_hours_per_valid_day", "dist_m",
]


# === SMALL HELPERS ===

def _filled(x) -> np.ndarray:
    """netCDF4 slice (plain or masked array) -> float ndarray with NaN for missing."""
    if np.ma.isMaskedArray(x):
        return x.filled(np.nan)
    return np.asarray(x, dtype=float)


def _decode_time_values(units: str, vals) -> pd.DatetimeIndex:
    """Decode a CF "<unit> since <base>" time axis without cftime. Handles
    days/hours/minutes/seconds (dailymax uses days, the hourly caches use hours)."""
    unit_word, _, base_str = units.partition("since")
    unit_word = unit_word.strip().lower()
    base = pd.Timestamp(base_str.strip())
    unit_map = {"days": "D", "day": "D", "hours": "h", "hour": "h", "minutes": "min", "seconds": "s"}
    td_unit = unit_map[unit_word]
    vals = np.asarray(vals, dtype=np.float64)
    return pd.DatetimeIndex(base + pd.to_timedelta(vals, unit=td_unit))


def _trim_to_whole_days(times: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, int]:
    """Drop a leading partial day (if the axis doesn't start at 00:00) and a
    trailing partial day, so len is a multiple of 24. Returns (trimmed, lead_drop)."""
    times = pd.DatetimeIndex(times)
    if len(times) == 0:
        return times, 0
    lead_drop = (24 - times[0].hour) % 24
    trimmed = times[lead_drop:]
    n = len(trimmed) - (len(trimmed) % 24)
    return trimmed[:n], lead_drop


def product_ready(product: str) -> bool:
    """True only once Stage 1 has finalized this product (per the log) and
    both of its cache files exist. Never trust a dailymax file whose finalize
    isn't confirmed in cache/s1_log.txt."""
    dmax_path = w.CACHE_DIR / f"dailymax_{product}.nc"
    st_path = w.CACHE_DIR / f"stations_hourly_{product}.nc"
    if not (dmax_path.exists() and st_path.exists()):
        return False
    log_path = w.CACHE_DIR / "s1_log.txt"
    if not log_path.exists():
        return False
    log_text = log_path.read_text(errors="ignore")
    return (f"{product}: finalize OK" in log_text) or (f"{product}: final caches already exist" in log_text)


# === GRID FIT CORE (shared by the worker and the timing harness) ===

def _fit_block_core(block: np.ndarray, days: pd.DatetimeIndex, wy_lo: int, wy_hi: int,
                     product_idx: int, win_idx: int, y0: int, x0: int, label: str) -> tuple[dict, np.ndarray]:
    """Fit every cell of `block` (n_days, nrow, ncol). `y0`/`x0` are this
    block's offset into the full grid, used for both seeding and reporting."""
    n_days, nrow, ncol = block.shape

    valid_dict = w.valid_water_years(days, block)   # {wy: bool array (nrow, ncol)}
    wy_of_day = w.water_year(days)
    wys_sorted = sorted(y for y in valid_dict if wy_lo <= y <= wy_hi)

    if wys_sorted:
        valid_stack = np.stack([np.asarray(valid_dict[y], dtype=bool) for y in wys_sorted], axis=0)
        wy_to_idx = {y: i for i, y in enumerate(wys_sorted)}
        day_wy_idx = np.fromiter((wy_to_idx.get(y, -1) for y in wy_of_day), dtype=np.int64, count=n_days)
        valid_mask = np.zeros((n_days, nrow, ncol), dtype=bool)
        ok_days = day_wy_idx >= 0
        if ok_days.any():
            valid_mask[ok_days] = valid_stack[day_wy_idx[ok_days]]
        n_years_grid = valid_stack.sum(axis=0).astype(np.float32)
    else:
        valid_mask = np.zeros((n_days, nrow, ncol), dtype=bool)
        n_years_grid = np.zeros((nrow, ncol), dtype=np.float32)

    masked = np.where(valid_mask, block, np.nan)

    arrs = {k: np.full((nrow, ncol), np.nan, dtype=np.float32) for k in GRID_OUT_KEYS}
    arrs["n_years"] = n_years_grid
    fit_code = np.full((nrow, ncol), 255, dtype=np.uint8)

    total = nrow * ncol
    count = 0
    for r in range(nrow):
        for c in range(ncol):
            gy, gx = y0 + r, x0 + c
            n_years_cell = int(n_years_grid[r, c])
            series = pd.Series(masked[:, r, c].astype(np.float64), index=days)
            rng = np.random.default_rng([w.RNG_SEED, product_idx, win_idx, gy, gx])
            res = _safe_pot_fit(series, n_years_cell, rng)
            fit_code[r, c] = FIT_CODE_MAP.get(res.get("status"), 255)
            if res.get("status") == "ok":
                for k in GRID_OUT_KEYS:
                    if k != "n_years":
                        arrs[k][r, c] = res[k]
            count += 1
            if count % 1000 == 0:
                print(f"  [{label}] {count}/{total} cells", flush=True)

    return arrs, fit_code


def _fit_row_block(args) -> tuple[int, int, dict, np.ndarray]:
    """Worker entry point: open the file itself, read only rows [y0:y1)."""
    (path_str, day_idx0, day_idx1, y0, y1, wy_lo, wy_hi, product_idx, win_idx, label) = args
    with netCDF4.Dataset(path_str) as ds:
        day_units = ds.variables["day"].units
        day_vals = ds.variables["day"][day_idx0:day_idx1]
        block = ds.variables["dmax"][day_idx0:day_idx1, y0:y1, :]
    days = _decode_time_values(day_units, day_vals)
    block = _filled(block).astype(np.float32)
    arrs, fit_code = _fit_block_core(block, days, wy_lo, wy_hi, product_idx, win_idx, y0, 0,
                                      label=f"{label} rows {y0}-{y1}")
    return y0, y1, arrs, fit_code


def _window_bounds(product: str, window: str) -> tuple[int, int]:
    return w.FULL_WY[product] if window == "full" else w.COMMON_WY


def _day_index_range(dailymax_path, wy_lo: int, wy_hi: int) -> tuple[np.ndarray, int, int]:
    """Day-axis timestamps, plus the [i0, i1) index range covering [wy_lo, wy_hi]."""
    with netCDF4.Dataset(dailymax_path) as ds:
        day_units = ds.variables["day"].units
        day_vals_full = ds.variables["day"][:]
    days_full = _decode_time_values(day_units, day_vals_full)
    wy_start = pd.Timestamp(year=wy_lo - 1, month=w.WY_START_MONTH, day=1)
    wy_end_excl = pd.Timestamp(year=wy_hi, month=w.WY_START_MONTH, day=1)
    i0 = int(np.searchsorted(days_full.to_numpy(), np.datetime64(wy_start)))
    i1 = int(np.searchsorted(days_full.to_numpy(), np.datetime64(wy_end_excl)))
    return days_full, i0, i1


def write_rl_map(path, x, y, arrs: dict, fit_code: np.ndarray, wy_lo: int, wy_hi: int,
                  product: str, window: str) -> None:
    ny, nx = len(y), len(x)
    if path.exists():
        path.unlink()
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        yv = ds.createVariable("y", "f8", ("y",)); yv[:] = np.asarray(y, dtype=np.float64)
        xv = ds.createVariable("x", "f8", ("x",)); xv[:] = np.asarray(x, dtype=np.float64)
        for k, arr in arrs.items():
            v = ds.createVariable(k, "f4", ("y", "x"), zlib=True, complevel=4, fill_value=np.float32(np.nan))
            v[:, :] = arr.astype(np.float32)
        cv = ds.createVariable("fit_code", "u1", ("y", "x"), zlib=True, complevel=4)
        cv[:, :] = fit_code.astype(np.uint8)
        cv.legend = "0=ok,1=dropped_<6yr,2=insufficient_peaks,255=other"
        ds.method = "POT-GPD-Lmom lam3 sep48h dailymax"
        ds.wy_range = f"{wy_lo}-{wy_hi}"
        ds.product = product
        ds.window = window
    size_mb = path.stat().st_size / 1e6
    print(f"  wrote {path} ({size_mb:.1f} MB)")


def fit_grid(product: str, window: str) -> dict:
    out_path = w.RESULT_DIR / f"rl_map_{product}_{window}.nc"
    wy_lo, wy_hi = _window_bounds(product, window)
    if out_path.exists() and not OVERWRITE_EXISTING:
        print(f"[{product}/{window}] already exists, skipping ({out_path})")
        return {"product": product, "window": window, "path": out_path, "seconds": 0.0, "skipped": True}

    t0 = time.time()
    dailymax_path = w.CACHE_DIR / f"dailymax_{product}.nc"
    with netCDF4.Dataset(dailymax_path) as ds:
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
    days_full, day_idx0, day_idx1 = _day_index_range(dailymax_path, wy_lo, wy_hi)
    ny, nx = len(y), len(x)
    win_idx = 0 if window == "full" else 1
    product_idx = w.ALL_PRODUCTS.index(product)

    n_workers = min(N_WORKERS, ny)
    chunk = math.ceil(ny / n_workers)
    blocks = [(r0, min(r0 + chunk, ny)) for r0 in range(0, ny, chunk)]
    label = f"{product}/{window}"
    args_list = [(str(dailymax_path), day_idx0, day_idx1, y0, y1, wy_lo, wy_hi, product_idx, win_idx, label)
                 for (y0, y1) in blocks]

    arrs = {k: np.full((ny, nx), np.nan, dtype=np.float32) for k in GRID_OUT_KEYS}
    fit_code = np.full((ny, nx), 255, dtype=np.uint8)

    print(f"[{label}] fitting {ny}x{nx} = {ny * nx} cells, {n_workers} workers, {len(blocks)} row blocks", flush=True)
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for y0, y1, block_arrs, block_code in ex.map(_fit_row_block, args_list):
            for k in GRID_OUT_KEYS:
                arrs[k][y0:y1, :] = block_arrs[k]
            fit_code[y0:y1, :] = block_code

    write_rl_map(out_path, x, y, arrs, fit_code, wy_lo, wy_hi, product, window)
    dt = time.time() - t0
    print(f"[{label}] done in {dt:.1f} s", flush=True)
    return {"product": product, "window": window, "path": out_path, "seconds": dt, "ny": ny, "nx": nx}


def time_20x20(product: str, window: str = "full", block_size: int = 20) -> dict:
    """Time the per-cell fit on a block_size x block_size corner of the grid
    and extrapolate to the full grid (serial and with N_WORKERS)."""
    wy_lo, wy_hi = _window_bounds(product, window)
    dailymax_path = w.CACHE_DIR / f"dailymax_{product}.nc"
    with netCDF4.Dataset(dailymax_path) as ds:
        ny = ds.dimensions["y"].size
        nx = ds.dimensions["x"].size
    _, day_idx0, day_idx1 = _day_index_range(dailymax_path, wy_lo, wy_hi)
    y1b, x1b = min(block_size, ny), min(block_size, nx)
    win_idx = 0 if window == "full" else 1
    product_idx = w.ALL_PRODUCTS.index(product)

    with netCDF4.Dataset(dailymax_path) as ds:
        day_units = ds.variables["day"].units
        day_vals = ds.variables["day"][day_idx0:day_idx1]
        block = ds.variables["dmax"][day_idx0:day_idx1, 0:y1b, 0:x1b]
    days = _decode_time_values(day_units, day_vals)
    block = _filled(block).astype(np.float32)

    t0 = time.time()
    _fit_block_core(block, days, wy_lo, wy_hi, product_idx, win_idx, 0, 0, label=f"{product}/{window} TIMING {y1b}x{x1b}")
    dt = time.time() - t0

    n_cells_block = y1b * x1b
    per_cell = dt / n_cells_block
    total_cells = ny * nx
    serial_est = per_cell * total_cells
    parallel_est = serial_est / min(N_WORKERS, ny)
    print(f"[{product}/{window}] TIMING: {n_cells_block} cells in {dt:.1f} s "
          f"({per_cell*1000:.1f} ms/cell); grid {ny}x{nx}={total_cells} cells -> "
          f"serial ~{serial_est/60:.1f} min, ~{N_WORKERS} workers ~{parallel_est/60:.1f} min")
    return {"product": product, "window": window, "n_cells_block": n_cells_block, "seconds": dt,
            "per_cell_s": per_cell, "total_cells": total_cells,
            "serial_est_s": serial_est, "parallel_est_s": parallel_est}


# === OBS / STATION CACHE LOADERS ===

def load_obs_cache():
    path = w.CACHE_DIR / "obs_hourly.nc"
    with netCDF4.Dataset(path) as ds:
        station_id = w._decode_station_ids(ds.variables["station_id"][:])
        network = w._decode_station_ids(ds.variables["network"][:])
        times = _decode_time_values(ds.variables["time"].units, ds.variables["time"][:])
        ws = _filled(ds.variables["ws"][:, :]).astype(np.float64)
    return station_id, network, times, ws


def load_stations_hourly(product: str):
    path = w.CACHE_DIR / f"stations_hourly_{product}.nc"
    with netCDF4.Dataset(path) as ds:
        station_id = w._decode_station_ids(ds.variables["station_id"][:])
        times = _decode_time_values(ds.variables["time"].units, ds.variables["time"][:])
        ws = _filled(ds.variables["ws"][:, :]).astype(np.float64)
        iy = np.asarray(ds.variables["iy"][:], dtype=np.int64)
        ix = np.asarray(ds.variables["ix"][:], dtype=np.int64)
        dist_m = np.asarray(ds.variables["dist_m"][:], dtype=np.float64)
    return station_id, times, ws, iy, ix, dist_m


# === STATION FITS ===

def compute_obs_full_results(station_table: pd.DataFrame, obs_cache):
    """Obs-only 'full' fit, one per station (used for station_rl.csv's obs rows
    and for homogeneity.csv). Returns the per-station fit dict plus the
    (dmax_all, days, nvalid_all, valid_dict) the homogeneity step reuses."""
    sid_list, _network_list, obs_times, obs_ws = obs_cache
    sid_to_i = {s: i for i, s in enumerate(sid_list)}

    dmax_all, days, nvalid_all = w.daily_max(obs_ws.T, obs_times)   # (n_days, n_station)
    valid_dict = w.valid_water_years(days, dmax_all)                # {wy: bool array(n_station)}
    wy_of_days = w.water_year(days)

    results = {}
    for i, row in enumerate(station_table.itertuples()):
        sid = row.station_id
        oi = sid_to_i[sid]
        years_ok = sorted(y for y, v in valid_dict.items()
                           if y <= w.OBS_MAX_WY and bool(np.asarray(v)[oi]))
        n_years = len(years_ok)
        wy_first = years_ok[0] if years_ok else np.nan
        wy_last = years_ok[-1] if years_ok else np.nan
        keep = np.isin(wy_of_days, years_ok)
        series = pd.Series(np.where(keep, dmax_all[:, oi], np.nan), index=days)
        rng = np.random.default_rng([w.RNG_SEED, 99, 0, i])
        res = _safe_pot_fit(series, n_years, rng)
        res["wy_first"], res["wy_last"] = wy_first, wy_last
        sel = keep & (nvalid_all[:, oi] >= w.MIN_HOURS_PER_DAY)
        res["median_hours"] = float(np.median(nvalid_all[sel, oi])) if sel.any() else np.nan
        results[sid] = res

    return results, dmax_all, days, nvalid_all, valid_dict


def matched_fit(obs_times, obs_ws_row, prod_times, prod_ws_row, product: str, station_index: int):
    """Hourly-masked, WY-intersected fit of obs and a product at one station."""
    common = obs_times.intersection(prod_times)
    common, _lead = _trim_to_whole_days(common)
    if len(common) == 0:
        return None

    obs_c = pd.Series(obs_ws_row, index=obs_times).reindex(common).to_numpy()
    prod_c = pd.Series(prod_ws_row, index=prod_times).reindex(common).to_numpy()
    mask = np.isfinite(obs_c) & np.isfinite(prod_c)
    obs_masked = np.where(mask, obs_c, np.nan)
    prod_masked = np.where(mask, prod_c, np.nan)

    dmax_obs, days, nvalid_obs = w.daily_max(obs_masked, common)
    dmax_prod, _days2, _nvalid_prod = w.daily_max(prod_masked, common)

    valid_obs = w.valid_water_years(days, dmax_obs)
    valid_prod = w.valid_water_years(days, dmax_prod)
    wy_lo_p, wy_hi_p = w.FULL_WY[product]
    joint_years = sorted(y for y in valid_obs if y in valid_prod
                          and bool(valid_obs[y]) and bool(valid_prod[y])
                          and wy_lo_p <= y <= wy_hi_p)
    n_years = len(joint_years)
    wy_first = joint_years[0] if joint_years else np.nan
    wy_last = joint_years[-1] if joint_years else np.nan

    wy_of_days = w.water_year(days)
    keep = np.isin(wy_of_days, joint_years)
    obs_series = pd.Series(np.where(keep, dmax_obs, np.nan), index=days)
    prod_series = pd.Series(np.where(keep, dmax_prod, np.nan), index=days)

    product_idx = w.ALL_PRODUCTS.index(product)
    rng_obs = np.random.default_rng([w.RNG_SEED, 98, product_idx, station_index])
    rng_mod = np.random.default_rng([w.RNG_SEED, 97, product_idx, station_index])
    res_obs = _safe_pot_fit(obs_series, n_years, rng_obs)
    res_mod = _safe_pot_fit(prod_series, n_years, rng_mod)

    sel = keep & (nvalid_obs >= w.MIN_HOURS_PER_DAY)
    median_hours = float(np.median(nvalid_obs[sel])) if sel.any() else np.nan

    return res_obs, res_mod, n_years, wy_first, wy_last, median_hours


def _row(sid, network, mode, product, source, res: dict, n_years=None, wy_first=np.nan, wy_last=np.nan,
         median_hours=np.nan, dist_m=np.nan) -> dict:
    return {
        "station_id": sid, "network": network, "mode": mode, "product": product, "source": source,
        "n_years": n_years if n_years is not None else res.get("n_years", np.nan),
        "wy_first": wy_first, "wy_last": wy_last,
        "n_peaks": res.get("n_peaks", np.nan),
        "threshold": res.get("threshold", np.nan), "sigma": res.get("sigma", np.nan), "xi": res.get("xi", np.nan),
        "rl_1": res.get("rl_1", np.nan), "lo_1": res.get("lo_1", np.nan), "hi_1": res.get("hi_1", np.nan),
        "rl_10": res.get("rl_10", np.nan), "lo_10": res.get("lo_10", np.nan), "hi_10": res.get("hi_10", np.nan),
        "rl_100": res.get("rl_100", np.nan), "lo_100": res.get("lo_100", np.nan), "hi_100": res.get("hi_100", np.nan),
        "rl_exp_1": res.get("rl_exp_1", np.nan), "rl_exp_10": res.get("rl_exp_10", np.nan),
        "rl_exp_100": res.get("rl_exp_100", np.nan),
        "method": res.get("method", np.nan),
        "status": res.get("status"),
        "median_hours_per_valid_day": median_hours,
        "dist_m": dist_m,
    }


def run_station_fits(product: str, obs_cache, obs_full_results: dict, station_table: pd.DataFrame) -> pd.DataFrame:
    sid_list, _network_list, obs_times, obs_ws = obs_cache
    sid_to_i_obs = {s: i for i, s in enumerate(sid_list)}

    prod_sid, prod_times, prod_ws, iy_arr, ix_arr, dist_arr = load_stations_hourly(product)
    sid_to_i_prod = {s: i for i, s in enumerate(prod_sid)}

    rl_full_path = w.RESULT_DIR / f"rl_map_{product}_full.nc"
    with netCDF4.Dataset(rl_full_path) as ds:
        map_vars = {k: _filled(ds.variables[k][:, :]) for k in GRID_OUT_KEYS}
        fit_code_map = np.asarray(ds.variables["fit_code"][:, :])
    wy_lo_p, wy_hi_p = w.FULL_WY[product]

    rows = []
    for i, row in enumerate(station_table.itertuples()):
        sid, network, status_obs = row.station_id, row.network, row.status

        # --- full mode, obs (product-independent; repeated in every part file) ---
        res_obs_full = obs_full_results[sid]
        rows.append(_row(sid, network, "full", "obs", "obs", res_obs_full,
                          n_years=res_obs_full.get("n_years", np.nan),
                          wy_first=res_obs_full.get("wy_first", np.nan),
                          wy_last=res_obs_full.get("wy_last", np.nan),
                          median_hours=res_obs_full.get("median_hours", np.nan)))

        j = sid_to_i_prod.get(sid)
        if j is None:
            print(f"  WARNING: {sid} missing from stations_hourly_{product}.nc")
            continue
        iy, ix, dist_m = int(iy_arr[j]), int(ix_arr[j]), float(dist_arr[j])

        # --- full mode, model: map lookup ---
        code = int(fit_code_map[iy, ix])
        res_model_full = {"status": CODE_TO_STATUS.get(code, "other")}
        if code == 0:
            for k in GRID_OUT_KEYS:
                res_model_full[k] = float(map_vars[k][iy, ix])
            res_model_full["method"] = "POT-GPD-Lmom lam3 sep48h dailymax"
        rows.append(_row(sid, network, "full", product, "model", res_model_full,
                          n_years=float(map_vars["n_years"][iy, ix]),
                          wy_first=wy_lo_p, wy_last=wy_hi_p, dist_m=dist_m))

        # --- matched mode: only for obs stations that are themselves "ok" ---
        if status_obs != "ok":
            rows.append(_row(sid, network, "matched", product, "obs", {"status": status_obs}))
            rows.append(_row(sid, network, "matched", product, "model", {"status": status_obs}, dist_m=dist_m))
            continue

        oi = sid_to_i_obs.get(sid)
        if oi is None:
            print(f"  WARNING: {sid} missing from obs_hourly.nc")
            continue
        result = matched_fit(obs_times, obs_ws[oi, :], prod_times, prod_ws[j, :], product, station_index=i)
        if result is None:
            rows.append(_row(sid, network, "matched", product, "obs", {"status": "no_overlap"}))
            rows.append(_row(sid, network, "matched", product, "model", {"status": "no_overlap"}, dist_m=dist_m))
            continue
        res_obs_m, res_mod_m, n_years_m, wy_first_m, wy_last_m, median_hours_m = result
        rows.append(_row(sid, network, "matched", product, "obs", res_obs_m,
                          n_years=n_years_m, wy_first=wy_first_m, wy_last=wy_last_m, median_hours=median_hours_m))
        rows.append(_row(sid, network, "matched", product, "model", res_mod_m,
                          n_years=n_years_m, wy_first=wy_first_m, wy_last=wy_last_m, dist_m=dist_m))

    df_out = pd.DataFrame(rows, columns=STATION_CSV_COLUMNS)
    out_path = w.RESULT_DIR / f"station_rl_part_{product}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df_out)} rows)")
    return df_out


def merge_station_rl_parts() -> "pd.DataFrame | None":
    parts = sorted(w.RESULT_DIR.glob("station_rl_part_*.csv"))
    if not parts:
        print("No station_rl_part_*.csv found yet, skipping merge")
        return None
    dfs = [pd.read_csv(p) for p in parts]
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["station_id", "mode", "product", "source"], keep="last")
    merged = merged.sort_values(["station_id", "mode", "product", "source"]).reset_index(drop=True)
    out_path = w.RESULT_DIR / "station_rl.csv"
    merged.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(merged)} rows) from {len(parts)} part file(s): "
          f"{[p.name for p in parts]}")
    missing = [p for p in w.ALL_PRODUCTS if not (w.RESULT_DIR / f"station_rl_part_{p}.csv").exists()]
    if missing:
        print(f"  station_rl.csv is PARTIAL: products not yet available: {missing}")
    return merged


# === HOMOGENEITY (Step 5, obs-only) ===

def run_homogeneity(station_table: pd.DataFrame, dmax_all: np.ndarray, days: pd.DatetimeIndex,
                     valid_dict: dict, obs_full_results: dict, obs_cache) -> pd.DataFrame:
    sid_list, _network_list, _obs_times, _obs_ws = obs_cache
    sid_to_i = {s: i for i, s in enumerate(sid_list)}
    wy_of_days = w.water_year(days)

    rows = []
    for i, row in enumerate(station_table.itertuples()):
        sid = row.station_id
        res_full = obs_full_results[sid]
        n_years = res_full.get("n_years", 0) or 0
        if n_years < 20:
            continue
        oi = sid_to_i[sid]
        years_ok = sorted(y for y, v in valid_dict.items()
                           if y <= w.OBS_MAX_WY and bool(np.asarray(v)[oi]))
        annual_max = np.array([np.nanmax(dmax_all[wy_of_days == y, oi]) for y in years_ok], dtype=float)
        years_arr = np.asarray(years_ok, dtype=int)

        pt = rt.pettitt_test(annual_max, years_arr)
        p_value = pt["p"]
        break_wy = pt["change_year"]
        sub_from_wy = break_wy if (break_wy is not None and p_value < 0.05) else 1997

        post_years = [y for y in years_ok if y >= sub_from_wy]
        n_years_post = len(post_years)
        keep_post = np.isin(wy_of_days, post_years)
        series_post = pd.Series(np.where(keep_post, dmax_all[:, oi], np.nan), index=days)
        rng_post = np.random.default_rng([w.RNG_SEED, 96, 0, i])   # not seeded by the plan; own choice, documented in report
        res_post = _safe_pot_fit(series_post, n_years_post, rng_post)

        rl_100_full = res_full.get("rl_100", np.nan)
        rl_100_post = res_post.get("rl_100", np.nan)
        if np.isfinite(rl_100_full) and rl_100_full != 0 and np.isfinite(rl_100_post):
            pct_diff = 100.0 * (rl_100_post - rl_100_full) / rl_100_full
        else:
            pct_diff = np.nan

        rows.append({
            "station_id": sid, "n_years": n_years, "break_wy": break_wy, "p_value": p_value,
            "sub_from_wy": sub_from_wy, "n_years_post": n_years_post,
            "rl_100_full": rl_100_full, "rl_100_post": rl_100_post, "pct_diff": pct_diff,
        })

    df = pd.DataFrame(rows, columns=["station_id", "n_years", "break_wy", "p_value", "sub_from_wy",
                                      "n_years_post", "rl_100_full", "rl_100_post", "pct_diff"])
    out_path = w.RESULT_DIR / "homogeneity.csv"
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df)} rows)")
    return df


# === SANITY CHECKS (Step 6) ===

def sanity_checks(products_done: list[str]) -> None:
    print("\n=== Sanity checks ===")
    violations_total = 0
    all_nan_maps = []

    for product in products_done:
        for window in WINDOWS:
            path = w.RESULT_DIR / f"rl_map_{product}_{window}.nc"
            if not path.exists():
                continue
            with netCDF4.Dataset(path) as ds:
                rl1 = _filled(ds.variables["rl_1"][:, :])
                rl10 = _filled(ds.variables["rl_10"][:, :])
                rl100 = _filled(ds.variables["rl_100"][:, :])
                n_years = _filled(ds.variables["n_years"][:, :])

            ok = np.isfinite(rl1) & np.isfinite(rl10) & np.isfinite(rl100)
            # GPD return levels are monotone in T by construction; a strict test trips on
            # float noise where xi << 0 collapses the curve (BC clamp), so allow MONO_TOL.
            viol = int(np.sum(ok & ~((rl1 <= rl10 + MONO_TOL) & (rl10 <= rl100 + MONO_TOL))))
            degenerate = int(np.sum(ok & (rl100 - rl1 < DEGENERATE_SPREAD)))
            violations_total += viol
            print(f"  [{product}/{window}] monotonicity violations: {viol} (n ok cells = {int(ok.sum())}); "
                  f"degenerate cells (rl_100 - rl_1 < {DEGENERATE_SPREAD} m/s): {degenerate}")

            if not np.isfinite(rl1).any():
                all_nan_maps.append(path.name)
                print(f"  WARNING: {path.name} rl_1 is all-NaN")

            vals = n_years[np.isfinite(n_years)]
            if vals.size:
                print(f"  [{product}/{window}] n_years distribution: min={vals.min():.0f} "
                      f"p25={np.percentile(vals, 25):.0f} median={np.median(vals):.0f} "
                      f"p75={np.percentile(vals, 75):.0f} max={vals.max():.0f}")
            expected = w.FULL_WY[product][1] - w.FULL_WY[product][0] + 1 if window == "full" else \
                w.COMMON_WY[1] - w.COMMON_WY[0] + 1
            print(f"  [{product}/{window}] expected n_years (nominal) = {expected}")

    station_csv = w.RESULT_DIR / "station_rl.csv"
    if station_csv.exists():
        df = pd.read_csv(station_csv)
        ok_rows = df[df["status"] == "ok"].dropna(subset=["rl_1", "rl_10", "rl_100"])
        viol_station = int((~((ok_rows["rl_1"] <= ok_rows["rl_10"] + MONO_TOL)
                              & (ok_rows["rl_10"] <= ok_rows["rl_100"] + MONO_TOL))).sum())
        violations_total += viol_station
        print(f"  [station_rl.csv] monotonicity violations: {viol_station} (n ok rows = {len(ok_rows)})")

    assert violations_total == 0, f"{violations_total} monotonicity violations found"
    assert not all_nan_maps, f"all-NaN maps: {all_nan_maps}"
    print("checks passed")


# === MAIN ===

def main() -> None:
    t_start = time.time()
    w.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if TIME_ONLY_PRODUCTS:
        for product in TIME_ONLY_PRODUCTS:
            if not product_ready(product):
                print(f"[{product}] cache not ready yet, cannot time")
                continue
            for window in WINDOWS:
                time_20x20(product, window)
        print(f"timing-only run done in {time.time() - t_start:.1f} s")
        return

    print(f"PRODUCTS_TO_FIT = {PRODUCTS_TO_FIT}")
    available = [p for p in PRODUCTS_TO_FIT if product_ready(p)]
    unavailable = [p for p in PRODUCTS_TO_FIT if p not in available]
    print(f"available now   = {available}")
    if unavailable:
        print(f"NOT YET READY, skipped this run: {unavailable}")

    timings = []
    for product in available:
        for window in WINDOWS:
            timings.append(fit_grid(product, window))

    station_table = pd.read_csv(w.RESULT_DIR / "station_table.csv")
    obs_cache = load_obs_cache()
    print("\n=== Obs full-record fits + homogeneity ===")
    obs_full_results, dmax_all, days, nvalid_all, valid_dict = compute_obs_full_results(station_table, obs_cache)

    print("\n=== Station fits ===")
    for product in available:
        part_path = w.RESULT_DIR / f"station_rl_part_{product}.csv"
        if part_path.exists() and not (OVERWRITE_EXISTING or OVERWRITE_STATIONS):
            print(f"[{product}] station_rl part already exists, skipping ({part_path})")
            continue
        run_station_fits(product, obs_cache, obs_full_results, station_table)

    merge_station_rl_parts()
    run_homogeneity(station_table, dmax_all, days, valid_dict, obs_full_results, obs_cache)

    print("\n=== Grid fit timings ===")
    for info in timings:
        status = "skipped (exists)" if info.get("skipped") else f"{info['seconds']:.1f} s"
        print(f"  {info['product']}/{info['window']}: {status}")

    sanity_checks(available)
    print(f"\ns3_fit_return_levels: done in {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()

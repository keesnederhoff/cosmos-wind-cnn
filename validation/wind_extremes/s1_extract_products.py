"""Stage 1: stream ERA5 / RTMA / CNN-raw / CNN-BC one water year at a time into
cached daily-maximum cubes and station hourly series.

Resumable: a WY is skipped if its tmp npz already exists. Final NetCDFs are
written only once every WY of a product is present and SMOKE_WY is None.
"""

# === CONFIGURATION ===
SMOKE_WY = None                 # int (e.g. 2010) to process a single WY and skip finalize, or None for production
PRODUCTS_TO_RUN = ["era5", "rtma", "cnn_raw", "cnn_bc", "cnn_q82", "cnn_q92", "cnn_ba"]   # finished products are skipped

# === IMPORTS ===
import re
import shutil
import time

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

import windx_common as w


# === TIME AXIS HELPERS ===

def _time_hours_int(ds: netCDF4.Dataset) -> tuple[np.ndarray, pd.Timestamp]:
    """Read the `time` variable of an open dataset as (int64 hours-since-epoch, epoch)."""
    tvar = ds.variables["time"]
    m = re.match(r"hours since (.+)", tvar.units)
    if not m:
        raise ValueError(f"unexpected time units: {tvar.units!r}")
    epoch = pd.Timestamp(m.group(1).strip())
    raw = np.asarray(tvar[:], dtype=np.float64)
    if not np.allclose(raw, np.round(raw)):
        raise ValueError(f"time axis is not integer hours: {tvar.units!r}")
    return np.round(raw).astype(np.int64), epoch


def _wy_hour_axis(wy: int) -> pd.DatetimeIndex:
    """Hourly UTC axis for one water year: 1 Oct 00:00 (wy-1) .. 30 Sep 23:00 (wy)."""
    start = pd.Timestamp(year=wy - 1, month=w.WY_START_MONTH, day=1)
    end = pd.Timestamp(year=wy, month=w.WY_START_MONTH, day=1)
    return pd.date_range(start, end, freq="h", inclusive="left")


def _hours_since(times: pd.DatetimeIndex, epoch: pd.Timestamp) -> np.ndarray:
    hours = (times - epoch) / pd.Timedelta(hours=1)
    return np.round(hours.to_numpy()).astype(np.int64)


def _to_ns_i8(idx: pd.DatetimeIndex) -> np.ndarray:
    """int64 nanoseconds since epoch, forced to ns regardless of the index's
    native resolution (pandas >= 2 defaults DatetimeIndex to 'us' or 's')."""
    return np.asarray(idx.values, dtype="datetime64[ns]").view(np.int64)


def _station_cells(x1d: np.ndarray, y1d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nearest_cell for every station, on this product's grid. Computed once per product."""
    stations = w.station_locations()
    iy = np.empty(len(stations), dtype=np.int64)
    ix = np.empty(len(stations), dtype=np.int64)
    dist = np.empty(len(stations), dtype=np.float64)
    for i, row in enumerate(stations.itertuples()):
        iy[i], ix[i], dist[i] = w.nearest_cell(x1d, y1d, row.x, row.y)
    return iy, ix, dist


def _grid_xy(product: str) -> tuple[np.ndarray, np.ndarray]:
    if product in w.CNN_FILE_BY_PRODUCT:
        path = w.CNN_FILE_BY_PRODUCT[product]
    else:
        path = {"rtma": w.RTMA_WS, "era5": w.ERA5_U}[product]
    with netCDF4.Dataset(path) as ds:
        x1d = np.asarray(ds.variables["x"][:], dtype=float)
        y1d = np.asarray(ds.variables["y"][:], dtype=float)
    return x1d, y1d


def _place_on_axis(sub_h: np.ndarray, target_h0: int, n_times: int, raw: np.ndarray) -> np.ndarray:
    """Scatter file rows `raw` (aligned with `sub_h`) onto a full-length NaN field,
    by integer-hour offset from `target_h0`. Hours missing from the file stay NaN."""
    field = np.full((n_times,) + raw.shape[1:], np.nan, dtype=np.float32)
    pos = (sub_h - target_h0).astype(np.int64)
    ok = (pos >= 0) & (pos < n_times)
    field[pos[ok]] = raw[ok]
    return field


# === READER GENERATOR ===

def iter_product_wy(product: str, wy_list: list[int]):
    """Yield (wy, times, field[T,ny,nx] float32, x1d, y1d, st_iy, st_ix, st_dist)
    for each WY in `wy_list`, in order. Opens the source file(s) once and reuses
    them across WYs. Hours absent from the source file are left NaN."""
    if product in w.CNN_FILE_BY_PRODUCT:
        path = w.CNN_FILE_BY_PRODUCT[product]
        q_idx = w.CNN_Q_INDEX_BY_PRODUCT[product]
        ds = netCDF4.Dataset(path)
        try:
            x1d = np.asarray(ds.variables["x"][:], dtype=float)
            y1d = np.asarray(ds.variables["y"][:], dtype=float)
            H, epoch = _time_hours_int(ds)
            st_iy, st_ix, st_dist = _station_cells(x1d, y1d)
            for wy in wy_list:
                times = _wy_hour_axis(wy)
                target_h = _hours_since(times, epoch)
                i0 = int(np.searchsorted(H, target_h[0]))
                i1 = int(np.searchsorted(H, target_h[-1], side="right"))
                sub_h = H[i0:i1]
                if q_idx is None:          # P50-only files (cnn_ba) have no quantile axis
                    raw = ds.variables["wind_speed"][i0:i1, :, :]
                else:
                    raw = ds.variables["wind_speed"][i0:i1, q_idx, :, :]
                raw = np.ma.filled(raw, np.nan).astype(np.float32)
                field = _place_on_axis(sub_h, target_h[0], len(times), raw)
                yield wy, times, field, x1d, y1d, st_iy, st_ix, st_dist
        finally:
            ds.close()

    elif product == "rtma":
        ds = netCDF4.Dataset(w.RTMA_WS)
        try:
            x1d = np.asarray(ds.variables["x"][:], dtype=float)
            y1d = np.asarray(ds.variables["y"][:], dtype=float)
            H, epoch = _time_hours_int(ds)
            st_iy, st_ix, st_dist = _station_cells(x1d, y1d)
            for wy in wy_list:
                times = _wy_hour_axis(wy)
                target_h = _hours_since(times, epoch)
                i0 = int(np.searchsorted(H, target_h[0]))
                i1 = int(np.searchsorted(H, target_h[-1], side="right"))
                sub_h = H[i0:i1]
                raw = ds.variables["wind_speed"][i0:i1, :, :]
                raw = np.ma.filled(raw, np.nan).astype(np.float32)
                field = _place_on_axis(sub_h, target_h[0], len(times), raw)
                yield wy, times, field, x1d, y1d, st_iy, st_ix, st_dist
        finally:
            ds.close()

    elif product == "era5":
        ds_u = netCDF4.Dataset(w.ERA5_U)
        ds_v = netCDF4.Dataset(w.ERA5_V)
        try:
            x1d = np.asarray(ds_u.variables["x"][:], dtype=float)
            y1d = np.asarray(ds_u.variables["y"][:], dtype=float)
            Hu, epoch_u = _time_hours_int(ds_u)
            Hv, epoch_v = _time_hours_int(ds_v)
            if epoch_u != epoch_v or not np.array_equal(Hu, Hv):
                raise ValueError("ERA5 U/V time axes differ")
            H, epoch = Hu, epoch_u
            st_iy, st_ix, st_dist = _station_cells(x1d, y1d)
            for wy in wy_list:
                times = _wy_hour_axis(wy)
                target_h = _hours_since(times, epoch)
                i0 = int(np.searchsorted(H, target_h[0]))
                i1 = int(np.searchsorted(H, target_h[-1], side="right"))
                sub_h = H[i0:i1]
                u = np.ma.filled(ds_u.variables["eastward_wind"][i0:i1, :, :], np.nan)
                v = np.ma.filled(ds_v.variables["northward_wind"][i0:i1, :, :], np.nan)
                raw = np.hypot(u, v).astype(np.float32)
                field = _place_on_axis(sub_h, target_h[0], len(times), raw)
                yield wy, times, field, x1d, y1d, st_iy, st_ix, st_dist
        finally:
            ds_u.close()
            ds_v.close()

    else:
        raise ValueError(f"unknown product: {product}")


# === PER-WY LOOP + FINALIZE ===

def process_product(product: str) -> None:
    print(f"=== {product} ===", flush=True)
    wy_lo, wy_hi = w.FULL_WY[product]
    all_wy = [SMOKE_WY] if SMOKE_WY is not None else list(range(wy_lo, wy_hi + 1))
    tmp_dir = w.CACHE_DIR / f"tmp_{product}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pending = [wy for wy in all_wy if not (tmp_dir / f"WY{wy}.npz").exists()]
    if pending:
        for wy, times, field, x1d, y1d, st_iy, st_ix, st_dist in iter_product_wy(product, pending):
            t0 = time.time()
            dmax, days, nvalid = w.daily_max(field, times)
            st_hourly = field[:, st_iy, st_ix].T  # (n_station, T)
            np.savez(
                tmp_dir / f"WY{wy}.npz",
                dmax=dmax.astype(np.float32),
                nvalid=nvalid,
                days=_to_ns_i8(days),
                st_hourly=st_hourly.astype(np.float32),
                times=_to_ns_i8(times),
            )
            print(f"{product} WY{wy} done in {time.time() - t0:.1f} s", flush=True)
    else:
        print(f"{product}: all {len(all_wy)} requested WY already cached", flush=True)

    if SMOKE_WY is None:
        _maybe_finalize(product, wy_lo, wy_hi, tmp_dir)


def _maybe_finalize(product: str, wy_lo: int, wy_hi: int, tmp_dir) -> None:
    all_wy = list(range(wy_lo, wy_hi + 1))
    npz_paths = [tmp_dir / f"WY{wy}.npz" for wy in all_wy]
    missing = [wy for wy, p in zip(all_wy, npz_paths) if not p.exists()]
    if missing:
        print(f"{product}: not finalizing, missing WY {missing}", flush=True)
        return

    dailymax_path = w.CACHE_DIR / f"dailymax_{product}.nc"
    stations_path = w.CACHE_DIR / f"stations_hourly_{product}.nc"
    if dailymax_path.exists() and stations_path.exists():
        print(f"{product}: final caches already exist, skipping finalize", flush=True)
        return

    print(f"{product}: finalizing {len(all_wy)} WY ...", flush=True)
    x1d, y1d = _grid_xy(product)
    stations = w.station_locations()
    st_iy, st_ix, st_dist = _station_cells(x1d, y1d)

    days_list, dmax_list, nvalid_list, times_list, sth_list = [], [], [], [], []
    for p in npz_paths:
        # Context manager closes the npz; an open handle blocks rmtree on Windows.
        with np.load(p) as z:
            days_list.append(z["days"])
            dmax_list.append(z["dmax"])
            nvalid_list.append(z["nvalid"])
            times_list.append(z["times"])
            sth_list.append(z["st_hourly"])

    days_all = np.concatenate(days_list).view("datetime64[ns]")
    dmax_all = np.concatenate(dmax_list, axis=0)
    nvalid_all = np.concatenate(nvalid_list, axis=0)
    times_all = np.concatenate(times_list).view("datetime64[ns]")
    st_hourly_all = np.concatenate(sth_list, axis=1)
    n_days_expected = len(days_all)
    ny, nx = len(y1d), len(x1d)

    ds_day = xr.Dataset(
        {
            "dmax": (("day", "y", "x"), dmax_all),
            "nvalid": (("day", "y", "x"), nvalid_all),
        },
        coords={"day": days_all, "y": y1d, "x": x1d},
        attrs={"product": product, "wy_range": f"{wy_lo}-{wy_hi}", "min_hours_per_day": w.MIN_HOURS_PER_DAY},
    )
    chunk_day = min(366, n_days_expected)
    encoding = {
        "dmax": {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (chunk_day, ny, nx)},
        "nvalid": {"zlib": True, "complevel": 4, "dtype": "uint8", "chunksizes": (chunk_day, ny, nx)},
    }
    ds_day.to_netcdf(dailymax_path, encoding=encoding)
    ds_day.close()

    ds_st = xr.Dataset(
        {
            "ws": (("station", "time"), st_hourly_all),
            "iy": (("station",), st_iy),
            "ix": (("station",), st_ix),
            "dist_m": (("station",), st_dist),
            "station_id": (("station",), stations["station_id"].to_numpy()),
            "network": (("station",), stations["network"].to_numpy()),
        },
        coords={"time": times_all},
        attrs={"product": product, "wy_range": f"{wy_lo}-{wy_hi}"},
    )
    enc_st = {"ws": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    ds_st.to_netcdf(stations_path, encoding=enc_st)
    ds_st.close()

    # Reopen-check the day length before deleting the tmp dir.
    with netCDF4.Dataset(dailymax_path) as chk:
        n_day_file = chk.dimensions["day"].size
    if n_day_file == n_days_expected:
        # tmp dir is left in place: recursive deletes on g: are left to Kees.
        print(f"{product}: finalize OK, {n_day_file} days; {tmp_dir} can now be deleted", flush=True)
    else:
        print(f"{product}: WARNING day length mismatch ({n_day_file} != {n_days_expected}); {tmp_dir} KEPT", flush=True)


if __name__ == "__main__":
    for product in PRODUCTS_TO_RUN:
        process_product(product)
    print("s1_extract_products: done", flush=True)

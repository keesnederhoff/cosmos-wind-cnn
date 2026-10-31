"""Stage 2: observations - QC, hourly means, validity, top-peak review table.

Reads IEM/NDBC/USGS raw archives, builds one common hourly series per station,
writes cache/obs_hourly.nc and results/station_table.csv, then (Step 7) a
top-peak review table at results/qc_top_peaks.csv. Product columns in that
table are filled only once the Stage-1 station caches exist; see Task 6 of
validation/plans/2026-10-02-wind-return-levels.md.
"""

# === CONFIGURATION ===
ONLY_STEP7 = False   # True: skip Steps 1-6, read cache/obs_hourly.nc + results/station_table.csv, run Step 7 only
TOP_N_PEAKS = 10     # declustered peaks per station in qc_top_peaks.csv

# === IMPORTS ===
import time

import numpy as np
import pandas as pd
import netCDF4

import evt as rx
import windx_common as w

HOURLY_START = pd.Timestamp("1914-05-01 00:00:00")
HOURLY_END = pd.Timestamp("2026-06-30 23:00:00")
HOURLY_AXIS = pd.date_range(HOURLY_START, HOURLY_END, freq="h")

STAGE1_CACHES = {
    "era5": w.CACHE_DIR / "stations_hourly_era5.nc",
    "cnn_raw": w.CACHE_DIR / "stations_hourly_cnn_raw.nc",
    "rtma": w.CACHE_DIR / "stations_hourly_rtma.nc",
}

OBS_HOURLY_NC = w.CACHE_DIR / "obs_hourly.nc"
STATION_TABLE_CSV = w.RESULT_DIR / "station_table.csv"
QC_TOP_PEAKS_CSV = w.RESULT_DIR / "qc_top_peaks.csv"


# === SMALL HELPERS ===

def _to_float_filled(x) -> np.ndarray:
    """netCDF4 slice (plain or masked array) -> float ndarray with NaN for missing."""
    if np.ma.isMaskedArray(x):
        x = x.filled(np.nan)
    return np.asarray(x, dtype=float)


def _decode_hours_since(tvar) -> pd.DatetimeIndex:
    """Decode an "hours since <base>" time variable without cftime (base is well
    within pandas/Gregorian range for every file this script touches)."""
    units = tvar.units
    base_str = units.split("since", 1)[1].strip()
    base = pd.Timestamp(base_str)
    hours = np.asarray(tvar[:], dtype=np.float64)
    return base + pd.to_timedelta(hours, unit="h")


def _decode_usgs_minutes(raw_minutes: np.ndarray) -> pd.DatetimeIndex:
    """USGS time is minutes since 1970-01-01 with float jitter; round to the second."""
    seconds = np.round(np.asarray(raw_minutes, dtype=np.float64) * 60.0)
    return pd.Timestamp("1970-01-01") + pd.to_timedelta(seconds, unit="s")


def qc_cap(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Readings outside [0, WS_MAX_RAW] -> NaN. Returns (clean, n_removed)."""
    bad = np.isfinite(raw) & ((raw < 0.0) | (raw > w.WS_MAX_RAW))
    out = raw.copy()
    out[bad] = np.nan
    return out, int(bad.sum())


def spike_test(hourly_full: pd.Series) -> tuple[pd.Series, int]:
    """Flag hour > SPIKE_FACTOR * mean(prev, next) AND > SPIKE_MIN -> NaN.

    If one neighbour is NaN the other alone is used (plain mean with skipna);
    if both are NaN the mean is NaN and the comparison is False, so the value
    is never flagged.
    """
    prev = hourly_full.shift(1)
    nxt = hourly_full.shift(-1)
    neighbor_mean = pd.concat([prev, nxt], axis=1).mean(axis=1, skipna=True)
    flag = (hourly_full > w.SPIKE_FACTOR * neighbor_mean) & (hourly_full > w.SPIKE_MIN) & neighbor_mean.notna()
    out = hourly_full.copy()
    out[flag] = np.nan
    return out, int(flag.sum())


def persistence_test(hourly_full: pd.Series) -> tuple[pd.Series, int]:
    """Hour > PERSIST_MIN whose higher neighbour is < PERSIST_FRAC x value -> NaN.

    Neighbour = nearest finite hour within 2 h on each side (gaps are common in
    ASOS records); an hour with no finite neighbour within 2 h on either side is
    an unsupported single report and is flagged too. Catches the 20.00/25.00 m/s
    doubled values and single-hour spikes that the 3x-mean spike test lets through.
    """
    prev = hourly_full.shift(1).fillna(hourly_full.shift(2))
    nxt = hourly_full.shift(-1).fillna(hourly_full.shift(-2))
    nb_max = pd.concat([prev, nxt], axis=1).max(axis=1, skipna=True)
    unsupported = nb_max.isna() | (nb_max < w.PERSIST_FRAC * hourly_full)
    flag = (hourly_full > w.PERSIST_MIN) & unsupported
    out = hourly_full.copy()
    out[flag] = np.nan
    return out, int(flag.sum())


def apply_qc_exclude(hourly_full: pd.Series, station_id: str) -> pd.Series:
    """Set hours listed in QC_EXCLUDE[station_id] ('YYYY-MM-DD HH' strings) to NaN."""
    hours = w.QC_EXCLUDE.get(station_id, [])
    if not hours:
        return hourly_full
    out = hourly_full.copy()
    for hstr in hours:
        ts = pd.to_datetime(hstr, format="%Y-%m-%d %H")
        if ts in out.index:
            out.loc[ts] = np.nan
        else:
            print(f"  WARNING: QC_EXCLUDE hour {hstr} for {station_id} not on the hourly axis")
    return out


def process_station(station_id: str, times: pd.DatetimeIndex, raw_ws: np.ndarray) -> tuple[pd.Series, int, int]:
    """Raw readings -> cap QC -> hourly mean -> spike test -> QC_EXCLUDE, reindexed to HOURLY_AXIS."""
    qc_ws, n_raw_removed = qc_cap(raw_ws)
    hourly_mean = w.hourly_mean_centred(times, qc_ws)
    hourly_full = hourly_mean.reindex(HOURLY_AXIS)
    hourly_full, n_spike_removed = spike_test(hourly_full)
    hourly_full, n_persist_removed = persistence_test(hourly_full)
    n_spike_removed += n_persist_removed          # station_table column counts both tests
    hourly_full = apply_qc_exclude(hourly_full, station_id)
    return hourly_full, n_raw_removed, n_spike_removed


# === LOADERS (Step 1) ===

def load_iem() -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    out = {}
    with netCDF4.Dataset(w.IEM_FILE) as ds:
        sid_list = w._decode_station_ids(ds.variables["station_id"][:])
        times = _decode_hours_since(ds.variables["time"])
        for i, sid in enumerate(sid_list):
            raw = _to_float_filled(ds.variables["wind_speed"][i, :])
            out[sid] = (times, raw)
    return out


def load_ndbc() -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    out = {}
    with netCDF4.Dataset(w.NDBC_FILE) as ds:
        sid_list = w._decode_station_ids(ds.variables["station_id"][:])
        times = _decode_hours_since(ds.variables["time"])
        for i, sid in enumerate(sid_list):
            raw = _to_float_filled(ds.variables["wind_speed"][i, :])
            out[sid] = (times, raw)
    return out


def load_usgs(station_id: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    fpath, _lat, _lon, height_m = w.USGS_FILES[station_id]
    with netCDF4.Dataset(fpath) as ds:
        raw_minutes = _to_float_filled(ds.variables["time"][:])
        times = _decode_usgs_minutes(raw_minutes)
        raw_ws = _to_float_filled(ds.variables["wind_speed"][:])
    scale = np.log(10.0 / w.Z0_WATER) / np.log(height_m / w.Z0_WATER)
    return times, raw_ws * scale


# === STEPS 1-6 ===

def run_steps_1_to_6() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DatetimeIndex, dict]:
    df = w.station_locations()
    n_station = len(df)
    print(f"Step 1: station_locations -> {n_station} stations "
          f"({df['network'].value_counts().to_dict()})")

    iem_data = load_iem()
    ndbc_data = load_ndbc()

    n_hours = len(HOURLY_AXIS)
    ws_matrix = np.full((n_station, n_hours), np.nan, dtype=np.float32)
    n_raw_removed = np.zeros(n_station, dtype=int)
    n_spike_removed = np.zeros(n_station, dtype=int)

    print("Steps 2-3: QC cap, hourly mean, spike test, QC_EXCLUDE")
    for i, row in df.iterrows():
        sid, net = row["station_id"], row["network"]
        if net == "IEM":
            times, raw = iem_data[sid]
        elif net == "NDBC":
            times, raw = ndbc_data[sid]
        elif net == "USGS":
            times, raw = load_usgs(sid)
        else:
            raise ValueError(f"unknown network {net}")

        hourly_full, n_raw_rm, n_spike_rm = process_station(sid, times, raw)
        ws_matrix[i, :] = hourly_full.to_numpy(dtype=np.float32)
        n_raw_removed[i] = n_raw_rm
        n_spike_removed[i] = n_spike_rm
        print(f"  {sid:10s} ({net:4s}) raw_removed={n_raw_rm:7d}  spike_removed={n_spike_rm:5d}")

    df["n_raw_removed"] = n_raw_removed
    df["n_spike_removed"] = n_spike_removed

    print("Step 4: daily_max + valid_water_years")
    dmax, days, _nvalid = w.daily_max(ws_matrix.T, HOURLY_AXIS)  # dmax shape (D, n_station)
    valid_dict = w.valid_water_years(days, dmax)                 # {wy: bool array(n_station)}

    wys_in_range = sorted(y for y in valid_dict if y <= w.OBS_MAX_WY)
    n_valid_wy = np.zeros(n_station, dtype=int)
    first_wy = np.full(n_station, np.nan)
    last_wy = np.full(n_station, np.nan)
    for y in wys_in_range:
        v = np.asarray(valid_dict[y])
        n_valid_wy += v.astype(int)
        first_wy = np.where(v & np.isnan(first_wy), y, first_wy)
        last_wy = np.where(v, y, last_wy)

    df["n_valid_wy"] = n_valid_wy
    df["first_wy"] = first_wy
    df["last_wy"] = last_wy
    df["status"] = np.where(df["n_valid_wy"] >= w.MIN_YEARS, "ok", "dropped_<6yr")
    with np.errstate(all="ignore"):
        df["max_hourly"] = np.nanmax(ws_matrix, axis=1)

    print("Step 5: writing cache/obs_hourly.nc")
    write_obs_hourly(df, ws_matrix)

    print("Step 6: writing results/station_table.csv")
    write_station_table(df)

    return df, ws_matrix, dmax, days, valid_dict


def write_obs_hourly(df: pd.DataFrame, ws_matrix: np.ndarray) -> None:
    n_station = len(df)
    n_hours = len(HOURLY_AXIS)
    with netCDF4.Dataset(OBS_HOURLY_NC, "w", format="NETCDF4") as ds:
        ds.createDimension("station", n_station)
        ds.createDimension("time", n_hours)

        t = ds.createVariable("time", "i8", ("time",))
        t.units = f"hours since {HOURLY_START.strftime('%Y-%m-%d %H:%M:%S')}"
        t.calendar = "proleptic_gregorian"
        t[:] = np.arange(n_hours, dtype=np.int64)

        ws_var = ds.createVariable("ws", "f4", ("station", "time"), zlib=True, complevel=4,
                                    chunksizes=(n_station, min(n_hours, 17520)))
        ws_var.units = "m s-1"
        ws_var.long_name = "QC'd hourly-mean wind speed at 10 m"
        ws_var[:, :] = ws_matrix

        sid_var = ds.createVariable("station_id", str, ("station",))
        sid_var[:] = np.asarray(df["station_id"], dtype=object)

        net_var = ds.createVariable("network", str, ("station",))
        net_var[:] = np.asarray(df["network"], dtype=object)

        for col in ("lat", "lon", "x", "y"):
            v = ds.createVariable(col, "f8", ("station",))
            v[:] = df[col].to_numpy(dtype=np.float64)

    size_mb = OBS_HOURLY_NC.stat().st_size / 1e6
    print(f"  wrote {OBS_HOURLY_NC} ({size_mb:.1f} MB)")


def write_station_table(df: pd.DataFrame) -> None:
    cols = ["station_id", "network", "lat", "lon", "x", "y", "n_raw_removed", "n_spike_removed",
            "n_valid_wy", "first_wy", "last_wy", "status", "max_hourly"]
    out = df[cols].copy()
    out.to_csv(STATION_TABLE_CSV, index=False)
    print(f"  wrote {STATION_TABLE_CSV} ({len(out)} rows)")


# === STEP 7: top-peak review table ===

def load_obs_hourly_cache() -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    with netCDF4.Dataset(OBS_HOURLY_NC) as ds:
        station_id = w._decode_station_ids(ds.variables["station_id"][:])
        network = w._decode_station_ids(ds.variables["network"][:])
        lat = _to_float_filled(ds.variables["lat"][:])
        lon = _to_float_filled(ds.variables["lon"][:])
        x = _to_float_filled(ds.variables["x"][:])
        y = _to_float_filled(ds.variables["y"][:])
        hourly_axis = _decode_hours_since(ds.variables["time"])
        ws_matrix = _to_float_filled(ds.variables["ws"][:, :])
    df = pd.DataFrame({"station_id": station_id, "network": network, "lat": lat, "lon": lon, "x": x, "y": y})
    return df, ws_matrix, hourly_axis


def load_stage1_product(path) -> tuple[dict[str, int], pd.DatetimeIndex, np.ndarray]:
    with netCDF4.Dataset(path) as ds:
        sid_list = w._decode_station_ids(ds.variables["station_id"][:])
        times = _decode_hours_since(ds.variables["time"])
        ws = _to_float_filled(ds.variables["ws"][:, :])
    sid_index = {s: i for i, s in enumerate(sid_list)}
    return sid_index, times, ws


def _lookup_product(product_cache, station_id: str, ts: pd.Timestamp) -> float:
    sid_index, times, ws = product_cache
    row = sid_index.get(station_id)
    if row is None:
        return np.nan
    try:
        pos = times.get_loc(ts)
    except KeyError:
        return np.nan
    return float(ws[row, pos])


def run_step7(df: pd.DataFrame, ws_matrix: np.ndarray, dmax: np.ndarray, days: pd.DatetimeIndex,
              valid_dict: dict, hourly_axis: pd.DatetimeIndex) -> None:
    caches_exist = all(p.exists() for p in STAGE1_CACHES.values())
    product_data = {}
    if caches_exist:
        for prod, path in STAGE1_CACHES.items():
            product_data[prod] = load_stage1_product(path)
        print("Step 7: stage-1 caches found, filling product columns")
    else:
        print("skipping qc_top_peaks: stage-1 caches missing")

    n_station, n_days = dmax.shape[1], dmax.shape[0]
    wy_of_days = w.water_year(days)

    # per-station, per-day validity mask (True only inside a valid WY <= OBS_MAX_WY)
    valid_day_2d = np.zeros((n_station, n_days), dtype=bool)
    for y, v in valid_dict.items():
        if y > w.OBS_MAX_WY:
            continue
        sel = wy_of_days == y
        valid_day_2d[:, sel] = np.asarray(v, dtype=bool)[:, None]

    dmax_station_major = dmax.T  # (n_station, n_days)
    ws_by_station = {i: pd.Series(ws_matrix[i, :], index=hourly_axis) for i in range(n_station)}

    rows = []
    for i, row in df.iterrows():
        if row.get("status") != "ok":
            continue
        sid = row["station_id"]
        masked = np.where(valid_day_2d[i, :], dmax_station_major[i, :], np.nan)
        series = pd.Series(masked, index=days)
        try:
            peaks = rx.decluster_peaks(series, w.SEP_HOURS, max_peaks=TOP_N_PEAKS)
        except ValueError:
            print(f"  WARNING: no peaks found for ok station {sid}")
            continue
        peaks_sorted = peaks.sort_values(ascending=False)

        hourly_series = ws_by_station[i]
        for rank, (day, obs_val) in enumerate(peaks_sorted.items(), start=1):
            day_end = day + pd.Timedelta(hours=23)
            sub = hourly_series.loc[day:day_end]
            t_peak = sub.idxmax() if sub.notna().any() else day

            era5_val = cnn_raw_val = rtma_val = np.nan
            if caches_exist:
                era5_val = _lookup_product(product_data["era5"], sid, t_peak)
                cnn_raw_val = _lookup_product(product_data["cnn_raw"], sid, t_peak)
                rtma_val = _lookup_product(product_data["rtma"], sid, t_peak)

            ratio_obs_era5 = obs_val / era5_val if np.isfinite(era5_val) and era5_val != 0 else np.nan
            review_flag = bool(np.isfinite(era5_val) and (obs_val > w.REVIEW_RATIO * era5_val) and (obs_val > 20.0))
            flag_no_ref = bool(not np.isfinite(era5_val) and (obs_val > 25.0))

            rows.append({
                "station_id": sid, "rank": rank, "time": t_peak, "obs": obs_val,
                "era5": era5_val, "cnn_raw": cnn_raw_val, "rtma": rtma_val,
                "ratio_obs_era5": ratio_obs_era5, "review_flag": review_flag,
                "flag_no_ref": flag_no_ref,
            })

    out = pd.DataFrame(rows)
    out.to_csv(QC_TOP_PEAKS_CSV, index=False)
    print(f"  wrote {QC_TOP_PEAKS_CSV} ({len(out)} rows)")


def run_step7_only() -> None:
    if not STATION_TABLE_CSV.exists():
        raise FileNotFoundError(f"ONLY_STEP7 needs {STATION_TABLE_CSV} - run s2_prepare_obs.py fully first")
    if not OBS_HOURLY_NC.exists():
        raise FileNotFoundError(f"ONLY_STEP7 needs {OBS_HOURLY_NC} - run s2_prepare_obs.py fully first")

    status_table = pd.read_csv(STATION_TABLE_CSV)[["station_id", "status"]]
    df_raw, ws_matrix, hourly_axis = load_obs_hourly_cache()
    df = df_raw.merge(status_table, on="station_id", how="left")

    dmax, days, _nvalid = w.daily_max(ws_matrix.T, hourly_axis)
    valid_dict = w.valid_water_years(days, dmax)

    run_step7(df, ws_matrix, dmax, days, valid_dict, hourly_axis)


# === MAIN ===

def main():
    t0 = time.time()
    if ONLY_STEP7:
        print("ONLY_STEP7 = True: skipping Steps 1-6, reading cache/obs_hourly.nc")
        run_step7_only()
    else:
        df, ws_matrix, dmax, days, valid_dict = run_steps_1_to_6()
        run_step7(df, ws_matrix, dmax, days, valid_dict, HOURLY_AXIS)
    print(f"s2_prepare_obs done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

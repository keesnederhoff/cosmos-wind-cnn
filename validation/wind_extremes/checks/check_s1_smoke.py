"""Stage 1 smoke gate (Task 4, Step 5): verify the cnn_raw WY2010 npz against a
direct read of CNN_RAW (v4 P50 file, no quantile axis). Both checks must print max|diff| = 0.000, or Stage 1 is
wrong and must be debugged before the production run."""

import netCDF4
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import windx_common as w

NPZ = w.CACHE_DIR / "tmp_cnn_raw" / "WY2010.npz"


def check_station_series() -> float:
    """OAK cached hourly series for 2010-01-10..2010-01-24 vs a direct CNN_RAW read."""
    stations = w.station_locations()
    row = stations.loc[stations["station_id"] == "OAK"].iloc[0]

    with netCDF4.Dataset(w.CNN_RAW) as ds:
        x1d = np.asarray(ds.variables["x"][:], dtype=float)
        y1d = np.asarray(ds.variables["y"][:], dtype=float)
        iy, ix, dist = w.nearest_cell(x1d, y1d, row.x, row.y)
        print(f"OAK -> iy={iy} ix={ix} dist_m={dist:.1f}")

        # hours-since-1940 index of 2010-01-10 00:00 and 2010-01-24 23:00 (inclusive)
        t0 = int((np.datetime64("2010-01-10T00:00:00") - np.datetime64("1940-01-01T00:00:00")) / np.timedelta64(1, "h"))
        t1 = int((np.datetime64("2010-01-24T23:00:00") - np.datetime64("1940-01-01T00:00:00")) / np.timedelta64(1, "h"))
        time_var = np.asarray(ds.variables["time"][:], dtype=np.int64)
        i0 = int(np.searchsorted(time_var, t0))
        i1 = int(np.searchsorted(time_var, t1, side="right"))
        direct = ds.variables["wind_speed"][i0:i1, iy, ix]
        direct = np.ma.filled(direct, np.nan)
        direct_hours = time_var[i0:i1]

    z = np.load(NPZ)
    stations_order = w.station_locations()  # same order as used when the npz was written
    sidx = stations_order.index[stations_order["station_id"] == "OAK"][0]
    cached_times = z["times"].view("datetime64[ns]")
    cached_series = z["st_hourly"][sidx, :]

    epoch = np.datetime64("1940-01-01T00:00:00")
    direct_times = epoch + direct_hours.astype("timedelta64[h]")

    cached_sel = cached_series[np.isin(cached_times, direct_times)]
    cached_times_sel = cached_times[np.isin(cached_times, direct_times)]
    order = np.argsort(cached_times_sel)
    cached_sel = cached_sel[order]

    assert cached_sel.shape == direct.shape, f"shape mismatch {cached_sel.shape} vs {direct.shape}"
    diff = np.abs(cached_sel.astype(float) - direct.astype(float))
    max_diff = float(np.nanmax(diff))
    print(f"OAK station series 2010-01-10..2010-01-24: n={len(direct)} max|diff| = {max_diff:.3f}")
    return max_diff


def check_daily_max() -> float:
    """dmax for 2010-01-18 (whole grid) vs np.nanmax of the 24 direct hours."""
    with netCDF4.Dataset(w.CNN_RAW) as ds:
        time_var = np.asarray(ds.variables["time"][:], dtype=np.int64)
        t0 = int((np.datetime64("2010-01-18T00:00:00") - np.datetime64("1940-01-01T00:00:00")) / np.timedelta64(1, "h"))
        t1 = int((np.datetime64("2010-01-18T23:00:00") - np.datetime64("1940-01-01T00:00:00")) / np.timedelta64(1, "h"))
        i0 = int(np.searchsorted(time_var, t0))
        i1 = int(np.searchsorted(time_var, t1, side="right"))
        assert i1 - i0 == 24, f"expected 24 hourly rows, got {i1 - i0}"
        day_block = ds.variables["wind_speed"][i0:i1, :, :]
        day_block = np.ma.filled(day_block, np.nan)
    direct_dmax = np.nanmax(day_block, axis=0)

    z = np.load(NPZ)
    days = z["days"].view("datetime64[ns]")
    day_idx = np.where(days == np.datetime64("2010-01-18"))[0][0]
    cached_dmax = z["dmax"][day_idx, :, :]

    diff = np.abs(cached_dmax.astype(float) - direct_dmax.astype(float))
    max_diff = float(np.nanmax(diff))
    print(f"dmax 2010-01-18 whole grid: max|diff| = {max_diff:.3f}")
    return max_diff


if __name__ == "__main__":
    d1 = check_station_series()
    d2 = check_daily_max()
    print(f"GATE RESULT: station max|diff|={d1:.3f}  dmax max|diff|={d2:.3f}")
    if d1 == 0.0 and d2 == 0.0:
        print("PASS")
    else:
        print("FAIL -- debug before production run")

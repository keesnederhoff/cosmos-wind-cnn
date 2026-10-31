"""Stage 1 smoke gate for cnn_q82 / cnn_q92 (Task: extra CNN quantile products).

Mirrors check_s1_smoke.py: verify the WY2010 npz for each new product against a
direct read of CNN_RAW at that product's quantile index. Both checks must
print max|diff| = 0.000, or Stage 1 is wrong and must be debugged before the
production run.
"""

import netCDF4
import numpy as np

import windx_common as w


def check_station_series(product: str) -> float:
    """OAK cached hourly series for 2010-01-10..2010-01-24 vs a direct CNN_RAW read
    at this product's quantile index."""
    q_idx = w.CNN_Q_INDEX_BY_PRODUCT[product]
    npz_path = w.CACHE_DIR / f"tmp_{product}" / "WY2010.npz"

    stations = w.station_locations()
    row = stations.loc[stations["station_id"] == "OAK"].iloc[0]

    with netCDF4.Dataset(w.CNN_RAW) as ds:
        x1d = np.asarray(ds.variables["x"][:], dtype=float)
        y1d = np.asarray(ds.variables["y"][:], dtype=float)
        iy, ix, dist = w.nearest_cell(x1d, y1d, row.x, row.y)
        print(f"[{product}] OAK -> iy={iy} ix={ix} dist_m={dist:.1f} q_idx={q_idx}")

        t0 = int((np.datetime64("2010-01-10T00:00:00") - np.datetime64("1940-01-01T00:00:00")) / np.timedelta64(1, "h"))
        t1 = int((np.datetime64("2010-01-24T23:00:00") - np.datetime64("1940-01-01T00:00:00")) / np.timedelta64(1, "h"))
        time_var = np.asarray(ds.variables["time"][:], dtype=np.int64)
        i0 = int(np.searchsorted(time_var, t0))
        i1 = int(np.searchsorted(time_var, t1, side="right"))
        direct = ds.variables["wind_speed"][i0:i1, q_idx, iy, ix]
        direct = np.ma.filled(direct, np.nan)
        direct_hours = time_var[i0:i1]

    z = np.load(npz_path)
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
    print(f"[{product}] OAK station series 2010-01-10..2010-01-24: n={len(direct)} max|diff| = {max_diff:.3f}")
    return max_diff


if __name__ == "__main__":
    results = {}
    for product in ["cnn_q82", "cnn_q92"]:
        results[product] = check_station_series(product)
    print("GATE RESULT: " + "  ".join(f"{p} max|diff|={d:.3f}" for p, d in results.items()))
    if all(d == 0.0 for d in results.values()):
        print("PASS")
    else:
        print("FAIL -- debug before production run")

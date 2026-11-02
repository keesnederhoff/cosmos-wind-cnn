"""v3 vs v4 return levels for the CNN-RTMA products the two versions share (cnn_raw, cnn_ba).

v4 is refitted on WY1941-2025 so that its window equals v3's, with the s3 settings unchanged (lambda 3, 48 h,
L-moment GPD, same seeds): grid fit via s3.fit_grid, station fits via s3.run_station_fits, both with the
product window and OBS_MAX_WY set to 2025. The refit goes to results/v4_wy1941_2025/ and does not touch the
WY1941-2026 results. v3 comes from the archived v3 results.

The table (results/v3_vs_v4_rl.csv) has one row per product x return period x version:
grid median and range over all fitted cells (full window), and matched-mode station skill on the stations
that are 'ok' (obs and model) in BOTH versions. Obs is each version's own matched obs fit, as in s4.
cnn_q89 / cnn_q95 are not compared: they are other tau levels than v3's q82 / q92.

Run after s1 (needs cache/dailymax_cnn_{raw,ba}.nc, stations_hourly_*, obs_hourly.nc) and s2.
"""

# === CONFIGURATION ===
V3_DIR = "archive/v3/wind_extremes/results"       # relative to COSMOS_WINDX_CNN_ROOT
PRODUCTS = ["cnn_raw", "cnn_ba"]                  # products present in both v3 and v4
REFIT_WY_HI = 2025                                # v3 window end; v4 full-window refit runs WY1941..REFIT_WY_HI
REFIT_DIRNAME = "v4_wy1941_2025"                  # sub-folder of results/ for the refit
OUT_NAME = "v3_vs_v4_rl.csv"
MIN_SKILL_N = 5

# === IMPORTS ===
import sys

import netCDF4
import numpy as np
import pandas as pd

import windx_common as w
import s3_fit_return_levels as s3
from skill_utils import compute_skill_scores


def refit_v4_in_v3_window() -> None:
    """Grid + station fits of PRODUCTS on WY1941..REFIT_WY_HI, written to results/<REFIT_DIRNAME>/."""
    out_dir = w.RESULT_DIR / REFIT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    station_table = pd.read_csv(w.RESULT_DIR / "station_table.csv")    # read before RESULT_DIR is redirected
    obs_cache = s3.load_obs_cache()

    for product in PRODUCTS:
        w.FULL_WY[product] = (w.FULL_WY[product][0], REFIT_WY_HI)
    w.OBS_MAX_WY = REFIT_WY_HI
    w.RESULT_DIR = out_dir        # s3 functions read w.RESULT_DIR at call time

    for product in PRODUCTS:
        s3.fit_grid(product, "full")
    obs_full_results, *_ = s3.compute_obs_full_results(station_table, obs_cache)
    for product in PRODUCTS:
        s3.run_station_fits(product, obs_cache, obs_full_results, station_table)


def grid_stats(path, rp: int) -> dict:
    with netCDF4.Dataset(path) as ds:
        arr = s3._filled(ds.variables[f"rl_{rp}"][:, :])
    fin = arr[np.isfinite(arr)]
    return {"n_cells": int(fin.size), "grid_median": float(np.median(fin)),
            "grid_min": float(fin.min()), "grid_max": float(fin.max())}


def matched_pairs(station_rl: pd.DataFrame, product: str, rp: int) -> pd.DataFrame:
    """station_id, obs, model (RP value) for matched-mode stations with both fits 'ok'."""
    m = station_rl[(station_rl["mode"] == "matched") & (station_rl["product"] == product)
                   & (station_rl["status"] == "ok")]
    obs = m[m["source"] == "obs"].set_index("station_id")[f"rl_{rp}"].rename("obs")
    mod = m[m["source"] == "model"].set_index("station_id")[f"rl_{rp}"].rename("model")
    return pd.concat([obs, mod], axis=1, join="inner").dropna()


def main() -> None:
    v3_dir = w.CNN_ROOT / V3_DIR
    out_path = w.RESULT_DIR / OUT_NAME         # fixed before refit_v4_in_v3_window redirects w.RESULT_DIR
    refit_v4_in_v3_window()
    v4_dir = w.RESULT_DIR                      # now the refit folder

    st_v3 = pd.read_csv(v3_dir / "station_rl.csv")
    st_v4 = pd.concat([pd.read_csv(v4_dir / f"station_rl_part_{p}.csv") for p in PRODUCTS], ignore_index=True)

    rows = []
    for product in PRODUCTS:
        for rp in w.RETURN_PERIODS:
            p3, p4 = matched_pairs(st_v3, product, rp), matched_pairs(st_v4, product, rp)
            common = p3.index.intersection(p4.index)
            per_version = {}
            for version, pairs, map_dir in (("v3", p3, v3_dir), ("v4", p4, v4_dir)):
                sub = pairs.loc[common]
                sk = compute_skill_scores(sub["obs"].to_numpy(), sub["model"].to_numpy(), min_n=MIN_SKILL_N)
                row = {"product": product, "rp": rp, "version": version, "window": f"WY1941-{REFIT_WY_HI}",
                       **grid_stats(map_dir / f"rl_map_{product}_full.nc", rp),
                       "n_stations_common": len(common), "n_stations_own": len(pairs),
                       "bias": sk["bias"], "rmse": sk["rmse"], "skill": sk["skill"],
                       "median_ratio": float(np.median(sub["model"] / sub["obs"]))}
                rows.append(row)
                per_version[version] = row
            diff = {"product": product, "rp": rp, "version": "v4-v3", "window": f"WY1941-{REFIT_WY_HI}"}
            for k in ("grid_median", "grid_min", "grid_max", "bias", "rmse", "skill", "median_ratio"):
                diff[k] = per_version["v4"][k] - per_version["v3"][k]
            diff["n_stations_common"] = len(common)
            rows.append(diff)

    cols = ["product", "rp", "version", "window", "n_cells", "grid_median", "grid_min", "grid_max",
            "n_stations_common", "n_stations_own", "bias", "rmse", "skill", "median_ratio"]
    out = pd.DataFrame(rows, columns=cols).astype({"n_cells": "Int64", "n_stations_own": "Int64"})
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())

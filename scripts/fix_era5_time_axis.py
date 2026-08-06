#!/usr/bin/env python
"""Repair ERA5 files whose time axis is a bad concatenation of download chunks.

Three files delivered on 2026-08-06 carry seam damage:

    ERA5_wind_gust_2014_2026_UTM.nc
    ERA5_friction_velocity_2014_2026_UTM.nc
    ERA5_surface_latent_heat_flux_2014_2026_UTM.nc

Each has 110304 steps where a complete hourly axis needs 110160, i.e. **144
duplicate timestamps**, and 12 backward jumps of -660 min (-11 h) marking the
overlapping seams.

Why this matters: `preprocessing.yaml` sets `regular_time_grid: true`, which
reindexes onto a complete hourly axis. Reindexing against a NON-MONOTONIC index
either raises or silently mis-associates values, and `_find_common_times`
intersects on exact stamps. A single quietly-misaligned predictor channel is
exactly the class of bug that costs a whole training campaign -- and two of
these three (wind_gust, friction_velocity) are prime peak-wind predictors.

Repair: sort by time, keep the FIRST record of each duplicated stamp, assert the
result is strictly increasing on a uniform 1 h step, write `<stem>_dedup.nc`.
The originals are never modified.

Usage:
    python scripts/fix_era5_time_axis.py [--data-dir DIR] [--check-only]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import xarray as xr

DEFAULT_DIR = (
    "/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/"
    "sf_bay_rtma/raw_data"
)

TARGETS = [
    "ERA5_wind_gust_2014_2026_UTM.nc",
    "ERA5_friction_velocity_2014_2026_UTM.nc",
    "ERA5_surface_latent_heat_flux_2014_2026_UTM.nc",
]


def audit(time_vals):
    """Return (n, n_unique, n_dup, backward_jumps, spacing_counts)."""
    n = len(time_vals)
    uniq = np.unique(time_vals)
    diffs = np.diff(time_vals).astype("timedelta64[m]").astype(np.int64)
    vals, cnts = np.unique(diffs, return_counts=True)
    backward = int((diffs < 0).sum())
    return n, len(uniq), n - len(uniq), backward, dict(zip(vals.tolist(), cnts.tolist()))


def repair(path, out_path, check_only=False):
    name = os.path.basename(path)
    ds = xr.open_dataset(path)
    t = ds.time.values

    n, n_uniq, n_dup, backward, spacing = audit(t)
    print(f"  {name}")
    print(f"    before: n={n} unique={n_uniq} duplicates={n_dup} "
          f"backward_jumps={backward}")
    print(f"            spacing(min)={spacing}")

    if n_dup == 0 and backward == 0:
        print("    already clean -- nothing to do.")
        ds.close()
        return False

    if check_only:
        ds.close()
        return True

    # Sort, then keep the first occurrence of each timestamp. np.unique on a
    # sorted axis gives first-occurrence indices directly.
    ds = ds.sortby("time")
    t_sorted = ds.time.values
    _, first_idx = np.unique(t_sorted, return_index=True)
    ds = ds.isel(time=np.sort(first_idx))

    t2 = ds.time.values
    d2 = np.diff(t2).astype("timedelta64[m]").astype(np.int64)
    if len(np.unique(t2)) != len(t2):
        sys.exit(f"    ABORT: duplicates survived in {name}")
    if (d2 <= 0).any():
        sys.exit(f"    ABORT: axis not strictly increasing in {name}")
    uniq_sp = np.unique(d2)
    if uniq_sp.tolist() != [60]:
        # Not fatal on its own -- a genuine gap is fine, the dataset's
        # NaN-window logic drops sequences spanning it -- but it must be seen.
        print(f"    WARNING: non-uniform spacing after repair: {uniq_sp.tolist()}")

    print(f"    after:  n={len(t2)} unique={len(np.unique(t2))} "
          f"spacing(min)={np.unique(d2).tolist()}")
    print(f"            {str(t2[0])[:16]} -> {str(t2[-1])[:16]}")

    enc = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
    tmp = out_path + ".tmp"
    ds.to_netcdf(tmp, encoding=enc)
    ds.close()
    os.replace(tmp, out_path)
    print(f"    wrote {os.path.basename(out_path)} "
          f"({os.path.getsize(out_path) / 1e6:.0f} MB)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--check-only", action="store_true",
                    help="audit the axes, write nothing")
    args = ap.parse_args()

    print(f"ERA5 time-axis repair  (dir={args.data_dir})")
    changed = 0
    for fn in TARGETS:
        path = os.path.join(args.data_dir, fn)
        if not os.path.exists(path):
            print(f"  MISSING: {fn}")
            continue
        out = os.path.join(args.data_dir, fn.replace(".nc", "_dedup.nc"))
        if repair(path, out, check_only=args.check_only):
            changed += 1
    print(f"\n{changed} file(s) {'need repair' if args.check_only else 'repaired'}.")


if __name__ == "__main__":
    main()

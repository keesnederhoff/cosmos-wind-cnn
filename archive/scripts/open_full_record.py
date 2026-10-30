#!/usr/bin/env python
"""Open the 4-year full-record inference segments as ONE clean lazy dataset.

Why this exists: a naive ``xr.open_mfdataset(sorted(glob(...)))`` on these
segments FAILS with

    ValueError: Resulting object does not have monotonic global indexes along
    dimension time

because the segments deliberately overlap. ``step_inference`` pads each
segment's load window back by ``sequence_length - 1`` hours, and ``end_date`` is
inclusive-of-day, so segment [1940,1944] spans 1940-01-01T07 -> 1944-01-01T23
while the next starts at 1943-12-31T19. The first 5 steps of every segment are
also NaN (sliding-window warm-up).

This helper drops the NaN pad, de-duplicates the boundary overlap, and returns a
lazy (dask-backed) dataset -- so you can subset a year/region/variable without
ever materializing the ~250 GB full record.

Usage:
    from scripts.open_full_record import open_full_record
    ds = open_full_record()                      # lazy, whole 1940-2027 record
    ds.sel(time="2024").to_netcdf("2024.nc")     # cut what you actually need

    # or from the shell:
    python scripts/open_full_record.py --summary
"""

import argparse
import glob
import os

import numpy as np
import xarray as xr

DEFAULT_DIR = ("/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/"
               "sf_bay_rtma/results/sw_bc24do0/output_inference")
WARMUP = 5  # sequence_length - 1: leading NaN steps written by run_streaming_inference


def open_full_record(seg_dir=DEFAULT_DIR, time_chunk=720):
    """Return the segments stitched into one monotonic, gap-free lazy Dataset."""
    pattern = os.path.join(seg_dir, "full_record_ERA5_*.nc")
    # Only the 4-year segments: exclude the 15-year 2011-2026 file, which would
    # overlap them, and any quarantined partial.
    files = sorted(f for f in glob.glob(pattern)
                   if os.path.basename(f) != "full_record_ERA5_20110101_20260101.nc")
    if not files:
        raise FileNotFoundError(f"no segments matched {pattern}")

    segs = []
    for f in files:
        d = xr.open_dataset(f, chunks={"time": time_chunk})
        segs.append(d.isel(time=slice(WARMUP, None)))   # drop NaN warm-up pad

    ds = xr.concat(segs, dim="time")
    # De-duplicate the ~24 h boundary overlap. Both copies are real predictions
    # of the same hour, so keeping the first is fine.
    _, idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=np.sort(idx))

    dt = np.diff(ds.time.values).astype("timedelta64[s]").astype(int)
    if not (dt > 0).all():
        raise ValueError("time axis still not monotonic after dedupe")
    return ds


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg-dir", default=DEFAULT_DIR)
    p.add_argument("--summary", action="store_true", help="print a summary and exit")
    p.add_argument("--year", help="write <year>.nc subset to --out")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ds = open_full_record(a.seg_dir)
    if a.summary or not a.year:
        gaps = np.unique(np.diff(ds.time.values).astype("timedelta64[h]").astype(int))
        print(f"  segments merged (lazy)")
        print(f"  vars : {[v for v in ds.data_vars]}")
        print(f"  dims : {dict(ds.sizes)}")
        print(f"  time : {str(ds.time.values[0])[:19]} -> {str(ds.time.values[-1])[:19]}")
        print(f"  time steps present, unique spacings (h): {gaps}")
    if a.year:
        out = a.out or f"full_record_{a.year}.nc"
        sub = ds.sel(time=a.year)
        enc = {v: {"zlib": True, "complevel": 4} for v in sub.data_vars if sub[v].ndim == 3}
        sub.to_netcdf(out, encoding=enc)
        print(f"  wrote {out} ({os.path.getsize(out)/1e9:.2f} GB, {sub.sizes['time']} steps)")


if __name__ == "__main__":
    main()

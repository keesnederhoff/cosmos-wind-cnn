#!/usr/bin/env python
"""Convert processed NetCDF splits to uncompressed memmap arrays.

For each split (train/val/test) under a data_processed dir, writes:
  <data_processed>/memmap/<split>/<var>.dat   raw float32, shape (T, H, W)
  <data_processed>/memmap/<split>/nan_at_time.npy   bool, shape (T,)
  <data_processed>/memmap/<split>/meta.json   {T, H, W, vars, dtype, ydim, xdim}

This is the one-time cost that buys a fast, RAM-shared (page-cache) data path
for WindDatasetMemmap. Reads are streamed in time-chunks so peak memory stays
small regardless of record length. The source NetCDF is read sequentially
(one var at a time, in time order), which is far cheaper than the random,
re-decompressing access pattern that made lazy training slow.

Usage:
  python scripts/convert_to_memmap.py --data-processed <dir> [--splits train val test] [--chunk 2000]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def convert_split(nc_path: Path, out_dir: Path, chunk: int = 2000, verbose: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(str(nc_path), cache=False)

    ydim = "y" if "y" in ds.dims else "latitude"
    xdim = "x" if "x" in ds.dims else "longitude"
    T = int(len(ds["time"]))
    H = int(len(ds[ydim]))
    W = int(len(ds[xdim]))

    # Only variables that span (time, y, x) — these are the model channels.
    data_vars = [v for v in ds.data_vars
                 if "time" in ds[v].dims and ydim in ds[v].dims and xdim in ds[v].dims]

    nan_at_time = np.zeros(T, dtype=bool)
    for var in data_vars:
        out_path = out_dir / (var + ".dat")
        mm = np.memmap(str(out_path), dtype="float32", mode="w+", shape=(T, H, W))
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            arr = (ds[var].isel(time=slice(t0, t1))
                   .transpose("time", ydim, xdim).values.astype("float32"))
            mm[t0:t1] = arr
            nan_at_time[t0:t1] |= np.isnan(arr).any(axis=(1, 2))
        mm.flush()
        del mm
        if verbose:
            print(f"  wrote {var}: ({T}, {H}, {W})", flush=True)

    np.save(str(out_dir / "nan_at_time.npy"), nan_at_time)
    meta = {"T": T, "H": H, "W": W, "vars": list(data_vars),
            "dtype": "float32", "ydim": ydim, "xdim": xdim}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    ds.close()
    if verbose:
        n_nan = int(nan_at_time.sum())
        print(f"  {out_dir.name}: {len(data_vars)} vars, {T} steps, "
              f"{n_nan} NaN timesteps", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-processed", required=True,
                    help="dir containing train.nc/val.nc/test.nc + normalization_stats.pkl")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--chunk", type=int, default=2000,
                    help="timesteps per streamed write block")
    args = ap.parse_args()

    dp = Path(args.data_processed)
    out_root = dp / "memmap"
    for split in args.splits:
        nc = dp / (split + ".nc")
        if not nc.exists():
            print(f"skip {split}: {nc} missing", flush=True)
            continue
        print(f"Converting {split} from {nc} ...", flush=True)
        convert_split(nc, out_root / split, chunk=args.chunk)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

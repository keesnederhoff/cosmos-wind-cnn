#!/usr/bin/env python
"""Fill the fixed NaN border on the 2026-08-06 ERA5 files.

The problem
-----------
Every ERA5 variable in the new drop carries a NaN border -- the left 2-3
columns, the right column and the bottom row of the 21x17 UTM grid, identical at
every timestep and identical across variables (83 of 357 cells, 23.25%). The
older 1940-2026 ERA5 files on the EXACT SAME grid have none, so this is an
artifact of the new download/regrid, not physics.

Why it is fatal untreated: the dataset drops any sequence window containing a
NaN, and these NaNs are present at 100% of timesteps -- so a single one of these
channels would silently produce a ZERO-SAMPLE training set.

Why filling is safe here
------------------------
The only part of the border that intersects the target domain is the western
edge. The ERA5 valid core starts at x = 320500 m; the RTMA target grid starts at
x = 318750 m. That is a 1750 m gap against a 25000 m ERA5 cell -- so the fill
extends the field by 7% of one cell, and affects exactly 1 of 162 RTMA columns.
The bottom NaN row (y = 3969000) sits below the RTMA domain (y >= 4015750) and
the right NaN column (x = 770500) sits east of it (x <= 721250); neither is ever
sampled.

Nearest-valid extrapolation (ffill/bfill along x, then y) is precisely what a
nearest-neighbour regridder would do at the boundary.

sea_surface_temperature is DELIBERATELY EXCLUDED: its NaN is a genuine land mask
(160 cells beyond the border artifact), and filling land with the nearest ocean
temperature would feed the model a physically false stability signal. Use
skin_temperature instead -- over water it is effectively SST, and it carries only
the border artifact.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import xarray as xr

DEFAULT_DIR = ("/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/"
               "sf_bay_rtma/raw_data")

# Files that must NOT be touched: the old (already clean) ERA5 record, anything
# RTMA, and SST for the reason given above.
EXCLUDE_SUBSTRINGS = ("1940_2026", "RTMA_", "sea_surface_temperature", "_filled")


def nearest_valid_index_map(invalid2d):
    """For every cell, the (y, x) index of the nearest VALID cell.

    `distance_transform_edt` measures distance from each non-zero element to the
    nearest zero, so passing the INVALID mask makes valid cells the zeros: every
    invalid cell is mapped to its nearest valid neighbour, and valid cells map to
    themselves (a no-op). This is exactly nearest-neighbour extrapolation.
    """
    from scipy import ndimage
    _, (iy, ix) = ndimage.distance_transform_edt(invalid2d, return_indices=True)
    return iy, ix


def fill_border(da):
    """Nearest-valid 2D extrapolation, applied to every timestep at once.

    The index map is built from the UNION of the NaN mask over all live
    timesteps, not from one representative slice. It used to be sampled from the
    mid-record slice on the stated assumption that the mask is time-invariant --
    true of the uniformly-bordered 2026-08-06 files, and FALSE of the merged
    2000-2026 files, where a clean 2000-2025 record has newly-downloaded 2026
    data appended to it. Mid-record lands in the clean part, `invalid.any()` is
    False, and the function returns the array untouched while reporting success.

    The gather is applied only WHERE THE VALUE IS ACTUALLY NaN. With a genuinely
    time-invariant mask that is identical to gathering everywhere; with a
    time-varying one it is the difference between patching holes and overwriting
    good data at the border columns for every timestep in the record.

    This avoids xarray's ffill/bfill, which needs the optional `bottleneck`
    package, and is far faster than a per-step interpolation.
    """
    vals = da.values                                   # (T, y, x)
    nan_all = np.isnan(vals)
    dead = nan_all.all(axis=(1, 2))
    live = ~dead
    invalid = (nan_all[live].any(axis=0) if live.any() else nan_all.any(axis=0))
    if not invalid.any():
        return da
    iy, ix = nearest_valid_index_map(invalid)
    filled = np.where(nan_all, vals[:, iy, ix], vals)
    # Preserve any wholly-NaN slices as NaN rather than fabricating values from
    # a gather over an all-NaN plane (they are outside the analysis window).
    if dead.any():
        filled[dead] = np.nan
    return da.copy(data=filled)


def process(path, out_path, dry_run=False, window=None):
    """Fill and verify. Verification is restricted to the ANALYSIS WINDOW.

    Some accumulated ERA5 variables (wind_gust, friction_velocity,
    surface_latent_heat_flux) have a wholly-NaN FIRST timestep -- there is no
    preceding accumulation interval for it. A spatial fill cannot repair an
    all-NaN slice, and it does not need to: that stamp is 2013-12-31T19:00, far
    outside the 2020+ window. So require zero NaN inside the window, and merely
    report fully-NaN slices elsewhere.

    window=None means VERIFY THE WHOLE TIME AXIS, and that is now the DEFAULT.
    It used to default to ("2020-01-01", "2025-12-31") -- the v3 training window
    -- which silently missed a defect confined to the TAIL of a file. The 2026
    portion of the 2000-2026 u/v/cloud files carries the border NaN while
    2000-2025 does not: the drop appended new data onto clean older data. This
    check therefore reported "already clean in window", and every 2026 inference
    frame came out wholly NaN -- one NaN column on the target grid propagates
    across the entire field through the U-Net. Narrow it with --window only when
    the record outside that window is genuinely unused.
    """
    name = os.path.basename(path)
    ds = xr.open_dataset(path)
    var = [v for v in ds.data_vars][0]
    da = ds[var]

    if "time" not in da.dims:
        ds.close()
        return None

    win = da if window is None else da.sel(time=slice(*window))
    probe = win.isel(time=slice(0, None, max(1, win.sizes["time"] // 200))).load()
    before = float(np.isnan(probe.values).mean()) * 100.0
    if before == 0.0:
        print(f"  {name[:58]:58s} already clean in window")
        ds.close()
        return False

    if dry_run:
        print(f"  {name[:58]:58s} {before:6.2f}% NaN in window -> would fill")
        ds.close()
        return True

    filled = fill_border(da)

    # Verify over the FULL window, not a subsample -- the gaps are short and a
    # coarse probe both misses them and mis-states their size.
    fwin = (filled if window is None else filled.sel(time=slice(*window))).load().values
    nan_any = np.isnan(fwin)
    dead = nan_any.all(axis=(1, 2))
    spatial_residual = float((nan_any.any(axis=(1, 2)) & ~dead).mean()) * 100.0

    if spatial_residual > 0.0:
        # A genuine spatial hole the fill could not close -- that IS a failure.
        print(f"  {name[:58]:58s} ABORT: {spatial_residual:.3f}% of steps still "
              f"have partial spatial NaN")
        ds.close()
        return False

    # Wholly-NaN timesteps are acceptable and are NOT a fill failure: the
    # accumulated ERA5 variables are missing the first 7 hours of each calendar
    # year. convert_to_memmap ORs nan_at_time across all variables, so these
    # steps drop for EVERY arm alike -- which keeps the predictor-block ablation
    # controlled rather than confounding it. Cost is ~0.1% of windows.
    dead_note = ""
    if dead.any():
        dead_note = (f"  [WARN {int(dead.sum())} fully-NaN steps "
                     f"({100 * dead.mean():.3f}%) inside window -- dropped by "
                     f"the NaN-window logic]")

    out = filled.to_dataset(name=var)
    for c in ("crs", "spatial_ref"):
        if c in ds:
            out[c] = ds[c]
    out[var].attrs = dict(da.attrs)
    out[var].attrs["border_nan_filled"] = (
        "nearest-valid extrapolation along x then y; the 2026-08-06 ERA5 drop "
        "carried a fixed NaN border of 83/357 cells at every timestep")
    out.attrs = dict(ds.attrs)

    enc = {var: {"zlib": True, "complevel": 1}}
    tmp = out_path + ".tmp"
    out.to_netcdf(tmp, encoding=enc)
    ds.close()
    os.replace(tmp, out_path)
    print(f"  {name[:58]:58s} {before:6.2f}% -> 0.00%  wrote "
          f"{os.path.basename(out_path)} "
          f"({os.path.getsize(out_path) / 1e6:.0f} MB){dead_note}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--window", nargs=2, metavar=("START", "END"), default=None,
                    help="Verify only this time slice. Default: the WHOLE time "
                         "axis -- a narrower window is how the 2026 border NaN "
                         "went unnoticed for a whole inference campaign.")
    ap.add_argument("--only", default=None,
                    help="Substring filter on the filename, so a targeted repair "
                         "does not rewrite 28 unrelated files.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "ERA5_*.nc")))
    files = [f for f in files
             if not any(s in os.path.basename(f) for s in EXCLUDE_SUBSTRINGS)]
    # Prefer the deduped copies where they exist (Phase 0b).
    dedup_stems = {os.path.basename(f).replace("_dedup.nc", ".nc")
                   for f in files if f.endswith("_dedup.nc")}
    files = [f for f in files
             if os.path.basename(f) not in dedup_stems or f.endswith("_dedup.nc")]

    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    print(f"ERA5 border-NaN fill  ({len(files)} candidate files)"
          f"  verify-window={args.window or 'FULL TIME AXIS'}")
    n = 0
    for f in files:
        out = f.replace(".nc", "_filled.nc")
        if process(f, out, dry_run=args.dry_run,
                   window=tuple(args.window) if args.window else None):
            n += 1
    print(f"\n{n} file(s) {'need filling' if args.dry_run else 'filled'}.")
    print("NOTE sea_surface_temperature deliberately excluded (real land mask); "
          "use skin_temperature.")


if __name__ == "__main__":
    main()

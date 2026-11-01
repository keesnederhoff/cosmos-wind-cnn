"""Stage 1 full-cache check (Task 4, Step 7). Run after the production s1 run
has produced all four dailymax_<product>.nc files. For each product, prints:
the number of days, the WY range, the overall fraction of NaN dmax entries,
and the number of WY (of FULL_WY[product]) that are valid domain-wide under
valid_water_years (i.e. every grid cell meets MIN_DAY_FRAC_WY).

Expected (v4): era5 / cnn_raw -> 86 valid WY (1941-2026), rtma -> 15 (2012-2026). CNN NaN fraction
should be < 0.5% (the annual ~7 h gust-seam gap must not push many cells below
MIN_HOURS_PER_DAY = 12).
"""

import numpy as np
import pandas as pd
import xarray as xr

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import windx_common as w


def check_product(product: str) -> None:
    path = w.CACHE_DIR / f"dailymax_{product}.nc"
    print(f"=== {product} ===")
    if not path.exists():
        print(f"  MISSING {path}")
        return

    with xr.open_dataset(path) as ds:
        days = pd.DatetimeIndex(ds["day"].values)
        dmax = ds["dmax"].values  # (day, y, x) float32; CNN full cube ~2.5 GB, fine to load once

    n_days = len(days)
    wy_lo, wy_hi = w.FULL_WY[product]
    nan_frac = float(np.mean(~np.isfinite(dmax)))

    vwy = w.valid_water_years(days, dmax)  # dict: wy -> bool array (y, x)
    n_valid = sum(1 for wy, arr in vwy.items() if wy_lo <= wy <= wy_hi and np.all(arr))

    print(f"  n_days        = {n_days}")
    print(f"  wy_range      = {wy_lo}-{wy_hi} ({wy_hi - wy_lo + 1} WY expected)")
    print(f"  nan_frac      = {nan_frac * 100:.3f} %")
    print(f"  n_valid_wy    = {n_valid} (domain-wide, MIN_DAY_FRAC_WY={w.MIN_DAY_FRAC_WY})")


if __name__ == "__main__":
    for product in w.PRODUCTS:
        check_product(product)

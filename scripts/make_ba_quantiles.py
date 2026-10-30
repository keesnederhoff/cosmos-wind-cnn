#!/usr/bin/env python
"""Apply the observation-based BA factor to the shipped speed-QUANTILE subset (v4).

v3 shipped BA only for the P50 u/v/speed products; the quantile files were not
adjusted. The BA factor is one multiplicative number per cell, fitted at the 90th
percentile and flat across q0.75-q0.99 at the stations, so it is applied to every
shipped level. Order and relative spread are preserved; the factor is >= 1
(never reduces wind speed, user rule 2026-10-03), so the adjusted quantiles are
never below the raw ones.

Input : CNN_RTMA_v4_<seed>_speed_quantiles_raw_<t0>_<t1>.nc (from combine_full_record.py)
Map   : BA_factor_map_<seed>_E3q90.nc (from ba_fit_factor.py)
Output: CNN_RTMA_v4_<seed>_speed_quantiles_BA_<t0>_<t1>.nc  (.tmp -> verify -> rename)

Which levels carry the factor is a judgement: only P50 is validated against
observations (BA exact at q90 at the stations). The attributes say so.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

ROOT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay')
CASE = ROOT / os.environ.get('CNN_CASE', 'sf_bay_rtma_v4')
PREFIX = os.environ.get('CNN_PREFIX', 'CNN_RTMA_v4')
BLOCK = 24 * 31


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed')
    ap.add_argument('--dir', default=str(CASE / 'results' / 'v1940_product'))
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    seed = a.seed or (CASE / 'results' / 'chosen_seed.txt').read_text().strip()
    d = Path(a.dir)
    raws = sorted(d.glob(f'{PREFIX}_{seed}_speed_quantiles_raw_*.nc'))
    if len(raws) != 1:
        sys.exit(f'STOP: expected one raw quantile product in {d}, found {len(raws)}')
    raw = raws[0]
    out = Path(str(raw).replace('_raw_', '_BA_'))
    bamap = d / f'BA_factor_map_{seed}_E3q90.nc'
    if out.exists() and not a.overwrite:
        sys.exit(f'STOP: {out} exists (use --overwrite)')

    src = netCDF4.Dataset(raw)
    src.set_auto_mask(False)
    m = xr.open_dataset(bamap)
    F = m['factor'].values.astype('float32')
    if not np.isfinite(F).all() or F.min() < 1.0 - 1e-6:
        sys.exit(f'STOP: BA factor must be finite and >= 1 (min {F.min():.4f})')
    ws = src.variables['wind_speed']
    if ws.shape[2:] != F.shape:
        sys.exit(f'STOP: grid {ws.shape[2:]} vs BA map {F.shape}')
    print(f'raw {raw.name}: {ws.shape}; BA factor {F.min():.3f}..{F.max():.3f}', flush=True)

    tmp = str(out) + '.tmp'
    dst = netCDF4.Dataset(tmp, 'w', format='NETCDF4')
    for k, dm in src.dimensions.items():
        dst.createDimension(k, None if dm.isunlimited() else len(dm))
    for k, v in src.variables.items():
        kw = {}
        if k == 'wind_speed':
            kw = dict(zlib=True, complevel=4, chunksizes=v.chunking(), fill_value=np.float32(np.nan))
        o = dst.createVariable(k, v.dtype, v.dimensions, **kw)
        o.setncatts({x: v.getncattr(x) for x in v.ncattrs() if x != '_FillValue'})
        if k != 'wind_speed':
            o[:] = v[:]
    dst.setncatts({x: src.getncattr(x) for x in src.ncattrs()})
    dst.title = str(src.title).replace('(raw)', '(observation-based bias adjustment, BA)')
    dst.history = f'{src.history} | {dt.date.today()} make_ba_quantiles.py'
    dst.variables['wind_speed'].bias_correction = (
        'multiplicative per-cell BA factor (obs/CNN at the 90th percentile at the stations, '
        f'floored at 1, Gaussian sigma from the map) applied to every shipped quantile level. '
        f'Only P50 is validated against observations. Map: {bamap.name}')
    nt = ws.shape[0]
    for t0 in range(0, nt, BLOCK):
        t1 = min(t0 + BLOCK, nt)
        dst.variables['wind_speed'][t0:t1] = ws[t0:t1] * F[None, None]
        if (t0 // BLOCK) % 20 == 0:
            print(f'  {t1}/{nt}', flush=True)
    dst.close()

    chk = netCDF4.Dataset(tmp); chk.set_auto_mask(False)
    rng = np.random.default_rng(0)
    ok = True
    for i in rng.integers(0, nt, 6):
        w = ws[int(i)] * F[None]
        g = chk.variables['wind_speed'][int(i)]
        both = np.isfinite(w) & np.isfinite(g)
        mx = float(np.abs(w - g)[both].max()) if both.any() else 0.0
        nm = int((np.isfinite(w) ^ np.isfinite(g)).sum())
        ok &= mx < 1e-5 and nm == 0
        print(f'  spot t={int(i)}: max|diff| {mx:.2e} NaN mismatch {nm}')
    if not ok:
        sys.exit('STOP: verification failed -- .tmp left for inspection')
    os.replace(tmp, out)
    print(f'BA QUANTILES OK -> {out.name} ({os.path.getsize(out) / 1e9:.1f} GB)')


if __name__ == '__main__':
    main()

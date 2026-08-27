#!/usr/bin/env python
"""Verify the re-run era inference: reproduction check, then full-record audit.

Two modes, because the two questions are different.

--compare  Does a re-inferred year reproduce the archived original? Inputs are
           unchanged for 2000-2025 except the gust border, which was masked and
           re-filled to match training (max input difference 1.9e-06). So the
           outputs should agree to something small. A LARGE difference means the
           staging changed the data in a way the pre-flight missed.

--audit    The Phase 4 gate on the finished product, per seed: exactly 27 year
           files, a monotonic gap-free hourly axis after dropping the 754
           boundary duplicates, and an all-NaN step fraction around 0.3-0.4%.
           The whole point is the 2026 file: a wholly-NaN year is the failure
           that cost this project an entire campaign, and it is invisible unless
           you look per-year.

Runs on a compute node -- these files are ~5.3 GB each and the login node
times out reading two of them.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr

RESULTS = ('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/'
           'sf_bay_rtma_v3/results')
SEEDS = ['r1_do010', 'r1b_do010_s2', 'r1b_do010_s3']
VARS = ['hr_u', 'hr_v', 'hr_speed_q', 'hr_gust_q']


def compare(arm, fname, stride):
    new = f'{RESULTS}/{arm}/output_inference/{fname}'
    old = f'{RESULTS}/{arm}/output_inference/_prev/{fname}'
    if not os.path.exists(old):
        print(f'  no archived twin at {old} -- nothing to compare')
        return True
    dn = xr.open_dataset(new, decode_timedelta=False)
    do = xr.open_dataset(old, decode_timedelta=False)
    tn, to = dn['time'].values, do['time'].values
    print(f'  new: n={tn.size} {str(tn[0])[:16]} -> {str(tn[-1])[:16]}')
    print(f'  old: n={to.size} {str(to[0])[:16]} -> {str(to[-1])[:16]}')
    if not np.array_equal(tn, to):
        print('  NOTE time axes differ; comparing the intersection')
    common = np.intersect1d(tn, to)
    sel = common[::stride]
    print(f'  comparing {sel.size} of {common.size} common steps '
          f'(stride {stride})')
    worst = 0.0
    for v in VARS:
        if v not in dn.data_vars or v not in do.data_vars:
            continue
        a = dn[v].sel(time=sel).values
        b = do[v].sel(time=sel).values
        d = np.abs(a - b)
        d[np.isnan(a) & np.isnan(b)] = 0.0
        mism = int((np.isnan(a) ^ np.isnan(b)).sum())
        fin = d[np.isfinite(d)]
        mx = float(fin.max()) if fin.size else float('nan')
        mn = float(fin.mean()) if fin.size else float('nan')
        worst = max(worst, mx if np.isfinite(mx) else 0.0)
        print(f'    {v:<11s} max|diff|={mx:.4e}  mean|diff|={mn:.4e}  '
              f'NaN-mismatch={mism}  NaNfrac new={np.isnan(a).mean():.4f} '
              f'old={np.isnan(b).mean():.4f}')
    dn.close()
    do.close()
    print(f'  worst max|diff| across vars: {worst:.4e}')
    return worst


def audit(arm):
    d = f'{RESULTS}/{arm}/output_inference'
    files = sorted(glob.glob(f'{d}/speed_full_record_ERA5_????0101_*.nc'))
    print(f'  {arm}: glob resolved {len(files)} files (expect 27)')
    if len(files) != 27:
        for f in files:
            print('     ', os.path.basename(f))
        return False

    ok = True
    times = []
    for f in files:
        ds = xr.open_dataset(f, decode_timedelta=False)
        t = ds['time'].values
        v = 'hr_u' if 'hr_u' in ds.data_vars else list(ds.data_vars)[0]
        a = ds[v].values
        flat = np.isnan(a.reshape(a.shape[0], -1))
        dead = flat.all(axis=1)
        frac = 100.0 * dead.mean()
        flag = ''
        if frac > 5.0:
            flag = '   <<< SUSPICIOUS'
            ok = False
        print(f'    {os.path.basename(f)[-26:]:26s} n={t.size:5d} '
              f'{str(t[0])[:13]} -> {str(t[-1])[:13]}  all-NaN={frac:6.3f}%{flag}')
        times.append(t)
        ds.close()

    allt = np.concatenate(times)
    uniq = np.unique(allt)
    dupes = allt.size - uniq.size
    dif = np.diff(uniq).astype('timedelta64[h]').astype(int)
    gaps = int((dif != 1).sum())
    print(f'  concatenated: {allt.size} steps, {dupes} boundary duplicates, '
          f'{uniq.size} unique, gaps={gaps}')
    print(f'  span: {str(uniq[0])[:16]} -> {str(uniq[-1])[:16]}')
    if gaps:
        print('  FAIL: gaps in the concatenated axis')
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--arm', default='r1_do010')
    ap.add_argument('--file', default='speed_full_record_ERA5_20000101_20010101.nc')
    ap.add_argument('--stride', type=int, default=24)
    a = ap.parse_args()

    rc = 0
    if a.compare:
        print('=' * 74)
        print(f'REPRODUCTION CHECK  {a.arm}  {a.file}')
        print('=' * 74)
        compare(a.arm, a.file, a.stride)
    if a.audit:
        print()
        print('=' * 74)
        print('FULL-RECORD AUDIT')
        print('=' * 74)
        for s in SEEDS:
            if not audit(s):
                rc = 1
            print()
    sys.exit(rc)


if __name__ == '__main__':
    main()

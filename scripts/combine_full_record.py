#!/usr/bin/env python
"""Combine the 1940-2026 year files of ONE seed into one NetCDF per parameter.

Outputs (float32 + zlib, CF, chunked 24 h x 1 quantile x full grid):
  CNN_RTMA_v3_<seed>_speed_quantiles_raw_<t0>_<t1>.nc
  CNN_RTMA_v3_<seed>_speed_quantiles_BC_<t0>_<t1>.nc

BIAS CORRECTION (user decision 2026-09-29). The v3 BC was never saved as a map:
bc_v3_test.py fits a per-cell quantile map (bias_correct.fit_quantile_maps, 200
levels) of |hr_u, hr_v| -- i.e. the P50 speed -- against RTMA speed on the VAL
window, in memory. Here that fit is repeated verbatim, SAVED, and PROVEN by
re-applying it to the archived test-window inference and reproducing the
archived BCVAL_ file. It is then applied WITHOUT refit to 1940-2026 as a
per-cell, per-hour RATIO  r = BC(P50) / P50  scaled onto ALL quantile levels --
exactly how the shipped BC scales u/v. Quantile order and relative spread are
preserved; only P50 has been validated against RTMA.

Streaming: year files overlap (each starts ~5 h before Jan 1 and ends Jan 1
23:00 of the next year). Steps already written are dropped; the overlapping
values must be identical (checked) and the axis must be hourly-contiguous.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bias_correct import fit_quantile_maps, apply_maps  # noqa: E402

ROOT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay')
CASE = ROOT / 'sf_bay_rtma_v3'
SHARED = CASE / 'results' / 'v3data' / 'data_processed'
VAL_FILE = 'speed_full_record_ERA5_20240314_20250206.nc'
TEST_FILE = 'speed_full_record_ERA5_20250206_20260101.nc'
NQUANT = 200
EPS = 1e-6                        # bc_v3_test.py / bias_correct.py convention
TUNITS = 'hours since 1940-01-01 00:00:00'


def _speed(ds, u='hr_u', v='hr_v'):
    return np.sqrt(ds[u].astype('float32') ** 2 + ds[v].astype('float32') ** 2)


def bc_ratio(s, cnn_q, rtma_q):
    s_bc = apply_maps(s, cnn_q, rtma_q)
    return np.where(s > EPS, np.maximum(s_bc, EPS) / np.maximum(s, EPS),
                    1.0).astype('float32')


def fit_and_prove(seed, map_path):
    inf_dir = CASE / 'results' / seed / 'output_inference'
    vp, tp = inf_dir / VAL_FILE, inf_dir / TEST_FILE
    bp = inf_dir / f'BCVAL_{TEST_FILE}'
    for p in (vp, tp, bp):
        if not p.exists():
            sys.exit(f'STOP: missing {p}')
    inf_val = xr.open_dataset(vp)
    val = xr.open_dataset(SHARED / 'val.nc')
    cnn_v, rtma_v = xr.align(_speed(inf_val), _speed(val), join='inner')
    n_fit = cnn_v.sizes['time']
    print(f'  BC fit on VAL: {n_fit} samples '
          f'({str(cnn_v.time.values[0])[:13]} -> {str(cnn_v.time.values[-1])[:13]})')
    if n_fit < 1000:
        sys.exit(f'STOP: only {n_fit} overlapping fit samples')
    cnn_q, rtma_q = fit_quantile_maps(cnn_v.values, rtma_v.values, NQUANT)

    # PROOF: reproduce the archived BCVAL_ test-window u/v
    it = xr.open_dataset(tp)
    u = it['hr_u'].values.astype('float32')
    v = it['hr_v'].values.astype('float32')
    r = bc_ratio(np.sqrt(u ** 2 + v ** 2), cnn_q, rtma_q)
    arch = xr.open_dataset(bp)
    du = np.abs(u * r - arch['hr_u'].values)
    dv = np.abs(v * r - arch['hr_v'].values)
    mx = float(np.nanmax([np.nanmax(du), np.nanmax(dv)]))
    nanm = int((np.isnan(u * r) ^ np.isnan(arch['hr_u'].values)).sum())
    print(f'  BC PROOF vs {bp.name}: max|diff| u/v {mx:.3e}, NaN mismatch {nanm}')
    if not (mx < 1e-4 and nanm == 0):
        sys.exit('STOP: refitted BC map does not reproduce the archived BCVAL file')

    xr.Dataset(
        {'cnn_q': (('level', 'y', 'x'), cnn_q), 'rtma_q': (('level', 'y', 'x'), rtma_q)},
        coords={'level': np.linspace(0.0, 1.0, NQUANT), 'y': it['y'].values,
                'x': it['x'].values},
        attrs={'seed': seed, 'fit_period': 'VAL window (inner join with v3data val.nc)',
               'fit_samples': n_fit, 'nquant': NQUANT,
               'method': 'bias_correct.fit_quantile_maps on |hr_u,hr_v| (P50) vs RTMA speed',
               'proof': f'reproduces {bp.name} to max|diff| {mx:.3e}'}
    ).to_netcdf(map_path)
    print(f'  saved BC map {map_path}')
    return cnn_q, rtma_q


def year_files(in_dir):
    files = sorted(glob.glob(str(in_dir / 'speed_full_record_ERA5_????0101_*.nc')))
    if not files:
        sys.exit(f'STOP: no year files in {in_dir}')
    return files


def create(path, like, quantiles, t_attrs, title, extra):
    nc = netCDF4.Dataset(path, 'w', format='NETCDF4')
    nc.createDimension('time', None)
    nc.createDimension('quantile', len(quantiles))
    nc.createDimension('y', like['y'].size)
    nc.createDimension('x', like['x'].size)
    t = nc.createVariable('time', 'f8', ('time',))
    t.units, t.calendar, t.standard_name = TUNITS, 'standard', 'time'
    t.long_name = 'time (UTC)'
    q = nc.createVariable('quantile', 'f8', ('quantile',))
    q[:] = quantiles
    q.long_name = 'quantile level (tau) of the predictive distribution'
    for c in ('y', 'x'):
        var = nc.createVariable(c, 'f8', (c,))
        var[:] = like[c].values
        var.units = 'm'
        var.standard_name = 'projection_y_coordinate' if c == 'y' else 'projection_x_coordinate'
    crs = nc.createVariable('crs', 'i4')
    if 'crs' in like:
        crs.setncatts({k: v for k, v in like['crs'].attrs.items()})
    ws = nc.createVariable('wind_speed', 'f4', ('time', 'quantile', 'y', 'x'),
                           zlib=True, complevel=4, chunksizes=(24, 1, like['y'].size,
                                                               like['x'].size),
                           fill_value=np.float32(np.nan))
    ws.units = 'm s-1'
    ws.standard_name = 'wind_speed'
    ws.long_name = '10 m wind speed, quantiles of the CNN predictive distribution'
    ws.grid_mapping = 'crs'
    ws.setncatts(extra)
    nc.setncatts(dict(t_attrs, title=title))
    return nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', required=True)
    ap.add_argument('--in-dir')
    ap.add_argument('--out-dir', default=str(CASE / 'results' / 'v1940_product'))
    a = ap.parse_args()
    in_dir = Path(a.in_dir or CASE / 'results' / 'v1940_grids' / a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('=' * 78 + '\nSTEP 1  BC map: refit on VAL, prove, save\n' + '=' * 78)
    map_path = out / f'BC_map_{a.seed}_valfit.nc'
    cnn_q, rtma_q = fit_and_prove(a.seed, map_path)

    files = year_files(in_dir)
    print('=' * 78 + f'\nSTEP 2  stream {len(files)} year files\n' + '=' * 78)
    first = xr.open_dataset(files[0])
    last = xr.open_dataset(files[-1])
    quantiles = first['quantile'].values
    tag0 = '19400101'
    tag1 = str(last['time'].values[-1])[:10].replace('-', '')
    last.close()
    prov = {k: v for k, v in first.attrs.items() if isinstance(v, (str, int, float))}
    common = {
        'institution': 'USGS Pacific Coastal and Marine Science Center',
        'source': ('cosmos-wind-cnn v3 quantile head (3D U-Net), ERA5 -> RTMA 2.5 km '
                   'downscaling; seed ' + a.seed + ', checkpoint best_speed.pth'),
        'inputs': ('ERA5 10 m u/v, cloud fraction, wind gust (border-filled as in '
                   'training), RTMA terrain; staged by scripts/stage_era5_1940.py'),
        'conventions_note': 'quantile levels are actual grid levels of the 19-level head; '
                            'there is no true P99 (top level tau=0.9737)',
        'caveats': ('Pre-1979 ERA5 assimilates far fewer observations; station skill '
                    'rises from ~0.34 (1940s) to ~0.57 (2020s). Hours without a full '
                    'ERA5 input window (annual 7 h gust seam) are NaN.'),
        'history': f'{dt.date.today()} combine_full_record.py from {in_dir}',
        'Conventions': 'CF-1.8',
    }
    common.update({f'model_{k}': v for k, v in prov.items()})
    first.close()

    names = {k: out / f'CNN_RTMA_v3_{a.seed}_speed_quantiles_{k}_{tag0}_{tag1}.nc'
             for k in ('raw', 'BC')}
    nc_raw = create(str(names['raw']) + '.tmp', xr.open_dataset(files[0]), quantiles,
                    common, 'SF Bay 10 m wind speed quantiles, CNN-RTMA v3 (raw)',
                    {'bias_correction': 'none'})
    nc_bc = create(str(names['BC']) + '.tmp', xr.open_dataset(files[0]), quantiles,
                   common, 'SF Bay 10 m wind speed quantiles, CNN-RTMA v3 (bias-corrected)',
                   {'bias_correction': (
                       'per-cell quantile map of P50 speed vs RTMA (200 levels), fitted '
                       'on the VAL window 2024-03-14..2025-02-06 and applied without '
                       'refit; the ratio BC(P50)/P50 scales every quantile level. Only '
                       'P50 is validated. Map: ' + map_path.name)})

    n_written, t_last, tail, worst = 0, None, None, 0.0
    for f in files:
        ds = xr.open_dataset(f)
        t = ds['time'].values
        if tail is not None:                          # overlap must agree
            c = np.intersect1d(tail['time'].values, t)
            if c.size:
                x0 = tail['hr_speed_q'].sel(time=c).values
                x1 = ds['hr_speed_q'].sel(time=c).values
                both = np.isfinite(x0) & np.isfinite(x1)
                if both.any():
                    worst = max(worst, float(np.abs(x0 - x1)[both].max()))
        keep = t > t_last if t_last is not None else np.ones(t.size, bool)
        if t_last is not None and keep.any():
            gap = (t[keep][0] - t_last).astype('timedelta64[h]').astype(int)
            if gap != 1:
                sys.exit(f'STOP: {Path(f).name} leaves a {gap} h step after {t_last}')
        sub = ds.isel(time=np.where(keep)[0])
        q = sub['hr_speed_q'].values.astype('float32')          # (T, Q, y, x)
        u = sub['hr_u'].values.astype('float32')
        v = sub['hr_v'].values.astype('float32')
        r = bc_ratio(np.sqrt(u ** 2 + v ** 2), cnn_q, rtma_q)
        tt = netCDF4.date2num(sub['time'].values.astype('datetime64[s]').astype(dt.datetime),
                              TUNITS, 'standard')
        i0, i1 = n_written, n_written + tt.size
        for nc, data in ((nc_raw, q), (nc_bc, q * r[:, None])):
            nc['time'][i0:i1] = tt
            nc['wind_speed'][i0:i1] = data
        n_written = i1
        t_last = sub['time'].values[-1]
        tail = ds.isel(time=slice(-48, None)).load()
        nanfrac = np.isnan(q).all(axis=(1, 2, 3)).mean()
        print(f'  {Path(f).name}: +{tt.size} h -> {n_written} '
              f'(dead steps {100 * nanfrac:.2f}%)', flush=True)
        ds.close()
    for nc in (nc_raw, nc_bc):
        nc.close()
    print(f'  max |diff| in year-boundary overlaps: {worst:.2e}')
    if worst > 1e-4:
        sys.exit('STOP: overlapping year files disagree')

    print('=' * 78 + '\nSTEP 3  verify\n' + '=' * 78)
    ok = True
    rng = np.random.default_rng(0)
    for k in ('raw', 'BC'):
        tmp = str(names[k]) + '.tmp'
        d = xr.open_dataset(tmp)
        tv = d['time'].values
        dd = np.diff(tv).astype('timedelta64[h]').astype(int)
        exp = int((tv[-1] - tv[0]).astype('timedelta64[h]').astype(int)) + 1
        print(f'  {k}: {tv.size} steps {str(tv[0])[:13]} -> {str(tv[-1])[:13]} '
              f'(expected {exp}), steps!=1h: {int((dd != 1).sum())}')
        ok &= tv.size == exp and (dd == 1).all()
        for f in rng.choice(files, min(3, len(files)), replace=False):
            src = xr.open_dataset(f)
            day = src['time'].values[rng.integers(24, src['time'].size - 48)]
            win = slice(day, day + np.timedelta64(23, 'h'))
            s = src.sel(time=win)
            want = s['hr_speed_q'].values
            if k == 'BC':
                uu, vv = s['hr_u'].values, s['hr_v'].values
                want = want * bc_ratio(np.sqrt(uu ** 2 + vv ** 2), cnn_q, rtma_q)[:, None]
            got = d['wind_speed'].sel(time=win).values
            both = np.isfinite(want) & np.isfinite(got)
            mx = float(np.abs(want - got)[both].max())
            nm = int((np.isfinite(want) ^ np.isfinite(got)).sum())
            print(f'    spot {str(day)[:10]}: max|diff| {mx:.2e} NaN mismatch {nm}')
            ok &= mx < 1e-6 and nm == 0
            src.close()
        d.close()
    if not ok:
        sys.exit('STOP: verification failed -- .tmp files left for inspection')
    for k in ('raw', 'BC'):
        os.replace(str(names[k]) + '.tmp', names[k])
        print(f'  -> {names[k].name} ({os.path.getsize(names[k]) / 1e9:.1f} GB)')
    print('COMBINE OK')


if __name__ == '__main__':
    main()

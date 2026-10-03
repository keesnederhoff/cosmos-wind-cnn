#!/usr/bin/env python
"""P50 (median) 10 m wind products for the full 1940 -> 2026 record, raw and BA.

Outputs (float32 + zlib, CF, chunked 24 h x full grid), in --out-dir:
  CNN_RTMA_v3_<seed>_P50_uvs_raw_<t0>_<t1>.nc   u10, v10, wind_speed
  CNN_RTMA_v3_<seed>_P50_uvs_BA_<t0>_<t1>.nc    the same, times F(x) from --ba-map

Source = the 87 year files in results/v1940_grids/<seed>/ (hr_u, hr_v, hr_speed_q).
hr_u/hr_v are P50 x direction, so wind_speed = sqrt(u^2 + v^2) IS the P50; the
first file asserts this against hr_speed_q at tau = 0.5. The BA factor is one
number per cell (scripts/ba_fit_factor.py: obs/CNN at the 90th percentile,
exact at every station, land / bay / ocean constant elsewhere) applied to all
three fields, so direction is unchanged.

Streaming pattern, year-boundary checks and the .tmp -> verify -> rename step
follow scripts/combine_full_record.py. --max-files N limits the run to the
first N year files (smoke test) and writes under <out-dir>/_smoke/.
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

ROOT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay')
CASE = ROOT / 'sf_bay_rtma_v3'
TUNITS = 'hours since 1940-01-01 00:00:00'
FIELDS = ('u10', 'v10', 'wind_speed')


def year_files(in_dir):
    files = sorted(glob.glob(str(in_dir / 'speed_full_record_ERA5_????0101_*.nc')))
    if not files:
        sys.exit(f'STOP: no year files in {in_dir}')
    return files


def load_factor(path, like):
    m = xr.open_dataset(path)
    if not (np.allclose(m.x.values, like.x.values) and np.allclose(m.y.values, like.y.values)):
        sys.exit('STOP: BA map grid differs from the year-file grid')
    F = m['factor'].values.astype('float32')
    if not np.isfinite(F).all() or F.min() <= 0:
        sys.exit('STOP: BA map has non-finite or non-positive factors')
    print(f'  BA map {Path(path).name}: factor {F.min():.3f}..{F.max():.3f}, sigma {m.attrs.get("sigma_km")} km, '
          f'constants land {m.attrs.get("c_land"):.3f} bay {m.attrs.get("c_bay"):.3f} ocean {m.attrs.get("c_ocean"):.3f}')
    return F, {k: v for k, v in m.attrs.items() if isinstance(v, (str, int, float))}


def create(path, like, t_attrs, title, extra):
    nc = netCDF4.Dataset(path, 'w', format='NETCDF4')
    nc.createDimension('time', None)
    nc.createDimension('y', like['y'].size)
    nc.createDimension('x', like['x'].size)
    t = nc.createVariable('time', 'f8', ('time',))
    t.units, t.calendar, t.standard_name = TUNITS, 'standard', 'time'
    t.long_name = 'time (UTC)'
    for c in ('y', 'x'):
        var = nc.createVariable(c, 'f8', (c,))
        var[:] = like[c].values
        var.units = 'm'
        var.standard_name = 'projection_y_coordinate' if c == 'y' else 'projection_x_coordinate'
    crs = nc.createVariable('crs', 'i4')
    if 'crs' in like:
        crs.setncatts({k: v for k, v in like['crs'].attrs.items()})
    meta = {'u10': ('eastward_wind', '10 m eastward wind (CNN P50 speed x direction)'),
            'v10': ('northward_wind', '10 m northward wind (CNN P50 speed x direction)'),
            'wind_speed': ('wind_speed', '10 m wind speed, median (P50) of the CNN predictive distribution')}
    for name, (std, long) in meta.items():
        v = nc.createVariable(name, 'f4', ('time', 'y', 'x'), zlib=True, complevel=4,
                              chunksizes=(24, like['y'].size, like['x'].size),
                              fill_value=np.float32(np.nan))
        v.units, v.standard_name, v.long_name, v.grid_mapping = 'm s-1', std, long, 'crs'
        v.setncatts(extra)
    nc.setncatts(dict(t_attrs, title=title))
    return nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default='r1b_do010_s3')
    ap.add_argument('--in-dir')
    ap.add_argument('--out-dir', default=str(CASE / 'results' / 'v1940_product'))
    ap.add_argument('--ba-map')
    ap.add_argument('--max-files', type=int, default=0, help='smoke test: first N year files only')
    ap.add_argument('--only-ba', action='store_true',
                    help='write only the BA file (the raw product does not depend on the map)')
    ap.add_argument('--overwrite', action='store_true',
                    help='replace an existing product of the same name (user-directed redo)')
    a = ap.parse_args()
    in_dir = Path(a.in_dir or CASE / 'results' / 'v1940_grids' / a.seed)
    out = Path(a.out_dir)
    if a.max_files:
        out = out / '_smoke'
    out.mkdir(parents=True, exist_ok=True)
    ba_map = a.ba_map or str(Path(a.out_dir) / f'BA_factor_map_{a.seed}_E3q90.nc')

    files = year_files(in_dir)
    if a.max_files:
        files = files[:a.max_files]
    first = xr.open_dataset(files[0])
    last = xr.open_dataset(files[-1])
    F, ba_attrs = load_factor(ba_map, first)
    iq = int(np.argmin(np.abs(first['quantile'].values - 0.5)))
    if abs(float(first['quantile'].values[iq]) - 0.5) > 1e-6:
        sys.exit('STOP: no tau = 0.5 level in the year files')
    tag0 = str(first['time'].values[0])[:10].replace('-', '') if a.max_files else '19400101'
    tag1 = str(last['time'].values[-1])[:10].replace('-', '')
    last.close()
    prov = {k: v for k, v in first.attrs.items() if isinstance(v, (str, int, float))}
    common = {
        'institution': 'USGS Pacific Coastal and Marine Science Center',
        'source': ('cosmos-wind-cnn v3 quantile head (3D U-Net), ERA5 -> RTMA 2.5 km '
                   'downscaling; seed ' + a.seed + ', checkpoint best_speed.pth'),
        'inputs': ('ERA5 10 m u/v, cloud fraction, wind gust (border-filled as in '
                   'training), RTMA terrain; staged by scripts/stage_era5_1940.py'),
        'caveats': ('Pre-1979 ERA5 assimilates far fewer observations; station skill '
                    'rises from ~0.34 (1940s) to ~0.57 (2020s). Hours without a full '
                    'ERA5 input window (annual 7 h gust seam) are NaN.'),
        'history': f'{dt.date.today()} make_p50_products.py from {in_dir}',
        'Conventions': 'CF-1.8',
    }
    common.update({f'model_{k}': v for k, v in prov.items()})
    kinds = ('BA',) if a.only_ba else ('raw', 'BA')
    names = {k: out / f'CNN_RTMA_v3_{a.seed}_P50_uvs_{k}_{tag0}_{tag1}.nc' for k in kinds}
    for p in names.values():
        if p.exists() and not a.overwrite:
            sys.exit(f'STOP: {p} exists -- refusing to overwrite a published product (use --overwrite)')
        elif p.exists():
            print(f'  --overwrite: {p.name} will be replaced when the new file verifies')
    floor = ba_attrs.get('floor', 0)
    floor_txt = (f' Factor floored at {floor:g}: the adjustment never reduces wind speed.'
                 if floor and float(floor) > 0 else '')
    writers = {}
    if 'raw' in kinds:
        writers['raw'] = create(str(names['raw']) + '.tmp', first, common,
                                'SF Bay 10 m wind, CNN-RTMA v3 P50 (raw)', {'bias_correction': 'none'})
    writers['BA'] = create(str(names['BA']) + '.tmp', first, common,
                           'SF Bay 10 m wind, CNN-RTMA v3 P50, observation-based bias adjustment (BA)',
                           {'bias_correction': (
                               'multiplicative factor per cell = obs/CNN at the 90th percentile, fitted at '
                               f'{len(ba_attrs.get("stations", "").split(","))} stations over Era 3 '
                               f'({ba_attrs.get("fit_window")}), exact at every station, decaying (sigma '
                               f'{ba_attrs.get("sigma_km")} km) to class constants land {ba_attrs.get("c_land"):.3f} / '
                               f'bay {ba_attrs.get("c_bay"):.3f} / ocean {ba_attrs.get("c_ocean"):.3f}; applied to '
                               f'u10, v10 and wind_speed alike (direction unchanged).{floor_txt} '
                               f'Map: {Path(ba_map).name}')})
    first.close()

    print('=' * 78 + f'\nstream {len(files)} year files -> {out}\n' + '=' * 78)
    n_written, t_last, tail, worst, worst_p50 = 0, None, None, 0.0, 0.0
    for f in files:
        ds = xr.open_dataset(f)
        t = ds['time'].values
        if tail is not None:                          # overlap must agree
            c = np.intersect1d(tail['time'].values, t)
            if c.size:
                x0 = tail['hr_u'].sel(time=c).values
                x1 = ds['hr_u'].sel(time=c).values
                both = np.isfinite(x0) & np.isfinite(x1)
                if both.any():
                    worst = max(worst, float(np.abs(x0 - x1)[both].max()))
        keep = t > t_last if t_last is not None else np.ones(t.size, bool)
        if t_last is not None and keep.any():
            gap = (t[keep][0] - t_last).astype('timedelta64[h]').astype(int)
            if gap != 1:
                sys.exit(f'STOP: {Path(f).name} leaves a {gap} h step after {t_last}')
        sub = ds.isel(time=np.where(keep)[0])
        u = sub['hr_u'].values.astype('float32')
        v = sub['hr_v'].values.astype('float32')
        s = np.sqrt(u ** 2 + v ** 2)
        if n_written == 0:                            # speed from u/v must be the P50 level
            q50 = sub['hr_speed_q'].values[:, iq].astype('float32')
            both = np.isfinite(q50) & np.isfinite(s)
            worst_p50 = float(np.abs(q50 - s)[both].max())
            print(f'  |sqrt(u^2+v^2) - P50| max {worst_p50:.2e} on {Path(f).name}')
            if worst_p50 > 1e-3:
                sys.exit('STOP: u/v speed is not the P50 level')
        tt = netCDF4.date2num(sub['time'].values.astype('datetime64[s]').astype(dt.datetime),
                              TUNITS, 'standard')
        i0, i1 = n_written, n_written + tt.size
        for k, nc in writers.items():
            fac = None if k == 'raw' else F
            nc['time'][i0:i1] = tt
            for name, arr in zip(FIELDS, (u, v, s)):
                nc[name][i0:i1] = arr if fac is None else arr * fac[None]
        n_written = i1
        t_last = sub['time'].values[-1]
        tail = ds.isel(time=slice(-48, None))[['hr_u']].load()
        nanfrac = np.isnan(s).all(axis=(1, 2)).mean()
        print(f'  {Path(f).name}: +{tt.size} h -> {n_written} (dead steps {100 * nanfrac:.2f}%)', flush=True)
        ds.close()
    for nc in writers.values():
        nc.close()
    print(f'  max |diff| in year-boundary overlaps: {worst:.2e}')
    if worst > 1e-4:
        sys.exit('STOP: overlapping year files disagree')

    print('=' * 78 + '\nverify\n' + '=' * 78)
    ok = True
    rng = np.random.default_rng(0)
    for k in kinds:
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
            ss = src.sel(time=win)
            uu, vv = ss['hr_u'].values, ss['hr_v'].values
            want = {'u10': uu, 'v10': vv, 'wind_speed': np.sqrt(uu ** 2 + vv ** 2)}
            for name in FIELDS:
                w = want[name] if k == 'raw' else want[name] * F[None]
                got = d[name].sel(time=win).values
                both = np.isfinite(w) & np.isfinite(got)
                mx = float(np.abs(w - got)[both].max())
                nm = int((np.isfinite(w) ^ np.isfinite(got)).sum())
                ok &= mx < 1e-5 and nm == 0
                print(f'    spot {str(day)[:10]} {name:10s}: max|diff| {mx:.2e} NaN mismatch {nm}')
            src.close()
        d.close()
    if not ok:
        sys.exit('STOP: verification failed -- .tmp files left for inspection')
    for k in kinds:
        os.replace(str(names[k]) + '.tmp', names[k])
        print(f'  -> {names[k].name} ({os.path.getsize(names[k]) / 1e9:.1f} GB)')
    print('P50 PRODUCTS OK')


if __name__ == '__main__':
    main()

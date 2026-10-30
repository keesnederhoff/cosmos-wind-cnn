#!/usr/bin/env python
"""Stage the 2026-08-27 ERA5 drop as the five inference inputs, to 2026-08-10.

The drop lives in validation/era5/ and extends the record from 2026-07-26T18 to
2026-08-10 (~14.3 days). Staging is NOT a copy, because the four dynamic inputs
are not in the same state:

u / v / cloud
    Verified clean over the WHOLE axis: 0 duplicate stamps, 0 backward jumps,
    0 gaps, and 0 border NaN at any timestep. Bit-identical to the files the
    existing product was inferred from over 2022-06 and 2024-01. Copied through
    unchanged.

gust
    Two defects. (1) The recurring accumulated-variable seam damage: 312
    duplicate stamps and 26 backward jumps -- the same failure as the 2026-08-06
    drop, in the same three variables. Repaired with the existing
    fix_era5_time_axis.repair().
    (2) It carries REAL ERA5 values in the 83 border cells (23.25% of the grid)
    that every previous gust file had as NaN and that the pipeline filled by
    nearest-valid gather. THE MODEL WAS TRAINED ON THE FILLED VERSION. Feeding
    it real border values would be untested extrapolation on the western ~11 of
    162 target columns -- 28 of the 285 ERA5 cells the target interpolation
    touches are in that border. So the border is masked back to NaN and re-filled
    with the same fill_era5_border_nan.fill_border(), reproducing exactly what
    training saw and keeping 2000-2025 reproducible.

The border mask is taken from the OLD raw gust file, so it is the mask the
original fill actually used rather than one re-derived from different data.
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from fill_era5_border_nan import fill_border          # noqa: E402
from fix_era5_time_axis import repair                 # noqa: E402

B = '/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay'
SRC = B + '/validation/era5'
DST = B + '/sf_bay_rtma/raw_data'
OLD_GUST_RAW = DST + '/ERA5_wind_gust_2000_2026_UTM.nc'
OLD_GUST_FILL = DST + '/ERA5_wind_gust_2000_2026_UTM_filled.nc'

TAG = '2000_20260810'
CLEAN = [
    ('eastward_wind', 'ERA5_eastward_wind_2000_2026_UTM.nc',
     f'ERA5_eastward_wind_{TAG}_UTM.nc'),
    ('northward_wind', 'ERA5_northward_wind_2000_2026_UTM.nc',
     f'ERA5_northward_wind_{TAG}_UTM.nc'),
    ('cloud_area_fraction', 'ERA5_cloud_area_fraction_2000_2026_UTM.nc',
     f'ERA5_cloud_area_fraction_{TAG}_UTM.nc'),
]
GUST_SRC = SRC + '/ERA5_wind_gust_2000_2026_UTM.nc'
GUST_DEDUP = DST + f'/ERA5_wind_gust_{TAG}_UTM_dedup.nc'
GUST_OUT = DST + f'/ERA5_wind_gust_{TAG}_UTM_filled.nc'

ok = True


def audit(path, var):
    ds = xr.open_dataset(path, decode_timedelta=False)
    t = ds['time'].values
    d = np.diff(t).astype('timedelta64[h]').astype(int)
    a = ds[var].values
    flat = np.isnan(a.reshape(a.shape[0], -1))
    ncell = flat.shape[1]
    dead = flat.sum(axis=1) == ncell
    live = ~dead
    union = int(flat[live].any(axis=0).sum()) if live.any() else -1
    res = dict(n=t.size, t0=str(t[0])[:16], t1=str(t[-1])[:16],
               dup=t.size - np.unique(t).size, back=int((d < 0).sum()),
               gaps=int((d != 1).sum()), border=union, dead=int(dead.sum()),
               last_live=str(t[live][-1])[:16] if live.any() else 'n/a')
    ds.close()
    return res


print('=' * 78)
print('STEP 1  u / v / cloud -- verify clean, then copy through unchanged')
print('=' * 78)
for var, src_name, dst_name in CLEAN:
    src = f'{SRC}/{src_name}'
    a = audit(src, var)
    print(f'  {var:22s} n={a["n"]:7d}  {a["t0"]} -> {a["t1"]}')
    print(f'  {"":22s} dup={a["dup"]} back={a["back"]} gaps={a["gaps"]} '
          f'border-NaN={a["border"]} dead={a["dead"]}')
    if a['dup'] or a['back'] or a['gaps'] or a['border'] or a['dead']:
        print(f'  {"":22s} ABORT: expected a pristine file, got defects')
        ok = False
        continue
    shutil.copyfile(src, f'{DST}/{dst_name}')
    print(f'  {"":22s} -> copied to {dst_name}')

if not ok:
    sys.exit('STOP: u/v/cloud did not verify clean')

print()
print('=' * 78)
print('STEP 2  gust -- repair the time axis')
print('=' * 78)
repair(GUST_SRC, GUST_DEDUP)

print()
print('=' * 78)
print('STEP 3  gust -- restore the training-time synthetic border')
print('=' * 78)
dr = xr.open_dataset(OLD_GUST_RAW, decode_timedelta=False)
raw = dr['wind_gust'].values
f0 = np.isnan(raw.reshape(raw.shape[0], -1))
live0 = f0.sum(axis=1) != f0.shape[1]
border = f0[live0].any(axis=0).reshape(raw.shape[1], raw.shape[2])
dr.close()
print(f'  border mask from {os.path.basename(OLD_GUST_RAW)}: '
      f'{int(border.sum())} of {border.size} cells '
      f'({100 * border.mean():.2f}%)')

ds = xr.open_dataset(GUST_DEDUP, decode_timedelta=False)
da = ds['wind_gust']
vals = da.values.copy()
dead = np.isnan(vals).all(axis=(1, 2))
pre_nan = int(np.isnan(vals).sum())
live_idx = np.where(~dead)[0]
sub = vals[live_idx]
sub[:, border] = np.nan
vals[live_idx] = sub
masked = da.copy(data=vals)
print(f'  masked border to NaN on {int((~dead).sum())} live steps '
      f'(NaN cells {pre_nan} -> {int(np.isnan(vals).sum())})')

filled = fill_border(masked)
fv = filled.values
res_dead = np.isnan(fv).all(axis=(1, 2))
res_partial = (np.isnan(fv).any(axis=(1, 2)) & ~res_dead).sum()
print(f'  after fill: partial-NaN steps={int(res_partial)} '
      f'fully-NaN steps={int(res_dead.sum())}')
if res_partial:
    sys.exit('STOP: fill left partial spatial NaN')

out = filled.to_dataset(name='wind_gust')
for c in ('crs', 'spatial_ref'):
    if c in ds:
        out[c] = ds[c]
out['wind_gust'].attrs = dict(da.attrs)
out['wind_gust'].attrs['border_nan_filled'] = (
    'border masked to the 83-cell training-time mask and re-filled by '
    'nearest-valid gather, so inference preprocessing matches training')
out.attrs = dict(ds.attrs)
tmp = GUST_OUT + '.tmp'
out.to_netcdf(tmp, encoding={'wind_gust': {'zlib': True, 'complevel': 1}})
ds.close()
os.replace(tmp, GUST_OUT)
print(f'  wrote {os.path.basename(GUST_OUT)} '
      f'({os.path.getsize(GUST_OUT) / 1e6:.0f} MB)')

print()
print('=' * 78)
print('STEP 4  PROOF: treated gust must reproduce the old filled gust')
print('=' * 78)
dn = xr.open_dataset(GUST_OUT, decode_timedelta=False)
do = xr.open_dataset(OLD_GUST_FILL, decode_timedelta=False)
to = do['time'].values
_, i0 = np.unique(to, return_index=True)
do = do.isel(time=np.sort(i0))
common = np.intersect1d(dn['time'].values, do['time'].values)
vn = dn['wind_gust'].sel(time=common).values
vo = do['wind_gust'].sel(time=common).values
d = np.abs(vn - vo)
d[np.isnan(vn) & np.isnan(vo)] = 0.0
mism = int((np.isnan(vn) ^ np.isnan(vo)).sum())
fin = d[np.isfinite(d)]
print(f'  common steps          : {common.size}')
print(f'  max |new - old|       : {fin.max():.3e}')
print(f'  NaN-pattern mismatch  : {mism}')
inter = ~border
print(f'  max |diff| interior   : {np.nanmax(d[:, inter]):.3e}')
print(f'  max |diff| border     : {np.nanmax(d[:, border]):.3e}')
verdict = fin.max() < 1e-3 and mism == 0
print(f'  VERDICT: {"PASS" if verdict else "FAIL"} '
      f'(border now agrees with training treatment)')
dn.close()
do.close()

print()
print('=' * 78)
print('STEP 5  final audit of all four staged inputs')
print('=' * 78)
for var, _, dst_name in CLEAN:
    a = audit(f'{DST}/{dst_name}', var)
    print(f'  {dst_name[:52]:52s} n={a["n"]:7d} last_live={a["last_live"]}')
a = audit(GUST_OUT, 'wind_gust')
print(f'  {os.path.basename(GUST_OUT)[:52]:52s} n={a["n"]:7d} '
      f'last_live={a["last_live"]}')
print(f'  gust dup={a["dup"]} back={a["back"]} gaps={a["gaps"]} '
      f'border={a["border"]} dead={a["dead"]}')
print()
print('EFFECTIVE RECORD END =', a['last_live'])
if not verdict:
    sys.exit('STOP: gust did not reproduce the training-time treatment')
print('STAGING OK')

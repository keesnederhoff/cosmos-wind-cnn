#!/usr/bin/env python
"""Stage the 2026-09-28 ERA5 drop as ONE continuous 1940 -> 2026-08-10 input set.

The 2026-09-28 drop (sf_bay_rtma/raw_data/*_1940_2025_UTM.nc) runs 1940-01-01 to
2025-12-31 23:00. The 2026 tail is taken from the 2026-08-27 staging
(*_2000_20260810_*), which is what the current product was inferred from, so
the staged files are that product's inputs with 1940-1999 prepended.

u / v / cloud
    Must be pristine over the whole axis (0 dup, 0 backward, 0 gaps, 0 NaN).
    Concatenated through unchanged. PROOF: bit-identical to the 2026-08-27
    inputs over the WHOLE 2000-2025 overlap, not a sample window.

gust
    The new download again carries REAL values in the 83 border cells that the
    model was trained with as NaN + nearest-valid fill (same trap as 2026-08-27).
    The border is masked back to NaN and re-filled with
    fill_era5_border_nan.fill_border(). The mask is taken from the OLD raw gust
    file, i.e. the mask the training-time fill actually used.
    PROOF: agrees with the 2026-08-27 filled gust to float noise over the whole
    overlap. Steps that were dead (NaN) in the old file but are live in the new
    one are reported as a GAIN; the reverse is a LOSS and fails the gate.

Per-year NaN census over the full 1940-2026 axis is printed at the end:
scope validators to the whole axis, never the training window (e64b669).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from fill_era5_border_nan import fill_border          # noqa: E402
from fix_era5_time_axis import repair                 # noqa: E402

B = '/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay'
DST = B + '/sf_bay_rtma/raw_data'
OLD_GUST_RAW = DST + '/ERA5_wind_gust_2000_2026_UTM.nc'   # border-mask source
TAIL_START = np.datetime64('2026-01-01T00:00')
OVL = slice('2000-01-01', '2025-12-31T23:00')
TAG = '1940_20260810'

CLEAN = [  # var, new drop, 2026-08-27 staged (tail + reference), output
    ('eastward_wind', 'ERA5_eastward_wind_1940_2025_UTM.nc',
     'ERA5_eastward_wind_2000_20260810_UTM.nc', f'ERA5_eastward_wind_{TAG}_UTM.nc'),
    ('northward_wind', 'ERA5_northward_wind_1940_2025_UTM.nc',
     'ERA5_northward_wind_2000_20260810_UTM.nc', f'ERA5_northward_wind_{TAG}_UTM.nc'),
    ('cloud_area_fraction', 'ERA5_cloud_area_fraction_1940_2025_UTM.nc',
     'ERA5_cloud_area_fraction_2000_20260810_UTM.nc',
     f'ERA5_cloud_area_fraction_{TAG}_UTM.nc'),
]
GUST_NEW = DST + '/ERA5_wind_gust_1940_2025_UTM.nc'
GUST_DEDUP = DST + '/ERA5_wind_gust_1940_2025_UTM_dedup.nc'
GUST_REF = DST + '/ERA5_wind_gust_2000_20260810_UTM_filled.nc'
GUST_OUT = DST + f'/ERA5_wind_gust_{TAG}_UTM_filled.nc'


def audit(path, var):
    ds = xr.open_dataset(path, decode_timedelta=False)
    t = ds['time'].values
    d = np.diff(t).astype('timedelta64[h]').astype(int)
    a = ds[var].values
    flat = np.isnan(a.reshape(a.shape[0], -1))
    dead = flat.all(axis=1)
    live = ~dead
    union = int(flat[live].any(axis=0).sum()) if live.any() else -1
    res = dict(n=t.size, t0=str(t[0])[:16], t1=str(t[-1])[:16],
               dup=t.size - np.unique(t).size, back=int((d < 0).sum()),
               gaps=int((d > 1).sum()), border=union, dead=int(dead.sum()),
               last_live=str(t[live][-1])[:16] if live.any() else 'n/a')
    ds.close()
    return res


def show(tag, a):
    print(f'  {tag:22s} n={a["n"]:7d}  {a["t0"]} -> {a["t1"]}')
    print(f'  {"":22s} dup={a["dup"]} back={a["back"]} gaps={a["gaps"]} '
          f'border-NaN={a["border"]} dead={a["dead"]}')


def splice(head_ds, tail_path, var):
    """head (1940-2025) + tail (>= 2026-01-01) with a seam check."""
    tl = xr.open_dataset(tail_path, decode_timedelta=False)
    tl = tl.sel(time=slice(TAIL_START, None))
    h1, t0 = head_ds['time'].values[-1], tl['time'].values[0]
    step = (t0 - h1).astype('timedelta64[h]').astype(int)
    print(f'  seam: head ends {str(h1)[:16]}, tail starts {str(t0)[:16]} '
          f'(step {step} h)')
    if step != 1:
        sys.exit(f'STOP: {var} seam is not a 1-h step')
    da = xr.concat([head_ds[var].load(), tl[var].load()], dim='time')
    tl.close()
    return da


def write(da, var, like, out_path, extra_attr=None):
    out = da.to_dataset(name=var)
    for c in ('crs', 'spatial_ref'):
        if c in like:
            out[c] = like[c]
    out[var].attrs = dict(like[var].attrs)
    if extra_attr:
        out[var].attrs.update(extra_attr)
    out.attrs = dict(like.attrs)
    out.attrs['staging'] = ('stage_era5_1940.py: 1940-2025 from the 2026-09-28 '
                            'drop + 2026 tail from the 2026-08-27 staging')
    tmp = out_path + '.tmp'
    out.to_netcdf(tmp, encoding={var: {'zlib': True, 'complevel': 1,
                                       'dtype': 'float32'}})
    os.replace(tmp, out_path)
    print(f'  wrote {os.path.basename(out_path)} '
          f'({os.path.getsize(out_path) / 1e6:.0f} MB)')


def compare(new_path, ref_path, var, window):
    dn = xr.open_dataset(new_path, decode_timedelta=False)
    do = xr.open_dataset(ref_path, decode_timedelta=False)
    common = np.intersect1d(dn['time'].sel(time=window).values,
                            do['time'].sel(time=window).values)
    vn = dn[var].sel(time=common).values
    vo = do[var].sel(time=common).values
    both = np.isfinite(vn) & np.isfinite(vo)
    mx = float(np.abs(vn - vo)[both].max()) if both.any() else np.nan
    gain = int((np.isfinite(vn) & np.isnan(vo)).sum())
    loss = int((np.isnan(vn) & np.isfinite(vo)).sum())
    ref_steps = do['time'].sel(time=window).size
    dn.close()
    do.close()
    return common.size, ref_steps, mx, gain, loss


ok = True
print('=' * 78)
print('STEP 1  u / v / cloud -- verify pristine, splice the 2026 tail, prove')
print('=' * 78)
for var, new_name, ref_name, out_name in CLEAN:
    a = audit(f'{DST}/{new_name}', var)
    show(var, a)
    if a['dup'] or a['back'] or a['gaps'] or a['border'] or a['dead']:
        print(f'  {"":22s} ABORT: expected a pristine file, got defects')
        ok = False
        continue
    hd = xr.open_dataset(f'{DST}/{new_name}', decode_timedelta=False)
    da = splice(hd, f'{DST}/{ref_name}', var)
    write(da, var, hd, f'{DST}/{out_name}')
    hd.close()
    n, nref, mx, gain, loss = compare(f'{DST}/{out_name}', f'{DST}/{ref_name}',
                                      var, slice('2000-01-01', None))
    good = n == nref and mx == 0.0 and gain == 0 and loss == 0
    print(f'  PROOF vs {ref_name}: common={n}/{nref} max|diff|={mx:.3e} '
          f'gain={gain} loss={loss} -> {"PASS" if good else "FAIL"}')
    ok &= good
if not ok:
    sys.exit('STOP: u/v/cloud did not verify')

print()
print('=' * 78)
print('STEP 2  gust -- time axis')
print('=' * 78)
a = audit(GUST_NEW, 'wind_gust')
show('wind_gust (new)', a)
gsrc = GUST_NEW
if a['dup'] or a['back']:
    repair(GUST_NEW, GUST_DEDUP)
    gsrc = GUST_DEDUP
if audit(gsrc, 'wind_gust')['gaps']:
    sys.exit('STOP: gust has gaps after repair')

print()
print('=' * 78)
print('STEP 3  gust -- restore the training-time synthetic border')
print('=' * 78)
dr = xr.open_dataset(OLD_GUST_RAW, decode_timedelta=False)
raw = dr['wind_gust'].values
f0 = np.isnan(raw.reshape(raw.shape[0], -1))
live0 = ~f0.all(axis=1)
border = f0[live0].any(axis=0).reshape(raw.shape[1], raw.shape[2])
dr.close()
print(f'  border mask from {os.path.basename(OLD_GUST_RAW)}: '
      f'{int(border.sum())} of {border.size} cells ({100 * border.mean():.2f}%)')
if int(border.sum()) != 83:
    sys.exit('STOP: expected the 83-cell training-time border mask')

ds = xr.open_dataset(gsrc, decode_timedelta=False)
da = ds['wind_gust'].load()
vals = da.values.copy()
dead = np.isnan(vals).all(axis=(1, 2))
live_idx = np.where(~dead)[0]
sub = vals[live_idx]
sub[:, border] = np.nan
vals[live_idx] = sub
filled = fill_border(da.copy(data=vals))
fv = filled.values
res_dead = np.isnan(fv).all(axis=(1, 2))
res_partial = int((np.isnan(fv).any(axis=(1, 2)) & ~res_dead).sum())
print(f'  after fill: partial-NaN steps={res_partial} '
      f'fully-NaN steps={int(res_dead.sum())}')
if res_partial:
    sys.exit('STOP: fill left partial spatial NaN')
head = filled.to_dataset(name='wind_gust')
gda = splice(head, GUST_REF, 'wind_gust')

# The new drop has the usual 7 dead seam hours at 2000-01-01T00-06 (as every
# year), but the 2026-08-27 inputs had LIVE values there -- their 2000-2026
# download started early enough to carry the accumulation. Found 2026-09-28 as
# 2499 = 7 x 357 'loss' cells. They are genuine ERA5 values the current product
# used, so copy them back: keeps year 2000 reproducible, loses nothing.
ref = xr.open_dataset(GUST_REF, decode_timedelta=False)['wind_gust'].load()
common = np.intersect1d(gda['time'].values, ref['time'].values)
nd = gda.sel(time=common).isnull().all(('y', 'x')).values
rl = ref.sel(time=common).notnull().all(('y', 'x')).values
patch = common[nd & rl]
print(f'  patched {patch.size} dead-new/live-ref steps from the 08-27 gust: '
      f'{[str(t)[:13] for t in patch]}')
if patch.size > 7:
    sys.exit('STOP: more dead-new/live-ref steps than the known 2000 seam')
gda.loc[dict(time=patch)] = ref.sel(time=patch).values
ref.close()
write(gda, 'wind_gust', ds, GUST_OUT, {
    'border_nan_filled': 'border masked to the 83-cell training-time mask and '
                         're-filled by nearest-valid gather, so inference '
                         'preprocessing matches training'})
ds.close()

print()
print('=' * 78)
print('STEP 4  PROOF: treated gust must reproduce the 2026-08-27 filled gust')
print('=' * 78)
n, nref, mx, gain, loss = compare(GUST_OUT, GUST_REF, 'wind_gust',
                                  slice('2000-01-01', None))
gverdict = mx < 1e-3 and loss == 0 and n == nref
print(f'  common steps {n}/{nref}  max|diff| on shared-finite {mx:.3e}')
print(f'  gain (new live, old dead) {gain} cells   loss (new dead, old live) {loss}')
print(f'  VERDICT: {"PASS" if gverdict else "FAIL"}')

print()
print('=' * 78)
print('STEP 5  full-axis audit + per-year NaN census of the four staged inputs')
print('=' * 78)
staged = [(v, f'{DST}/{o}') for v, _, _, o in CLEAN] + [('wind_gust', GUST_OUT)]
census = {}
for var, p in staged:
    a = audit(p, var)
    show(os.path.basename(p)[:22], a)
    ok &= not (a['dup'] or a['back'] or a['gaps'] or a['border'])
    d = xr.open_dataset(p, decode_timedelta=False)
    v = d[var].values
    yrs = pd.DatetimeIndex(d['time'].values).year
    dd = np.isnan(v).all(axis=(1, 2))
    pp = np.isnan(v).any(axis=(1, 2)) & ~dd
    census[var] = pd.DataFrame({'dead': dd, 'partial': pp}, index=yrs) \
                    .groupby(level=0).sum()
    d.close()
tab = pd.concat(census, axis=1)
bad = tab[(tab.xs('partial', axis=1, level=1) > 0).any(axis=1)
          | (tab.xs('dead', axis=1, level=1) > 7).any(axis=1)]
print('  years with any partial-NaN step or >7 dead steps in any input:')
print(bad.to_string() if len(bad) else '  none')
print('  totals:')
print(tab.sum().to_string())
print()
print('STAGING OK' if (ok and gverdict) else 'STAGING FAILED')
sys.exit(0 if (ok and gverdict) else 1)

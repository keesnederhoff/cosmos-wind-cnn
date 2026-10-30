#!/usr/bin/env python
"""Rotate RTMA grid-relative 10 m u/v (v4 drop) to earth-relative (true east/north).

WHY. The 2026-10-27 audit of sf_bay_rtma/raw_data found the RTMA u/v vectors rotated
~11.9 deg (10.6-12.9, smooth W->E) from the v3 files, which reproduce RTMA's native
wind_direction exactly. The v4 export kept the NCEP Lambert-grid-relative components.
Rotation by the local grid convergence angle restores true east/north.

METHOD. Convergence alpha = n * (lon - lon0), n = sin(phi1) = sin(25 deg), lon0 = -95 deg
(NDFD CONUS 2.5 km Lambert conformal). Cell lon from the UTM10 x/y. The sign is not
trusted: both signs are tried and the one that matches v3 wins. If the analytic map does
not pass the gate, an EMPIRICAL angle map (circular mean of angle(v3) - angle(v4) over
many steps, speed-weighted) is used instead, and gated again.

GATE (exit 1 on failure, no output is left behind as a finished file). Over N random
timesteps present in BOTH v3 and v4, rotated-v4 vs v3:
    rms(du, dv)            < 0.03 m/s   (v4 speed itself differs from native by ~1e-2)
    max |du|, |dv|         < 0.10 m/s
    mean |dir - v3 dir|    < 0.2 deg  where speed > 2 m/s
The tail (v4 steps after v3's end) cannot be checked against v3; it is rotated with the
same map, which is time-invariant (verified by the same gate across 2011-2026).

Output (next to the originals; originals untouched):
    RTMA_SFbay_2p5km_{eastward,northward}_wind_2011_2026_UTM10_earthrel.nc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
from pyproj import Transformer

ROOT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay')
RAW = ROOT / 'sf_bay_rtma' / 'raw_data'
V3 = ROOT / 'archive' / 'sf_bay_rtma_v2' / 'raw_data'
STEM = 'RTMA_SFbay_2p5km_{}_2011_2026_UTM10'
CHUNK = 1680                                  # 10 x the on-disk time chunk (168)

RMS_MAX, ABS_MAX, DIR_MAX = 0.03, 0.10, 0.2


def log(*a):
    print(*a, flush=True)


def cell_lon(x, y):
    tr = Transformer.from_crs('EPSG:32610', 'EPSG:4326', always_xy=True)
    xx, yy = np.meshgrid(x.astype('float64'), y.astype('float64'))
    lon, _ = tr.transform(xx, yy)
    return lon


def rotate(u, v, alpha_deg):
    a = np.deg2rad(alpha_deg).astype('float32')
    c, s = np.cos(a), np.sin(a)
    return u * c - v * s, u * s + v * c


def read_steps(ds, var, idx):
    """Read arbitrary time indices (sorted) one at a time -> (n, y, x) float32."""
    return np.stack([ds.variables[var][int(i)] for i in idx]).astype('float32')


def common_steps(t4, t3, n, seed=0):
    ts, i4, i3 = np.intersect1d(t4, t3, return_indices=True)
    pick = np.sort(np.random.default_rng(seed).choice(len(ts), min(n, len(ts)), replace=False))
    return ts[pick], i4[pick], i3[pick]


def dir_index(t3d, ts):
    """Indices of ts in the v3 wind_direction time axis (its own axis, NOT the u file's)."""
    j = np.searchsorted(t3d, ts)
    assert j.max() < len(t3d) and np.array_equal(t3d[j], ts), 'wind_direction file lacks gate steps'
    return j


def gate(alpha, ds4u, ds4v, ds3u, ds3v, ds3d, i4, i3, i3d, label):
    u4 = read_steps(ds4u, 'eastward_wind', i4)
    v4 = read_steps(ds4v, 'northward_wind', i4)
    u3 = read_steps(ds3u, 'eastward_wind', i3)
    v3 = read_steps(ds3v, 'northward_wind', i3)
    wd = read_steps(ds3d, 'wind_direction', i3d)
    ur, vr = rotate(u4, v4, alpha[None])
    du, dv = ur - u3, vr - v3
    m = np.isfinite(du) & np.isfinite(dv)
    rms = float(np.sqrt(np.mean(np.concatenate([du[m], dv[m]]) ** 2)))
    mx = float(max(np.abs(du[m]).max(), np.abs(dv[m]).max()))
    sp = np.hypot(ur, vr)
    from_dir = (270.0 - np.degrees(np.arctan2(vr, ur))) % 360.0
    dd = np.abs((from_dir - wd + 180.0) % 360.0 - 180.0)
    mm = m & (sp > 2) & np.isfinite(wd)
    dmean = float(dd[mm].mean())
    ok = rms < RMS_MAX and mx < ABS_MAX and dmean < DIR_MAX
    log(f'  gate[{label}] n={len(i4)} rms={rms:.4f} max={mx:.4f} dir_vs_native={dmean:.3f} deg '
        f'-> {"PASS" if ok else "FAIL"}')
    return ok, rms


def empirical_alpha(ds4u, ds4v, ds3u, ds3v, i4, i3):
    u4 = read_steps(ds4u, 'eastward_wind', i4); v4 = read_steps(ds4v, 'northward_wind', i4)
    u3 = read_steps(ds3u, 'eastward_wind', i3); v3 = read_steps(ds3v, 'northward_wind', i3)
    d = np.arctan2(v3, u3) - np.arctan2(v4, u4)
    w = np.where((np.hypot(u3, v3) > 3) & np.isfinite(d), np.hypot(u3, v3), 0.0)
    d = np.where(np.isfinite(d), d, 0.0)
    return np.degrees(np.arctan2((w * np.sin(d)).sum(0), (w * np.cos(d)).sum(0)))


def make_out(src_path, out_path, var):
    src = netCDF4.Dataset(src_path)
    src.set_auto_mask(False)
    out = netCDF4.Dataset(out_path, 'w', format='NETCDF4')
    for k, d in src.dimensions.items():
        out.createDimension(k, None if d.isunlimited() else len(d))
    for k, v in src.variables.items():
        kw = {}
        if k == var:
            kw = dict(zlib=True, complevel=1, shuffle=True, chunksizes=(168, 123, 162),
                      fill_value=np.float32(np.nan))
        o = out.createVariable(k, v.dtype, v.dimensions, **kw)
        o.setncatts({a: v.getncattr(a) for a in v.ncattrs() if a != '_FillValue'})
        if k != var:
            o[:] = v[:]
    out.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
    out.history = (str(src.history) + ' | rotated to earth-relative by '
                   'rotate_rtma_earthrel.py (Lambert grid convergence)')
    out.variables[var].long_name = out.variables[var].long_name + ' (earth-relative)'
    return src, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-gate', type=int, default=400)
    ap.add_argument('--gate-only', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--verify-only', action='store_true',
                    help='re-prove the EXISTING *_earthrel.nc files against v3 (chain resume)')
    a = ap.parse_args()

    pu, pv = RAW / (STEM.format('eastward_wind') + '.nc'), RAW / (STEM.format('northward_wind') + '.nc')
    ou, ov = (RAW / (STEM.format('eastward_wind') + '_earthrel.nc'),
              RAW / (STEM.format('northward_wind') + '_earthrel.nc'))
    for o in (ou, ov):
        if o.exists() and not a.overwrite and not a.gate_only and not a.verify_only:
            sys.exit(f'REFUSING to overwrite {o} (use --overwrite)')

    d4u, d4v = netCDF4.Dataset(pu), netCDF4.Dataset(pv)
    d3u = netCDF4.Dataset(V3 / (STEM.format('eastward_wind') + '.nc'))
    d3v = netCDF4.Dataset(V3 / (STEM.format('northward_wind') + '.nc'))
    d3d = netCDF4.Dataset(V3 / (STEM.format('wind_direction') + '.nc'))
    for _d in (d4u, d4v, d3u, d3v, d3d):
        _d.set_auto_mask(False)           # NaN stays NaN; no masked-array surprises
    t4 = d4u.variables['time'][:]; t3 = d3u.variables['time'][:]
    log(f'v4 {len(t4)} steps, v3 {len(t3)} steps; units {d4u.variables["time"].units} / '
        f'{d3u.variables["time"].units}')
    assert d4u.variables['time'].units == d3u.variables['time'].units, 'time units differ'
    assert np.array_equal(t4, d4v.variables['time'][:]), 'v4 u/v time axes differ'
    t3d = d3d.variables['time'][:]
    ts, i4, i3 = common_steps(t4, t3, a.n_gate)
    i3d = dir_index(t3d, ts)
    tsj, j4, j3 = common_steps(t4, t3, 5, seed=1)
    j3d = dir_index(t3d, tsj)

    if a.verify_only:
        ru, rv = netCDF4.Dataset(ou), netCDF4.Dataset(ov)
        for _d in (ru, rv):
            _d.set_auto_mask(False)
        same = (ru.variables['eastward_wind'].shape == d4u.variables['eastward_wind'].shape
                and np.array_equal(ru.variables['time'][:], t4))
        okv, _ = gate(np.zeros((len(d4u.variables['y']), len(d4u.variables['x'])), 'float32'),
                      ru, rv, d3u, d3v, d3d, i4, i3, i3d, 'EXISTING earthrel files vs v3')
        sys.exit(0 if (okv and same) else 1)

    x = d4u.variables['x'][:]; y = d4u.variables['y'][:]
    lon = cell_lon(x, y)
    n = np.sin(np.deg2rad(25.0))
    conv = n * (lon + 95.0)                       # degrees; negative west of -95
    log(f'analytic convergence {conv.min():.2f}..{conv.max():.2f} deg (n={n:.4f})')

    # pick the sign on 5 steps, then run the full gate
    best = min((1.0, -1.0), key=lambda s: gate(s * conv, d4u, d4v, d3u, d3v, d3d, j4, j3, j3d,
                                               f'sign {s:+.0f} (5 steps)')[1])
    alpha = best * conv
    ok, _ = gate(alpha, d4u, d4v, d3u, d3v, d3d, i4, i3, i3d, f'analytic sign {best:+.0f}')
    src = 'analytic'
    if not ok:
        log('analytic map failed the gate -> empirical map from v3 - v4')
        _, k4, k3 = common_steps(t4, t3, 600, seed=2)
        alpha = empirical_alpha(d4u, d4v, d3u, d3v, k4, k3)
        log(f'empirical alpha {alpha.min():.2f}..{alpha.max():.2f} deg '
            f'(analytic*sign range {np.abs(conv).min():.2f}..{np.abs(conv).max():.2f})')
        ok, _ = gate(alpha, d4u, d4v, d3u, d3v, d3d, i4, i3, i3d, 'empirical')
        src = 'empirical'
    if not ok:
        sys.exit('GATE FAILED: rotated v4 does not reproduce v3 on the overlap. Nothing written.')
    if a.gate_only:
        log('gate-only: PASS'); return

    su, uo = make_out(pu, ou, 'eastward_wind')
    sv, vo = make_out(pv, ov, 'northward_wind')
    for o in (uo, vo):
        o.rotation_source = src
        o.rotation_deg_range = f'{float(alpha.min()):.3f}..{float(alpha.max()):.3f}'
    nt = len(t4)
    for t0 in range(0, nt, CHUNK):
        t1 = min(t0 + CHUNK, nt)
        u = d4u.variables['eastward_wind'][t0:t1].astype('float32')
        v = d4v.variables['northward_wind'][t0:t1].astype('float32')
        ur, vr = rotate(u, v, alpha[None])
        uo.variables['eastward_wind'][t0:t1] = ur
        vo.variables['northward_wind'][t0:t1] = vr
        if (t0 // CHUNK) % 10 == 0:
            log(f'  wrote {t1}/{nt}')
    for o in (uo, vo, su, sv):
        o.close()
    # re-open and re-prove from the files actually written
    ru, rv = netCDF4.Dataset(ou), netCDF4.Dataset(ov)
    ru.set_auto_mask(False); rv.set_auto_mask(False)
    assert ru.variables['eastward_wind'].shape == d4u.variables['eastward_wind'].shape
    ok2, _ = gate(np.zeros_like(alpha), ru, rv, d3u, d3v, d3d, i4, i3, i3d, 'WRITTEN FILES vs v3')
    if not ok2:
        for p in (ou, ov):
            p.rename(p.with_suffix('.nc.FAILED'))
        sys.exit('POST-WRITE GATE FAILED; outputs renamed *.FAILED')
    log(f'DONE rotation ({src}); wrote {ou.name}, {ov.name}')


if __name__ == '__main__':
    main()

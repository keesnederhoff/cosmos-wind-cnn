#!/usr/bin/env python
"""Follow-up tests on the v4 product. Reads existing outputs; changes no product.

A. Tapered BA: speed' = s * (1 + (f - 1) * w(s)), w ramps 0 -> 1 between s0 and s1, with f the
   LEAVE-ONE-STATION-OUT field factor (pred_gauss_loo). s0 = s1 = 0 is the current flat BA.
B. Quantile spread: fit one inflation c_tau per level on WY2025 stations,
   q' = P50 + c_tau (q - P50), evaluate coverage and pinball loss on other windows.
C. Baselines: same reliability for the v3 station points, and v4 reliability against RTMA itself
   over the grid (WY2026), which removes the point-versus-cell mismatch of station obs.
Output: <results>/v1940_product/FOLLOWUP_EVAL.txt
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay')
RES = ROOT / 'sf_bay_rtma_v4' / 'results'
V3RES = ROOT / 'sf_bay_rtma_v3' / 'results'
TABLE = V3RES / 'v1940_points' / 'station_table.nc'
V3SEED = 'r1b_do010_s3'
WINDOWS = {
    'WY2025 (fit)': ('2024-10-01', '2025-09-30T23'),
    'WY2026 (test)': ('2025-10-01', '2026-10-01T23'),
    '2011-2019': ('2011-01-01', '2019-12-31T23'),
    '2000-2010': ('2000-01-01', '2010-12-31T23'),
}
TAPERS = [(0, 0), (2, 6), (3, 8), (4, 10), (6, 12)]
OUT: list[str] = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def pairs(vmm, files, groups):
    parts = [xr.open_dataset(f)[['hr_speed_q']].load() for f in files
             if int(Path(f).name.split('_')[-2][:4]) >= 2000]
    ds = xr.concat(parts, dim='time')
    _, first = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=np.sort(first))
    taus = ds['quantile'].values
    im = int(np.argmin(np.abs(taus - 0.5)))
    rows = []
    for sid, cfg in vmm.STATIONS.items():
        if groups.get(sid) not in ('IEM', 'NDBC') or sid not in ds.station.values:
            continue
        obs = vmm.load_station(sid, cfg)
        if obs is None or not np.isfinite(obs['speed10']).any():
            continue
        q = ds['hr_speed_q'].sel(station=sid)
        mv, ov, tt = vmm.match_timeseries(q.values[:, im], q.time.values, obs['speed10'], obs['time'])
        if mv is None:
            continue
        df = pd.DataFrame(q.sel(time=tt).values, columns=[f'q{t:.4f}' for t in taus])
        df['obs'], df['time'], df['station'] = ov, pd.DatetimeIndex(tt), sid
        rows.append(df)
    D = pd.concat(rows, ignore_index=True)
    cols = [f'q{t:.4f}' for t in taus]
    D = D[np.isfinite(D['obs']) & np.isfinite(D[cols]).all(axis=1)]
    return D, taus, cols


def win(D, name):
    a, b = WINDOWS[name]
    return D[(D.time >= a) & (D.time <= b)]


def taper_test(D, cols, taus):
    f = pd.read_csv(sorted(glob.glob(str(RES / 'v1940_product' / 'BA_station_factors*.csv')))[0])
    fac = f.set_index('station')['pred_gauss_loo'].clip(lower=1.0)
    p50 = cols[int(np.argmin(np.abs(taus - 0.5)))]
    d = D[D.station.isin(fac.dropna().index)].copy()
    d['f'] = d.station.map(fac)
    d['s'] = d[p50]
    say('\n=== A. TAPERED BA (leave-one-station-out factor, station mean / median skill) ===')
    say(f'{len(fac.dropna())} stations; skill = 1 - MSE/var(obs) per station; top = obs >= station P90')
    for name in WINDOWS:
        w = win(d, name)
        if not len(w):
            continue
        say(f'\n{name}: n={len(w)}')
        say('  variant        mean_sk  med_sk   bias   rmse  top_bias  top_rmse')
        rows = [('raw', w['s'].values)]
        for s0, s1 in TAPERS:
            r = np.ones(len(w)) if s1 == 0 else np.clip((w['s'].values - s0) / (s1 - s0), 0, 1)
            rows.append((f'flat BA' if s1 == 0 else f'taper {s0}-{s1}', w['s'].values * (1 + (w['f'].values - 1) * r)))
        for tag, m in rows:
            e = w[['station', 'obs']].assign(m=m)
            sk, bi, rm, tb, tr = [], [], [], [], []
            for _, g in e.groupby('station'):
                if len(g) < 500 or g.obs.var() <= 0:
                    continue
                er = g.m - g.obs
                sk.append(1 - (er ** 2).mean() / g.obs.var())
                bi.append(er.mean())
                rm.append(np.sqrt((er ** 2).mean()))
                t = g.obs >= g.obs.quantile(0.9)
                tb.append(er[t].mean())
                tr.append(np.sqrt((er[t] ** 2).mean()))
            say(f'  {tag:13s} {np.mean(sk):7.3f} {np.median(sk):7.3f} {np.mean(bi):6.3f} {np.mean(rm):6.3f} '
                f'{np.mean(tb):8.3f} {np.mean(tr):8.3f}')


def pinball(q, o, t):
    e = o - q
    return float(np.mean(np.maximum(t * e, (t - 1) * e)))


def spread_test(D, cols, taus):
    im = int(np.argmin(np.abs(taus - 0.5)))
    fit = win(D, 'WY2025 (fit)')
    say('\n=== B. QUANTILE SPREAD (inflation fitted on WY2025 only) ===')
    say(f'P(obs < 0.5 m/s) on WY2025 = {(fit.obs < 0.5).mean():.3f} (calm reports limit the lower quantiles)')
    grid = np.linspace(0, 6, 241)
    cs = np.ones(len(taus))
    for i, t in enumerate(taus):
        if i == im:
            continue
        d = (fit[cols[i]] - fit[cols[im]]).values
        cov = [np.mean(fit.obs.values <= fit[cols[im]].values + c * d) for c in grid]
        cs[i] = grid[int(np.argmin(np.abs(np.array(cov) - t)))]
    say('levels ', np.round(taus, 4).tolist())
    say('c_tau  ', np.round(cs, 2).tolist())
    for name in WINDOWS:
        w = win(D, name)
        if not len(w):
            continue
        p = w[cols[im]].values
        raw, rec, pb0, pb1 = [], [], [], []
        for i, t in enumerate(taus):
            q0 = w[cols[i]].values
            q1 = p + cs[i] * (q0 - p)
            raw.append(np.mean(w.obs.values <= q0))
            rec.append(np.mean(w.obs.values <= q1))
            pb0.append(pinball(q0, w.obs.values, t))
            pb1.append(pinball(q1, w.obs.values, t))
        say(f'\n{name}: n={len(w)} stations={w.station.nunique()}')
        say('  nominal ', ' '.join(f'{v:6.3f}' for v in taus))
        say('  raw     ', ' '.join(f'{v:6.3f}' for v in raw))
        say('  inflated', ' '.join(f'{v:6.3f}' for v in rec))
        say(f'  mean pinball raw {np.mean(pb0):.4f} inflated {np.mean(pb1):.4f}; '
            f'top level raw {pb0[-1]:.4f} inflated {pb1[-1]:.4f}')
    return cs


def v3_baseline(vmm, groups):
    say('\n=== C1. v3 STATION RELIABILITY (same windows) ===')
    files = sorted(glob.glob(str(V3RES / 'v1940_points' / V3SEED / 'points_speed_full_record_ERA5_*.nc')))
    if not files:
        say('no v3 point files')
        return
    with xr.open_dataset(files[0]) as d:
        if 'hr_speed_q' not in d:
            say('v3 point files have no hr_speed_q:', list(d.data_vars))
            return
    D, taus, cols = pairs(vmm, files, groups)
    for name in WINDOWS:
        w = win(D, name)
        if len(w):
            say(f'{name}: n={len(w)}')
            say('  nominal ', ' '.join(f'{v:6.3f}' for v in taus))
            say('  v3      ', ' '.join(f'{np.mean(w.obs.values <= w[c].values):6.3f}' for c in cols))


def rtma_baseline():
    say('\n=== C2. v4 RELIABILITY AGAINST RTMA OVER THE GRID, WY2026 (every 7th hour) ===')
    seed = (RES / 'chosen_seed.txt').read_text().strip()
    inf = xr.open_dataset(RES / seed / 'output_inference' / 'speed_full_record_ERA5_20251001_20261002.nc')
    tru = xr.open_dataset(RES / 'v4data' / 'data_processed' / 'test.nc')
    t = np.intersect1d(inf.time.values, tru.time.values)[::7]
    q = inf['hr_speed_q'].sel(time=t)
    qd = [d for d in q.dims if d not in ('time', 'y', 'x')][0]
    q = q.transpose('time', qd, 'y', 'x')
    s = np.hypot(tru['hr_u'].sel(time=t).values, tru['hr_v'].sel(time=t).values)
    tv = q[qd].values
    if tv.size > 8:
        idx = [int(np.argmin(np.abs(tv - v))) for v in (0.049, 0.108, 0.245, 0.5, 0.755, 0.892, 0.951, 0.990)]
    else:
        idx = list(range(tv.size))
    say('hours', len(t), 'levels available', tv.size)
    say('  nominal ', ' '.join(f'{tv[i]:6.3f}' for i in idx))
    vals = []
    for i in idx:
        a = q.isel({qd: i}).values
        m = np.isfinite(a) & np.isfinite(s)
        vals.append(np.mean(s[m] <= a[m]))
    say('  vs RTMA ', ' '.join(f'{v:6.3f}' for v in vals))


def main():
    os.environ.setdefault('VAL_USGS', '0')
    sys.path.insert(0, str(HERE.parent / 'validation'))
    import validate_met_models as vmm                      # noqa: E402

    tab = xr.open_dataset(TABLE)
    groups = dict(zip(tab.station.values, tab.group.values))
    seed = (RES / 'chosen_seed.txt').read_text().strip()
    D, taus, cols = pairs(vmm, sorted(glob.glob(str(RES / 'v1940_points' / seed / 'points_speed_full_record_ERA5_*.nc'))), groups)
    say('v4 pairs', len(D), 'stations', D.station.nunique(), 'levels', np.round(taus, 4).tolist())
    taper_test(D, cols, taus)
    spread_test(D, cols, taus)
    v3_baseline(vmm, groups)
    rtma_baseline()
    (RES / 'v1940_product' / 'FOLLOWUP_EVAL.txt').write_text('\n'.join(OUT) + '\n')
    say('wrote FOLLOWUP_EVAL.txt')


if __name__ == '__main__':
    main()

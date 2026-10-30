#!/usr/bin/env python
"""File-based gates between the stages of the unattended chain.

Usage: python scripts/gate.py {prep|train|points|grids|dirtrack|products}
Exit 0 = PASS, 1 = FAIL (the next stage is `afterok` on this job, so a FAIL stops the
chain; `--kill-on-invalid-dep=yes` then cancels everything downstream).

Nothing here trusts sacct State (it has reported COMPLETED for crashes): every check
reads files that the stage had to produce.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay')
RES = ROOT / 'sf_bay_rtma' / 'results'
SEEDS = ['s1', 's2', 's3']
FAILS: list[str] = []


def ok(msg):
    print(f'  PASS  {msg}', flush=True)


def bad(msg):
    print(f'  FAIL  {msg}', flush=True)
    FAILS.append(msg)


def check(cond, msg):
    (ok if cond else bad)(msg)
    return bool(cond)


def tfirst_last(path):
    with xr.open_dataset(path) as d:
        t = d['time'].values
    return t[0], t[-1], t.size


# --------------------------------------------------------------------------- prep
def gate_prep():
    dp = RES / 'data' / 'data_processed'
    exp = {  # split -> (first >=, first <=, last >=, last <=)  (gap_hours=48 drops the tail of train/val)
        'train': ('2019-01-01', '2019-01-02T12', '2024-09-28T12', '2024-09-30T00'),
        'val':   ('2024-10-01T00', '2024-10-01T12', '2025-09-28T12', '2025-09-30T00'),
        'test':  ('2025-10-01T00', '2025-10-01T12', '2026-09-30T00', '2026-10-02'),
    }
    storm_min = {'train': 300, 'val': 60, 'test': 60}
    for sp, (f0, f1, l0, l1) in exp.items():
        mm = dp / 'memmap' / sp
        meta = json.loads((mm / 'meta.json').read_text())
        t0, t1, n = tfirst_last(dp / f'{sp}.nc')
        check(meta['T'] == n, f'{sp}: memmap T {meta["T"]} == nc steps {n}')
        check(np.datetime64(f0) <= t0 <= np.datetime64(f1), f'{sp}: first step {str(t0)[:16]} in [{f0},{f1}]')
        check(np.datetime64(l0) <= t1 <= np.datetime64(l1), f'{sp}: last step {str(t1)[:16]} in [{l0},{l1}]')
        need = {'hr_u', 'hr_v', 'hr_gust', 'lr_u', 'lr_v', 'lr_gust', 'lr_cloud', 'static_terrain'}
        check(need <= set(meta['vars']), f'{sp}: channels {sorted(meta["vars"])}')
        nan_t = np.load(mm / 'nan_at_time.npy')
        usable = 1.0 - nan_t.mean()
        check(usable > 0.97, f'{sp}: usable steps {100 * usable:.2f}% ({int(nan_t.sum())} NaN steps of {n})')
        T, H, W = meta['T'], meta['H'], meta['W']
        u = np.memmap(mm / 'hr_u.dat', dtype='float32', mode='r', shape=(T, H, W))
        v = np.memmap(mm / 'hr_v.dat', dtype='float32', mode='r', shape=(T, H, W))
        storm = 0
        for a in range(0, T, 1500):
            s = np.hypot(u[a:a + 1500], v[a:a + 1500])
            storm += int((np.nanmax(s.reshape(s.shape[0], -1), axis=1) > 10.0).sum())
        check(storm >= storm_min[sp], f'{sp}: {storm} hours with RTMA max wind > 10 m/s (min {storm_min[sp]})')
        # earth-relative sanity: RTMA u/v must not be the raw grid-relative ones. The
        # mean heading of strong wind in SF Bay is westerly (from ~270-300 deg); a
        # 12 deg rotation is not detectable by that alone, so the rotation job's own
        # gate is the authority. Here: only require both components finite & plausible.
        smp = np.asarray(u[:: max(1, T // 200)])
        check(np.isfinite(smp).mean() > 0.97 and np.nanmax(np.abs(smp)) < 60, f'{sp}: hr_u finite & |u|<60')
    st = dp / 'normalization_stats.pkl'
    check(st.exists() and st.stat().st_size > 0, 'normalization_stats.pkl present')
    check((dp / 'target_grid_reference.nc').exists(), 'target_grid_reference.nc present')


# -------------------------------------------------------------------------- train
def gate_train():
    import torch
    for sd in SEEDS:
        cd = RES / sd / 'checkpoint'
        got = {}
        for name in ('best_speed.pth', 'best_direction.pth', 'best_model.pth'):
            p = cd / name
            if not check(p.exists() and p.stat().st_size > 1e6, f'{sd}: {name} exists'):
                continue
            ck = torch.load(p, map_location='cpu', weights_only=False)
            got[name] = ck
        if 'best_speed.pth' in got:
            ck = got['best_speed.pth']
            e = int(ck['epoch'])
            check(e >= 2, f'{sd}: best_speed selected at epoch {e} (0/1 = the v3 over-sharpening pathology)')
            cfg = ck.get('config', {})
            if cfg:
                check(str(cfg.get('head')).lower() == 'quantile', f'{sd}: head={cfg.get("head")}')
                check(int(cfg.get('n_quantiles', -1)) == 51, f'{sd}: n_quantiles={cfg.get("n_quantiles")}')
                check(abs(float(cfg.get('dropout_rate', -1)) - 0.1) < 1e-9, f'{sd}: dropout={cfg.get("dropout_rate")}')
            else:
                print(f'  WARN  {sd}: checkpoint carries no config dict', flush=True)
            if 'best_direction.pth' in got:
                ed = int(got['best_direction.pth']['epoch'])
                print(f'  INFO  {sd}: best_speed epoch {e}, best_direction epoch {ed} '
                      f'(v3 recipe: 7 / 27)', flush=True)
                if ed <= e:
                    print(f'  WARN  {sd}: direction epoch not later than speed epoch', flush=True)


# ------------------------------------------------------------------------- points
def gate_points():
    exp_total = 87
    for sd in SEEDS:
        files = sorted(glob.glob(str(RES / 'v1940_points' / sd / 'points_speed_full_record_ERA5_*.nc')))
        check(len(files) == exp_total, f'{sd}: {len(files)} point files (expect {exp_total})')
        rates = []
        for f in files:
            with xr.open_dataset(f) as d:
                s = d['speed10'].values            # (time, station)
            allnan = np.isnan(s).all(axis=1)
            rates.append(allnan.mean())
            if allnan.all():
                bad(f'{sd}: {Path(f).name} is entirely NaN (the 2026 all-NaN failure mode)')
        r = np.array(rates) if rates else np.array([1.0])
        check(r.max() < 0.02, f'{sd}: all-NaN step rate per file max {100 * r.max():.2f}% '
                              f'median {100 * np.median(r):.2f}% (v3: 0.2-0.5%)')


# ------------------------------------------------------------------- grids / dirs
def _year_files(d, prefix):
    return sorted(glob.glob(str(d / f'{prefix}_full_record_ERA5_????0101_*.nc')))


def gate_grids(prefix='speed'):
    seed = (RES / 'chosen_seed.txt').read_text().strip()
    check(seed in SEEDS, f'chosen seed {seed}')
    files = _year_files(RES / 'v1940_grids' / seed, prefix)
    if not check(len(files) == 87, f'{prefix}: {len(files)} year files (expect 87)'):
        return
    times, epochs = [], set()
    for f in files:
        with xr.open_dataset(f) as d:
            times.append(d['time'].values)
            epochs.add(int(d.attrs.get('checkpoint_epoch', -1)))
    t = np.concatenate(times)
    _, first = np.unique(t, return_index=True)
    t = t[np.sort(first)]
    dd = np.diff(t).astype('timedelta64[h]').astype(int)
    check((dd == 1).all(), f'{prefix}: hourly-contiguous after dedupe ({t.size} h, '
                           f'{int((dd != 1).sum())} breaks)')
    check(str(t[0]) < '1940-01-02' and str(t[-1]) >= '2026-10-01T10',
          f'{prefix}: span {str(t[0])[:13]} -> {str(t[-1])[:13]}')
    check(len(epochs) == 1, f'{prefix}: single checkpoint epoch across files {sorted(epochs)}')


def gate_dirtrack():
    gate_grids('direction')


# ---------------------------------------------------------------------- products
def gate_products():
    seed = (RES / 'chosen_seed.txt').read_text().strip()
    d = RES / 'v1940_product'
    rng = np.random.default_rng(0)

    def one(pat):
        h = sorted(d.glob(pat))
        check(len(h) == 1, f'{pat}: {len(h)} file(s)')
        return h[0] if h else None

    raw = one(f'CNN_RTMA_{seed}_P50_uvs_raw_*.nc')
    ba = one(f'CNN_RTMA_{seed}_P50_uvs_BA_*.nc')
    qraw = one(f'CNN_RTMA_{seed}_speed_quantiles_raw_*.nc')
    qbc = one(f'CNN_RTMA_{seed}_speed_quantiles_BC_*.nc')
    qba = one(f'CNN_RTMA_{seed}_speed_quantiles_BA_*.nc')
    if FAILS:
        return
    for p in (raw, ba):
        with xr.open_dataset(p) as x:
            t = x['time'].values
            dd = np.diff(t).astype('timedelta64[h]').astype(int)
            check((dd == 1).all(), f'{p.name}: contiguous {t.size} h')
            for i in rng.integers(0, t.size, 6):
                u, v, s = (x[k].isel(time=int(i)).values for k in ('u10', 'v10', 'wind_speed'))
                m = np.isfinite(s)
                if m.any():
                    check(np.nanmax(np.abs(np.hypot(u, v) - s)[m]) < 2e-3, f'{p.name}: sqrt(u2+v2)==speed @ {i}')
    with xr.open_dataset(raw) as a, xr.open_dataset(ba) as b:
        for i in rng.integers(0, a.sizes['time'], 6):
            sa, sb = a['wind_speed'].isel(time=int(i)).values, b['wind_speed'].isel(time=int(i)).values
            m = np.isfinite(sa) & (sa > 0.5)
            if m.any():
                check((sb[m] / sa[m]).min() >= 1.0 - 1e-5, f'BA/raw >= 1 (never reduces) @ {i}')
    with xr.open_dataset(qraw) as a:
        q = a['quantile'].values
        check(q.size == 8 and abs(q[-1] - 0.9902) < 1e-3 and np.isclose(q, 0.5).any(),
              f'quantile levels {np.round(q, 4).tolist()} (8 levels, P50 exact, top ~0.9902)')
    for p in (qbc, qba):
        with xr.open_dataset(p) as x:
            check(x.sizes['quantile'] == 8, f'{p.name}: 8 levels')


if __name__ == '__main__':
    gate = sys.argv[1] if len(sys.argv) > 1 else ''
    fn = {'prep': gate_prep, 'train': gate_train, 'points': gate_points, 'grids': gate_grids,
          'dirtrack': gate_dirtrack, 'products': gate_products}.get(gate)
    if fn is None:
        sys.exit(__doc__)
    print(f'=== gate: {gate} ===', flush=True)
    fn()
    print(f'=== gate {gate}: {"FAIL (" + str(len(FAILS)) + ")" if FAILS else "PASS"} ===', flush=True)
    sys.exit(1 if FAILS else 0)

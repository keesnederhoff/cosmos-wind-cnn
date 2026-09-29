#!/usr/bin/env python
"""Pick the v3 seed for the 1940-2026 product from station time series.

Input: results/v1940_points/<seed>/points_speed_full_record_ERA5_*.nc written
by extract_station_points.py (ERAS_POINTS=1). Observations come from the
validation engine itself (validate_met_models.load_station: IEM/NDBC archive,
height-corrected to 10 m) and are matched with its match_timeseries (nearest
within 1 h), so the numbers follow the published validation's conventions.

Score: POOLED Murphy skill of 10 m wind speed, IEM + NDBC only (USGS moorings
are 1.2-4.9 m anemometers and stay out of any headline):
    skill = 1 - sum(SE) / sum((obs - station_mean_obs)^2)
per era and over the whole record. Pseudo-rows (*_MEAN) never enter.

Decision rule: seeds are scored against the SAME observations, so the
comparison is PAIRED. For each calendar year the pooled skill of every seed is
computed; the gap between the top seed and each other seed is the mean of the
year-wise differences, SE = std / sqrt(n_years). The top seed wins only if its
all-record gap to the runner-up exceeds 2 SE; otherwise the reference seed
r1_do010 is kept. Prints a TIE rather than a ranking inside the noise.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
P = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/'
         'sf_bay_rtma_v3/results/v1940_points')
SEEDS = ['r1_do010', 'r1b_do010_s2', 'r1b_do010_s3']
REFERENCE = 'r1_do010'
ERAS = [('E-2', 1940, 1959), ('E-1', 1960, 1979), ('E0', 1980, 1999),
        ('E1', 2000, 2010), ('E2', 2011, 2019), ('E3', 2020, 2026)]
GROUPS = ('IEM', 'NDBC')


def load_points(seed, expect):
    files = sorted(glob.glob(str(P / seed / 'points_speed_full_record_ERA5_*.nc')))
    if expect and len(files) != expect:
        sys.exit(f'STOP: {seed} has {len(files)} point files, expected {expect}')
    parts, prev = [], None
    worst = 0.0
    for f in files:
        d = xr.open_dataset(f)[['speed10']].load()
        if prev is not None:                       # year-boundary overlap check
            c = np.intersect1d(prev.time.values, d.time.values)
            if c.size:
                a = prev.speed10.sel(time=c).values
                b = d.speed10.sel(time=c).values
                both = np.isfinite(a) & np.isfinite(b)
                if both.any():
                    worst = max(worst, float(np.abs(a - b)[both].max()))
        parts.append(d)
        prev = d
    ds = xr.concat(parts, dim='time')
    t = ds.time.values
    _, first = np.unique(t, return_index=True)
    ds = ds.isel(time=np.sort(first))
    d = np.diff(ds.time.values).astype('timedelta64[h]').astype(int)
    if (d < 1).any() or (expect and (d > 1).any()):
        sys.exit(f'STOP: {seed} point record not hourly-contiguous after dedupe')
    print(f'  {seed}: {len(files)} files, {ds.time.size} h '
          f'{str(ds.time.values[0])[:13]} -> {str(ds.time.values[-1])[:13]}, '
          f'max boundary-overlap |diff| {worst:.2e}')
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect', type=int, default=87,
                    help='point files per seed; 0 = dry run on a partial set')
    ap.add_argument('--out', default=str(P / 'seed_selection'))
    a = ap.parse_args()

    os.environ.setdefault('VAL_USGS', '0')
    sys.path.insert(0, str(HERE.parent / 'validation'))
    import validate_met_models as vmm          # noqa: E402

    pts = {s: load_points(s, a.expect) for s in SEEDS}
    tab = xr.open_dataset(P / 'station_table.nc')
    groups = dict(zip(tab.station.values, tab.group.values))

    rows = []
    for sid, cfg in vmm.STATIONS.items():
        if groups.get(sid) not in GROUPS:
            continue
        obs = vmm.load_station(sid, cfg)
        if obs is None or not np.isfinite(obs['speed10']).any():
            continue
        for s in SEEDS:
            m = pts[s].speed10.sel(station=sid)
            mv, ov, tt = vmm.match_timeseries(m.values, m.time.values,
                                              obs['speed10'], obs['time'])
            if mv is None:
                continue
            rows.append(pd.DataFrame({'seed': s, 'station': sid,
                                      'group': groups[sid], 'time': tt,
                                      'mod': mv, 'obs': ov}))
    df = pd.concat(rows, ignore_index=True)
    df = df[np.isfinite(df['mod']) & np.isfinite(df['obs'])]
    # identical sample across seeds: keep (station, time) present for all
    key = df.groupby(['station', 'time'])['seed'].transform('nunique')
    df = df[key == len(SEEDS)].copy()
    df['year'] = pd.DatetimeIndex(df['time']).year
    df['se'] = (df['mod'] - df['obs']) ** 2
    # station climatology over the WHOLE record (fixed reference, all seeds)
    mu = df[df.seed == SEEDS[0]].groupby('station')['obs'].mean()
    df['var'] = (df['obs'] - df['station'].map(mu)) ** 2

    def pooled(g):
        return 1.0 - g['se'].sum() / g['var'].sum()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for name, y0, y1 in ERAS + [('ALL', 1940, 2026)]:
        sub = df[(df.year >= y0) & (df.year <= y1)]
        ref = sub[sub.seed == SEEDS[0]]
        row = {'era': name, 'years': f'{y0}-{y1}',
               'n_stations': ref.station.nunique(), 'n_pairs': len(ref)}
        for s in SEEDS:
            row[s] = pooled(sub[sub.seed == s])
        table.append(row)
    T = pd.DataFrame(table)
    pd.set_option('display.width', 200)
    print('\nPOOLED MURPHY SKILL, 10 m speed, IEM+NDBC, identical samples per seed')
    print(T.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    T.to_csv(out / 'seed_skill_by_era.csv', index=False)

    yearly = df.groupby(['year', 'seed']).apply(pooled).unstack('seed')
    yearly.to_csv(out / 'seed_skill_by_year.csv')
    allrow = T[T.era == 'ALL'].iloc[0]
    order = sorted(SEEDS, key=lambda s: -allrow[s])
    top = order[0]
    print(f'\nPAIRED year-wise gaps vs top seed {top} (n_years={len(yearly)}):')
    decisive = True
    for s in order[1:]:
        dif = (yearly[top] - yearly[s]).dropna()
        gap, se = dif.mean(), dif.std(ddof=1) / np.sqrt(len(dif))
        ok = gap > 2 * se
        decisive &= ok
        print(f'  {top} - {s}: {gap:+.4f}  2SE {2 * se:.4f}  '
              f'{"CLEARS" if ok else "inside noise"}')
    pick = top if decisive else REFERENCE
    verdict = (f'PICK {pick}' if decisive
               else f'TIE -> keep reference {REFERENCE} (top on raw score: {top})')
    print(f'\nVERDICT: {verdict}')
    (out / 'VERDICT.txt').write_text(verdict + '\n')


if __name__ == '__main__':
    main()

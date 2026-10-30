#!/usr/bin/env python
"""Final stage of the chain: write RESULTS.md and the DONE marker.

Never fails the chain by itself (runs `afterany`): everything it cannot find is reported
as MISSING, which is exactly the information the person reading it needs.

Contents: per-stage status from chain_status.log (stages with no line at all = killed
by walltime / node death), the per-year RTMA/ERA5 skill table (locates the RTMA quality
break), the current-vs-v3 era tables from the validation CSVs, the quantile reliability, the
chosen seed, and the quota.
"""
from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay')
RES = ROOT / 'sf_bay_rtma' / 'results'
VAL = ROOT / 'validation' / 'results'
STAGES = ['rotate', 'prep', 'gate_prep', 'train_0', 'train_1', 'train_2', 'gate_train',
          'points', 'gate_points', 'seedsel', 'grids', 'dirtrack', 'gate_grids', 'gate_dirtrack',
          'combine', 'ba_fit', 'ba_quantiles', 'p50', 'gate_products', 'reliability', 'val']
FENCE = chr(96) * 3
SPEED = 'Wind Speed [m/s]'
TOP = 'Wind Speed [m/s] (q90-100)'
DIRN = 'Wind Direction [deg]'
LABELS = {'CNN-RTMA': 'V3-RTMA', 'CNN-RTMA-BA': 'V3-RTMA-BA', 'V4-RTMA': 'CNN-RTMA', 'V4-RTMA-BA': 'CNN-RTMA-BA'}


def relabel(df):
    """Stored 2026-10 scores use the old model labels; map them to the current ones."""
    if 'V4-RTMA' in set(df['model']):
        df = df.assign(model=df['model'].map(lambda m: LABELS.get(m, m)))
    return df


def read_stats(d):
    f = VAL / d / 'validation_statistics.csv'
    if not f.exists():
        return None
    df = relabel(pd.read_csv(f))
    return df[df['source'].isin(['IEM', 'NDBC'])]


def table(d, var=SPEED):
    df = read_stats(d)
    if df is None:
        return f'MISSING: {d}\n'
    g = df[df.variable == var].groupby('model').agg(
        stations=('station', 'nunique'), skill=('skill', 'mean'), bias=('bias', 'mean'),
        rmse=('rmse', 'mean'))
    return FENCE + '\n' + g.sort_values('skill', ascending=False).round(3).to_string() + '\n' + FENCE + '\n'


def main():
    L = [f'# CNN results ({dt.datetime.now():%Y-%m-%d %H:%M})', '']
    seed = RES / 'chosen_seed.txt'
    L.append(f'**Chosen seed:** {seed.read_text().strip() if seed.exists() else "MISSING"}')
    sv = RES / 'v1940_points' / 'seed_selection' / 'VERDICT.txt'
    if sv.exists():
        L.append(f'**Seed verdict:** {sv.read_text().strip()}')
    L += ['', '## Stage status (chain_status.log)', '']
    log = RES / 'chain_status.log'
    lines = log.read_text().splitlines() if log.exists() else []
    seen = {}
    for ln in lines:
        p = ln.split()
        if len(p) >= 3 and p[0] == 'STAGE':
            seen.setdefault(p[1], []).append(p[2])
    for s in STAGES:
        st = seen.get(s)
        L.append(f'- `{s}`: ' + (', '.join(st) if st else '**NO LINE (never ran, or killed by walltime/node death)**'))
    L += ['', '## RTMA / ERA5 skill per calendar year (station-mean Murphy skill, IEM+NDBC)', '',
          'Locates the RTMA quality break (user believes 2020). Not gating.', '']
    rows = []
    for y in range(2011, 2027):
        df = read_stats(f'obsEY_{y}')
        if df is None:
            continue
        g = df[df.variable == SPEED].groupby('model')['skill'].mean()
        rows.append({'year': y, 'stations': df.station.nunique(), **g.round(3).to_dict()})
    L.append((FENCE + '\n' + pd.DataFrame(rows).to_string(index=False) + '\n' + FENCE + '\n')
             if rows else 'MISSING\n')
    L += ['## Held-out WY2026 (2025-10-01 -> 2026-08-10): current vs v3 vs RTMA vs ERA5', '',
          '(station-mean of per-station skill; top-10% skill is on the observed top decile and '
          'is negative for everything, including RTMA -- read relative, never absolute)', '',
          table('obsWY2026'), table('obsWY2026', TOP)]
    for e, d in (('Em2 1940-59', 'obsEm2_1940-1959'), ('Em1 1960-79', 'obsEm1_1960-1979'),
                 ('E0 1980-99', 'obsE0_1980-1999'), ('E1 2000-10', 'obsE1_2000-2010'),
                 ('E2 2011-19', 'obsE2_2011-2019'), ('E3 2020-26', 'obsE3_2020-2026')):
        L += [f'## Era {e}: speed skill', '', table(d)]
    L += ['## Direction RMSE (E2)', '', table('obsE2_2011-2019', DIRN)]
    rel = RES / 'v1940_product' / 'QUANTILE_RELIABILITY.txt'
    L += ['## Quantile reliability', '', '```', rel.read_text() if rel.exists() else 'MISSING', '```', '']
    ba = RES / 'v1940_product' / 'BA_LOSO.txt'
    L += ['## BA leave-one-station-out', '', '```', ba.read_text()[:3000] if ba.exists() else 'MISSING', '```', '']
    q = subprocess.run(['lfs', 'quota', '-g', 'pcmsc', '/caldera/projects/usgs/hazards/pcmsc'],
                       capture_output=True, text=True).stdout
    L += ['## Quota', '', '```', q.strip(), '```', '']
    (RES / 'RESULTS.md').write_text('\n'.join(L))
    (RES / 'DONE').write_text(f'{dt.datetime.now().isoformat()}\n')
    print('wrote', RES / 'RESULTS.md')


if __name__ == '__main__':
    main()

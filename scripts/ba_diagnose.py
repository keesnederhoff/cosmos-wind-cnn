#!/usr/bin/env python
"""Why does current BA trail v3 BA on all-hours skill? Reads existing outputs only.

1. Station factor comparison (v3 vs current BA_station_factors_E3q90.csv).
2. WY2026 skill/bias per speed bin of the observations, all four CNN products.
3. Per-station WY2026 gap current BA - v3 BA, against the factor change.
"""
import glob
from pathlib import Path

import pandas as pd

from report import relabel

ROOT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay')
VAL = ROOT / 'validation' / 'results'
V3 = ROOT / 'archive' / 'sf_bay_rtma_v3' / 'results' / 'v1940_product'
CUR = ROOT / 'sf_bay_rtma' / 'results' / 'v1940_product'
MODELS = ['V3-RTMA', 'CNN-RTMA', 'V3-RTMA-BA', 'CNN-RTMA-BA', 'RTMA-SFbay']
pd.set_option('display.width', 200, 'display.max_columns', 30, 'display.max_rows', 80)

f3 = pd.read_csv(sorted(glob.glob(str(V3 / 'BA_station_factors*.csv')))[0])
f4 = pd.read_csv(sorted(glob.glob(str(CUR / 'BA_station_factors*.csv')))[0])
print('factor columns:', list(f4.columns))

df = relabel(pd.read_csv(VAL / 'obsWY2026' / 'validation_statistics.csv'))
df = df[df['source'].isin(['IEM', 'NDBC']) & df['model'].isin(MODELS)]
print('\nWY2026 by observed speed bin (station mean)')
sp = df[df['variable'].str.startswith('Wind Speed [m/s]')]
for stat in ('bias', 'rmse', 'skill'):
    print(f'\n{stat}')
    print(sp.pivot_table(index='variable', columns='model', values=stat, aggfunc='mean')[MODELS].round(3))

al = sp[sp['variable'] == 'Wind Speed [m/s]'].pivot_table(index='station', columns='model', values='skill')
al['gap_BA'] = al['CNN-RTMA-BA'] - al['V3-RTMA-BA']
al['gap_raw'] = al['CNN-RTMA'] - al['V3-RTMA']
al['gain_v3'] = al['V3-RTMA-BA'] - al['V3-RTMA']
al['gain_cur'] = al['CNN-RTMA-BA'] - al['CNN-RTMA']
print('\nPer-station WY2026: mean gaps', al[['gap_raw', 'gap_BA', 'gain_v3', 'gain_cur']].mean().round(4).to_dict())
print('stations where current BA > v3 BA:', int((al['gap_BA'] > 0).sum()), 'of', len(al))
print(al[['V3-RTMA', 'CNN-RTMA', 'V3-RTMA-BA', 'CNN-RTMA-BA', 'gap_raw', 'gap_BA']].sort_values('gap_BA').round(3).head(10))
print(al[['V3-RTMA', 'CNN-RTMA', 'V3-RTMA-BA', 'CNN-RTMA-BA', 'gap_raw', 'gap_BA']].sort_values('gap_BA').round(3).tail(5))

BAD = ['TIBC1', 'RCMC1', 'PXOC1']
print('\nRobustness of the all-station means, by era (IEM+NDBC)')
for era in ('obsWY2026', 'obsE3_2020-2026', 'obsE2_2011-2019', 'obsE1_2000-2010'):
    f = VAL / era / 'validation_statistics.csv'
    if not f.exists():
        print(era, 'MISSING')
        continue
    e = relabel(pd.read_csv(f))
    e = e[e['source'].isin(['IEM', 'NDBC']) & (e['variable'] == 'Wind Speed [m/s]')]
    p = e.pivot_table(index='station', columns='model', values='skill')
    pr = p.dropna(subset=['V3-RTMA', 'CNN-RTMA', 'V3-RTMA-BA', 'CNN-RTMA-BA'])
    rest = pr.drop(index=[s for s in BAD if s in pr.index])
    row = {'n': len(pr)}
    for a, b, tag in (('CNN-RTMA', 'V3-RTMA', 'raw'), ('CNN-RTMA-BA', 'V3-RTMA-BA', 'BA')):
        row[f'{tag}_mean'] = round((pr[a] - pr[b]).mean(), 4)
        row[f'{tag}_median'] = round((pr[a] - pr[b]).median(), 4)
        row[f'{tag}_excl3'] = round((rest[a] - rest[b]).mean(), 4)
        row[f'{tag}_wins'] = int((pr[a] > pr[b]).sum())
    print(era, row)
    print(e[e['station'].isin(BAD) & e['model'].isin(MODELS)].pivot_table(
        index='station', columns='model', values=['skill', 'bias'])[[('skill', 'V3-RTMA'), ('skill', 'CNN-RTMA'),
        ('skill', 'RTMA-SFbay') if ('skill', 'RTMA-SFbay') in e.pivot_table(index='station', columns='model', values=['skill']).columns else ('skill', 'CNN-RTMA'),
        ('bias', 'V3-RTMA'), ('bias', 'CNN-RTMA')]].round(3))
print(f4[f4['station'].isin(BAD)][['station', 'klass', 'cell_class', 'x_km', 'y_km', 'n_pairs', 'obs_mean', 'cnn_mean', 'factor', 'factor_raw']].round(3))

key = [c for c in f4.columns if c.lower() in ('station', 'station_id', 'id', 'name', 'stid')]
num = [c for c in f4.columns if 'factor' in c.lower() or c.lower() in ('f', 'ratio')]
print('\nkey', key, 'factor columns', num)
if key and num:
    m = f3.merge(f4, on=key[0], suffixes=('_v3', '_cur'))
    c3, c4 = f'{num[0]}_v3', f'{num[0]}_cur'
    m['d'] = m[c4] - m[c3]
    print('matched stations', len(m), 'mean factor v3', round(m[c3].mean(), 3), 'current', round(m[c4].mean(), 3))
    cls = [c for c in f4.columns if 'class' in c.lower() or 'type' in c.lower()]
    if cls:
        print(m.groupby(f'{cls[0]}_cur')[[c3, c4]].agg(['mean', 'count']).round(3))
    j = al.join(m.set_index(key[0])['d'], how='inner')
    print('stations joined to skill:', len(j), ' corr(factor change, gap_BA) =', round(j['d'].corr(j['gap_BA']), 3))

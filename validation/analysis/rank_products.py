"""Rank wind products per era from the overview validation CSVs."""
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root on path
import config

BASE = config.OUTPUT_ROOT
ERAS = {
    'Era 1  1990-2010': 'era1_1990-2010',
    'Era 2  2011-2021': 'era2_2011-2021',
    'Era 3  2022-present': 'era3_2022-present',
}
VAR = 'Wind Speed [m/s]'

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 200)

for label, d in ERAS.items():
    fp = BASE / d / 'validation_statistics.csv'
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    if not fp.exists():
        print("  no CSV"); continue
    df = pd.read_csv(fp)
    sub = df[df['variable'] == VAR].copy()
    sub = sub[~sub['station'].str.contains('MEAN')]  # drop validator summary rows
    if sub.empty:
        print("  no wind-speed rows"); continue

    QUALITY = {'NDBC', 'IEM', 'USGS'}

    def rank(frame, tier_name):
        if frame.empty:
            print(f"\n  [{tier_name}] no stations"); return
        agg = (frame.groupby('model')
                  .agg(skill=('skill', 'mean'),
                       skill_med=('skill', 'median'),
                       rmse=('rmse', 'mean'),
                       bias=('bias', 'mean'),
                       corr=('corr', 'mean'),
                       SI=('scatter_index', 'mean'),
                       n_sta=('station', 'nunique'),
                       n_obs=('n', 'sum'))
                  .sort_values('skill', ascending=False))
        out = agg.round({'skill': 3, 'skill_med': 3, 'rmse': 2,
                         'bias': 2, 'corr': 3, 'SI': 2})
        out['n_obs'] = (out['n_obs'] / 1e3).round(0).astype(int).astype(str) + 'k'
        sta_list = ','.join(sorted(frame['station'].unique()))
        print(f"\n  [{tier_name}] stations: {sta_list}")
        print("  ranked by mean Murphy skill (bias = model-obs [m/s]):")
        print(out.to_string().replace('\n', '\n  '))

    rank(sub[sub['source'].isin(QUALITY)], 'QUALITY: NDBC+IEM+USGS')
    if (sub['source'] == 'CWOP').any():
        rank(sub[sub['source'] == 'CWOP'], 'CWOP citizen (noisy, low-variance)')

    print("\n  Best model per station (by skill):")
    for sta, g in sub.groupby('station'):
        b = g.loc[g['skill'].idxmax()]
        print(f"    {sta:<10} [{b['source']:<4}] {b['model']:<11} "
              f"skill={b['skill']:+.3f}  rmse={b['rmse']:.2f}  bias={b['bias']:+.2f}  n={int(b['n'])//1000}k")
print()

#!/usr/bin/env python
"""Headline table + figure: RTMA-SFbay vs CNN-RTMA vs CNN-RTMA-BA per era.

Reads results/obs<era>_<years>_<tag>/validation_statistics.csv written by
run_validation_ba.slurm, pools IEM + NDBC stations (USGS reported separately,
pseudo rows *_MEAN dropped) and prints, per era:
  pooled Murphy skill      1 - sum(n rmse^2) / sum(n obs_std^2)   (as combined_skill.py)
  mean bias                n-weighted
  q90 / q99 ratio          station MEDIAN of model/obs at the 10 % / 1 % exceedance level
  direction RMSE           pooled, must be identical for CNN-RTMA and CNN-RTMA-BA
plus, for E3, the leave-one-station-out row of CNN-RTMA-BA taken from the fit's
BA_LOSO.txt -- the in-sample E3 score of BA is zero-bias at the stations by
construction and must not be read as skill.

Outputs: <out>/ba_summary.csv, ba_summary.md, ba_summary.png
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('COSMOS_VALIDATION_OUTPUT_ROOT',
                           '/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation/results'))
PRODUCT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/sf_bay_rtma_v3/results/v1940_product')
ERAS = [('Em2', 'obsEm2_1940-1959'), ('Em1', 'obsEm1_1960-1979'), ('E0', 'obsE0_1980-1999'),
        ('E1', 'obsE1_2000-2010'), ('E2', 'obsE2_2011-2019'), ('E3', 'obsE3_2020-2026')]
MODELS = ['ERA5', 'RTMA-SFbay', 'CNN-RTMA', 'CNN-RTMA-BA']
COLORS = {'ERA5': 'tab:blue', 'RTMA-SFbay': 'tab:cyan', 'CNN-RTMA': '#d62728',
          'CNN-RTMA-BA': 'darkviolet', 'CNN-RTMA-BA (leave-one-out)': 'plum'}
SPEED, DIRN = 'Wind Speed [m/s]', 'Wind Direction [deg]'


def clean(df, sources, circular=False):
    d = df[df['source'].isin(sources) & ~df['station'].str.endswith('_MEAN')]
    d = d[d['n'] >= 50]
    return d if circular else d[d['obs_std'] > 0.05]      # direction rows carry no obs_std


def pooled(d):
    if d.empty:
        return None
    n = d['n'].values.astype(float)
    W = n.sum()
    return dict(n_stations=int(len(d)), n_pairs=int(W),
                skill=1 - np.sum(n * d['rmse'].values ** 2) / np.sum(n * d['obs_std'].values ** 2),
                bias=float(np.sum(n * d['bias'].values) / W),
                q90_ratio=float(d['q90_ratio'].median()) if 'q90_ratio' in d else np.nan,
                q99_ratio=float(d['q99_ratio'].median()) if 'q99_ratio' in d else np.nan)


def loso_row():
    """CNN-RTMA-BA leave-one-station-out numbers from the fit (Era 3 only)."""
    p = PRODUCT / 'BA_LOSO.txt'
    if not p.exists():
        return None
    txt = p.read_text()
    m = re.search(r'chosen ([0-9.]+)', txt)
    sig = m.group(1) if m else None
    for line in txt.splitlines():
        if sig and line.strip().startswith(f'gaussian sigma={float(sig):g} km + class'):
            parts = line.split()
            # ... field  skill  bias  abs_bias  P50  P90  P99  P90_sd  P99_sd  n_stations
            vals = [float(v) for v in parts[-9:]]
            return dict(era='E3', model='CNN-RTMA-BA (leave-one-out)', n_stations=int(vals[-1]),
                        n_pairs=np.nan, skill=vals[0], bias=vals[1], q90_ratio=vals[4],
                        q99_ratio=vals[5], dir_rmse=np.nan, note='from BA_LOSO.txt, station factors predicted without the station')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='ba_q90')
    ap.add_argument('--out', default=str(ROOT / 'rankings_ba'))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, usgs_rows = [], []
    for era, base in ERAS:
        f = ROOT / f'{base}_{a.tag}' / 'validation_statistics.csv'
        if not f.exists():
            print(f'  {era}: missing {f}'); continue
        df = pd.read_csv(f, dtype={'station': str})
        for model in MODELS:
            sp = clean(df[(df['model'] == model) & (df['variable'] == SPEED)], ('IEM', 'NDBC'))
            st = pooled(sp)
            if st is None:
                continue
            dr = clean(df[(df['model'] == model) & (df['variable'] == DIRN)], ('IEM', 'NDBC'), circular=True)
            n = dr['n'].values.astype(float)
            dir_rmse = float(np.sqrt(np.sum(n * dr['rmse'].values ** 2) / n.sum())) if len(dr) else np.nan
            rows.append(dict(era=era, model=model, **st, dir_rmse=dir_rmse, note=''))
            us = pooled(clean(df[(df['model'] == model) & (df['variable'] == SPEED)], ('USGS',)))
            if us is not None:
                usgs_rows.append(dict(era=era, model=model, **us))
    lr = loso_row()
    if lr:
        rows.append(lr)
    T = pd.DataFrame(rows)
    if T.empty:
        raise SystemExit('no results found')
    T['era'] = pd.Categorical(T['era'], [e for e, _ in ERAS], ordered=True)
    T = T.sort_values(['era', 'model'], key=lambda s: s if s.name == 'era' else s.map(
        {m: i for i, m in enumerate(MODELS + ['CNN-RTMA-BA (leave-one-out)'])}))
    T.to_csv(out / 'ba_summary.csv', index=False, float_format='%.4f')
    pd.set_option('display.width', 220)
    fmt = lambda v: f'{v:.3f}'
    print('\nPOOLED IEM+NDBC, 10 m wind speed (ratios = station median model/obs at the exceedance level)')
    print(T[['era', 'model', 'n_stations', 'skill', 'bias', 'q90_ratio', 'q99_ratio', 'dir_rmse']]
          .to_string(index=False, float_format=fmt))
    if usgs_rows:
        U = pd.DataFrame(usgs_rows)
        print('\nUSGS moorings (own group; 1.2-4.9 m anemometers log-law to 10 m; BA patched them -> in-sample)')
        print(U[['era', 'model', 'n_stations', 'skill', 'bias', 'q90_ratio', 'q99_ratio']].to_string(index=False, float_format=fmt))
        U.to_csv(out / 'ba_summary_usgs.csv', index=False, float_format='%.4f')
    # markdown
    md = ['# RTMA vs CNN-RTMA vs CNN-RTMA-BA, 10 m wind speed vs IEM+NDBC stations', '',
          'Pooled Murphy skill (ref = station climatology); bias in m/s; q90/q99 = station median of',
          'model/obs at the 10 % / 1 % exceedance level (1.00 = matched). E3 is IN-SAMPLE for CNN-RTMA-BA',
          '(the factor was fitted there, exact at every station); the leave-one-station-out row is the',
          'honest E3 number. E2 (2011-2019) is the out-of-sample era where RTMA also exists.', '']
    for era, _ in ERAS:
        s = T[T.era == era]
        if s.empty:
            continue
        md += [f'## {era}', '', '| model | stations | skill | bias | q90 | q99 | dir RMSE |', '|---|---|---|---|---|---|---|']
        for _, r in s.iterrows():
            md.append(f'| {r.model} | {r.n_stations} | {r.skill:.3f} | {r.bias:+.2f} | {r.q90_ratio:.2f} | '
                      f'{r.q99_ratio:.2f} | {r.dir_rmse:.1f} |' if np.isfinite(r.dir_rmse) else
                      f'| {r.model} | {r.n_stations} | {r.skill:.3f} | {r.bias:+.2f} | {r.q90_ratio:.2f} | {r.q99_ratio:.2f} | - |')
        md.append('')
    (out / 'ba_summary.md').write_text('\n'.join(md))
    plot(T, out)
    print(f'\nwrote {out / "ba_summary.csv"}, ba_summary.md, ba_summary.png')


def plot(T, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eras = [e for e, _ in ERAS if e in set(T.era.astype(str))]
    models = [m for m in MODELS + ['CNN-RTMA-BA (leave-one-out)'] if m in set(T.model)]
    panels = [('skill', 'Pooled Murphy skill (10 m speed)', None),
              ('bias', 'Mean bias (m/s)', 0.0),
              ('q90_ratio', 'Model / obs at 10 % exceedance', 1.0),
              ('q99_ratio', 'Model / obs at 1 % exceedance', 1.0)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 3.1 * len(panels)), sharex=True)
    w = 0.8 / len(models)
    for ax, (col, title, ref) in zip(axes, panels):
        for j, m in enumerate(models):
            vals = [T[(T.era == e) & (T.model == m)][col].squeeze() if not T[(T.era == e) & (T.model == m)].empty else np.nan
                    for e in eras]
            xs = np.arange(len(eras)) + (j - (len(models) - 1) / 2) * w
            bars = ax.bar(xs, vals, w, color=COLORS.get(m, 'gray'), label=m, edgecolor='k', linewidth=0.4)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.2f}', ha='center',
                            va='bottom' if v >= 0 else 'top', fontsize=7)
        if ref is not None:
            ax.axhline(ref, color='k', lw=0.6)
        ax.set_title(title, fontsize=10, loc='left'); ax.grid(axis='y', alpha=0.3)
    labels = []
    for e in eras:
        s = T[(T.era == e) & (T.model == 'CNN-RTMA')]
        labels.append(f'{e}\n{int(s.n_stations.iloc[0])} stn' if not s.empty else e)
    axes[-1].set_xticks(np.arange(len(eras))); axes[-1].set_xticklabels(labels)
    axes[0].legend(ncol=len(models), fontsize=8, loc='upper left')
    fig.suptitle('IEM + NDBC stations, 10 m wind speed; E3 is in-sample for CNN-RTMA-BA (fit era)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'ba_summary.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()

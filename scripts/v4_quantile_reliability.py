#!/usr/bin/env python
"""Reliability of the v4 predictive quantiles against station observations.

Question: when the model says "the 99th percentile of the wind speed is q", is the
observed speed <= q about 99% of the time? Uses the step-1 station point files of the
CHOSEN seed (hr_speed_q at the 45 stations, bilinear stencils exactly as the validation
engine derives them), IEM + NDBC only (USGS anemometers are 1.2-4.9 m).

For every written level tau: empirical = P(obs <= q_tau), over
  * all hours, and
  * "storm hours", defined on the FORECAST (model P50 > 6 m/s) -- conditioning on the
    observed value would bias the empirical frequency by construction.
per window: WY2025 (val), WY2026 (test), 2020-2026, 2011-2019 and 2000-2010 (never seen).
Also the central-interval coverage (e.g. [0.049, 0.951] nominal 0.902).
Output: <case>/results/v1940_product/QUANTILE_RELIABILITY.csv and .txt

v3 reference (Q=19, ep7): the nominal 98% interval covered 88%; nominal 99th = empirical
96th in storms (storm PIT mean 0.68).
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
TABLE = ROOT / 'sf_bay_rtma_v3' / 'results' / 'v1940_points' / 'station_table.nc'
WINDOWS = {
    'WY2025 (val)': ('2024-10-01', '2025-09-30T23'),
    'WY2026 (test)': ('2025-10-01', '2026-10-01T23'),
    '2020-2026': ('2020-01-01', '2026-10-01T23'),
    '2011-2019 (unseen)': ('2011-01-01', '2019-12-31T23'),
    '2000-2010 (unseen)': ('2000-01-01', '2010-12-31T23'),
}


def main():
    os.environ.setdefault('VAL_USGS', '0')
    sys.path.insert(0, str(HERE.parent / 'validation'))
    import validate_met_models as vmm                      # noqa: E402

    seed = (RES / 'chosen_seed.txt').read_text().strip()
    files = sorted(glob.glob(str(RES / 'v1940_points' / seed / 'points_speed_full_record_ERA5_*.nc')))
    parts = [xr.open_dataset(f)[['hr_speed_q']].load() for f in files if
             int(Path(f).name.split('_')[-2][:4]) >= 2000]
    ds = xr.concat(parts, dim='time')
    _, first = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=np.sort(first))
    taus = ds['quantile'].values
    tab = xr.open_dataset(TABLE)
    groups = dict(zip(tab.station.values, tab.group.values))
    print(f'seed {seed}: {len(files)} files, levels {np.round(taus, 4).tolist()}', flush=True)

    rows = []                                              # station, time, obs, p50, q[...]
    for sid, cfg in vmm.STATIONS.items():
        if groups.get(sid) not in ('IEM', 'NDBC'):
            continue
        obs = vmm.load_station(sid, cfg)
        if obs is None or not np.isfinite(obs['speed10']).any():
            continue
        q = ds['hr_speed_q'].sel(station=sid)             # (time, quantile)
        iq = int(np.argmin(np.abs(taus - 0.5)))
        mv, ov, tt = vmm.match_timeseries(q.values[:, iq], q.time.values,
                                          obs['speed10'], obs['time'])
        if mv is None:
            continue
        qq = q.sel(time=tt).values
        df = pd.DataFrame(qq, columns=[f'q{t:.4f}' for t in taus])
        df['obs'], df['time'], df['station'] = ov, tt, sid
        rows.append(df)
    D = pd.concat(rows, ignore_index=True)
    D = D[np.isfinite(D['obs']) & np.isfinite(D.iloc[:, :len(taus)]).all(axis=1)]
    p50 = D[f'q{taus[int(np.argmin(np.abs(taus - 0.5)))]:.4f}']
    D['time'] = pd.DatetimeIndex(D['time'])
    out = []
    for wname, (a, b) in WINDOWS.items():
        w = D[(D.time >= a) & (D.time <= b)]
        for cond, sub in (('all hours', w), ('storm (fcst P50>6)', w[p50.loc[w.index] > 6.0])):
            if len(sub) < 500:
                continue
            for t in taus:
                out.append({'window': wname, 'subset': cond, 'tau': round(float(t), 4),
                            'empirical': float((sub['obs'] <= sub[f'q{t:.4f}']).mean()),
                            'n': len(sub), 'stations': sub.station.nunique()})
            lo, hi = taus[0], taus[-1]
            out.append({'window': wname, 'subset': cond, 'tau': f'interval[{lo:.3f},{hi:.3f}]',
                        'empirical': float(((sub['obs'] >= sub[f'q{lo:.4f}'])
                                            & (sub['obs'] <= sub[f'q{hi:.4f}'])).mean()),
                        'n': len(sub), 'stations': sub.station.nunique(), 'nominal': float(hi - lo)})
    R = pd.DataFrame(out)
    d = RES / 'v1940_product'
    d.mkdir(parents=True, exist_ok=True)
    R.to_csv(d / 'QUANTILE_RELIABILITY.csv', index=False)
    pd.set_option('display.width', 200)
    txt = ['QUANTILE RELIABILITY (IEM+NDBC stations; empirical P(obs <= q_tau) vs nominal tau)', '']
    for (w, c), g in R[R.tau.map(lambda v: not isinstance(v, str))].groupby(['window', 'subset'], sort=False):
        txt.append(f'{w} | {c} | n={g.n.iloc[0]} stations={g.stations.iloc[0]}')
        txt.append('   nominal  ' + ' '.join(f'{v:6.3f}' for v in g.tau))
        txt.append('   empirical' + ' '.join(f'{v:6.3f}' for v in g.empirical))
    for _, r in R[R.tau.map(lambda v: isinstance(v, str))].iterrows():
        txt.append(f'{r.window} | {r.subset} | {r.tau} empirical {r.empirical:.3f} vs nominal {r.nominal:.3f}')
    (d / 'QUANTILE_RELIABILITY.txt').write_text('\n'.join(txt) + '\n')
    print('\n'.join(txt))


if __name__ == '__main__':
    main()

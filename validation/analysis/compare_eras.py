"""Compare the 2026-08-27 re-run era skill against the 2026-08-14 baseline.

Convention 3 (station-mean, each station's own climatology as reference), the
same one the parked baseline used, so the numbers are directly comparable.

READ CNN-ERA5, NOT CNN. ERA5's own skill is era-dependent, so raw CNN skill
conflates model quality with input quality.

ALL_STATIONS_MEAN and USGS_MEAN are pseudo-rows -- filtering the source column
to IEM/NDBC is what keeps them out and stops the double-count.
"""
import pandas as pd

R = ('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/'
     'validation/results')

PAIRS = [
    ('E1 2000-2010', 'obsE1_2000-2010', 'obsE1_2000-2010_v20260810'),
    ('E2 2011-2019', 'obsE2_2011-2019', 'obsE2_2011-2019_v20260810'),
    ('E3 2020-2026', 'obsE3_2020-2026', 'obsE3_2020-2026_v20260810'),
]
CNN = ['V3-ERAS-s1', 'V3-ERAS-s2', 'V3-ERAS-s3']


def load(d):
    df = pd.read_csv(f'{R}/{d}/validation_statistics.csv')
    df = df[df['source'].isin(['IEM', 'NDBC'])]
    return df


def summarise(df, var):
    s = df[df['variable'] == var]
    out = {}
    for m in s['model'].unique():
        sub = s[s['model'] == m]
        out[m] = (sub['skill'].mean(), sub['station'].nunique(), sub['n'].sum())
    return out


first = load(PAIRS[0][1])
print('variables present:', sorted(first['variable'].unique()))
VAR = 'Wind Speed [m/s]'
assert VAR in set(first['variable']), f'{VAR} not in CSV'
print(f'using variable: {VAR}\n')

print(f'{"era":<14}{"model":<14}{"baseline":>10}{"re-run":>10}{"delta":>9}'
      f'{"stations":>10}{"n(new)":>10}')
print('-' * 77)
for lab, dold, dnew in PAIRS:
    try:
        o = summarise(load(dold), VAR)
        n = summarise(load(dnew), VAR)
    except FileNotFoundError as e:
        print(f'{lab:<14} MISSING {e}')
        continue

    def cnnmean(d):
        vals = [d[m][0] for m in CNN if m in d]
        return sum(vals) / len(vals) if vals else float('nan')

    rows = []
    for m in ('ERA5', 'CONUS404', 'RTMA-SFbay'):
        if m in o or m in n:
            rows.append((m, o.get(m, (float('nan'),))[0],
                         n.get(m, (float('nan'),))[0],
                         n.get(m, (0, 0, 0))[1], n.get(m, (0, 0, 0))[2]))
    rows.append(('CNN (3-seed)', cnnmean(o), cnnmean(n),
                 n.get(CNN[0], (0, 0, 0))[1], n.get(CNN[0], (0, 0, 0))[2]))

    for i, (m, ov, nv, st, nn) in enumerate(rows):
        print(f'{lab if i == 0 else "":<14}{m:<14}{ov:>10.4f}{nv:>10.4f}'
              f'{nv - ov:>+9.4f}{int(st):>10d}{int(nn):>10d}')
    # the headline: added value over the input
    oe, ne = o.get('ERA5', (float("nan"),))[0], n.get('ERA5', (float("nan"),))[0]
    print(f'{"":<14}{"CNN - ERA5":<14}{cnnmean(o) - oe:>10.4f}'
          f'{cnnmean(n) - ne:>10.4f}{(cnnmean(n) - ne) - (cnnmean(o) - oe):>+9.4f}')
    print()

# per-seed spread on the re-run, so a shift can be read against seed noise
print('per-seed CNN skill, re-run:')
for lab, _, dnew in PAIRS:
    n = summarise(load(dnew), VAR)
    vals = {m: n[m][0] for m in CNN if m in n}
    if vals:
        sp = max(vals.values()) - min(vals.values())
        print(f'  {lab:<14}' + '  '.join(f'{k[-2:]}={v:.4f}' for k, v in vals.items())
              + f'   spread={sp:.4f}')

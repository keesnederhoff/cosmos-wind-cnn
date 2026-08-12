#!/usr/bin/env python
"""Day-3 scoring: grid metrics, PIT calibration, and direction, on the test window.

Everything is scored through ONE code path in physical m/s at ONE set of grid
points (seed 42, the same draw era_compare.py and the v2 re-score used), so the
numbers here are comparable to every other number in the project.

Three questions, in order of what they decide:

1. GRID SKILL -- does the fixed recipe beat the deterministic control C1 and the
   qw_exp=2 / no-dropout incumbent qh_P0? Reported in BOTH conventions
   (Murphy 1-MSE/MSE and 1-RMSE/RMSE) and BOTH aggregations (pooled and
   median-across-points), because mixing them has already produced one false
   comparison in this project.

2. CALIBRATION -- the falsifiable test of the quantile premise. A flat PIT with
   a beaten deterministic control means the head works and selection was the
   problem. A U-shaped PIT means the head never learned dispersion and the
   deterministic recommendation stands regardless of what the skill table says.

3. DIRECTION -- best_direction.pth (epoch 27) against best_speed.pth (epoch 7).
   Direction was still improving in every arm when training stopped, so this
   measures what shipping a single checkpoint costs on the CNN's strongest field.
   Reported over hours with RTMA speed > 3 m/s as well as all hours: direction
   error is meaningless in near-calm and the all-hours number is mostly noise.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, '/home/cnederhoff/cosmos/cosmos-wind-cnn/scripts')
sys.path.insert(0, '/home/cnederhoff/cosmos/cosmos-wind-cnn/src')
from era_compare import metrics, median_skill, only_var, RTMA_U, RTMA_V, ERA5_U, ERA5_V
from cosmos_wind_cnn.training.quantile_losses import pit_values, interval_coverage

R = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/'
         'sf_bay_rtma_v3/results')
WINDOW = ('2025-02-06', '2025-12-31 23:00')
N_POINTS, SEED = 200, 42
TEST = 'full_record_ERA5_20250206_20260101.nc'

# label -> (run, filename, kind). kind drives which blocks are computed.
PRODUCTS = [
    # the fixed shipping recipe, best_speed.pth (epoch 7)
    ('recipe_s1',    'r1_do010',     f'speed_{TEST}',     'quantile'),
    ('recipe_s2',    'r1b_do010_s2', f'speed_{TEST}',     'quantile'),
    ('recipe_s3',    'r1b_do010_s3', f'speed_{TEST}',     'quantile'),
    # same weights family, twCRPS-selected checkpoint (epoch 10) -- selection cost
    ('selrule_bestmodel_s1', 'r1_do010', TEST,            'quantile'),
    # direction-selected checkpoint (epoch 27), tau=0.5 only
    # Re-scored at the DENSE grid after the day-4 obs board reversed the
    # day-3 grid verdict: epoch 27 tops the CNN field at stations (pooled
    # Murphy 0.5193 vs the epoch-7 recipe's 0.5043, both aggregations
    # agreeing) while scoring WORST on gridded storm skill (0.0350 vs
    # 0.1750). Scored as 'quantile' so calibration is available for it too:
    # epoch 27 cannot be recommended as a probabilistic product on the
    # strength of a point estimate alone.
    ('dir_s1',       'r1_do010',     f'direction_{TEST}', 'quantile'),
    ('dir_s2',       'r1b_do010_s2', f'direction_{TEST}', 'quantile'),
    ('dir_s3',       'r1b_do010_s3', f'direction_{TEST}', 'quantile'),
    # deterministic control -- currently the obs-track leader
    ('C1_det_s1',    'c1_det_P0_s1', TEST,                'det'),
    ('C1_det_s2',    'c1_det_P0_s2', TEST,                'det'),
    ('C1_det_s3',    'c1_det_P0_s3', TEST,                'det'),
    # campaign incumbent: qw_exp=2.0, dropout 0, best_model at epoch 1
    ('qh_P0_s1',     'qh_P0_s1',     TEST,                'quantile'),
    ('qh_P0_s2',     'qh_P0_s2',     TEST,                'quantile'),
    ('qh_P0_s3',     'qh_P0_s3',     TEST,                'quantile'),
]

LEVELS = (0.50, 0.80, 0.90, 0.98)


def circ_rmse(pred_dir, true_dir, mask):
    """RMSE of the wrapped direction difference, in degrees."""
    d = (pred_dir - true_dir + 180.0) % 360.0 - 180.0
    d = d[mask]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else float('nan')


def main():
    ref = xr.open_dataset(R / 'c1_det_P0_s1' / 'output_inference' / TEST)
    ny, nx = ref.sizes['y'], ref.sizes['x']
    ygrid, xgrid = ref.y.values, ref.x.values
    ref.close()

    rng = np.random.default_rng(SEED)
    iys = rng.integers(0, ny, N_POINTS)
    ixs = rng.integers(0, nx, N_POINTS)
    pt = dict(y=xr.DataArray(iys, dims='pt'), x=xr.DataArray(ixs, dims='pt'))
    print(f'grid {ny} x {nx}, {N_POINTS} points, seed {SEED}')

    print('loading RTMA truth...')
    ru = xr.open_dataset(RTMA_U, chunks={'time': 2000})
    rv = xr.open_dataset(RTMA_V, chunks={'time': 2000})
    if not (np.allclose(ru.y.values, ygrid) and np.allclose(ru.x.values, xgrid)):
        raise RuntimeError('RTMA grid does not match the inference grid')
    tu = ru[only_var(ru)].isel(**pt).sel(time=slice(*WINDOW)).load()
    tv = rv[only_var(rv)].isel(**pt).sel(time=slice(*WINDOW)).load()
    truth = np.hypot(tu, tv)

    print('loading ERA5 reference...')
    eu = xr.open_dataset(ERA5_U, chunks={'time': 2000})
    ev = xr.open_dataset(ERA5_V, chunks={'time': 2000})
    ptc = dict(y=xr.DataArray(ygrid[iys], dims='pt'),
               x=xr.DataArray(xgrid[ixs], dims='pt'))
    era5 = np.hypot(eu[only_var(eu)].interp(**ptc).sel(time=slice(*WINDOW)),
                    ev[only_var(ev)].interp(**ptc).sel(time=slice(*WINDOW))).load()

    out = {}
    for label, run, fname, kind in PRODUCTS:
        f = R / run / 'output_inference' / fname
        if not f.exists():
            print(f'  MISSING {label}: {f}')
            continue
        print(f'\n=== {label}  ({run}/{fname})')
        ds = xr.open_dataset(f, chunks={'time': 720}).sel(time=slice(*WINDOW))

        mu = ds['hr_u'].isel(**pt).load()
        mv = ds['hr_v'].isel(**pt).load()

        t = (pd.DatetimeIndex(mu.time.values)
             .intersection(pd.DatetimeIndex(truth.time.values))
             .intersection(pd.DatetimeIndex(era5.time.values)))
        sel = dict(time=t)
        U = mu.sel(**sel).transpose('time', 'pt').values
        V = mv.sel(**sel).transpose('time', 'pt').values
        TU = tu.sel(**sel).transpose('time', 'pt').values
        TV = tv.sel(**sel).transpose('time', 'pt').values
        TR = truth.sel(**sel).transpose('time', 'pt').values
        ER = era5.sel(**sel).transpose('time', 'pt').values
        print(f'  {len(t)} common steps {t[0]} -> {t[-1]}')

        blk = {'run': run, 'file': fname, 'kind': kind, 'hours': len(t)}

        # -- direction, always available (hr_u/hr_v are written for every kind)
        pd_deg = np.degrees(np.arctan2(V, U))
        td_deg = np.degrees(np.arctan2(TV, TU))
        fin = np.isfinite(pd_deg) & np.isfinite(td_deg) & np.isfinite(TR)
        blk['dir_rmse_deg_all'] = circ_rmse(pd_deg, td_deg, fin)
        blk['dir_rmse_deg_gt3'] = circ_rmse(pd_deg, td_deg, fin & (TR > 3.0))
        blk['dir_rmse_deg_gt10'] = circ_rmse(pd_deg, td_deg, fin & (TR > 10.0))

        # -- speed skill. For the quantile products the best estimate is P50 from
        # -- hr_speed_q; hypot(hr_u, hr_v) is the same field reconstructed through
        # -- the direction head, so scoring P50 directly avoids that round trip.
        if kind == 'quantile' and 'hr_speed_q' in ds:
            taus = ds['quantile'].values.astype(float)
            j50 = int(np.argmin(np.abs(taus - 0.5)))
            speed = (ds['hr_speed_q'].isel(quantile=j50, **pt)
                     .sel(**sel).transpose('time', 'pt').values)
            blk['speed_source'] = f'hr_speed_q@tau={taus[j50]:.4f}'
        else:
            speed = np.hypot(U, V)
            blk['speed_source'] = 'hypot(hr_u,hr_v)'

        blk.update(metrics(speed.ravel(), TR.ravel(), ER.ravel()))
        blk.update(median_skill(speed, TR, ER))

        # -- calibration, dense-grid quantile products only
        if kind == 'quantile' and 'hr_speed_q' in ds and ds.sizes['quantile'] > 1:
            taus = ds['quantile'].values.astype(float)
            Q = (ds['hr_speed_q'].isel(**pt).sel(**sel)
                 .transpose('time', 'pt', 'quantile').values)
            Q = Q.reshape(-1, Q.shape[-1])
            Y = TR.ravel()
            ok = np.isfinite(Y) & np.isfinite(Q).all(axis=1)
            Q, Y = Q[ok], Y[ok]
            pit = pit_values(Q, Y)
            hist, _ = np.histogram(pit, bins=20, range=(0, 1))
            hist = (hist / hist.sum()).round(5)
            blk['pit_hist_20'] = hist.tolist()
            # A flat histogram has every bin at 0.05. Report the departure as a
            # single number so seeds can be compared without eyeballing 20 bins.
            blk['pit_flatness_l1'] = float(np.abs(hist - 0.05).sum())
            blk['pit_mean'] = float(pit.mean())
            blk['coverage'] = {f'{L:.2f}': interval_coverage(Q, Y, taus, L)
                               for L in LEVELS}
            # Reliability in the upper tail: the fraction of observations below
            # the predicted tau-quantile should equal tau.
            blk['reliability'] = {}
            for tt in (0.9, 0.95, 0.99):
                j = int(np.clip(np.searchsorted(taus, tt), 1, len(taus) - 1))
                t0, t1 = taus[j - 1], taus[j]
                w = (tt - t0) / (t1 - t0) if t1 > t0 else 0.0
                qv = Q[:, j - 1] * (1 - w) + Q[:, j] * w
                blk['reliability'][f'{tt:.2f}'] = float(np.mean(Y <= qv))
            # Storm subset -- the tail is where the head is supposed to earn its
            # keep, and where a pooled PIT can look fine while the tail is broken.
            hi = Y > 10.0
            if hi.sum() > 1000:
                ph = pit_values(Q[hi], Y[hi])
                hh, _ = np.histogram(ph, bins=20, range=(0, 1))
                hh = (hh / hh.sum()).round(5)
                blk['pit_hist_20_gt10'] = hh.tolist()
                blk['pit_mean_gt10'] = float(ph.mean())
                blk['n_gt10_cal'] = int(hi.sum())

        ds.close()
        out[label] = blk

        for k in ('rmse', 'bias', 'std_ratio', 'skill_vs_era5', 'skillr_vs_era5',
                  'skillr_med_allhours', 'skill_gt10', 'skillr_gt10',
                  'skillr_med_gt10', 'bias_gt10', 'dir_rmse_deg_gt3'):
            if k in blk:
                print(f'    {k:22s} {blk[k]:9.4f}')
        if 'coverage' in blk:
            print('    coverage      ' +
                  '  '.join(f'{L}:{v:.3f}' for L, v in blk['coverage'].items()))
            print(f'    pit_mean {blk["pit_mean"]:.4f}  '
                  f'flatness_L1 {blk["pit_flatness_l1"]:.4f}  '
                  f'pit_mean_gt10 {blk.get("pit_mean_gt10", float("nan")):.4f}')

    dest = R / 'day3_scores.json'
    dest.write_text(json.dumps(
        {'window': WINDOW, 'n_points': N_POINTS, 'seed': SEED,
         'products': out}, indent=2))
    print(f'\nwrote {dest}')

    print('\n' + '=' * 108)
    print('%-22s %9s %9s %9s %9s %9s %9s %8s' %
          ('product', 'skillr', 'medskil', 'skillr10', 'medsk10', 'stdrat',
           'dirRMSE3', 'PITmean'))
    print('=' * 108)
    for k, b in out.items():
        print('%-22s %9.4f %9.4f %9.4f %9.4f %9.4f %9.2f %8s' % (
            k,
            b.get('skillr_vs_era5', float('nan')),
            b.get('skillr_med_allhours', float('nan')),
            b.get('skillr_gt10', float('nan')),
            b.get('skillr_med_gt10', float('nan')),
            b.get('std_ratio', float('nan')),
            b.get('dir_rmse_deg_gt3', float('nan')),
            ('%.4f' % b['pit_mean']) if 'pit_mean' in b else '-'))


if __name__ == '__main__':
    main()

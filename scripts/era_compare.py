#!/usr/bin/env python
"""Seen/unseen era skill comparison for the v3 long-record inference.

Scores the CNN median (P50) against RTMA truth at random grid points, with ERA5
as the reference forecast, separately for:

  UNSEEN  2014-01-01 .. 2019-12-31   never trained on
  SEEN    2020-01-01 .. 2025-12-31   training/val/test window

Caveat this script cannot remove: RTMA's own quality is worse pre-2020 (obs
Murphy 0.403 vs 0.578), so a lower UNSEEN score against RTMA is ambiguous
between "model generalises worse" and "target is noisier". The ERA5-relative
skill is the more robust of the two numbers reported here because both products
are scored against the same (imperfect) target; obs validation settles it.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RESULTS = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/'
               'sf_bay_rtma_v3/results')
RAW = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/'
           'sf_bay_rtma_v3/raw_data')
RTMA_U = RAW / 'RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10.nc'
RTMA_V = RAW / 'RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10.nc'
ERA5_U = RAW / 'ERA5_eastward_wind_1940_2026_UTM.nc'
ERA5_V = RAW / 'ERA5_northward_wind_1940_2026_UTM.nc'

ERAS = {
    'UNSEEN_2014_2019': ('2014-01-01', '2019-12-31 23:00'),
    'SEEN_2020_2025':   ('2020-01-01', '2025-12-31 23:00'),
    # sub-windows of SEEN, so "did it generalise" can be separated from "did it
    # memorise": TRAIN is in-sample for the weights, TEST is the held-out split
    # that Phase 6b scored, and only TEST is comparable to the v2 numbers.
    'SEEN_TRAIN_2020_2024': ('2020-01-01', '2024-03-13 23:00'),
    'SEEN_VAL_2024':        ('2024-03-14', '2025-02-05 23:00'),
    'SEEN_TEST_2025':       ('2025-02-06', '2025-12-31 23:00'),
}


def only_var(ds):
    """The single data variable in a raw file (they hold exactly one)."""
    names = [v for v in ds.data_vars if ds[v].ndim >= 2]
    if len(names) != 1:
        raise ValueError(f"expected 1 field, found {names}")
    return names[0]


def yearly_files(run):
    """The 12 single-year inference files, in order.

    Named explicitly rather than globbed: the run dir also holds the val/test
    window files and a BCVAL_ product, and a size- or mtime-based heuristic
    would pick those up.
    """
    d = RESULTS / run / 'output_inference'
    out = []
    for year in range(2014, 2026):
        start = '20131231' if year == 2014 else f'{year - 1}1230'
        f = d / f'full_record_ERA5_{start}_{year + 1}0101.nc'
        if f.exists():
            out.append((year, f))
        else:
            print(f"  MISSING {year}: {f.name}")
    return out


def load_cnn_points(run, iys, ixs):
    """CNN speed products at the sampled points, years concatenated, deduped.

    Two distinct products come out of the same checkpoint and they are NOT
    interchangeable:
      p50  -- median of the quantile speed head, predicted in m/s directly
      uv   -- |(hr_u, hr_v)|, the vector head's magnitude. This is what
              bias_correct.py scored, so it is the product every previously
              reported v2/v3 skill number refers to.
    Vector averaging shrinks magnitude, so the two disagree by construction.
    """
    frames = []
    for year, f in yearly_files(run):
        ds = xr.open_dataset(f, chunks={'time': 720})
        pt = dict(y=xr.DataArray(iys, dims='pt'), x=xr.DataArray(ixs, dims='pt'))
        sub = ds['hr_speed_q'].isel(**pt)
        # drop the scalar `quantile` coord each slice inherits, else the two
        # carry conflicting values and the Dataset merge raises MergeError
        p50 = sub.isel(quantile=0).drop_vars('quantile', errors='ignore').load()
        p90 = sub.isel(quantile=1).drop_vars('quantile', errors='ignore').load()
        u = ds['hr_u'].isel(**pt).load()
        v = ds['hr_v'].isel(**pt).load()
        uv = np.hypot(u, v)
        frames.append(xr.Dataset({'p50': p50, 'p90': p90, 'uv': uv}))
        ds.close()
        print(f"    {year}: {p50.sizes['time']} steps")
    cat = xr.concat(frames, dim='time').sortby('time')
    # 2-day segment overlap: keep the first occurrence of each timestamp
    keep = ~pd.Index(cat.time.values).duplicated()
    return cat.isel(time=keep)


def metrics(mod, tru, ref):
    """Skill of `mod` against truth `tru`, referenced to `ref` (ERA5).

    Reports BOTH skill conventions in use in this project, because they are not
    the same number and mixing them has already produced one false comparison:
      skill_*      Murphy, 1 - MSE_mod/MSE_ref  (the Phase 2 plan's definition)
      skillr_*     1 - RMSE_mod/RMSE_ref        (what bias_correct.py:_skill_block
                   computes, hence what every previously quoted v2/v3 number is)
    With r = RMSE_mod/RMSE_ref the two are 1-r^2 and 1-r, so the Murphy value is
    always the larger of the two for a skilful model.
    """
    ok = np.isfinite(mod) & np.isfinite(tru) & np.isfinite(ref)
    m, t, r = mod[ok], tru[ok], ref[ok]
    if m.size < 1000:
        return {'n': int(m.size)}

    em, er = m - t, r - t
    out = {
        'n': int(m.size),
        'rmse': float(np.sqrt(np.mean(em ** 2))),
        'bias': float(np.mean(em)),
        'std_ratio': float(np.std(m) / np.std(t)),
        'era5_rmse': float(np.sqrt(np.mean(er ** 2))),
        'era5_bias': float(np.mean(er)),
        'era5_std_ratio': float(np.std(r) / np.std(t)),
        'skill_vs_era5': float(1.0 - np.mean(em ** 2) / np.mean(er ** 2)),
        'skillr_vs_era5': float(
            1.0 - np.sqrt(np.mean(em ** 2)) / np.sqrt(np.mean(er ** 2))),
    }
    # Energy-weighted skill: weight each error by the observed speed^q, so the
    # score is dominated by the hours that matter for the application.
    for q in (1, 2, 3):
        w = t ** q
        num, den = np.sum(w * em ** 2), np.sum(w * er ** 2)
        out[f'skill_ew_q{q}'] = float(1.0 - num / den)
        out[f'skillr_ew_q{q}'] = float(1.0 - np.sqrt(num) / np.sqrt(den))
    # Fixed 10 m/s threshold (not a top-decile, which collapses the reference
    # variance and made RTMA score -3.4 against its own observations).
    hi = t > 10.0
    if hi.sum() > 100:
        out['n_gt10'] = int(hi.sum())
        out['skill_gt10'] = float(
            1.0 - np.mean(em[hi] ** 2) / np.mean(er[hi] ** 2))
        out['skillr_gt10'] = float(
            1.0 - np.sqrt(np.mean(em[hi] ** 2)) / np.sqrt(np.mean(er[hi] ** 2)))
        out['rmse_gt10'] = float(np.sqrt(np.mean(em[hi] ** 2)))
        out['bias_gt10'] = float(np.mean(em[hi]))
        out['era5_rmse_gt10'] = float(np.sqrt(np.mean(er[hi] ** 2)))
    return out


def median_skill(mod2, tru2, ref2):
    """Per-point skill, then median across points.

    This is the aggregation bias_correct.py:_skill_block uses (there, across all
    grid cells). Pooling every point into one array instead lets the windiest,
    highest-variance cells dominate and reads systematically higher, so the two
    aggregations are reported side by side rather than one standing in for the
    other. Arrays are (time, pt).
    """
    ss, ssx = [], []
    for j in range(mod2.shape[1]):
        m, t, r = mod2[:, j], tru2[:, j], ref2[:, j]
        ok = np.isfinite(m) & np.isfinite(t) & np.isfinite(r)
        m, t, r = m[ok], t[ok], r[ok]
        if m.size < 100:
            continue
        re = np.sqrt(np.mean((r - t) ** 2))
        if re > 0:
            ss.append(1.0 - np.sqrt(np.mean((m - t) ** 2)) / re)
        hi = t > 10.0
        if hi.sum() >= 10:
            ree = np.sqrt(np.mean((r[hi] - t[hi]) ** 2))
            if ree > 0:
                ssx.append(1.0 - np.sqrt(np.mean((m[hi] - t[hi]) ** 2)) / ree)
    med = lambda a: float(np.median(a)) if a else float('nan')
    return {'skillr_med_allhours': med(ss), 'skillr_med_gt10': med(ssx),
            'n_pts_scored': len(ss)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='qh_P0_s1')
    ap.add_argument('--n-points', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print(f"Era comparison for {args.run}")

    ref = xr.open_dataset(RESULTS / args.run / 'output_inference' /
                          yearly_files(args.run)[0][1].name)
    ny, nx = ref.sizes['y'], ref.sizes['x']
    ygrid, xgrid = ref.y.values, ref.x.values
    ref.close()

    rng = np.random.default_rng(args.seed)
    iys = rng.integers(0, ny, args.n_points)
    ixs = rng.integers(0, nx, args.n_points)

    print("  loading CNN...")
    cnn = load_cnn_points(args.run, iys, ixs)
    print(f"  CNN: {cnn.sizes['time']} unique steps "
          f"{str(cnn.time.values[0])[:13]} -> {str(cnn.time.values[-1])[:13]}")

    # -- RTMA truth: same native grid as the inference output, so index directly
    print("  loading RTMA truth...")
    ru = xr.open_dataset(RTMA_U, chunks={'time': 2000})
    rv = xr.open_dataset(RTMA_V, chunks={'time': 2000})
    if not (np.allclose(ru.y.values, ygrid) and np.allclose(ru.x.values, xgrid)):
        raise RuntimeError("RTMA grid does not match the inference grid")
    pt = dict(y=xr.DataArray(iys, dims='pt'), x=xr.DataArray(ixs, dims='pt'))
    tu = ru[only_var(ru)].isel(**pt)
    tv = rv[only_var(rv)].isel(**pt)
    truth = np.hypot(tu, tv).load()

    # -- ERA5 reference: coarse grid, interpolate straight to the sample points
    print("  loading ERA5...")
    eu = xr.open_dataset(ERA5_U, chunks={'time': 2000})
    ev = xr.open_dataset(ERA5_V, chunks={'time': 2000})
    ptc = dict(y=xr.DataArray(ygrid[iys], dims='pt'),
               x=xr.DataArray(xgrid[ixs], dims='pt'))
    e_u = eu[only_var(eu)].interp(**ptc)
    e_v = ev[only_var(ev)].interp(**ptc)
    era5 = np.hypot(e_u, e_v).load()

    # -- common time axis
    t = (pd.DatetimeIndex(cnn.time.values)
         .intersection(pd.DatetimeIndex(truth.time.values))
         .intersection(pd.DatetimeIndex(era5.time.values)))
    print(f"  common: {len(t)} steps {t[0]} -> {t[-1]}")

    # transpose explicitly: the boolean era mask below indexes axis 0
    cnn = cnn.sel(time=t).transpose('time', 'pt')
    truth = truth.sel(time=t).transpose('time', 'pt')
    era5 = era5.sel(time=t).transpose('time', 'pt')

    results = {}
    for era, (a, b) in ERAS.items():
        m = (t >= pd.Timestamp(a)) & (t <= pd.Timestamp(b))
        if m.sum() == 0:
            print(f"  {era}: no data")
            continue
        tru_e = truth.values[m].ravel()
        ref_e = era5.values[m].ravel()
        blk = {'hours': int(m.sum()), 'window': [a, b]}
        for prod in ('p50', 'uv'):
            blk[prod] = metrics(cnn[prod].values[m].ravel(), tru_e, ref_e)
        blk['p50'].update(median_skill(cnn['p50'].values[m],
                                       truth.values[m], era5.values[m]))
        results[era] = blk

        print(f"\n  {era}  ({blk['hours']} h)")
        keys = ('rmse', 'bias', 'std_ratio', 'skill_vs_era5', 'skillr_vs_era5',
                'skillr_med_allhours', 'skill_ew_q2', 'skillr_ew_q2',
                'skill_gt10', 'skillr_gt10', 'skillr_med_gt10', 'bias_gt10')
        print(f"    {'metric':18s} {'P50':>10s} {'|u,v|':>10s}")
        print(f"    {'era5_rmse':18s} {blk['p50'].get('era5_rmse', 0):10.4f}")
        for k in keys:
            va, vb = blk['p50'].get(k), blk['uv'].get(k)
            if va is None:
                continue
            # the median-aggregated keys are computed for P50 only
            tail = f" {vb:10.4f}" if isinstance(vb, float) else ""
            print(f"    {k:18s} {va:10.4f}{tail}")

    out = RESULTS / args.run / 'output_evaluation' / 'era_comparison.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'run': args.run, 'n_points': args.n_points,
                   'seed': args.seed, 'eras': results}, f, indent=2)
    print(f"\n  wrote {out}")


if __name__ == '__main__':
    main()

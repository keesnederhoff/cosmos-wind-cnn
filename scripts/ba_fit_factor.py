#!/usr/bin/env python
"""Observation-based factor correction (BA) for the v3 CNN-RTMA P50 product.

Station analysis of Era 3 (2020-01-01 -> 2026-08-10) showed the CNN's shortfall
against observations is MULTIPLICATIVE and nearly flat from the 75th to the
99th percentile, that the 90th percentile (10 % exceedance) is the stable
anchor (lowest year-to-year noise, clear of airport calm reporting), and that
station factors are NOT spatially correlated (stations 3-6 km apart differ as
much as stations 50 km apart). Hence one factor per location:

    f_s      = obs_P90 / cnn_P90 on paired hours, per station
    c_class  = median f_s over land (IEM) / bay (NOS shoreline NDBC) / ocean (46012, 46026)
    F(x) = c(x) + sum_i a_i k_i(x),   k_i(x) = exp(-d_i^2 / (2 sigma^2))
    a    = K^-1 (f - c_station),      K_ij = k_j sampled through station i's bilinear stencil

i.e. Gaussian radial-basis interpolation, exact at EVERY station as the
validation engine reads it (the user's requirement: no station keeps a bias),
Gaussian decay to the class constant of the cell away from them. Stations
closer than one grid cell are pooled first (the grid can carry one value
there). sigma is taken from a sweep (2.5 / 3.5 / 5 km): the first that does
not overshoot the station range by more than 10 %.

FLOOR (user decision 2026-10-03): the adjustment NEVER reduces wind speed. Station
factors below --floor (default 1.0) are raised to it before the fit, and the
final field is clipped to >= floor. Six Era-3 stations had obs/CNN < 1 (TIBC1,
RCMC1, PXOC1, FTPC1, EDU, VCB); they end up with factor 1, not a reduction. Patches at ALL stations incl. the 4 USGS moorings (user decision
2026-10-02); the moorings do not enter the class constants (1.2 m anemometers,
short records). WT_MW101 / WT_MW201 are the same mooring in consecutive
deployments and are pooled.

Inputs : results/v1940_points/<seed>/points_speed_full_record_ERA5_202*.nc (speed10 = P50
         through the validation engine's own bilinear stencils), station_table.nc,
         observations via validate_met_models.load_station (VAL_USGS=1,
         VAL_STATION_HEIGHTS=1 -> real anemometer heights, log-law to 10 m),
         RTMA_SFbay_2p5km_static_landsea_static_UTM10.nc for the class map.
Outputs: results/v1940_product/BA_factor_map_<seed>_E3q90.nc, BA_station_factors_E3q90.csv,
         BA_LOSO.txt, BA_factor_map.png, BA_class_map.png.

Light (~2 GB); still run it as a CPU batch job (scripts/cpu_ba_fit.slurm), not on
the login node.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay')
CASE = ROOT / 'sf_bay_rtma_v3'
POINTS = CASE / 'results' / 'v1940_points'
LANDSEA = ROOT / 'sf_bay_rtma' / 'raw_data' / 'RTMA_SFbay_2p5km_static_landsea_static_UTM10.nc'
E3 = ('2020-01-01', '2026-08-10T23:00')
OCEAN_BUOYS = ('46012', '46026')
MERGE = {'WT_MW201': 'WT_MW101'}          # same mooring, consecutive deployments
QLEV = [0.5, 0.75, 0.9, 0.95, 0.99]
CLASS_ID = {'land': 0, 'bay': 1, 'ocean': 2}
# Bay polygon (UTM10 km): west edge is the Golden Gate; encloses SF Bay, San Pablo
# Bay, Suisun Bay and the Delta; everything west of it is ocean.
BAY_POLY_KM = [(546.0, 4178.0), (546.0, 4188.0), (538.0, 4196.0), (536.0, 4215.0), (560.0, 4250.0),
               (650.0, 4250.0), (650.0, 4195.0), (605.0, 4160.0), (590.0, 4125.0),
               (570.0, 4125.0), (552.0, 4150.0)]


def station_class(sid, group):
    if group == 'IEM':
        return 'land'
    if group == 'USGS':
        return 'usgs'
    return 'ocean' if sid in OCEAN_BUOYS else 'bay'


def load_e3_points(seed):
    files = sorted(glob.glob(str(POINTS / seed / 'points_speed_full_record_ERA5_20[2-9]*.nc')))
    if not files:
        sys.exit(f'STOP: no Era-3 point files for {seed} in {POINTS}')
    ds = xr.concat([xr.open_dataset(f)[['speed10']].load() for f in files], dim='time')
    _, first = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=np.sort(first)).sel(time=slice(*E3))
    d = np.diff(ds.time.values).astype('timedelta64[h]').astype(int)
    if (d != 1).any():
        sys.exit('STOP: Era-3 point record is not hourly-contiguous')
    print(f'  points: {len(files)} files, {ds.time.size} h '
          f'{str(ds.time.values[0])[:13]} -> {str(ds.time.values[-1])[:13]}')
    return ds


def pair_stations(ds, tab, vmm, sink, min_pairs):
    """Paired (obs, CNN P50) hours per station over Era 3; merged moorings pooled."""
    groups = dict(zip(tab.station.values, tab.group.values))
    pairs = {}
    for sid in tab.station.values:
        cfg = vmm.STATIONS.get(sid)
        if cfg is None:
            print(f'  {sid}: not in validation STATIONS -- skipped'); continue
        with contextlib.redirect_stdout(sink):
            obs = vmm.load_station(sid, cfg)
        if obs is None:
            print(f'  {sid}: no observations loaded -- skipped'); continue
        o = (pd.DataFrame({'time': pd.to_datetime(obs['time']), 'obs': obs['speed10']})
             .dropna().sort_values('time'))
        m = pd.DataFrame({'time': pd.to_datetime(ds.time.values),
                          'cnn': ds.speed10.sel(station=sid).values})
        d = pd.merge_asof(o, m, on='time', direction='nearest',
                          tolerance=pd.Timedelta('1h')).dropna()
        key = MERGE.get(sid, sid)
        d['station'] = key
        pairs[key] = pd.concat([pairs[key], d]) if key in pairs else d
    out = {}
    for key, d in pairs.items():
        if len(d) < min_pairs:
            print(f'  {key}: only {len(d)} Era-3 pairs (< {min_pairs}) -- skipped'); continue
        out[key] = d.sort_values('time').reset_index(drop=True)
    return out, groups


def station_table(pairs, groups, tab, level):
    xs = dict(zip(tab.station.values, tab.x_utm.values / 1e3))
    ys = dict(zip(tab.station.values, tab.y_utm.values / 1e3))
    xk = {k: float(np.mean([xs[s] for s in ALIASES.get(k, [k])])) for k in pairs}
    yk = {k: float(np.mean([ys[s] for s in ALIASES.get(k, [k])])) for k in pairs}
    rows = []
    for sid, d in pairs.items():
        qo, qm = d.obs.quantile(QLEV).values, d.cnn.quantile(QLEV).values
        yearly = []
        for y, s in d.groupby(d.time.dt.year):
            if len(s) >= 2000:
                yearly.append(s.obs.quantile(level) / s.cnn.quantile(level))
        sid0 = ALIASES.get(sid, [sid])[0]
        row = dict(station=sid, group=groups[sid], klass=station_class(sid0, groups[sid]),
                   x_km=xk[sid], y_km=yk[sid], n_pairs=len(d),
                   t0=str(d.time.min())[:10], t1=str(d.time.max())[:10],
                   obs_mean=d.obs.mean(), cnn_mean=d.cnn.mean(), bias=(d.cnn - d.obs).mean(),
                   factor=qo[QLEV.index(level)] / qm[QLEV.index(level)],
                   factor_sd_years=float(np.std(yearly)) if len(yearly) >= 3 else np.nan,
                   n_years=len(yearly))
        for q, a, b in zip(QLEV, qo, qm):
            row[f'r{q:.2f}'] = a / b
        rows.append(row)
    return pd.DataFrame(rows).set_index('station')


def class_constants(T, rng, nboot=2000):
    out = {}
    for c in ('land', 'bay', 'ocean'):
        f = np.log(T.factor[T.klass == c].values)
        if f.size == 0:
            sys.exit(f'STOP: no stations in class {c}')
        boots = [np.median(rng.choice(f, f.size)) for _ in range(nboot)] if f.size > 2 else [f.mean()] * 2
        out[c] = dict(value=float(np.exp(np.median(f))), n=int(f.size),
                      lo=float(np.exp(np.percentile(boots, 5))),
                      hi=float(np.exp(np.percentile(boots, 95))),
                      fmin=float(np.exp(f.min())), fmax=float(np.exp(f.max())))
    return out


def class_map(T):
    """0 land, 1 bay, 2 ocean on the target grid.

    Water = binary land/sea static (1 = land where RTMA terrain h > 0). Bay = water
    inside BAY_POLY_KM (Golden Gate -> San Pablo -> Suisun -> Delta -> South Bay);
    ocean = water outside it that is connected to the west domain edge; remaining
    water (reservoirs, low bayshore cells with h = 0 outside the polygon) = land.
    Connectivity alone cannot separate Bay from ocean here: the Gate is open at
    2.5 km and pier stations sit on land cells.
    """
    from matplotlib.path import Path as MPath
    ls = xr.open_dataset(LANDSEA)
    land = ls['static_landsea'].values > 0.5                      # (y, x), y ascending
    x, y = ls.x.values / 1e3, ls.y.values / 1e3
    water = ~land
    X, Y = np.meshgrid(x, y)
    inside = MPath(BAY_POLY_KM).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    lab, n = ndimage.label(water & ~inside)
    ocean_labels = set(np.unique(lab[:, 0])) - {0}                 # touching the west edge
    cls = np.zeros(land.shape, np.int8)                             # land (incl. inland water)
    cls[np.isin(lab, list(ocean_labels))] = CLASS_ID['ocean']
    cls[water & inside] = CLASS_ID['bay']
    print(f'  class map: land fraction {land.mean():.3f}; {n} water bodies outside the Bay polygon')
    for k, v in CLASS_ID.items():
        print(f'    {k:5s}: {int((cls == v).sum()):5d} cells')
    print(f'    inland water assigned to land: {int((water & (cls == 0)).sum())} cells')
    for sid, r in T.iterrows():                                     # cell class under each station
        iy, ix = int(np.abs(y - r.y_km).argmin()), int(np.abs(x - r.x_km).argmin())
        T.loc[sid, 'cell_class'] = [k for k, v in CLASS_ID.items() if v == cls[iy, ix]][0]
    return cls, x, y


ALIASES = {}          # merged station key -> stencil-table ids it pools (filled at run time)


def stencil_ids(sid, tab):
    """Station ids in the stencil table that feed one (possibly merged) station."""
    ids = ALIASES.get(sid, [sid])
    return [s for s in ids if s in tab.station.values]


def auto_merge(pairs, tab, groups, dist_km):
    """Pool stations closer than one grid cell into a pseudo-station: the 2.5 km
    grid can carry only one factor there, and two conflicting values inside one
    cell make the interpolation blow up (MZXC1/UPBC1 0.5 km apart, OKXC1/OMHC1
    1.1 km apart in Era 3)."""
    xk = dict(zip(tab.station.values, tab.x_utm.values / 1e3))
    yk = dict(zip(tab.station.values, tab.y_utm.values / 1e3))
    for key in list(pairs):
        ALIASES.setdefault(key, [key] + [k for k, v in MERGE.items() if v == key])
    changed = True
    while changed:
        changed = False
        keys = list(pairs)
        pos = {k: (np.mean([xk[s] for s in ALIASES[k]]), np.mean([yk[s] for s in ALIASES[k]])) for k in keys}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                d = np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
                if d < dist_km and groups[ALIASES[a][0]] == groups[ALIASES[b][0]]:
                    new = f'{a}+{b}'
                    print(f'  merging {a} and {b} ({d:.2f} km apart, inside one grid cell) -> {new}')
                    pairs[new] = pd.concat([pairs.pop(a), pairs.pop(b)]).assign(station=new)
                    ALIASES[new] = ALIASES.pop(a) + ALIASES.pop(b)
                    groups[new] = groups[ALIASES[new][0]]
                    changed = True
                    break
            if changed:
                break
    return pairs, groups


_STENCIL = {}


def sample_stencil(field, tab, sid):
    """Field value at a station through the validation engine's bilinear stencil
    (mean over the merged aliases)."""
    vals = []
    for s in stencil_ids(sid, tab):
        if s not in _STENCIL:
            st = tab.sel(station=s)
            _STENCIL[s] = (st.iy.values.copy(), st.ix.values.copy(), st.w.values.copy())
        iy, ix, w = _STENCIL[s]
        vals.append(float(np.sum(w * field[iy, ix])))
    return float(np.mean(vals))


def factor_field(T, consts, cls, x, y, sigma, tab, quiet=False):
    """F on the grid (y, x): class constant of the cell + Gaussian radial-basis
    interpolation of the station residuals f_i - c_i (linear space).

    The interpolation system is written in STENCIL space -- K_ij = basis j
    sampled through station i's bilinear stencil -- and in linear F, so the
    field is exact at every station *as the validation engine reads it*
    (bilinear sample of speed x F). Also returns the plain Gaussian weight sum
    (how station-driven a cell is) and the coefficients (overshoot diagnostic)."""
    cvals = np.array([consts['land']['value'], consts['bay']['value'], consts['ocean']['value']])
    cfield = cvals[cls]
    P = T[['x_km', 'y_km']].values
    # residual against the class constant AS SAMPLED THROUGH THE STENCIL: a shoreline
    # station's quad mixes land and water cells, and only this makes it exact.
    ci = np.array([sample_stencil(cfield, tab, sid) for sid in T.index])
    resid = T.factor.values - ci
    X, Y = np.meshgrid(x, y)
    B = np.stack([np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * sigma ** 2)) for px, py in P])
    K = np.array([[sample_stencil(B[j], tab, sid) for j in range(len(P))] for sid in T.index])
    coef = np.linalg.solve(K + 1e-10 * np.eye(len(P)), resid)
    field = cfield + np.tensordot(coef, B, axes=1)
    wsum = B.sum(0)
    if not quiet:
        print(f'  RBF solve (sigma {sigma:g} km): {len(P)} stations, cond {np.linalg.cond(K):.2e}, '
              f'max |coef| {np.abs(coef).max():.2f} vs max |residual| {np.abs(resid).max():.3f}')
    return field, wsum, coef


def pooled_scores(pairs, T, factor_by_station):
    """Pooled Murphy skill etc. of P50 x factor over IEM+NDBC stations."""
    core = [s for s in pairs if T.klass[s] != 'usgs']
    D = pd.concat([pairs[s] for s in core], ignore_index=True)
    mu = D.groupby('station')['obs'].transform('mean')
    den = ((D.obs - mu) ** 2).sum()
    s = D.cnn * D.station.map(factor_by_station).astype(float)
    g = pd.DataFrame({'station': D.station, 's': s, 'obs': D.obs, 'e': s - D.obs})
    gb = g.groupby('station')
    qs = gb['s'].quantile([.5, .9, .99]).unstack(); qo = gb['obs'].quantile([.5, .9, .99]).unstack()
    rr = qs / qo
    sb = gb['e'].mean()
    return dict(skill=1 - (g.e ** 2).sum() / den, bias=sb.mean(), abs_bias=sb.abs().mean(),
                P50=rr[.5].median(), P90=rr[.9].median(), P99=rr[.99].median(),
                P90_sd=rr[.9].std(), P99_sd=rr[.99].std(), n_stations=len(core))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default='r1b_do010_s3')
    ap.add_argument('--level', type=float, default=0.90)
    ap.add_argument('--sigma-km', default='5,4,3.5,3,2.5',
                    help='comma list, largest first; all are diagnosed, the first that passes the '
                         'overshoot rule (field within the station range +-10 %%) is written')
    ap.add_argument('--min-pairs', type=int, default=1000)
    ap.add_argument('--floor', type=float, default=1.0,
                    help='never reduce wind speed: station factors and the field are clipped to '
                         '>= this value (user decision 2026-10-03); 0 disables')
    ap.add_argument('--overshoot-pct', type=float, default=20.0,
                    help='allowed excursion of the field beyond the station factor range')
    ap.add_argument('--merge-km', type=float, default=2.5,
                    help='pool stations closer than this (one grid cell) into one pseudo-station')
    ap.add_argument('--no-heights', action='store_true', help='keep the 10 m group defaults')
    ap.add_argument('--out', default=str(CASE / 'results' / 'v1940_product'))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f'E3q{int(round(a.level * 100))}'

    os.environ['VAL_USGS'] = '1'
    os.environ['VAL_STATION_HEIGHTS'] = '0' if a.no_heights else '1'
    os.environ.setdefault('COSMOS_VALIDATION_DATA_ROOT', str(ROOT / 'validation'))
    os.environ.setdefault('COSMOS_VALIDATION_OUTPUT_ROOT', str(ROOT / 'validation'))
    sys.path.insert(0, str(HERE.parent / 'validation'))
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        import validate_met_models as vmm                   # noqa: E402
    heights = {k: v['anemometer_height_m'] for k, v in vmm.STATIONS.items()}
    print('=' * 78 + f'\nSTEP 1  pair Era-3 observations with CNN P50 ({a.seed})\n' + '=' * 78)
    print(f'  VAL_STATION_HEIGHTS={os.environ["VAL_STATION_HEIGHTS"]}; non-10 m stations: '
          + ', '.join(f'{k} {v:g}' for k, v in sorted(heights.items()) if v != 10.0))
    tab = xr.open_dataset(POINTS / 'station_table.nc')
    ds = load_e3_points(a.seed)
    pairs, groups = pair_stations(ds, tab, vmm, sink, a.min_pairs)
    pairs, groups = auto_merge(pairs, tab, groups, a.merge_km)
    T = station_table(pairs, groups, tab, a.level)
    T['factor_raw'] = T['factor']
    if a.floor > 0:
        T['factor'] = T['factor'].clip(lower=a.floor)
        clamped = T.index[T.factor_raw < a.floor]
        print(f'  floor {a.floor:g}: never reduce wind speed -> {len(clamped)} station factors raised to '
              f'{a.floor:g}: ' + ', '.join(f'{s} ({T.factor_raw[s]:.2f})' for s in clamped))
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    print(f'\n  {len(T)} stations paired; factor = obs/CNN at q{a.level:.2f} (factor_raw = unfloored)')
    print(T[['klass', 'n_pairs', 'obs_mean', 'bias', 'r0.50', 'r0.75', 'r0.90', 'r0.95', 'r0.99',
             'factor_raw', 'factor', 'factor_sd_years']].sort_values(['klass', 'factor_raw'])
          .to_string(float_format=lambda v: f'{v:.3f}'))

    print('\n' + '=' * 78 + '\nSTEP 2  class constants (median of station factors, log space)\n' + '=' * 78)
    rng = np.random.default_rng(0)
    consts = class_constants(T, rng)
    for c, v in consts.items():
        print(f'  {c:5s} {v["value"]:.3f}  (n={v["n"]}, bootstrap 5-95% {v["lo"]:.3f}-{v["hi"]:.3f}, '
              f'stations {v["fmin"]:.2f}..{v["fmax"]:.2f})')
    usgs = T.factor[T.klass == 'usgs']
    if len(usgs):
        print(f'  usgs  {np.exp(np.log(usgs).median()):.3f}  (n={len(usgs)}, sensitivity only: '
              f'{", ".join(f"{s} {f:.2f}" for s, f in usgs.items())})')

    print('\n' + '=' * 78 + '\nSTEP 3  class map\n' + '=' * 78)
    cls, x, y = class_map(T)
    sigmas = [float(s) for s in str(a.sigma_km).split(',')]
    tol = a.overshoot_pct / 100.0
    lo_ok, hi_ok = (1 - tol) * T.factor.min(), (1 + tol) * T.factor.max()
    print(f'  overshoot rule: field within [{lo_ok:.3f}, {hi_ok:.3f}] (station range +-{a.overshoot_pct:g} %; '
          'cells must dip below the lowest station value to hit it through a 4-cell stencil)')

    # class-constant-only leave-one-out prediction is sigma-independent
    pred_cls = {}
    for sid, r in T.iterrows():
        c_others = class_constants(T.drop(sid), rng, nboot=2)
        pred_cls[sid] = max(c_others['bay' if r.klass == 'usgs' else r.klass]['value'], a.floor)
    y_true = np.log(T.factor.values)
    core = (T.klass != 'usgs').values

    def rmse_pct(pred):
        e = np.log(pd.Series(pred).reindex(T.index).values) - y_true
        return 100 * np.sqrt(np.mean(e[core] ** 2)), 100 * np.sqrt(np.mean(e ** 2))

    base_rows = [dict(field='raw', **pooled_scores(pairs, T, {s: 1.0 for s in T.index})),
                 dict(field=f'station factor q{a.level:.2f} (in-sample)',
                      **pooled_scores(pairs, T, T.factor.to_dict())),
                 dict(field='class constant (leave-one-out)', **pooled_scores(pairs, T, pred_cls))]
    print('\n' + '=' * 78 + '\nSTEP 4  factor field per sigma: exactness, overshoot, leave-one-out\n' + '=' * 78)
    results, chosen = {}, None
    for sigma in sigmas:
        print(f'\n--- sigma = {sigma:g} km')
        F, wsum, coef = factor_field(T, consts, cls, x, y, sigma, tab)
        if a.floor > 0:
            n_clip = int((F < a.floor).sum())
            F = np.maximum(F, a.floor)         # the field itself never reduces wind speed
            print(f'  floor {a.floor:g} applied to {n_clip} cells')
        if F.min() <= 0:
            sys.exit(f'STOP: factor field goes non-positive at sigma {sigma:g} km')
        logF = np.log(F)
        C = pd.DataFrame([dict(station=sid, klass=r.klass, factor=r.factor,
                               field=sample_stencil(F, tab, sid),
                               wsum=sample_stencil(wsum, tab, sid)) for sid, r in T.iterrows()]
                         ).set_index('station')
        C['dev_pct'] = 100 * (C.field / C.factor - 1)
        pred_gau = {}
        for sid in T.index:
            others = T.drop(sid)
            lf, _, _ = factor_field(others, class_constants(others, rng, nboot=2), cls, x, y,
                                    sigma, tab, quiet=True)
            pred_gau[sid] = max(float(sample_stencil(lf, tab, sid)), 0.05, a.floor)
        overshoot = (F.min() < lo_ok) or (F.max() > hi_ok)
        r_cls, r_cls_all = rmse_pct(pred_cls); r_gau, r_gau_all = rmse_pct(pred_gau)
        print(f'  field min {F.min():.3f} max {F.max():.3f} -> {"OVERSHOOT" if overshoot else "ok"}; '
              f'cells with station weight > 0.5: {int((wsum > 0.5).sum())}')
        print(f'  station exactness through the stencil: max |dev| {C.dev_pct.abs().max():.2f} %')
        print(f'  leave-one-out RMSE (log ratio): class only {r_cls:.1f} %, gaussian+class {r_gau:.1f} % '
              f'(IEM+NDBC n={core.sum()}); incl. USGS {r_cls_all:.1f} / {r_gau_all:.1f} %')
        row = dict(field=f'gaussian sigma={sigma:g} km + class (leave-one-out)',
                   **pooled_scores(pairs, T, pred_gau))
        results[sigma] = dict(F=F, logF=logF, wsum=wsum, C=C, pred_gau=pred_gau, row=row,
                              overshoot=overshoot, loso=(r_cls, r_gau, r_cls_all, r_gau_all))
        if chosen is None and not overshoot:
            chosen = sigma
    if chosen is None:
        sys.exit('STOP: every sigma overshoots -- inspect the station cluster, do not write a map')
    sigma = chosen
    R = results[sigma]; F, logF, wsum, C, pred_gau = R['F'], R['logF'], R['wsum'], R['C'], R['pred_gau']
    print(f'\n  CHOSEN sigma = {sigma:g} km (largest in the list without overshoot)')
    print('\n  station check (chosen sigma): field through the validation stencil vs fitted factor')
    print(C.to_string(float_format=lambda v: f'{v:.3f}'))
    S = pd.DataFrame(base_rows + [results[s]['row'] for s in sigmas])
    print('\n  Era-3 pooled scores, IEM+NDBC, P50 x factor (model/obs ratios are station medians):')
    print(S.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    with open(out / 'BA_LOSO.txt', 'w') as f:
        f.write(f'BA factor correction, seed {a.seed}, level q{a.level:.2f}, sigmas {sigmas} km '
                f'(chosen {sigma:g}), Era 3 {E3[0]}..{E3[1][:10]}, '
                f'VAL_STATION_HEIGHTS={os.environ["VAL_STATION_HEIGHTS"]}\n')
        f.write('class constants: ' + ', '.join(f'{c} {v["value"]:.3f} (n={v["n"]})' for c, v in consts.items()) + '\n')
        for s in sigmas:
            r = results[s]
            f.write(f'sigma {s:g} km: field {r["F"].min():.3f}..{r["F"].max():.3f} '
                    f'{"OVERSHOOT" if r["overshoot"] else "ok"}; LOSO RMSE class-only {r["loso"][0]:.1f} %, '
                    f'gaussian+class {r["loso"][1]:.1f} % (IEM+NDBC); incl. USGS {r["loso"][2]:.1f} / {r["loso"][3]:.1f} %\n')
        f.write('\n' + S.to_string(index=False, float_format=lambda v: f'{v:.3f}') + '\n')

    print('\n' + '=' * 78 + '\nSTEP 5  write outputs\n' + '=' * 78)
    T['pred_class_loo'] = pd.Series(pred_cls); T['pred_gauss_loo'] = pd.Series(pred_gau)
    T['field_at_station'] = C.field; T['anemometer_height_m'] = pd.Series(heights).reindex(T.index)
    a.sigma_km = sigma
    T.to_csv(out / f'BA_station_factors_{tag}.csv', float_format='%.4f')
    ds_out = xr.Dataset(
        {'factor': (('y', 'x'), F.astype('float32')),
         'log_factor': (('y', 'x'), logF.astype('float32')),
         'klass': (('y', 'x'), cls),
         'station_weight': (('y', 'x'), wsum.astype('float32'))},
        coords={'y': ls_coord(y), 'x': ls_coord(x)},
        attrs={'title': 'CNN-RTMA v3 observation-based factor correction (BA)',
               'seed': a.seed, 'anchor_quantile': a.level, 'sigma_km': a.sigma_km,
               'floor': a.floor,
               'floor_note': 'factor >= floor everywhere: the adjustment never reduces wind speed '
                             '(stations where obs/CNN < floor get factor = floor)',
               'fit_window': f'{E3[0]} .. {E3[1][:10]}',
               'c_land': consts['land']['value'], 'c_bay': consts['bay']['value'],
               'c_ocean': consts['ocean']['value'],
               'n_land': consts['land']['n'], 'n_bay': consts['bay']['n'], 'n_ocean': consts['ocean']['n'],
               'stations': ', '.join(f'{s}:{f:.3f}' for s, f in T.factor.items()),
               'station_heights': os.environ['VAL_STATION_HEIGHTS'],
               'method': ('factor = obs_P90/cnn_P90 per station (paired hours, obs log-law to 10 m); '
                          'F = c(class) + sum a_i exp(-d_i^2/2 sigma^2), a = K^-1 (f - c), K in bilinear-stencil '
                          'space: Gaussian radial-basis interpolation, exact at every station (stations within '
                          'one grid cell pooled); '
                          'apply as P50 x F, u/v x F'),
               'class_flags': '0 land (incl. inland water), 1 bay, 2 ocean',
               'history': f'{pd.Timestamp.today():%Y-%m-%d} scripts/ba_fit_factor.py'})
    ds_out['factor'].attrs = {'long_name': 'multiplicative correction factor for CNN P50 wind speed'}
    ds_out['x'].attrs = {'units': 'm', 'standard_name': 'projection_x_coordinate'}
    ds_out['y'].attrs = {'units': 'm', 'standard_name': 'projection_y_coordinate'}
    map_path = out / f'BA_factor_map_{a.seed}_{tag}.nc'
    ds_out.to_netcdf(map_path)
    print(f'  wrote {map_path}')
    plot_maps(F, cls, wsum, x, y, T, consts, out, tag)
    print('BA FIT OK')


def ls_coord(km):
    return (np.asarray(km) * 1e3).astype('float64')


def plot_maps(F, cls, wsum, x, y, T, consts, out, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ls = xr.open_dataset(LANDSEA)['static_landsea'].values
    mk = {'land': 'o', 'bay': 's', 'ocean': 'D', 'usgs': '^'}
    fig, ax = plt.subplots(figsize=(9, 8))
    pc = ax.pcolormesh(x, y, F, cmap='RdBu_r', vmin=0.7, vmax=1.5, shading='nearest')
    ax.contour(x, y, ls, levels=[0.5], colors='k', linewidths=0.4)
    for c, m in mk.items():
        s = T[T.klass == c]
        ax.scatter(s.x_km, s.y_km, c=s.factor, cmap='RdBu_r', vmin=0.7, vmax=1.5, marker=m,
                   s=55, edgecolors='k', linewidths=0.8, label=c)
    ax.set_xlim(500, 640); ax.set_ylim(4120, 4280); ax.set_aspect('equal')
    ax.set_title(f'BA factor F(x) ({tag}, floor 1 = never reduce): land {consts["land"]["value"]:.2f} '
                 f'bay {consts["bay"]["value"]:.2f} ocean {consts["ocean"]["value"]:.2f}')
    ax.legend(loc='lower left'); fig.colorbar(pc, ax=ax, label='obs / CNN at q0.90')
    fig.savefig(out / 'BA_factor_map.png', dpi=130, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].pcolormesh(x, y, cls, cmap='viridis', vmin=0, vmax=2, shading='nearest')
    ax[0].contour(x, y, ls, levels=[0.5], colors='w', linewidths=0.3)
    ax[0].scatter(T.x_km, T.y_km, c='r', s=8); ax[0].set_aspect('equal')
    ax[0].set_title('class: 0 land, 1 bay, 2 ocean')
    pc = ax[1].pcolormesh(x, y, wsum, cmap='magma', vmin=0, vmax=1, shading='nearest')
    ax[1].contour(x, y, ls, levels=[0.5], colors='w', linewidths=0.3); ax[1].set_aspect('equal')
    ax[1].set_title('station weight sum (1 = station-driven, 0 = class constant)')
    fig.colorbar(pc, ax=ax[1])
    fig.savefig(out / 'BA_class_map.png', dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'  wrote {out / "BA_factor_map.png"}, {out / "BA_class_map.png"}')


if __name__ == '__main__':
    main()

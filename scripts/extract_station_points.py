#!/usr/bin/env python
"""Sample a v3 inference year file at the validation stations.

Two modes.

--make-table GRID_FILE
    Build the station table ONCE from the validation engine itself
    (validation/validate_met_models.py: STATIONS with VAL_USGS=1, the same
    coordinate sources, latlon_to_utm10 and _stencil_1d with the configured
    INTERPOLATION_METHOD). Writes station ids/groups/coords AND the bilinear
    stencils against GRID_FILE's x/y. Needs COSMOS_VALIDATION_DATA_ROOT /
    COSMOS_VALIDATION_OUTPUT_ROOT.

--grid FILE --out POINTS_FILE [--ref ARCHIVED_FILE]
    Apply the stored stencils to every field of one inference file. No import
    of the validation engine, so GPU tasks stay light; the grid x/y are asserted
    equal to the ones the stencils were built on. With --ref, the same stencils
    are applied to an ARCHIVED year file and the max |diff| over the common
    steps is printed and stored as attrs -- the reproduction check for the
    2000-2026 overlap. Exit 3 if it exceeds --ref-tol.

Output: (time, station) u10/v10/speed10 exactly as the validation derives them
(bilinear u and v, speed = |(u, v)|), plus hr_speed_q (time, quantile, station)
and hr_gust_q (time, gust_quantile, station), float32 + zlib.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
DEFAULT_TABLE = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/'
                     'sf_bay_rtma_v3/results/v1940_points/station_table.nc')


def make_table(grid_file, table_path):
    os.environ.setdefault('VAL_USGS', '1')
    sys.path.insert(0, str(HERE.parent / 'validation'))
    import config as vcfg                      # noqa: E402
    import validate_met_models as vmm          # noqa: E402

    g = xr.open_dataset(grid_file)
    x1d, y1d = g['x'].values, g['y'].values
    g.close()
    if not (np.all(np.diff(x1d) > 0) and np.all(np.diff(y1d) > 0)):
        sys.exit('STOP: grid x/y must be ascending for _stencil_1d')

    rows = []
    for sid, cfg in vmm.STATIONS.items():
        if cfg.get('source') == 'pws':
            lat, lon = cfg['lat'], cfg['lon']          # == load_pws_archive's
            ox, oy = vmm.latlon_to_utm10(lat, lon)
        else:                                          # USGS: engine's own loader
            st = vmm.load_station(sid, cfg)
            if st is None:
                print(f'  skip {sid}: loader returned None')
                continue
            lat, lon, ox, oy = st['lat'], st['lon'], st['x_utm'], st['y_utm']
        iy, ix, w = vmm._stencil_1d(x1d, y1d, ox, oy, vcfg.INTERPOLATION_METHOD)
        rows.append((sid, cfg.get('group', ''), lat, lon, ox, oy, iy, ix, w))
        print(f'  {sid:14s} {cfg.get("group", ""):5s} x={ox:.0f} y={oy:.0f} '
              f'iy={iy.tolist()} ix={ix.tolist()}')

    n = len(rows)
    ds = xr.Dataset(
        {'group': ('station', np.array([r[1] for r in rows], dtype=object)),
         'lat': ('station', np.array([r[2] for r in rows])),
         'lon': ('station', np.array([r[3] for r in rows])),
         'x_utm': ('station', np.array([r[4] for r in rows])),
         'y_utm': ('station', np.array([r[5] for r in rows])),
         'iy': (('station', 'corner'), np.stack([r[6] for r in rows]).astype('i4')),
         'ix': (('station', 'corner'), np.stack([r[7] for r in rows]).astype('i4')),
         'w': (('station', 'corner'), np.stack([r[8] for r in rows])),
         'grid_x': ('gx', x1d), 'grid_y': ('gy', y1d)},
        coords={'station': np.array([r[0] for r in rows], dtype=object)},
        attrs={'interpolation_method': vcfg.INTERPOLATION_METHOD,
               'built_from': str(grid_file),
               'source': 'validation/validate_met_models.py STATIONS (VAL_USGS=1)'})
    table_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(table_path)
    print(f'wrote {table_path}: {n} stations')


def sample(path, tab):
    """Return a dict of point arrays for one inference file."""
    ds = xr.open_dataset(path)
    if not (np.array_equal(ds['x'].values, tab['grid_x'].values)
            and np.array_equal(ds['y'].values, tab['grid_y'].values)):
        sys.exit(f'STOP: {path} grid differs from the station-table grid')
    iy = tab['iy'].values.ravel()
    ix = tab['ix'].values.ravel()
    w = tab['w'].values                                  # (S, 4)
    S = w.shape[0]
    # One contiguous bounding-box read, then index in memory. Handing the
    # fancy index to the netCDF backend reads point-by-point and is ~10x slower.
    y0, y1 = int(iy.min()), int(iy.max()) + 1
    x0, x1 = int(ix.min()), int(ix.max()) + 1

    def pts(var):
        box = ds[var].isel(y=slice(y0, y1), x=slice(x0, x1)).values
        a = box[..., iy - y0, ix - x0].astype(float)     # (time, [q], S*4)
        a = a.reshape(a.shape[:-1] + (S, 4))
        out = np.einsum('...sc,sc->...s', a, w)
        out[np.abs(out) > 1e30] = np.nan
        return out

    res = {'time': ds['time'].values}
    u, v = pts('hr_u'), pts('hr_v')
    res['u10'], res['v10'] = u, v
    res['speed10'] = np.sqrt(u ** 2 + v ** 2)
    if 'hr_speed_q' in ds:
        res['hr_speed_q'] = pts('hr_speed_q')
        res['quantile'] = ds['quantile'].values
    if 'hr_gust_q' in ds:
        res['hr_gust_q'] = pts('hr_gust_q')
        res['gust_quantile'] = ds['gust_quantile'].values
    res['attrs'] = dict(ds.attrs)
    ds.close()
    return res


def extract(grid, out, table, ref, ref_tol):
    tab = xr.open_dataset(table)
    r = sample(grid, tab)
    st = tab['station'].values
    dv = {k: (('time', 'station'), r[k].astype('f4'))
          for k in ('u10', 'v10', 'speed10')}
    coords = {'time': r['time'], 'station': st,
              'group': ('station', tab['group'].values)}
    if 'hr_speed_q' in r:
        dv['hr_speed_q'] = (('time', 'quantile', 'station'), r['hr_speed_q'].astype('f4'))
        coords['quantile'] = r['quantile']
    if 'hr_gust_q' in r:
        dv['hr_gust_q'] = (('time', 'gust_quantile', 'station'), r['hr_gust_q'].astype('f4'))
        coords['gust_quantile'] = r['gust_quantile']
    attrs = {k: v for k, v in r['attrs'].items() if isinstance(v, (str, int, float))}
    attrs['source_grid_file'] = str(grid)
    attrs['station_table'] = str(table)

    status = 0
    if ref:
        rr = sample(ref, tab)
        common, i_n, i_r = np.intersect1d(r['time'], rr['time'], return_indices=True)
        msg = [f'  REF {Path(ref).name}: common steps {common.size}']
        for k in ('u10', 'v10', 'hr_speed_q'):
            if k not in r or k not in rr:
                continue
            a, b = r[k][i_n], rr[k][i_r]
            both = np.isfinite(a) & np.isfinite(b)
            mx = float(np.abs(a - b)[both].max()) if both.any() else float('nan')
            # GAIN = new finite where the archive is NaN. Expected: the first
            # seq_len-1 steps of a window the archive started cold (e.g.
            # 2000-01-01T00-04) now have ERA5 history. LOSS = the reverse, and
            # fails the check alongside any value difference.
            gain = int((np.isfinite(a) & np.isnan(b)).sum())
            loss = int((np.isnan(a) & np.isfinite(b)).sum())
            attrs[f'ref_maxdiff_{k}'] = mx
            attrs[f'ref_gain_{k}'] = gain
            attrs[f'ref_loss_{k}'] = loss
            msg.append(f'    {k:10s} max|diff|={mx:.3e} gain={gain} loss={loss}')
            if not (mx <= ref_tol) or loss:
                status = 3
        attrs['ref_file'] = str(ref)
        print('\n'.join(msg))
        print(f'  REF verdict: {"PASS" if status == 0 else "FAIL"} (tol {ref_tol})')

    ds = xr.Dataset(dv, coords=coords, attrs=attrs)
    enc = {k: {'zlib': True, 'complevel': 4} for k in dv}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.nc.tmp')
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, out)
    fin = np.isfinite(r['speed10'])
    print(f'  wrote {out.name}: {r["time"].size} steps x {st.size} stations, '
          f'finite {100 * fin.mean():.2f}%  {os.path.getsize(out) / 1e6:.1f} MB')
    return status


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-table', metavar='GRID_FILE')
    ap.add_argument('--table', default=str(DEFAULT_TABLE))
    ap.add_argument('--grid')
    ap.add_argument('--out')
    ap.add_argument('--ref')
    ap.add_argument('--ref-tol', type=float, default=1e-3)
    a = ap.parse_args()
    if a.make_table:
        make_table(a.make_table, Path(a.table))
        sys.exit(0)
    if not (a.grid and a.out):
        ap.error('--grid and --out are required unless --make-table')
    sys.exit(extract(a.grid, a.out, a.table, a.ref, a.ref_tol))

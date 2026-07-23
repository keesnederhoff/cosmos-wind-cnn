#!/usr/bin/env python
"""Derive high-resolution static predictor channels from the RTMA terrain field.

The downscaling model is predictor-limited: ERA5 enters at ~31 km (native 17x21)
and is interpolated onto the 2.5 km target grid, so the ONLY genuinely
high-resolution input is terrain. SF Bay winds at 2.5 km are largely
terrain/coastline forced (Golden Gate and Carquinez channeling, gap flows,
land-sea roughness contrast), so these channels add real fine-scale information
without any new data.

Writes ONE NetCDF per channel, with the data variable named exactly the
preprocessing `file_dict` key. That matters: `NetCDFPreprocessor._identify_variable`
resolves a key like `static_dhdx` by first checking its pattern table (which does
not know 'dhdx'), then falling back to an exact name match in `ds.data_vars`, and
finally to `list(ds.data_vars)[0]`. Naming the variable exactly the key hits the
exact-match branch; one variable per file makes the final fallback harmless too.

Channels (all on the native RTMA grid, coords copied verbatim from the source):
    static_dhdx      d(elev)/dx                        [m/m]
    static_dhdy      d(elev)/dy                        [m/m]
    static_landsea   1 = land (h > 0), 0 = water       [-]
    static_tpi       h - gaussian_smooth(h)            [m]
    static_distcoast distance to nearest water cell    [km]

No aspect: it is circular (the 359/0 deg seam), so z-score normalization of it is
meaningless. dh/dx + dh/dy encode the same directional information linearly.

Every channel MUST be NaN-free. A static is broadcast over the full time axis, so
a single NaN pixel sets `nan_at_time` True at EVERY timestep, which empties
valid_indices and yields a silent zero-sample training set.

Usage:
    python scripts/make_derived_terrain.py \
        --raw-data /path/to/sf_bay_rtma/raw_data \
        [--terrain RTMA_SFbay_2p5km_surface_height_static_UTM10.nc] \
        [--smooth-sigma 5.0] [--dry-run]
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt, gaussian_filter

TERRAIN_DEFAULT = 'RTMA_SFbay_2p5km_surface_height_static_UTM10.nc'
TERRAIN_VAR = 'surface_height'


def _grid_spacing(coord):
    """Uniform spacing of a 1-D coordinate, in metres (asserts near-uniformity)."""
    d = np.diff(np.asarray(coord, dtype='float64'))
    if not np.allclose(d, d[0], rtol=1e-6):
        raise ValueError(f"coordinate is not uniformly spaced: min={d.min()} max={d.max()}")
    return float(abs(d[0]))


def build_channels(h, dy, dx, smooth_sigma):
    """Return {name: (2-D array, units, long_name)} derived from elevation `h`."""
    # np.gradient returns d/d(row)=d/dy then d/d(col)=d/dx for a (y, x) array.
    dhdy, dhdx = np.gradient(h, dy, dx)

    land = (h > 0.0)
    landsea = land.astype('float32')

    tpi = h - gaussian_filter(h, sigma=smooth_sigma, mode='nearest')

    # Distance (in cells) from every cell to the nearest water cell, then -> km.
    # EDT computes distance to the nearest ZERO, so pass the land mask: water
    # cells are 0 and therefore distance 0.
    if land.all():
        raise ValueError("no water cells found (h > 0 everywhere) -- distcoast undefined")
    distcoast = distance_transform_edt(land, sampling=(dy, dx)) / 1000.0

    return {
        'static_dhdx': (dhdx.astype('float32'), 'm m**-1',
                        'Terrain slope, eastward component (d elevation / dx)'),
        'static_dhdy': (dhdy.astype('float32'), 'm m**-1',
                        'Terrain slope, northward component (d elevation / dy)'),
        'static_landsea': (landsea, '1',
                           'Land-sea mask derived from terrain (1 = land where h > 0, 0 = water)'),
        'static_tpi': (tpi.astype('float32'), 'm',
                       'Topographic position index (elevation minus smoothed elevation)'),
        'static_distcoast': (distcoast.astype('float32'), 'km',
                             'Distance to nearest water cell'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-data', required=True,
                   help='raw_data dir holding the terrain file; outputs written here')
    p.add_argument('--terrain', default=TERRAIN_DEFAULT, help='terrain filename')
    p.add_argument('--smooth-sigma', type=float, default=5.0,
                   help='gaussian sigma in CELLS for the TPI smoother (5 cells = 12.5 km)')
    p.add_argument('--dry-run', action='store_true',
                   help='compute and report stats but write nothing')
    args = p.parse_args()

    raw = Path(args.raw_data)
    src_path = raw / args.terrain
    if not src_path.exists():
        raise FileNotFoundError(f"terrain file not found: {src_path}")

    src = xr.open_dataset(src_path)
    if TERRAIN_VAR not in src.data_vars:
        raise KeyError(f"'{TERRAIN_VAR}' not in {src_path} (has {list(src.data_vars)})")
    da = src[TERRAIN_VAR]
    if da.dims != ('y', 'x'):
        raise ValueError(f"expected dims ('y','x'), got {da.dims}")

    h = np.asarray(da.values, dtype='float64')
    if np.isnan(h).any():
        raise ValueError("source terrain contains NaN -- derived channels would inherit it")

    dx = _grid_spacing(src['x'].values)
    dy = _grid_spacing(src['y'].values)

    print(f"Source : {src_path}")
    print(f"  shape={h.shape}  dx={dx:.1f} m  dy={dy:.1f} m")
    print(f"  elevation: min={h.min():.2f} max={h.max():.2f} m")
    n_water = int((h <= 0).sum())
    print(f"  cells h<=0: {n_water} / {h.size} ({100.0 * n_water / h.size:.1f}%)  "
          f"(h==0: {int((h == 0).sum())}, h<0: {int((h < 0).sum())})")
    print(f"  TPI smoother sigma: {args.smooth_sigma} cells "
          f"(~{args.smooth_sigma * dx / 1000.0:.1f} km)\n")

    channels = build_channels(h, dy, dx, args.smooth_sigma)

    for name, (arr, units, long_name) in channels.items():
        # Hard gate: a NaN here silently empties the training set downstream.
        n_nan = int(np.isnan(arr).sum())
        if n_nan:
            raise ValueError(f"{name}: {n_nan} NaN values -- refusing to write")
        if arr.shape != h.shape:
            raise ValueError(f"{name}: shape {arr.shape} != terrain {h.shape}")

        print(f"{name:18} min={arr.min():10.4f}  max={arr.max():10.4f}  "
              f"mean={arr.mean():10.4f}  nan=0")

        out = xr.Dataset(
            {name: (('y', 'x'), arr)},
            # Coords copied verbatim from the terrain file: training aligns statics
            # with broadcast_like (label matching, NO interpolation), so any coord
            # drift would silently produce an all-NaN channel.
            coords={'y': src['y'], 'x': src['x']},
        )
        out[name].attrs = {'long_name': long_name, 'units': units,
                           'derived_from': args.terrain}
        out.attrs = {
            'title': f'RTMA 2.5km SF-Bay derived static predictor: {name}',
            'source': str(args.terrain),
            'output_projection': src.attrs.get('output_projection',
                                               'UTM Zone 10N (EPSG:32610)'),
        }
        dest = raw / f'RTMA_SFbay_2p5km_{name}_static_UTM10.nc'
        if args.dry_run:
            print(f"  [dry-run] would write {dest}")
        else:
            out.to_netcdf(dest)
            print(f"  wrote {dest}")
        out.close()

    src.close()
    print("\nDone." + ("  (dry run -- nothing written)" if args.dry_run else ""))


if __name__ == '__main__':
    main()

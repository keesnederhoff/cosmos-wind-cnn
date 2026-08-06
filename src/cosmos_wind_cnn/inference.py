"""Shared inference core: sliding-window dataset + bounded-RAM streamed NetCDF inference."""
import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from cosmos_wind_cnn.utils.config import var_units_for


class SlidingWindowDataset(Dataset):
    """In-memory sliding-window dataset for inference (normalizes inputs, drops NaN windows)."""

    def __init__(self, data, input_vars, stats, sequence_length):
        self.input_vars = input_vars
        self.sequence_length = sequence_length
        n_times = data.sizes['time']

        self.arrays = {}
        nan_at_time = np.zeros(n_times, dtype=bool)
        for var in input_vars:
            arr = data[var].values.astype(np.float32)
            nan_at_time |= np.isnan(arr).any(axis=(1, 2))
            mean, std = stats[var]['mean'], stats[var]['std']
            self.arrays[var] = (arr - mean) / (std + 1e-8)

        self.n_times = n_times
        self.valid_indices = [
            i for i in range(n_times - sequence_length + 1)
            if not nan_at_time[i:i + sequence_length].any()
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        slices = [self.arrays[v][start:start + self.sequence_length]
                  for v in self.input_vars]
        return torch.from_numpy(np.stack(slices, axis=1)), start


def _write_cf_grid(nc, crs_str, has_x=True, has_y=True):
    """Add CF metadata so the output grid + projection are auto-recognized.

    Sets CF attributes on the x/y coordinate variables and, when ``crs_str`` is
    given (e.g. ``'EPSG:32610'``), writes a CF ``grid_mapping`` variable named
    ``crs`` (projection parameters + WKT, via pyproj). Returns the grid-mapping
    variable name to attach to each data variable, or ``None`` if no CRS given.
    """
    import pyproj
    crs = pyproj.CRS.from_user_input(crs_str) if crs_str else None
    projected = crs.is_projected if crs is not None else True
    if has_x:
        xv = nc.variables['x']
        xv.axis = 'X'
        xv.units, xv.standard_name, xv.long_name = (
            ('m', 'projection_x_coordinate', 'x coordinate of projection')
            if projected else ('degrees_east', 'longitude', 'longitude'))
    if has_y:
        yv = nc.variables['y']
        yv.axis = 'Y'
        yv.units, yv.standard_name, yv.long_name = (
            ('m', 'projection_y_coordinate', 'y coordinate of projection')
            if projected else ('degrees_north', 'latitude', 'latitude'))
    if crs is None:
        return None
    gm = nc.createVariable('crs', 'i4')
    for key, value in crs.to_cf().items():
        gm.setncattr(key, value)
    gm.spatial_ref = crs.to_wkt()   # GDAL / QGIS convention
    epsg = crs.to_epsg()
    if epsg:
        gm.epsg_code = int(epsg)
    return 'crs'


def run_streaming_inference(model, full_ds, input_vars, output_vars, stats,
                            sequence_length, output_path, *, device,
                            batch_size=64, num_workers=8, time_chunk=10000,
                            attrs=None, head_spec=None):
    """Stream sliding-window inference over `full_ds`, writing predictions to a
    NetCDF at `output_path` one time-chunk at a time (bounded RAM).

    `full_ds` is an xarray Dataset of the `input_vars` on the target grid (may be
    lazy; loaded per chunk). `attrs` (dict) is written as NetCDF global attributes.
    Returns (n_predicted, n_total).
    """
    attrs = attrs or {}
    n_total = len(full_ds.time)
    time_coords = full_ds.time.values
    y_coords = full_ds.y.values if 'y' in full_ds.coords else None
    x_coords = full_ds.x.values if 'x' in full_ds.coords else None
    height = full_ds.sizes.get('y', full_ds.sizes.get('latitude'))
    width = full_ds.sizes.get('x', full_ds.sizes.get('longitude'))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    VAR_UNITS = var_units_for(output_vars)
    target_offset = sequence_length - 1

    epoch0 = np.datetime64('1900-01-01T00:00:00')
    time_hours = (time_coords.astype('datetime64[ns]') - epoch0) / np.timedelta64(1, 'h')

    nc = netCDF4.Dataset(str(output_path), 'w', format='NETCDF4')
    nc.createDimension('time', n_total)
    nc.createDimension('y', height)
    nc.createDimension('x', width)
    tv = nc.createVariable('time', 'f8', ('time',))
    tv.units = 'hours since 1900-01-01'
    tv.calendar = 'gregorian'
    tv[:] = time_hours
    if y_coords is not None:
        nc.createVariable('y', 'f8', ('y',))[:] = y_coords
    if x_coords is not None:
        nc.createVariable('x', 'f8', ('x',))[:] = x_coords

    # CF coordinate + grid-mapping metadata so GIS/CF tools (QGIS, GDAL,
    # rioxarray, cartopy) auto-recognize the grid and projection. Driven by the
    # optional 'crs' entry in attrs (e.g. 'EPSG:32610').
    grid_mapping = _write_cf_grid(nc, attrs.pop('crs', None),
                                  has_x=x_coords is not None,
                                  has_y=y_coords is not None)

    t_chunk_nc = max(1, min(720, n_total))
    out_nc = {}

    # ---- probabilistic head: quantile variables + P50-derived hr_u / hr_v ----
    q_idx = g_idx = None
    if head_spec is not None:
        n_speed, n_gust = head_spec['n_speed'], head_spec['n_gust']
        s_taus = np.asarray(head_spec['speed_taus'], dtype='f8')
        g_taus = (np.asarray(head_spec['gust_taus'], dtype='f8')
                  if n_gust else None)

        # Optionally write only a subset of levels. CRPS/PIT/coverage need the
        # dense grid; the shipped product needs only a few levels and 19 levels
        # over 6 years is ~80 GB per run.
        sub = head_spec.get('quantile_subset')
        if sub:
            raw = [int(np.argmin(np.abs(s_taus - t))) for t in sub]
            q_idx = np.unique(raw)
            g_idx = (np.array(sorted({int(np.argmin(np.abs(g_taus - t)))
                                      for t in sub})) if n_gust else None)

            # Be explicit about what was actually written. The model can only
            # emit levels it was trained on: on the Q=19 midpoint grid the top
            # level is tau=0.9737, so a request for 0.99 lands there. Silently
            # substituting would let a field get labelled P99 when it is P97.4.
            print("    quantile subset  requested -> written:")
            for want, i in zip(sub, raw):
                gap = abs(float(s_taus[i]) - float(want))
                flag = "   <-- NEAREST AVAILABLE" if gap > 0.005 else ""
                print(f"      {want:.4f} -> {float(s_taus[i]):.4f}{flag}")
            if len(q_idx) < len(raw):
                dropped = len(raw) - len(q_idx)
                print(f"    WARNING: {dropped} requested level(s) collapsed onto a "
                      f"level already selected -- {len(q_idx)} distinct level(s) "
                      f"written, not {len(raw)}.")
            if max(abs(float(s_taus[i]) - float(w))
                   for w, i in zip(sub, raw)) > 0.02:
                print(f"    WARNING: the trained grid tops out at "
                      f"tau={float(s_taus[-1]):.4f}; higher levels are NOT "
                      f"available without retraining at a larger n_quantiles.")
        else:
            q_idx = np.arange(n_speed)
            g_idx = np.arange(n_gust) if n_gust else None

        nc.createDimension('quantile', len(q_idx))
        nc.createVariable('quantile', 'f8', ('quantile',))[:] = s_taus[q_idx]
        vq = nc.createVariable('hr_speed_q', 'f4', ('time', 'quantile', 'y', 'x'),
                               zlib=True, complevel=1,
                               chunksizes=(max(1, t_chunk_nc // 4), len(q_idx),
                                           height, width),
                               fill_value=np.float32(np.nan))
        vq.units = 'm s-1'
        vq.long_name = 'predictive quantiles of 10 m wind speed'
        out_nc['hr_speed_q'] = vq

        if n_gust:
            nc.createDimension('gust_quantile', len(g_idx))
            nc.createVariable('gust_quantile', 'f8', ('gust_quantile',))[:] = g_taus[g_idx]
            vg = nc.createVariable('hr_gust_q', 'f4',
                                   ('time', 'gust_quantile', 'y', 'x'),
                                   zlib=True, complevel=1,
                                   chunksizes=(max(1, t_chunk_nc // 4), len(g_idx),
                                               height, width),
                                   fill_value=np.float32(np.nan))
            vg.units = 'm s-1'
            vg.long_name = 'predictive quantiles of 10 m wind gust'
            out_nc['hr_gust_q'] = vg

        # hr_u / hr_v from the MEDIAN speed and the predicted direction. Emitting
        # these keeps every existing consumer -- obs validation, bias_correct.py,
        # the combine tooling, the grid-point evaluator -- working unchanged.
        for var in ('hr_u', 'hr_v'):
            v = nc.createVariable(var, 'f4', ('time', 'y', 'x'), zlib=True,
                                  complevel=1,
                                  chunksizes=(t_chunk_nc, height, width),
                                  fill_value=np.float32(np.nan))
            v.units = 'm s-1'
            v.long_name = f'{var[-1]}-component from the median predicted speed'
            if grid_mapping:
                v.grid_mapping = grid_mapping
                v.coordinates = 'x y'
            out_nc[var] = v
        if grid_mapping:
            for k in ('hr_speed_q', 'hr_gust_q'):
                if k in out_nc:
                    out_nc[k].grid_mapping = grid_mapping
                    out_nc[k].coordinates = 'x y'
    else:
        for var in output_vars:
            v = nc.createVariable(var, 'f4', ('time', 'y', 'x'), zlib=True, complevel=1,
                                  chunksizes=(t_chunk_nc, height, width),
                                  fill_value=np.float32(np.nan))
            if var in VAR_UNITS:
                v.units = VAR_UNITS[var]
            if grid_mapping:
                v.grid_mapping = grid_mapping
                v.coordinates = 'x y'
            out_nc[var] = v
    for key, value in attrs.items():
        setattr(nc, key, value)

    n_windows = max(0, n_total - sequence_length + 1)
    n_predicted = 0
    n_nan_outputs = 0
    with torch.no_grad():
        for s0 in tqdm(range(0, n_windows, time_chunk), desc='    Inference'):
            e0 = min(s0 + time_chunk, n_windows)
            in_hi = min(e0 + target_offset, n_total)
            block = full_ds.isel(time=slice(s0, in_hi)).load()
            ds_block = SlidingWindowDataset(block, input_vars, stats, sequence_length)
            nb = e0 - s0
            if head_spec is None:
                pred_block = {var: np.full((nb, height, width), np.nan, dtype=np.float32)
                              for var in output_vars}
            else:
                pred_block = {
                    'hr_speed_q': np.full((nb, len(q_idx), height, width), np.nan,
                                          dtype=np.float32),
                    'hr_u': np.full((nb, height, width), np.nan, dtype=np.float32),
                    'hr_v': np.full((nb, height, width), np.nan, dtype=np.float32),
                }
                if head_spec['n_gust']:
                    pred_block['hr_gust_q'] = np.full(
                        (nb, len(g_idx), height, width), np.nan, dtype=np.float32)
            if len(ds_block) > 0:
                loader = DataLoader(ds_block, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=torch.cuda.is_available())
                for batch_inputs, batch_starts in loader:
                    outputs = model(batch_inputs.to(device))
                    bnan = (~torch.isfinite(outputs)).sum().item()
                    if bnan > 0:
                        n_nan_outputs += bnan
                        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
                    outputs = outputs.cpu().numpy()
                    for b, local_start in enumerate(batch_starts.numpy()):
                        j = int(local_start)
                        if head_spec is None:
                            for c, var in enumerate(output_vars):
                                mean, std = stats[var]['mean'], stats[var]['std']
                                pred_block[var][j] = outputs[b, c] * (std + 1e-8) + mean
                        else:
                            # Head output is ALREADY physical -- no denormalisation.
                            ns = head_spec['n_speed']
                            s_q = outputs[b, :ns]
                            d = outputs[b, ns:ns + 2]
                            pred_block['hr_speed_q'][j] = s_q[q_idx]
                            med = s_q[ns // 2]          # tau = 0.5 on a midpoint grid
                            pred_block['hr_u'][j] = med * d[0]
                            pred_block['hr_v'][j] = med * d[1]
                            if head_spec['n_gust']:
                                pred_block['hr_gust_q'][j] = outputs[b, ns + 2:][g_idx]
                del loader
            t0 = s0 + target_offset
            t1 = e0 + target_offset
            for var, arr in pred_block.items():
                out_nc[var][t0:t1] = arr
            n_predicted += int(np.isfinite(
                pred_block['hr_u' if head_spec is not None
                           else output_vars[0]]).any(axis=(1, 2)).sum())
            del block, ds_block, pred_block

    nc.close()
    if n_nan_outputs > 0:
        print(f"    WARNING: {n_nan_outputs:,} non-finite outputs replaced with 0.")
    return n_predicted, n_total

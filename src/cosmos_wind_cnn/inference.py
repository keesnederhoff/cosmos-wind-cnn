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
                            attrs=None):
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
            pred_block = {var: np.full((e0 - s0, height, width), np.nan, dtype=np.float32)
                          for var in output_vars}
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
                        for c, var in enumerate(output_vars):
                            mean, std = stats[var]['mean'], stats[var]['std']
                            pred_block[var][j] = outputs[b, c] * (std + 1e-8) + mean
                del loader
            t0 = s0 + target_offset
            t1 = e0 + target_offset
            for var in output_vars:
                out_nc[var][t0:t1, :, :] = pred_block[var]
            n_predicted += int(np.isfinite(
                next(iter(pred_block.values()))).any(axis=(1, 2)).sum())
            del block, ds_block, pred_block

    nc.close()
    if n_nan_outputs > 0:
        print(f"    WARNING: {n_nan_outputs:,} non-finite outputs replaced with 0.")
    return n_predicted, n_total

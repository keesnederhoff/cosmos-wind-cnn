import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import netCDF4

from cosmos_wind_cnn.inference import SlidingWindowDataset, run_streaming_inference


def _make_ds(n_time=8, h=3, w=3, var_names=('lr_u', 'lr_v')):
    t0 = np.datetime64('2000-01-01T00')
    times = t0 + np.arange(n_time) * np.timedelta64(1, 'h')
    coords = {'time': times, 'y': np.arange(h, dtype='f8'), 'x': np.arange(w, dtype='f8')}
    data = {v: (('time', 'y', 'x'),
                (np.arange(n_time * h * w, dtype='f4').reshape(n_time, h, w) + i))
            for i, v in enumerate(var_names)}
    return xr.Dataset(data, coords=coords)


def _unit_stats(names):
    return {n: {'mean': 0.0, 'std': 1.0} for n in names}


def test_sliding_window_dataset_shapes_and_window_count():
    ds = _make_ds(n_time=8)
    d = SlidingWindowDataset(ds, ['lr_u', 'lr_v'], _unit_stats(['lr_u', 'lr_v']), sequence_length=3)
    assert len(d) == 8 - 3 + 1
    x, start = d[0]
    assert tuple(x.shape) == (3, 2, 3, 3)
    assert int(start) == 0


def test_sliding_window_dataset_drops_nan_windows():
    ds = _make_ds(n_time=6)
    ds['lr_u'].values[2, :, :] = np.nan
    d = SlidingWindowDataset(ds, ['lr_u', 'lr_v'], _unit_stats(['lr_u', 'lr_v']), sequence_length=3)
    assert len(d) == 1


class _LastFrameModel(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.n_out = n_out

    def forward(self, x):  # x: (B, seq, n_in, H, W)
        return x[:, -1, : self.n_out, :, :]


def test_run_streaming_inference_writes_expected_structure(tmp_path):
    ds = _make_ds(n_time=6)
    input_vars = ['lr_u', 'lr_v']
    output_vars = ['hr_u', 'hr_v']
    stats = _unit_stats(input_vars + output_vars)
    out = tmp_path / 'out.nc'
    model = _LastFrameModel(n_out=2).eval()

    n_pred, n_total = run_streaming_inference(
        model, ds, input_vars, output_vars, stats, sequence_length=3,
        output_path=out, device=torch.device('cpu'),
        batch_size=4, num_workers=0, time_chunk=100,
        attrs={'run_name': 'test', 'hr_source': 'CONUS404', 'lr_source': 'ERA5'},
    )

    assert n_total == 6
    assert n_pred > 0
    nc = netCDF4.Dataset(str(out))
    nc.set_auto_mask(False)  # unwritten positions have _FillValue=NaN; read raw NaN, not masked
    try:
        assert set(output_vars).issubset(set(nc.variables.keys()))
        assert nc.variables['hr_u'].shape == (6, 3, 3)
        assert nc.run_name == 'test'
        assert nc.lr_source == 'ERA5'
        vals = nc.variables['hr_u'][:]
        # predictions land at t = window_start + (seq-1); first 2 rows stay fill (NaN)
        assert np.isnan(vals[0]).all() and np.isnan(vals[1]).all()
        assert np.isfinite(vals[2]).all()
    finally:
        nc.close()


def test_run_streaming_inference_multichunk_denorm(tmp_path):
    # n_time=6, seq=3 -> windows start 0,1,2,3 (predictions at t=2,3,4,5).
    # time_chunk=2 forces TWO chunks across the window-start range, exercising the
    # chunk boundary; non-unit stats exercise the denormalization scaling.
    ds = _make_ds(n_time=6)
    input_vars = ['lr_u', 'lr_v']
    output_vars = ['hr_u', 'hr_v']
    stats = {
        'lr_u': {'mean': 10.0, 'std': 2.0}, 'lr_v': {'mean': 20.0, 'std': 4.0},
        'hr_u': {'mean': 5.0, 'std': 3.0},  'hr_v': {'mean': 7.0, 'std': 6.0},
    }
    out = tmp_path / 'mc.nc'
    model = _LastFrameModel(n_out=2).eval()

    run_streaming_inference(
        model, ds, input_vars, output_vars, stats, sequence_length=3,
        output_path=out, device=torch.device('cpu'),
        batch_size=2, num_workers=0, time_chunk=2, attrs={},
    )

    eps = 1e-8
    nc = netCDF4.Dataset(str(out))
    nc.set_auto_mask(False)
    try:
        # _LastFrameModel returns the normalized last input frame of channel c;
        # so output hr_<v>[t] == denorm_hr( (lr_<v>[t] - mean_lr) / (std_lr + eps) ).
        for t in (2, 3, 4, 5):  # every predicted timestep, spanning both chunks
            for in_v, out_v in (('lr_u', 'hr_u'), ('lr_v', 'hr_v')):
                norm = (ds[in_v].values[t] - stats[in_v]['mean']) / (stats[in_v]['std'] + eps)
                expected = norm * (stats[out_v]['std'] + eps) + stats[out_v]['mean']
                assert np.allclose(nc.variables[out_v][t], expected, rtol=1e-4), (out_v, t)
    finally:
        nc.close()

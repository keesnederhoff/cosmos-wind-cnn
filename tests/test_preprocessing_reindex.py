import numpy as np
import xarray as xr
from cosmos_wind_cnn.data.preprocessing import NetCDFPreprocessor


def _toy_ds_with_gap():
    # Hours 0,1,3 present (hour 2 missing) over a 2x2 grid
    times = np.array(['2011-01-01T00', '2011-01-01T01', '2011-01-01T03'],
                     dtype='datetime64[ns]')
    data = np.ones((3, 2, 2), dtype='float32')
    return xr.Dataset(
        {'rtma_u': (('time', 'y', 'x'), data)},
        coords={'time': times, 'y': [0, 1], 'x': [0, 1]},
    )


def test_reindex_regular_hourly_fills_gap_with_nan():
    ds = _toy_ds_with_gap()
    out = NetCDFPreprocessor._reindex_regular_hourly(ds)
    assert out.sizes['time'] == 4
    expected_times = np.array(
        ['2011-01-01T00', '2011-01-01T01', '2011-01-01T02', '2011-01-01T03'],
        dtype='datetime64[ns]')
    assert np.array_equal(out['time'].values, expected_times)
    assert bool(np.isnan(out['rtma_u'].isel(time=2)).all())
    assert float(out['rtma_u'].isel(time=0).mean()) == 1.0


def test_preprocessor_reads_prefix_and_grid_config():
    pre = NetCDFPreprocessor({
        'data_dir': '.', 'target_prefix': 'rtma_',
        'input_prefix': 'era5_', 'regular_time_grid': True,
    })
    assert pre.target_prefix == 'rtma_'
    assert pre.input_prefix == 'era5_'
    assert pre.regular_time_grid is True


def test_preprocessor_defaults_hr_lr():
    pre = NetCDFPreprocessor({'data_dir': '.'})
    assert pre.target_prefix == 'hr_'
    assert pre.input_prefix == 'lr_'
    assert pre.regular_time_grid is False


def test_static_input_broadcast_over_time(tmp_path):
    """A no-time 'other' field (e.g. terrain) is broadcast onto (time, y, x)."""
    times = np.array(['2011-01-01T00', '2011-01-01T01',
                      '2011-01-01T02', '2011-01-01T03'], dtype='datetime64[ns]')
    yy, xx = [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]

    def _save(name, var, data, with_time=True):
        coords = {'y': yy, 'x': xx}
        dims = ('y', 'x')
        if with_time:
            coords = {'time': times, **coords}
            dims = ('time', 'y', 'x')
        xr.Dataset({var: (dims, data)}, coords=coords).to_netcdf(tmp_path / name)

    _save('hr_u.nc', 'u', np.ones((4, 3, 3), dtype='f4'))
    _save('lr_u.nc', 'u', np.full((4, 3, 3), 2.0, dtype='f4'))
    terrain = np.arange(9, dtype='f4').reshape(3, 3)            # static, spatially varying
    _save('terrain.nc', 'surface_height', terrain, with_time=False)

    pre = NetCDFPreprocessor({'data_dir': str(tmp_path)})       # hr_/lr_ defaults
    out = pre.load_and_align_datasets({
        'hr_u': 'hr_u.nc', 'lr_u': 'lr_u.nc', 'static_terrain': 'terrain.nc',
    })

    assert out['static_terrain'].dims == ('time', 'y', 'x')
    assert out.sizes['time'] == 4
    for t in range(4):                                          # constant over time
        assert np.array_equal(out['static_terrain'].isel(time=t).values, terrain)

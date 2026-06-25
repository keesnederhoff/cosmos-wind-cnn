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


def test_preprocessor_defaults_backward_compatible():
    pre = NetCDFPreprocessor({'data_dir': '.'})
    assert pre.target_prefix == 'conus404_'
    assert pre.input_prefix == 'era5_'
    assert pre.regular_time_grid is False

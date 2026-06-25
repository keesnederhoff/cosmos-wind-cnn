from cosmos_wind_cnn.utils.config import (
    classify_file_keys, var_units_for, wind_var_names,
)


def test_classify_file_keys_rtma_prefix():
    file_dict = {
        'rtma_u': 'a.nc', 'era5_u': 'b.nc',
        'rtma_pressure': 'c.nc', 'era5_pressure': 'd.nc',
        'static_terrain': 'e.nc',
    }
    target, inp, other = classify_file_keys(
        file_dict, target_prefix='rtma_', input_prefix='era5_'
    )
    assert target == ['rtma_u', 'rtma_pressure']
    assert inp == ['era5_u', 'era5_pressure']
    assert other == ['static_terrain']


def test_classify_file_keys_defaults_conus404():
    file_dict = {'conus404_u': 'a.nc', 'era5_u': 'b.nc'}
    target, inp, other = classify_file_keys(file_dict)
    assert target == ['conus404_u']
    assert inp == ['era5_u']
    assert other == []


def test_var_units_for_rtma_and_conus404():
    units = var_units_for(['rtma_u', 'rtma_v', 'rtma_air_temp',
                           'rtma_dew_temp', 'rtma_pressure', 'rtma_rain'])
    assert units == {
        'rtma_u': 'm s**-1', 'rtma_v': 'm s**-1',
        'rtma_air_temp': 'K', 'rtma_dew_temp': 'K',
        'rtma_pressure': 'Pa', 'rtma_rain': 'mm hr**-1',
    }
    rad = var_units_for(['conus404_solar', 'conus404_thermal'])
    assert rad == {'conus404_solar': 'W m**-2', 'conus404_thermal': 'W m**-2'}


def test_var_units_for_skips_unknown():
    assert var_units_for(['rtma_visibility']) == {}


def test_wind_var_names_rtma():
    variable_pairs = {
        'wind_u': {'high_res': 'rtma_u', 'low_res': 'era5_u'},
        'wind_v': {'high_res': 'rtma_v', 'low_res': 'era5_v'},
        'pressure': {'high_res': 'rtma_pressure', 'low_res': 'era5_pressure'},
    }
    assert wind_var_names(variable_pairs) == ('rtma_u', 'rtma_v', 'era5_u', 'era5_v')


def test_wind_var_names_none_when_absent():
    variable_pairs = {'pressure': {'high_res': 'rtma_pressure', 'low_res': 'era5_pressure'}}
    assert wind_var_names(variable_pairs) is None

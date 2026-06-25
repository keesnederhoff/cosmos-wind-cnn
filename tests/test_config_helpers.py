from cosmos_wind_cnn.utils.config import (
    classify_file_keys, var_units_for, wind_var_names,
)


def test_classify_file_keys_explicit_prefix():
    file_dict = {
        'hr_u': 'a.nc', 'lr_u': 'b.nc',
        'hr_pressure': 'c.nc', 'lr_pressure': 'd.nc',
        'static_terrain': 'e.nc',
    }
    target, inp, other = classify_file_keys(
        file_dict, target_prefix='hr_', input_prefix='lr_'
    )
    assert target == ['hr_u', 'hr_pressure']
    assert inp == ['lr_u', 'lr_pressure']
    assert other == ['static_terrain']


def test_classify_file_keys_defaults_hr_lr():
    file_dict = {'hr_u': 'a.nc', 'lr_u': 'b.nc'}
    target, inp, other = classify_file_keys(file_dict)
    assert target == ['hr_u']
    assert inp == ['lr_u']
    assert other == []


def test_var_units_for_hr_and_lr():
    units = var_units_for(['hr_u', 'hr_v', 'hr_air_temp',
                           'hr_dew_temp', 'hr_pressure', 'hr_rain'])
    assert units == {
        'hr_u': 'm s**-1', 'hr_v': 'm s**-1',
        'hr_air_temp': 'K', 'hr_dew_temp': 'K',
        'hr_pressure': 'Pa', 'hr_rain': 'mm hr**-1',
    }
    rad = var_units_for(['lr_solar', 'lr_thermal'])
    assert rad == {'lr_solar': 'W m**-2', 'lr_thermal': 'W m**-2'}


def test_var_units_for_skips_unknown():
    assert var_units_for(['hr_visibility']) == {}


def test_wind_var_names_hr_lr():
    variable_pairs = {
        'wind_u': {'high_res': 'hr_u', 'low_res': 'lr_u'},
        'wind_v': {'high_res': 'hr_v', 'low_res': 'lr_v'},
        'pressure': {'high_res': 'hr_pressure', 'low_res': 'lr_pressure'},
    }
    assert wind_var_names(variable_pairs) == ('hr_u', 'hr_v', 'lr_u', 'lr_v')


def test_wind_var_names_none_when_absent():
    variable_pairs = {'pressure': {'high_res': 'hr_pressure', 'low_res': 'lr_pressure'}}
    assert wind_var_names(variable_pairs) is None

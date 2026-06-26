from pathlib import Path
import pytest
from cosmos_wind_cnn.utils.config import get_data_dir, get_run_dirs


def test_get_data_dir_uses_data_root(monkeypatch):
    monkeypatch.setenv('COSMOS_DATA_ROOT', '/storage')
    assert get_data_dir('case_studies/sf_bay_conus404') == \
        Path('/storage') / 'sf_bay_conus404' / 'raw_data'


def test_get_data_dir_errors_when_unset(monkeypatch):
    monkeypatch.delenv('COSMOS_DATA_ROOT', raising=False)
    with pytest.raises(RuntimeError, match='COSMOS_DATA_ROOT'):
        get_data_dir('case_studies/sf_bay_conus404')


def test_get_run_dirs_layout(monkeypatch):
    monkeypatch.setenv('COSMOS_RESULTS_ROOT', '/storage')
    d = get_run_dirs('case_studies/sf_bay_rtma', '3732177')
    base = Path('/storage') / 'sf_bay_rtma' / 'results' / '3732177'
    assert d['run_root'] == base
    assert d['checkpoint'] == base / 'checkpoint'
    assert d['data_processed'] == base / 'data_processed'
    assert d['logs'] == base / 'logs'
    assert d['output_inference'] == base / 'output_inference'
    assert d['output_evaluation'] == base / 'output_evaluation'


def test_get_run_dirs_errors_when_unset(monkeypatch):
    monkeypatch.delenv('COSMOS_RESULTS_ROOT', raising=False)
    with pytest.raises(RuntimeError, match='COSMOS_RESULTS_ROOT'):
        get_run_dirs('case_studies/sf_bay_rtma', '123')

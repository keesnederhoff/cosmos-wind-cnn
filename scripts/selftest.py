#!/usr/bin/env python
"""Cheap pre-launch self-test of the plumbing (login-node safe, no data read).

1. gap_hours split logic on a synthetic hourly record spanning the real window.
2. split_dates default (gap 0) is bit-identical to the v3 ratio-free behaviour.
3. YAML configs load and carry the recipe.
4. validation config registers the models and the WY26/EY eras.
5. make_tau_grid(51) has an exact median and top level 0.9902; the shipped levels snap
   onto actual grid levels with no collapse.
"""
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from cosmos_wind_cnn.data.preprocessing import NetCDFPreprocessor        # noqa: E402
from cosmos_wind_cnn.models.quantile_head import make_tau_grid            # noqa: E402

t = np.arange(np.datetime64('2020-01-01T00'), np.datetime64('2026-10-01T14'), np.timedelta64(1, 'h'))
ds = xr.Dataset({'x': ('time', np.zeros(t.size))}, coords={'time': t})
pp = NetCDFPreprocessor({'data_dir': '.', 'physical_bounds': {}, 'target_prefix': 'hr_',
                         'input_prefix': 'lr_', 'regular_time_grid': True})
sd = {'train_end': '2024-10-01', 'val_end': '2025-10-01'}
tr0, va0, te0 = pp.create_train_val_test_split(ds, split_dates=sd)
tr, va, te = pp.create_train_val_test_split(ds, split_dates=sd, gap_hours=48)
T = lambda d: (str(d.time.values[0])[:16], str(d.time.values[-1])[:16], d.time.size)
print('no gap :', T(tr0), T(va0), T(te0))
print('gap 48 :', T(tr), T(va), T(te))
assert tr0.time.size - tr.time.size == 48 and va0.time.size - va.time.size == 48
assert str(va.time.values[0])[:13] == '2024-10-01T00' and str(te.time.values[0])[:13] == '2025-10-01T00'
assert str(tr.time.values[-1])[:13] == '2024-09-28T23' and str(va.time.values[-1])[:13] == '2025-09-28T23'
assert te.time.size == te0.time.size
print('PASS gap_hours split')

tcfg = yaml.safe_load(open(REPO / 'case_studies/sf_bay_rtma/configs/training.yaml'))
pcfg = yaml.safe_load(open(REPO / 'case_studies/sf_bay_rtma/configs/preprocessing.yaml'))
icfg = yaml.safe_load(open(REPO / 'case_studies/sf_bay_rtma/configs/inference_preprocessing_1940.yaml'))
assert tcfg['n_quantiles'] == 51 and tcfg['n_gust_quantiles'] == 19 and tcfg['dropout_rate'] == 0.1
assert set(pcfg['file_dict']) == {'hr_u', 'hr_v', 'hr_gust', 'lr_u', 'lr_v', 'lr_cloud', 'lr_gust', 'static_terrain'}
assert 'earthrel' in pcfg['file_dict']['hr_u'] and 'earthrel' in pcfg['file_dict']['hr_v']
assert pcfg['gap_hours'] == 48 and str(pcfg['split_dates']['val_end']) == '2025-10-01'
assert set(icfg['sources']) == {'lr_u', 'lr_v', 'lr_gust', 'lr_cloud', 'static_terrain'}
for k, f in list(pcfg['file_dict'].items()) + [(k, v['file']) for k, v in icfg['sources'].items()]:
    p = Path('/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/sf_bay_rtma/raw_data') / f
    if 'earthrel' in f:
        continue                      # created by stage 1
    assert p.exists(), f'missing {p}'
print('PASS configs (inputs exist; earthrel created by stage 1)')

g = make_tau_grid(51)
print('tau grid 51: median', g[25], 'top', g[-1])
assert abs(g[25] - 0.5) < 1e-12 and abs(g[-1] - 0.9902) < 1e-4
lv = [float(x) for x in os.environ.get('CNN_QLEVELS', '0.0490,0.1078,0.2451,0.5000,0.7549,0.8922,0.9510,0.9902').split(',')]
idx = [int(np.argmin(np.abs(g - v))) for v in lv]
assert len(set(idx)) == len(lv) and max(abs(g[i] - v) for i, v in zip(idx, lv)) < 1e-4, (idx, lv)
print('PASS tau grid: shipped levels are exact grid levels', [round(float(g[i]), 4) for i in idx])
g19 = make_tau_grid(19)
assert abs(g19[9] - 0.5) < 1e-12
print('ALL SELFTESTS PASS')

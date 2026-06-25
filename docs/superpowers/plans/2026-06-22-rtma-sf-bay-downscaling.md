# RTMA SF Bay Downscaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained `sf_bay_rtma` case study that downscales ERA5 (~31 km) to RTMA 2.5 km for 6 meteorological variables (2011–2026), keeping the existing CONUS404 `sf_bay` study intact.

**Architecture:** Three small, backward-compatible generalizations to shared code remove the hardcoded `conus404_`/`era5_` assumptions (configurable target/input prefixes; an opt-in regular-hourly reindex to handle RTMA's ~1% missing hours; prefix-agnostic units and evaluation). Then a new case study directory with its own configs, staged data, and README. No changes to the model, loss, dataset, regridder, or trainer.

**Tech Stack:** Python 3.11, xarray, numpy, PyTorch (3D U-Net), pytest, YAML configs. Runs on Windows (dev) and Linux/SLURM (Tallgrass HPC).

**Reference spec:** `docs/superpowers/specs/2026-06-22-rtma-sf-bay-downscaling-design.md`

---

## File Structure

**New files:**
- `tests/test_config_helpers.py` — unit tests for the three pure helpers
- `tests/test_preprocessing_reindex.py` — unit test for the regular-hourly reindex
- `case_studies/sf_bay_rtma/configs/preprocessing.yaml`
- `case_studies/sf_bay_rtma/configs/training.yaml`
- `case_studies/sf_bay_rtma/configs/inference_preprocessing.yaml`
- `case_studies/sf_bay_rtma/README.md`
- `case_studies/sf_bay_rtma/results/.gitkeep`
- `case_studies/sf_bay_rtma/data/raw/.gitkeep` (data files are git-ignored)

**Modified files:**
- `src/cosmos_wind_cnn/utils/config.py` — add `classify_file_keys`, `var_units_for`, `wind_var_names`
- `src/cosmos_wind_cnn/data/preprocessing.py` — configurable prefixes + optional `_reindex_regular_hourly`; constructor reads new keys
- `scripts/preprocess_training.py` — pass new config keys into `NetCDFPreprocessor`
- `scripts/run_training_pipeline.py` — pass new keys to `NetCDFPreprocessor`; units via `var_units_for`; evaluation via `wind_var_names`
- `docs/adding_case_study.md` — note the configurable `target_prefix` / `regular_time_grid` options
- `case_studies/sf_bay_rtma/README.md` — domain details (created above)

---

## Task 1: Pure helper functions in `utils/config.py`

**Files:**
- Modify: `src/cosmos_wind_cnn/utils/config.py` (append functions after `parse_variable_config`)
- Test: `tests/test_config_helpers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Git/cosmos-wind-cnn && python -m pytest tests/test_config_helpers.py -v`
Expected: FAIL with `ImportError: cannot import name 'classify_file_keys'`

- [ ] **Step 3: Implement the helpers**

Append to `src/cosmos_wind_cnn/utils/config.py`:

```python
def classify_file_keys(file_dict, target_prefix: str = 'conus404_',
                       input_prefix: str = 'era5_'):
    """
    Partition file_dict keys into (target, input, other) by prefix, preserving order.

    target keys define the high-resolution reference grid; input keys are the coarse
    fields interpolated onto it. Anything matching neither prefix is returned as 'other'.
    """
    target_keys = [k for k in file_dict if k.startswith(target_prefix)]
    input_keys = [k for k in file_dict if k.startswith(input_prefix)]
    other_keys = [k for k in file_dict
                  if k not in target_keys and k not in input_keys]
    return target_keys, input_keys, other_keys


# Units keyed by variable-name suffix (prefix-agnostic: works for conus404_*, rtma_*, ...)
_UNIT_BY_SUFFIX = {
    'air_temp': 'K', 'dew_temp': 'K', 'pressure': 'Pa',
    'solar': 'W m**-2', 'thermal': 'W m**-2', 'rain': 'mm hr**-1',
    'u': 'm s**-1', 'v': 'm s**-1',
}


def var_units_for(var_names):
    """Map each variable name to a unit string by matching its suffix.

    Longer suffixes are matched first so 'air_temp' is not shadowed by 'temp'-style
    fragments. Names with no known suffix are omitted from the result.
    """
    suffixes = sorted(_UNIT_BY_SUFFIX, key=len, reverse=True)
    units = {}
    for name in var_names:
        for suffix in suffixes:
            if name == suffix or name.endswith('_' + suffix):
                units[name] = _UNIT_BY_SUFFIX[suffix]
                break
    return units


def wind_var_names(variable_pairs):
    """Return (u_target, v_target, u_input, v_input) from a training config's
    variable_pairs, or None if a u/v wind pair is not present.

    Recognises pair names 'wind_u'/'u' and 'wind_v'/'v'.
    """
    out = {}
    for pair_name, pair in variable_pairs.items():
        if pair_name in ('wind_u', 'u'):
            out['u_target'] = pair['high_res']
            out['u_input'] = pair['low_res']
        elif pair_name in ('wind_v', 'v'):
            out['v_target'] = pair['high_res']
            out['v_input'] = pair['low_res']
    if all(k in out for k in ('u_target', 'v_target', 'u_input', 'v_input')):
        return out['u_target'], out['v_target'], out['u_input'], out['v_input']
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config_helpers.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_config_helpers.py src/cosmos_wind_cnn/utils/config.py
git commit -m "feat: add prefix-agnostic config helpers (classify_file_keys, var_units_for, wind_var_names)"
```

---

## Task 2: Configurable prefixes + regular-hourly reindex in `preprocessing.py`

**Files:**
- Modify: `src/cosmos_wind_cnn/data/preprocessing.py` (`__init__`, `load_and_align_datasets`, add `_reindex_regular_hourly`)
- Test: `tests/test_preprocessing_reindex.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_preprocessing_reindex.py`:

```python
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
    # 0,1,2,3 -> 4 timesteps, hour 2 inserted
    assert out.sizes['time'] == 4
    expected_times = np.array(
        ['2011-01-01T00', '2011-01-01T01', '2011-01-01T02', '2011-01-01T03'],
        dtype='datetime64[ns]')
    assert np.array_equal(out['time'].values, expected_times)
    # inserted hour is all-NaN; present hours are unchanged
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocessing_reindex.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_reindex_regular_hourly'` and `target_prefix`

- [ ] **Step 3a: Extend `__init__`**

In `src/cosmos_wind_cnn/data/preprocessing.py`, replace the body of `__init__` (currently ending at `self.physical_bounds = config.get('physical_bounds', {})`):

```python
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['data_dir'])
        # Compression settings (can be overridden in config)
        self.compression_level = config.get('compression_level', 1)  # 0-9, lower = faster
        self.use_compression = config.get('use_compression', True)
        # Per-variable physical bounds {var_key: {'min': float, 'max': float}}
        self.physical_bounds = config.get('physical_bounds', {})
        # Prefixes identifying target (high-res reference grid) and input (coarse) keys.
        # Defaults keep the original CONUS404/ERA5 behaviour.
        self.target_prefix = config.get('target_prefix', 'conus404_')
        self.input_prefix = config.get('input_prefix', 'era5_')
        # Reindex all variables onto a complete hourly axis (NaN-filling missing hours)
        # before splitting. Needed for products with time gaps (e.g. RTMA); off by default.
        self.regular_time_grid = config.get('regular_time_grid', False)
```

- [ ] **Step 3b: Add the import and the `_reindex_regular_hourly` static method**

At the top of the file, update the imports to pull in the helper:

```python
from cosmos_wind_cnn.utils.config import classify_file_keys
```

Add this static method to the `NetCDFPreprocessor` class (place it next to `_standardize_coords`):

```python
    @staticmethod
    def _reindex_regular_hourly(ds: xr.Dataset) -> xr.Dataset:
        """Reindex onto a complete hourly time axis, NaN-filling any missing hours.

        Missing hours become explicit NaN rows so the dataset's NaN-window dropping
        (WindDataset3D / inference sliding window) excludes sequence windows that would
        otherwise silently span a time discontinuity.
        """
        t = ds['time'].values
        full = np.arange(t.min(), t.max() + np.timedelta64(1, 'h'),
                         np.timedelta64(1, 'h'))
        return ds.reindex(time=full)
```

- [ ] **Step 3c: Generalize the prefix logic in `load_and_align_datasets`**

Replace the block that currently reads:

```python
        # Identify CONUS404 variables to use as the spatial reference grid.
        # Keys starting with 'conus404_' are the high-resolution target grid.
        conus_keys = [k for k in file_dict if k.startswith('conus404_')]
        era5_keys  = [k for k in file_dict if k.startswith('era5_')]

        if not conus_keys:
            raise ValueError(
                "No CONUS404 variables found in file_dict. "
                "Keys must start with 'conus404_' for spatial reference."
            )
```

with:

```python
        # Identify target variables (high-resolution reference grid) and input
        # (coarse) variables by configurable prefix. Defaults: conus404_ / era5_.
        target_keys, input_keys, other_keys = classify_file_keys(
            file_dict, self.target_prefix, self.input_prefix
        )

        if not target_keys:
            raise ValueError(
                f"No target variables found in file_dict. "
                f"Keys must start with '{self.target_prefix}' for spatial reference."
            )
```

Then in the same method update the three loop sections to the new names:

1. Replace `conus_reference_da = None` with `target_reference_da = None`.
2. In Step 1, change `for var_name in conus_keys:` to `for var_name in target_keys:`, and every `conus_reference_da` to `target_reference_da`.
3. In Step 2, change `for var_name in era5_keys:` to `for var_name in input_keys:`, and every `conus_reference_da` to `target_reference_da` (including the interpolation print line and the `.interp(y=..., x=...)` call).
4. In Step 3, change `remaining = [k for k in file_dict if k not in conus_keys and k not in era5_keys]` to `remaining = other_keys` and `for var_name in remaining:` to `for var_name in other_keys:`.

- [ ] **Step 3d: Apply the optional reindex after building `combined`**

In `load_and_align_datasets`, immediately after `combined = xr.Dataset(data_vars)` and before the summary print, insert:

```python
        # Optional: reindex onto a complete hourly grid (NaN-fill gaps) for products
        # with missing hours (e.g. RTMA). Off by default — CONUS404 is gap-free.
        if self.regular_time_grid:
            n_before = combined.sizes['time']
            combined = self._reindex_regular_hourly(combined)
            n_after = combined.sizes['time']
            print(f"  Regular hourly grid: {n_before} -> {n_after} timesteps "
                  f"({n_after - n_before} gap hours NaN-filled)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocessing_reindex.py tests/test_config_helpers.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add tests/test_preprocessing_reindex.py src/cosmos_wind_cnn/data/preprocessing.py
git commit -m "feat: configurable target/input prefixes and optional regular-hourly reindex in preprocessing"
```

---

## Task 3: Update constructor call sites + generalize units/evaluation in the pipeline

**Files:**
- Modify: `scripts/preprocess_training.py` (`NetCDFPreprocessor({...})` construction)
- Modify: `scripts/run_training_pipeline.py` (`step_preprocess` construction; `VAR_UNITS` in `step_inference`; names in `step_evaluate_grid_points`)

- [ ] **Step 1: Pass new config keys in `preprocess_training.py`**

In `scripts/preprocess_training.py`, replace:

```python
    preprocessor = NetCDFPreprocessor({
        'data_dir': str(data_dir),
        'physical_bounds': config.get('physical_bounds', {}),
    })
```

with:

```python
    preprocessor = NetCDFPreprocessor({
        'data_dir': str(data_dir),
        'physical_bounds': config.get('physical_bounds', {}),
        'target_prefix': config.get('target_prefix', 'conus404_'),
        'input_prefix': config.get('input_prefix', 'era5_'),
        'regular_time_grid': config.get('regular_time_grid', False),
    })
```

- [ ] **Step 2: Pass new config keys in `run_training_pipeline.py` `step_preprocess`**

In `scripts/run_training_pipeline.py`, inside `step_preprocess`, replace the identical `NetCDFPreprocessor({...})` block with the same five-key version above.

- [ ] **Step 3: Generalize `VAR_UNITS` in `step_inference`**

In `scripts/run_training_pipeline.py`, add `var_units_for` and `wind_var_names` to the existing import:

```python
from cosmos_wind_cnn.utils.config import (
    load_config, parse_variable_config, get_run_dirs, var_units_for, wind_var_names,
)
```

Then replace the hardcoded `VAR_UNITS` dict in `step_inference`:

```python
    VAR_UNITS = {
        'conus404_u': 'm s**-1', 'conus404_v': 'm s**-1',
        'conus404_air_temp': 'K', 'conus404_dew_temp': 'K',
        'conus404_pressure': 'Pa', 'conus404_solar': 'W m**-2',
        'conus404_thermal': 'W m**-2', 'conus404_rain': 'mm hr**-1',
    }
```

with:

```python
    VAR_UNITS = var_units_for(output_vars)
```

- [ ] **Step 4: Generalize variable names in `step_evaluate_grid_points`**

In `scripts/run_training_pipeline.py`, near the top of `step_evaluate_grid_points` (after `output_dir.mkdir(...)`), load the archived training config and derive the wind names:

```python
    train_config = load_config(run_dirs['checkpoint'] / 'training.yaml')
    names = wind_var_names(train_config['variable_pairs'])
    if names is None:
        print("    No wind pair in training config -- skipping evaluation.")
        return
    u_tgt, v_tgt, u_in, v_in = names
```

Replace the required-variable checks:

```python
    # Check required variables
    for var in ['conus404_u', 'conus404_v']:
        if var not in inference_ds:
            print(f"    {var} not in inference output -- skipping evaluation.")
            return
    for var in ['conus404_u', 'conus404_v', 'era5_u', 'era5_v']:
        if var not in processed_ds:
            print(f"    {var} not in processed data -- skipping evaluation.")
            return
```

with:

```python
    # Check required variables
    for var in [u_tgt, v_tgt]:
        if var not in inference_ds:
            print(f"    {var} not in inference output -- skipping evaluation.")
            return
    for var in [u_tgt, v_tgt, u_in, v_in]:
        if var not in processed_ds:
            print(f"    {var} not in processed data -- skipping evaluation.")
            return
```

Replace the per-point extraction block:

```python
        # Model predictions
        mod_u = inference_ds['conus404_u'].isel(y=iy, x=ix).values[inf_idx].astype(float)
        mod_v = inference_ds['conus404_v'].isel(y=iy, x=ix).values[inf_idx].astype(float)

        # CONUS404 truth
        tru_u = processed_ds['conus404_u'].isel(y=iy, x=ix).values[proc_idx].astype(float)
        tru_v = processed_ds['conus404_v'].isel(y=iy, x=ix).values[proc_idx].astype(float)

        # ERA5
        e5_u = processed_ds['era5_u'].isel(y=iy, x=ix).values[proc_idx].astype(float)
        e5_v = processed_ds['era5_v'].isel(y=iy, x=ix).values[proc_idx].astype(float)
```

with:

```python
        # Model predictions
        mod_u = inference_ds[u_tgt].isel(y=iy, x=ix).values[inf_idx].astype(float)
        mod_v = inference_ds[v_tgt].isel(y=iy, x=ix).values[inf_idx].astype(float)

        # High-res truth (target)
        tru_u = processed_ds[u_tgt].isel(y=iy, x=ix).values[proc_idx].astype(float)
        tru_v = processed_ds[v_tgt].isel(y=iy, x=ix).values[proc_idx].astype(float)

        # ERA5 (coarse input)
        e5_u = processed_ds[u_in].isel(y=iy, x=ix).values[proc_idx].astype(float)
        e5_v = processed_ds[v_in].isel(y=iy, x=ix).values[proc_idx].astype(float)
```

- [ ] **Step 5: Verify the existing CONUS404 path still imports and resolves**

Run: `python -c "import scripts.run_training_pipeline" 2>&1 | head -5` from the repo root after `pip install -e .`
Expected: no ImportError (module imports cleanly). If `scripts` is not importable as a package, instead run:
`python -c "from cosmos_wind_cnn.utils.config import var_units_for, wind_var_names; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add scripts/preprocess_training.py scripts/run_training_pipeline.py
git commit -m "feat: drive units and grid-point evaluation from config (prefix-agnostic pipeline)"
```

---

## Task 4: Create the `sf_bay_rtma` case study (configs + README + scaffolding)

**Files:**
- Create: `case_studies/sf_bay_rtma/configs/preprocessing.yaml`
- Create: `case_studies/sf_bay_rtma/configs/training.yaml`
- Create: `case_studies/sf_bay_rtma/configs/inference_preprocessing.yaml`
- Create: `case_studies/sf_bay_rtma/README.md`
- Create: `case_studies/sf_bay_rtma/results/.gitkeep`, `case_studies/sf_bay_rtma/data/raw/.gitkeep`

- [ ] **Step 1: Create directory scaffolding**

```bash
cd D:/Git/cosmos-wind-cnn
mkdir -p case_studies/sf_bay_rtma/configs
mkdir -p case_studies/sf_bay_rtma/data/raw
mkdir -p case_studies/sf_bay_rtma/results
touch case_studies/sf_bay_rtma/data/raw/.gitkeep
touch case_studies/sf_bay_rtma/results/.gitkeep
```

- [ ] **Step 2: Write `configs/preprocessing.yaml`**

```yaml
# SF Bay RTMA Case Study - Preprocessing Configuration
#
# Target: RTMA 2.5 km (observation-constrained analysis), 2011-2026
# Input:  ERA5 ~31 km, interpolated onto the RTMA grid

# Prefix roles (generalised pipeline): target_ = high-res reference grid, input_ = coarse
target_prefix: 'rtma_'
input_prefix: 'era5_'

# Reindex onto a complete hourly axis (NaN-fill ~1% missing RTMA hours) before splitting.
# The dataset's NaN-window dropping then excludes any sequence window spanning a gap.
regular_time_grid: true

# File mappings: variable_key -> filename in data/raw/
file_dict:

  # Wind U / V
  rtma_u: 'RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10.nc'
  era5_u: 'ERA5_eastward_wind_1940_2026_UTM.nc'
  rtma_v: 'RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10.nc'
  era5_v: 'ERA5_northward_wind_1940_2026_UTM.nc'

  # Air temperature
  rtma_air_temp: 'RTMA_SFbay_2p5km_air_temperature_2011_2026_UTM10.nc'
  era5_air_temp: 'ERA5_air_temperature_1940_2026_UTM.nc'

  # Dew point temperature
  rtma_dew_temp: 'RTMA_SFbay_2p5km_dew_point_temperature_2011_2026_UTM10.nc'
  era5_dew_temp: 'ERA5_dew_point_temperature_1940_2026_UTM.nc'

  # Air pressure (RTMA = MSL pressure; ERA5 = msl-derived — consistent)
  rtma_pressure: 'RTMA_SFbay_2p5km_air_pressure_fixed_height_2011_2026_UTM10.nc'
  era5_pressure: 'ERA5_air_pressure_fixed_height_1940_2026_UTM.nc'

  # Precipitation (mm/hr)
  rtma_rain: 'RTMA_SFbay_2p5km_precipitation_2011_2026_UTM10.nc'
  era5_rain: 'ERA5_precipitation_1940_2026_UTM.nc'

# Time period — full ERA5 ∩ RTMA overlap
start_date: '2011-01-01'
end_date: '2026-06-18'

# Data split ratios
train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
compression_level: 1

# ── Physical value bounds (values outside [min, max] -> NaN) ──────────────────
physical_bounds:
  rtma_u:        {min: -100,  max: 100}
  era5_u:        {min: -100,  max: 100}
  rtma_v:        {min: -100,  max: 100}
  era5_v:        {min: -100,  max: 100}

  rtma_air_temp: {min: 220, max: 330}
  era5_air_temp: {min: 220, max: 330}

  rtma_dew_temp: {min: 200, max: 315}
  era5_dew_temp: {min: 200, max: 315}

  rtma_pressure: {min: 85000, max: 110000}
  era5_pressure: {min: 85000, max: 110000}

  rtma_rain:     {min: 0, max: 200}
  era5_rain:     {min: 0, max: 200}
```

- [ ] **Step 3: Write `configs/training.yaml`**

```yaml
# SF Bay RTMA Case Study - Training Configuration

# Variable Pairs (high-resolution target : low-resolution input)
variable_pairs:
  wind_u:
    high_res: 'rtma_u'
    low_res: 'era5_u'
  wind_v:
    high_res: 'rtma_v'
    low_res: 'era5_v'
  air_temperature:
    high_res: 'rtma_air_temp'
    low_res: 'era5_air_temp'
  dew_point_temperature:
    high_res: 'rtma_dew_temp'
    low_res: 'era5_dew_temp'
  pressure:
    high_res: 'rtma_pressure'
    low_res: 'era5_pressure'
  rainfall:
    high_res: 'rtma_rain'
    low_res: 'era5_rain'

# Model Architecture (same as sf_bay)
base_channels: 16
sequence_length: 6
forecast_horizon: 0
dropout_rate: 0.1

# Training Parameters
batch_size: 32
num_epochs: 200
learning_rate: 0.0003
weight_decay: 0.001
num_workers: 8

# Data Loading
load_in_memory: true
train_stride: 1
val_stride: 6

# Loss Function Weights
loss_alpha: 1.0
loss_beta: 0.5
loss_gamma: 0.3

# Learning Rate Scheduler
scheduler_patience: 5
scheduler_factor: 0.5

# Early Stopping
early_stopping_patience: 20

# Save checkpoint every N epochs
save_every: 10

# Coordinate Reference System for output NetCDFs
crs: 'EPSG:32610'  # UTM Zone 10N
```

- [ ] **Step 4: Write `configs/inference_preprocessing.yaml`**

```yaml
# SF Bay RTMA Case Study - Inference Preprocessing Configuration
#
# Regrid coarse ERA5 (or CMIP6 later) onto the RTMA target grid for inference.
# The model's targets are RTMA, but its inputs are always the coarse fields.

sources:
  era5_u:
    file: 'ERA5_eastward_wind_1940_2026_UTM.nc'
    source_var: null
  era5_v:
    file: 'ERA5_northward_wind_1940_2026_UTM.nc'
    source_var: null
  era5_air_temp:
    file: 'ERA5_air_temperature_1940_2026_UTM.nc'
    source_var: null
  era5_dew_temp:
    file: 'ERA5_dew_point_temperature_1940_2026_UTM.nc'
    source_var: null
  era5_pressure:
    file: 'ERA5_air_pressure_fixed_height_1940_2026_UTM.nc'
    source_var: null
  era5_rain:
    file: 'ERA5_precipitation_1940_2026_UTM.nc'
    source_var: null

# Full ERA5 record (long hindcast). Note: the model is trained only on 2011-2026,
# so inference before ~2011 extrapolates — treat early years as lower-confidence.
start_date: '1940-01-01'
end_date: '2027-01-01'

interpolation_method: 'linear'
compression_level: 1

physical_bounds:
  era5_u:        {min: -100,  max: 100}
  era5_v:        {min: -100,  max: 100}
  era5_air_temp: {min: 220,   max: 330}
  era5_dew_temp: {min: 200,   max: 315}
  era5_pressure: {min: 85000, max: 110000}
  era5_rain:     {min: 0,     max: 200}
```

- [ ] **Step 5: Write `README.md`**

```markdown
# SF Bay RTMA Case Study

Statistical downscaling of meteorological variables for the San Francisco Bay
region, using RTMA as the high-resolution training target (sister study to the
CONUS404-based `sf_bay`).

## Data

- **High-resolution target:** RTMA at 2.5 km (SF Bay domain), UTM Zone 10N, hourly
- **Low-resolution input:** ERA5 at ~31 km, interpolated to the same UTM10N grid
- **Training period:** 2011-2026 (RTMA availability; ~1% missing hours, mostly 2013)
- **Inference period:** 1940-2026 (full ERA5 record; pre-2011 extrapolates)
- **Grid:** 162 (x) x 123 (y) @ 2.5 km

## Variables (6 pairs)

| Variable | RTMA source | ERA5 source |
|----------|-------------|-------------|
| Eastward wind (U) | eastward_wind | u10 |
| Northward wind (V) | northward_wind | v10 |
| Air temperature | air_temperature | t2m |
| Dew point temperature | dew_point_temperature | d2m |
| Air pressure (MSL) | air_pressure_fixed_height | msl |
| Precipitation | precipitation | tp |

No shortwave/longwave radiation (RTMA has none). RTMA precipitation skill for
extremes is uncertain — validate `rtma_rain` against gauges before using it for
compound-flood forcing.

## Usage

```bash
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name first_run \
    --gpus 4
```

See `docs/adding_case_study.md` for the full workflow and
`docs/superpowers/specs/2026-06-22-rtma-sf-bay-downscaling-design.md` for the design.
```

- [ ] **Step 6: Commit**

```bash
git add case_studies/sf_bay_rtma
git commit -m "feat: add sf_bay_rtma case study (configs, README, scaffolding)"
```

---

## Task 5: Stage the RTMA + ERA5 data into `data/raw`

**Files:**
- Create (temporary, not committed): `case_studies/sf_bay_rtma/stage_data.py`
- Populate: `case_studies/sf_bay_rtma/data/raw/` (12 NetCDF files; git-ignored)

- [ ] **Step 1: Write the staging script**

Create `case_studies/sf_bay_rtma/stage_data.py`:

```python
"""One-off: copy RTMA target files (M:) + ERA5 input files (sf_bay) into data/raw."""
import shutil
from pathlib import Path

RTMA_SRC = Path("m:/emeryville_crescent/03_model_setup/meteo")
ERA5_SRC = Path(__file__).resolve().parents[1] / "sf_bay" / "data" / "raw"
DEST = Path(__file__).resolve().parent / "data" / "raw"
DEST.mkdir(parents=True, exist_ok=True)

RTMA_FILES = [
    "RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_air_temperature_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_dew_point_temperature_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_air_pressure_fixed_height_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_precipitation_2011_2026_UTM10.nc",
]
ERA5_FILES = [
    "ERA5_eastward_wind_1940_2026_UTM.nc",
    "ERA5_northward_wind_1940_2026_UTM.nc",
    "ERA5_air_temperature_1940_2026_UTM.nc",
    "ERA5_dew_point_temperature_1940_2026_UTM.nc",
    "ERA5_air_pressure_fixed_height_1940_2026_UTM.nc",
    "ERA5_precipitation_1940_2026_UTM.nc",
]

def copy_all(src_dir, files):
    for name in files:
        src, dst = src_dir / name, DEST / name
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"  [skip] {name} (already staged)")
            continue
        size_gb = src.stat().st_size / 1e9
        print(f"  [copy] {name} ({size_gb:.1f} GB) ...", flush=True)
        shutil.copy2(src, dst)

print("Staging RTMA targets...")
copy_all(RTMA_SRC, RTMA_FILES)
print("Staging ERA5 inputs...")
copy_all(ERA5_SRC, ERA5_FILES)
print("Done. Files in:", DEST)
```

- [ ] **Step 2: Run the staging script**

Run: `cd D:/Git/cosmos-wind-cnn && python case_studies/sf_bay_rtma/stage_data.py`
Expected: 12 `[copy]`/`[skip]` lines, ending `Done.` (this copies ~60+ GB and may take a while over the network drive).

- [ ] **Step 3: Verify all 12 files are present**

Run: `python -c "from pathlib import Path; d=Path('case_studies/sf_bay_rtma/data/raw'); fs=sorted(p.name for p in d.glob('*.nc')); print(len(fs)); [print(' ', f) for f in fs]"`
Expected: `12` followed by the 6 RTMA + 6 ERA5 filenames.

- [ ] **Step 4: Remove the throwaway staging script (data files stay, git-ignored)**

```bash
rm case_studies/sf_bay_rtma/stage_data.py
```

No commit needed (data is git-ignored; `.gitkeep` already committed in Task 4).

---

## Task 6: Dry-run verification (short slice)

**Files:** none modified — temporary edit to `configs/preprocessing.yaml` reverted at the end.

- [ ] **Step 1: Temporarily shorten the period for a fast dry run**

In `case_studies/sf_bay_rtma/configs/preprocessing.yaml`, change `end_date: '2026-06-18'` to `end_date: '2011-03-01'` (≈2 months including the 2013 gap-free early window).

- [ ] **Step 2: Run preprocessing only**

Run:
```bash
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name dryrun \
    --skip-train --skip-inference --skip-eval
```
Expected output includes:
- `No target variables found` does **not** appear; preprocessing proceeds.
- A `Regular hourly grid: N -> M timesteps (K gap hours NaN-filled)` line with K ≥ 0.
- `Saved reference grid` (~162×123), and `train.nc`/`val.nc`/`test.nc` written with 6 variables.
- Normalization stats printed for all 6 `rtma_*` variables.

- [ ] **Step 3: Verify processed shapes and the gap-drop logic**

Run:
```bash
python -c "import xarray as xr; ds=xr.open_dataset('case_studies/sf_bay_rtma/results/dryrun/data_processed/train.nc'); print(sorted(ds.data_vars)); print(dict(ds.sizes))"
```
Expected: the 6 `rtma_*` variables, grid `y=123, x=162`, and a non-zero `time`.

- [ ] **Step 4: Smoke-test the dataset NaN-window dropping**

Run:
```bash
python -c "
from cosmos_wind_cnn.data.dataset import WindDatasetInMemory
vp = ['rtma_u','rtma_v','rtma_air_temp','rtma_dew_temp','rtma_pressure','rtma_rain']
ds = WindDatasetInMemory('case_studies/sf_bay_rtma/results/dryrun/data_processed/train.nc',
    'case_studies/sf_bay_rtma/results/dryrun/data_processed/normalization_stats.pkl',
    input_vars=['era5_u','era5_v','era5_air_temp','era5_dew_temp','era5_pressure','era5_rain'],
    output_vars=vp, sequence_length=6, forecast_horizon=0, stride=1)
x,y = ds[0]; print('samples', len(ds), 'input', tuple(x.shape), 'target', tuple(y.shape))
"
```
Expected: `samples > 0`, input shape `(6, 6, 123, 162)`, target shape `(6, 123, 162)`. If gaps fell in-window, a `Dropped N/... samples` line appears — confirming the NaN fix works.

- [ ] **Step 5: Restore the full period**

Revert `end_date` in `case_studies/sf_bay_rtma/configs/preprocessing.yaml` back to `'2026-06-18'`.

- [ ] **Step 6: Remove the dry-run artefacts**

```bash
rm -rf case_studies/sf_bay_rtma/results/dryrun
```

- [ ] **Step 7: Run the full test suite once more**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS.

No commit (verification only; config restored to its committed state).

---

## Task 7: Docs + memory update

**Files:**
- Modify: `docs/adding_case_study.md` (mention the new config options)

- [ ] **Step 1: Document the new config options**

In `docs/adding_case_study.md`, in the "Configure preprocessing" section (after the `train_ratio`/`val_ratio`/`test_ratio` block), add:

```markdown
### Optional: non-CONUS404 targets and gappy products

The pipeline defaults to `conus404_` (target) and `era5_` (input) key prefixes.
To use a different target product, set the prefixes explicitly in
`preprocessing.yaml`:

```yaml
target_prefix: 'rtma_'        # high-resolution reference grid
input_prefix: 'era5_'         # coarse input
regular_time_grid: true       # reindex onto a complete hourly axis, NaN-filling gaps
```

`regular_time_grid` is needed for products with missing hours (e.g. RTMA): missing
timestamps become NaN rows, which the dataset's NaN-window dropping then excludes,
so no sequence window silently spans a time gap. See `case_studies/sf_bay_rtma` for a
worked example.
```

- [ ] **Step 2: Commit the docs**

```bash
git add docs/adding_case_study.md
git commit -m "docs: document configurable target prefix and regular_time_grid options"
```

- [ ] **Step 3: Update project memory (outside the repo)**

Update `C:\Users\keesn\.claude\projects\C--Users-keesn--claude-projects-cosmos-wind-cnn\memory\data-sources-and-period.md` to note that SF Bay now has a second target option (RTMA 2.5 km, 2011–2026, 6 vars, no radiation, ~1% gaps) in the `sf_bay_rtma` case study, and add a one-line pointer in `MEMORY.md`. (This is a manual memory edit, not a repo commit.)

---

## Self-Review Notes

- **Spec coverage:** §4 mapping → Task 4 configs; §5 period/split/gap → Task 2 reindex + Task 4 `regular_time_grid`/dates + Task 6 verification; §6 code changes → Tasks 1–3; §7 new files → Tasks 4–5; §8 verification → Task 6; §9 limitations → documented in README + inference config comments.
- **Backward compatibility:** defaults (`conus404_`/`era5_`, `regular_time_grid=False`) reproduce the existing `sf_bay` behavior; verified by `test_preprocessor_defaults_backward_compatible` and the unchanged CONUS404 `VAR_UNITS` mapping via `var_units_for`.
- **Type/name consistency:** `classify_file_keys`, `var_units_for`, `wind_var_names`, `_reindex_regular_hourly`, and config keys `target_prefix`/`input_prefix`/`regular_time_grid` are used identically across Tasks 1–4.
```

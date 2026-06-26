"""
NetCDF data preprocessing utilities
"""

import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from dask.diagnostics import ProgressBar
from cosmos_wind_cnn.utils.config import classify_file_keys


class NetCDFPreprocessor:
    """
    Handle multiple NetCDF files (ERA5, CONUS404, etc.)
    and combine them into a unified dataset
    """

    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['data_dir'])
        # Compression settings (can be overridden in config)
        self.compression_level = config.get('compression_level', 1)  # 0-9, lower = faster
        self.use_compression = config.get('use_compression', True)
        # Per-variable physical bounds {var_key: {'min': float, 'max': float}}
        # loaded from preprocessing.yaml → physical_bounds section
        self.physical_bounds = config.get('physical_bounds', {})
        # Prefixes identifying target (high-res reference grid) and input (coarse) keys.
        # Defaults: hr_ / lr_.
        self.target_prefix = config.get('target_prefix', 'hr_')
        self.input_prefix = config.get('input_prefix', 'lr_')
        # Reindex all variables onto a complete hourly axis (NaN-filling missing hours)
        # before splitting. Needed for products with time gaps (e.g. RTMA); off by default.
        self.regular_time_grid = config.get('regular_time_grid', False)

    def load_and_align_datasets(
        self,
        file_dict: Dict[str, str],
        start_date: str = None,
        end_date: str = None,
    ) -> xr.Dataset:
        """
        Load multiple NetCDF files and align them spatially/temporally.
        Uses dask for lazy loading to handle large datasets efficiently.

        LR (coarse) variables are interpolated onto the HR (fine) grid
        so that all variables share a common spatial grid.

        Args:
            file_dict: Dictionary like {
                'lr_u': 'LR_u_wind_2020_2023.nc',
                'hr_u': 'HR_u_wind_2020_2023.nc',
            }
            start_date: Optional start date string (e.g. '2010-01-01') to
                restrict the time period considered for the overlap.
            end_date: Optional end date string (e.g. '2021-12-31') to
                restrict the time period.

        Returns:
            Combined xarray Dataset with all variables on the HR grid.
        """
        raw_datasets = {}
        var_names_map = {}

        # First pass: load all datasets lazily with dask
        print("Loading datasets with dask (lazy loading)...")
        for var_name, filename in file_dict.items():
            filepath = self.data_dir / filename
            print(f"  Opening {var_name} from {filepath.name}")

            # Use chunks='auto' for dask lazy loading
            ds = xr.open_dataset(filepath, chunks='auto')
            raw_datasets[var_name] = ds

            # Get the actual variable name in the file
            actual_var_name = self._identify_variable(ds, var_name)
            var_names_map[var_name] = actual_var_name

        # Find common time range across all datasets
        print("\nFinding common time range...")
        time_coord_name = self._find_time_coordinate(raw_datasets)
        common_times = self._find_common_times(
            raw_datasets, time_coord_name, start_date=start_date, end_date=end_date
        )

        print(f"  Common time steps: {len(common_times)}")
        if len(common_times) > 0:
            print(f"  Time range: {common_times[0]} to {common_times[-1]}")

        # Identify target variables (high-resolution reference grid) and input
        # (coarse) variables by configurable prefix. Defaults: hr_ / lr_.
        target_keys, input_keys, other_keys = classify_file_keys(
            file_dict, self.target_prefix, self.input_prefix
        )

        if not target_keys:
            raise ValueError(
                f"No target variables found in file_dict. "
                f"Keys must start with '{self.target_prefix}' for spatial reference."
            )

        # Build combined dataset lazily - extract variables one by one
        print("\nBuilding combined dataset (lazy)...")

        # --- Step 1: load all HR (target) DataArrays (they define the target grid) ---
        data_vars = {}
        target_reference_da = None  # used as spatial template for interp_like

        for var_name in target_keys:
            ds = raw_datasets[var_name]
            actual_var_name = var_names_map[var_name]

            da = ds[actual_var_name].sel({time_coord_name: common_times})
            da = self._standardize_coords(da)
            da = self._mask_fill_values(da, var_name)

            if target_reference_da is None:
                target_reference_da = da  # spatial template for LR interpolation

            data_vars[var_name] = da

        # --- Step 2: load LR DataArrays and interpolate onto HR grid ---
        for var_name in input_keys:
            ds = raw_datasets[var_name]
            actual_var_name = var_names_map[var_name]

            da = ds[actual_var_name].sel({time_coord_name: common_times})
            da = self._standardize_coords(da)
            da = self._mask_fill_values(da, var_name)

            # Interpolate the coarse input onto the high-res target grid.
            # interp matches on named coordinates ('x', 'y') using linear
            # interpolation; the input already covers the target domain so no
            # extrapolation fill is needed.
            print(f"  Interpolating {var_name} ({da.sizes['y']}x{da.sizes['x']}) "
                  f"-> target grid ({target_reference_da.sizes['y']}x{target_reference_da.sizes['x']})...")
            da = da.interp(
                y=target_reference_da['y'],
                x=target_reference_da['x'],
                method='linear',
            )

            data_vars[var_name] = da

        # --- Step 3: handle any remaining variables not prefixed with lr_/hr_ ---
        for var_name in other_keys:
            ds = raw_datasets[var_name]
            actual_var_name = var_names_map[var_name]
            da = ds[actual_var_name].sel({time_coord_name: common_times})
            da = self._standardize_coords(da)
            da = self._mask_fill_values(da, var_name)
            data_vars[var_name] = da

        # Merge into a single dataset (all arrays now share time/y/x)
        combined = xr.Dataset(data_vars)

        # Optional: reindex onto a complete hourly grid (NaN-fill gaps) for products
        # with missing hours (e.g. RTMA). Off by default — HR data is typically gap-free.
        if self.regular_time_grid:
            n_before = combined.sizes['time']
            combined = self._reindex_regular_hourly(combined)
            n_after = combined.sizes['time']
            print(f"  Regular hourly grid: {n_before} -> {n_after} timesteps "
                  f"({n_after - n_before} gap hours NaN-filled)")

        print(f"\nCombined dataset (lazy-loaded):")
        print(f"  Variables: {list(combined.data_vars)}")
        print(f"  Grid shape: {len(combined.y)} x {len(combined.x)}")
        print(f"  Time points: {len(combined.time)}")

        return combined

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

    def _standardize_coords(self, da: xr.DataArray) -> xr.DataArray:
        """Standardize coordinate names for a DataArray"""
        rename_dict = {}

        # Time coordinate
        for name in ['time', 'TIME', 't', 'Time']:
            if name in da.coords and name != 'time':
                rename_dict[name] = 'time'
                break

        # Y coordinate (latitude)
        for name in ['lat', 'latitude', 'y', 'LAT', 'Y']:
            if name in da.coords and name != 'y':
                rename_dict[name] = 'y'
                break

        # X coordinate (longitude)
        for name in ['lon', 'longitude', 'x', 'LON', 'X']:
            if name in da.coords and name != 'x':
                rename_dict[name] = 'x'
                break

        if rename_dict:
            da = da.rename(rename_dict)

        return da

    def _mask_fill_values(self, da: xr.DataArray, var_name: str) -> xr.DataArray:
        """
        Replace out-of-range values with NaN.

        Priority:
          1. Per-variable bounds from preprocessing.yaml → physical_bounds
             (specific min/max for each variable key, e.g. hr_v: {min: -100, max: 100})
          2. Generic fallback: |value| > 1e10  (catches undeclared WRF fill values
             such as the ~1e37 that CONUS404 northward wind was using)

        Reports how many values were masked.
        """
        bounds = self.physical_bounds.get(var_name)

        if bounds:
            lo = bounds.get('min', -1e10)
            hi = bounds.get('max',  1e10)
            bad = da.isnull() | (da < lo) | (da > hi)
            range_str = f'[{lo}, {hi}]'
        else:
            # Generic fallback — catches gross fill values even without explicit bounds
            lo, hi = -1e10, 1e10
            bad = da.isnull() | (np.abs(da) > 1e10)
            range_str = '|value| > 1e10  (no bounds configured — using generic threshold)'

        n_bad = int(bad.sum().compute()) if hasattr(bad.sum(), 'compute') else int(bad.sum())
        if n_bad > 0:
            total = int(da.size)
            pct = 100.0 * n_bad / total
            print(f"  WARNING [{var_name}]: {n_bad:,} / {total:,} values ({pct:.2f}%) "
                  f"outside {range_str} → replaced with NaN")
            da = da.where((da >= lo) & (da <= hi))

        return da

    def _find_time_coordinate(self, datasets: Dict) -> str:
        """Find the time coordinate name"""
        possible_names = ['time', 'TIME', 't', 'Time']
        for ds in datasets.values():
            for name in possible_names:
                if name in ds.coords:
                    return name
        return 'time'  # default

    def _find_common_times(
        self,
        datasets: Dict,
        time_coord: str,
        start_date: str = None,
        end_date: str = None,
    ) -> np.ndarray:
        """Find overlapping time coordinates across all datasets.

        Args:
            datasets: dict of variable_name -> xr.Dataset
            time_coord: name of the time coordinate
            start_date: optional ISO date string to restrict overlap start
            end_date: optional ISO date string to restrict overlap end
        """
        time_sets = []
        for var_name, ds in datasets.items():
            if time_coord in ds.coords:
                times = set(ds[time_coord].values)
                time_sets.append(times)
                print(f"  {var_name}: {len(times)} time steps")

        # Find intersection of all time sets
        if len(time_sets) == 0:
            return np.array([])

        common = time_sets[0]
        for ts in time_sets[1:]:
            common = common.intersection(ts)

        # Sort the common times
        common_sorted = np.array(sorted(list(common)))

        # Apply optional date filters
        if start_date is not None:
            t_start = np.datetime64(start_date)
            common_sorted = common_sorted[common_sorted >= t_start]
            print(f"  Filtered to >= {start_date}: {len(common_sorted)} time steps remain")
        if end_date is not None:
            t_end = np.datetime64(end_date)
            common_sorted = common_sorted[common_sorted <= t_end]
            print(f"  Filtered to <= {end_date}: {len(common_sorted)} time steps remain")

        return common_sorted

    def _identify_variable(self, ds: xr.Dataset, var_type: str) -> str:
        """
        Identify the actual variable name in the NetCDF file
        """
        # Common naming patterns — first match wins; add new names here as needed
        patterns = {
            'u':          ['u', 'u10', 'U', 'U10', 'uwnd', 'u_wind', 'eastward_wind'],
            'v':          ['v', 'v10', 'V', 'V10', 'vwnd', 'v_wind', 'northward_wind'],
            'air_temp':   ['t2m', 'air_temperature', 'temp', 'temperature', 't', 'T'],
            'dew_temp':   ['d2m', 'dew_point_temperature', 'dew_temp', 'dpt'],
            'temperature':['t2m', 'air_temperature', 'dew_point_temperature', 'temp', 'temperature', 't', 'T'],
            'solar':      ['surface_solar_radiation', 'surface_solar_radiation_downwards',
                           'ssrd', 'radiation', 'solar', 'shortwave'],
            'thermal':    ['surface_thermal_radiation', 'surface_thermal_radiation_downwards',
                           'strd', 'thermal', 'longwave'],
            'radiation':  ['surface_solar_radiation', 'ssrd', 'radiation', 'solar', 'shortwave'],
            'pressure':   ['air_pressure_fixed_height', 'sp', 'msl', 'slp', 'pressure'],
            'rain':       ['precipitation', 'rainfall', 'tp', 'rain', 'precip',
                           'precipitation_flux', 'total_precipitation'],
            'cloud':      ['cloud_area_fraction', 'cloud', 'cloud_area', 'tcc'],
            'evap':       ['evaporation', 'e', 'evap'],
            'heat':       ['surface_sensible_heat_flux', 'sshf', 'sensible_heat', 'heat'],
        }

        # Determine variable type from key name
        base_type = var_type.split('_')[-1]  # Get 'u' from 'lr_u' or 'hr_u'

        if base_type not in patterns:
            # If exact match exists, use it
            if var_type in ds.data_vars:
                return var_type
            # Otherwise try first variable
            return list(ds.data_vars)[0]

        # Search for matching pattern
        for pattern in patterns[base_type]:
            if pattern in ds.data_vars:
                return pattern

        # Fallback: return first data variable
        print(f"  Warning: Could not identify variable for {var_type}, using {list(ds.data_vars)[0]}")
        return list(ds.data_vars)[0]

    def create_train_val_test_split(
        self,
        ds: xr.Dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """
        Split dataset chronologically (maintains lazy loading)
        """
        n_times = len(ds.time)

        train_end = int(n_times * train_ratio)
        val_end = int(n_times * (train_ratio + val_ratio))

        train_ds = ds.isel(time=slice(0, train_end))
        val_ds = ds.isel(time=slice(train_end, val_end))
        test_ds = ds.isel(time=slice(val_end, None))

        print(f"\nDataset split:")
        print(f"  Train: {len(train_ds.time)} timesteps")
        print(f"  Val:   {len(val_ds.time)} timesteps")
        print(f"  Test:  {len(test_ds.time)} timesteps")

        return train_ds, val_ds, test_ds

    def _standardize_variable_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Standardize variable attributes so long_name and standard_name
        match the variable key name (e.g., hr_air_temp).
        """
        for var_name in ds.data_vars:
            ds[var_name].attrs['long_name'] = var_name
            ds[var_name].attrs['standard_name'] = var_name
        return ds

    def save_processed_data(self, ds: xr.Dataset, output_path: str, compression_level: int = None):
        """
        Save processed dataset with optional compression and progress bar.

        Args:
            ds: Dataset to save
            output_path: Path to save to
            compression_level: 0 = no compression (fastest), 1-9 = zlib levels (default: 1)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use provided level or fall back to instance setting
        comp_level = compression_level if compression_level is not None else self.compression_level

        # Standardize variable attributes (long_name, standard_name = var key)
        ds = self._standardize_variable_attrs(ds)

        # Configure encoding
        encoding = {}
        for var in ds.data_vars:
            if comp_level > 0:
                encoding[var] = {
                    'zlib': True,
                    'complevel': comp_level,
                }
            else:
                # No compression - fastest writes
                encoding[var] = {
                    'zlib': False,
                }

        comp_str = f"compression level {comp_level}" if comp_level > 0 else "no compression"
        print(f"Saving to {output_path} ({comp_str})...")

        # Use dask progress bar for visual feedback
        with ProgressBar():
            ds.to_netcdf(output_path, encoding=encoding)

        # Show file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Done! File size: {size_mb:.1f} MB")

    def calculate_and_save_stats(self, train_ds: xr.Dataset, output_path: str):
        """Calculate normalization statistics from training data (uses dask)"""
        stats = {}

        print("\nCalculating normalization statistics...")
        for var in train_ds.data_vars:
            print(f"  Processing {var}...", end=" ", flush=True)
            # Compute stats lazily with dask
            da = train_ds[var]
            with ProgressBar():
                stats[var] = {
                    'mean': float(da.mean().compute()),
                    'std': float(da.std().compute()),
                    'min': float(da.min().compute()),
                    'max': float(da.max().compute())
                }
            print(f"  mean={stats[var]['mean']:.4f}, std={stats[var]['std']:.4f}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(stats, f)

        print(f"\nSaved statistics to {output_path}")
        return stats

"""
NetCDF data preprocessing utilities
"""

import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pickle


class NetCDFPreprocessor:
    """
    Handle multiple NetCDF files (ERA5, CONUS404, etc.)
    and combine them into a unified dataset
    """

    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['data_dir'])

    def load_and_align_datasets(self, file_dict: Dict[str, str]) -> xr.Dataset:
        """
        Load multiple NetCDF files and align them spatially/temporally

        Args:
            file_dict: Dictionary like {
                'era5_u': 'ERA5_u_wind_2020_2023.nc',
                'era5_v': 'ERA5_v_wind_2020_2023.nc',
                'conus404_u': 'CONUS404_u_wind_2020_2023.nc',
                'conus404_v': 'CONUS404_v_wind_2020_2023.nc',
                'temperature': 'ERA5_temperature_2020_2023.nc',
                'radiation': 'ERA5_solar_radiation_2020_2023.nc'
            }

        Returns:
            Combined xarray Dataset with all variables
        """
        datasets = {}
        raw_datasets = {}

        # First pass: load all datasets
        for var_name, filename in file_dict.items():
            filepath = self.data_dir / filename
            print(f"Loading {var_name} from {filepath}")

            ds = xr.open_dataset(filepath)
            raw_datasets[var_name] = ds

            # Get the actual variable name in the file
            actual_var_name = self._identify_variable(ds, var_name)

            # Extract just the data variable and rename it
            datasets[var_name] = ds[actual_var_name]

        # Find common time range across all datasets
        print("\nFinding common time range...")
        time_coord_name = self._find_time_coordinate(raw_datasets)
        common_times = self._find_common_times(raw_datasets, time_coord_name)
        
        print(f"  Common time steps: {len(common_times)}")
        if len(common_times) > 0:
            print(f"  Time range: {common_times[0]} to {common_times[-1]}")

        # Select only common times for each dataset
        for var_name in datasets:
            datasets[var_name] = datasets[var_name].sel({time_coord_name: common_times})

        # Combine all variables into one dataset
        combined = xr.Dataset(datasets)

        # Align coordinates (interpolate if needed)
        combined = self._align_coordinates(combined)

        # Handle missing values
        combined = self._handle_missing_values(combined)

        return combined

    def _find_time_coordinate(self, datasets: Dict) -> str:
        """Find the time coordinate name"""
        possible_names = ['time', 'TIME', 't', 'Time']
        for ds in datasets.values():
            for name in possible_names:
                if name in ds.coords:
                    return name
        return 'time'  # default

    def _find_common_times(self, datasets: Dict, time_coord: str) -> np.ndarray:
        """Find overlapping time coordinates across all datasets"""
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
        return common_sorted

    def _identify_variable(self, ds: xr.Dataset, var_type: str) -> str:
        """
        Identify the actual variable name in the NetCDF file
        ERA5 uses 'u10', 'v10', 't2m', etc.
        CONUS404 might use different names
        """
        # Common naming patterns
        patterns = {
            'u': ['u', 'u10', 'U', 'uwnd', 'u_wind', 'eastward_wind'],
            'v': ['v', 'v10', 'V', 'vwnd', 'v_wind', 'northward_wind'],
            'temperature': ['t', 't2m', 'temp', 'temperature', 'T', 'air_temperature', 'dew_point_temperature'],
            'radiation': ['ssrd', 'radiation', 'solar', 'shortwave'],
            'pressure': ['sp', 'pressure', 'msl', 'slp'],
            'cloud': ['cloud', 'cloud_area', 'cloud_area_fraction', 'tcc']
        }

        # Determine variable type
        base_type = var_type.split('_')[-1]  # Get 'u' from 'era5_u'

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
        print(f"Warning: Could not identify variable for {var_type}, using {list(ds.data_vars)[0]}")
        return list(ds.data_vars)[0]

    def _align_coordinates(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Align spatial and temporal coordinates across datasets
        Since your data is already on the same grid, we just standardize coordinate names
        """
        # Standardize coordinate names
        coord_mapping = {
            'latitude': ['lat', 'latitude', 'y', 'LAT'],
            'longitude': ['lon', 'longitude', 'x', 'LON'],
            'time': ['time', 'TIME', 't']
        }

        # Rename coordinates to standard names
        for standard_name, possible_names in coord_mapping.items():
            for name in possible_names:
                if name in ds.coords:
                    ds = ds.rename({name: standard_name})
                    break

        print(f"\nFinal dataset:")
        print(f"  Grid shape: {len(ds.latitude)} x {len(ds.longitude)}")
        print(f"  Time points: {len(ds.time)}")

        return ds

    def _handle_missing_values(self, ds: xr.Dataset) -> xr.Dataset:
        """Fill or remove missing values"""
        # Check for NaN values
        print("\nChecking for NaN values...")
        has_nan = False
        for var in ds.data_vars:
            nan_count = np.isnan(ds[var].values).sum()
            if nan_count > 0:
                print(f"  {var}: {nan_count} NaN values")
                has_nan = True

        if not has_nan:
            print("  No NaN values found!")
            return ds

        # Forward fill
        print("  Applying forward fill...")
        ds = ds.ffill(dim='time', limit=2)

        # Interpolate
        print("  Applying interpolation...")
        ds = ds.interpolate_na(dim='time', method='linear', limit=5)

        # Drop times with remaining NaN
        print("  Dropping remaining NaN times...")
        ds = ds.dropna(dim='time', how='any')

        return ds

    def create_train_val_test_split(
        self,
        ds: xr.Dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """
        Split dataset chronologically
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

    def save_processed_data(self, ds: xr.Dataset, output_path: str):
        """Save processed dataset"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)
        print(f"Saved to {output_path}")

    def calculate_and_save_stats(self, train_ds: xr.Dataset, output_path: str):
        """Calculate normalization statistics from training data"""
        stats = {}

        print("\nCalculating normalization statistics:")
        for var in train_ds.data_vars:
            stats[var] = {
                'mean': float(train_ds[var].mean().values),
                'std': float(train_ds[var].std().values),
                'min': float(train_ds[var].min().values),
                'max': float(train_ds[var].max().values)
            }
            print(f"  {var}: mean={stats[var]['mean']:.4f}, std={stats[var]['std']:.4f}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(stats, f)

        print(f"\nSaved statistics to {output_path}")
        return stats

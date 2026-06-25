# Data Preparation

This document describes how to prepare ERA5 and CONUS404 data for use with the wind downscaling CNN.

## Overview

The CNN expects all input and target data on the **same spatial grid** in a projected coordinate system (e.g., UTM). The preparation pipeline involves:

1. Extracting a spatial subset from both datasets for your domain of interest
2. Reprojecting CONUS404 from its native Lambert Conformal Conic grid to UTM
3. Interpolating ERA5 to the same UTM grid
4. Saving each variable as a separate NetCDF file

## Source Datasets

### ERA5 Reanalysis (~31 km)

- Source: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- Variables: 10m U/V wind, 2m temperature, surface pressure, solar radiation, cloud cover
- Temporal resolution: Hourly
- Spatial resolution: ~0.25 deg (~31 km)
- Coverage: 1940-present

### CONUS404 (4 km)

- Source: [USGS CONUS404 on AWS / ScienceBase](https://www.sciencebase.gov/catalog/item/6372ab09d34e4844940b59c3)
- Variables: Wind, temperature, pressure, radiation at 4 km resolution
- Temporal resolution: Hourly
- Spatial resolution: 4 km (native Lambert Conformal Conic)
- Coverage: 1979-2021

## Preparation Steps

### 1. Define the domain

Choose your study area and target UTM zone. For example:

| Case Study | UTM Zone | EPSG Code | Approximate Bounds (UTM, meters) |
|------------|----------|-----------|----------------------------------|
| SF Bay | 10N | EPSG:32610 | x: 540000-620000, y: 4140000-4240000 |
| Puget Sound | 10N | EPSG:32610 | x: 490000-580000, y: 5200000-5320000 |

### 2. Process CONUS404

CONUS404 is on a Lambert Conformal Conic grid. Convert to UTM:

```python
import xarray as xr
from pyproj import Transformer

# Load CONUS404 variable
ds = xr.open_dataset('CONUS404_raw.nc')

# Define coordinate transformation
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)

# Transform coordinates and interpolate to regular UTM grid
# (implementation depends on your grid setup)
```

Key considerations:
- CONUS404 uses a staggered grid; use mass-point variables (no stagger) when possible
- Interpolate to a regular UTM grid with uniform spacing (e.g., 4 km)
- Preserve the hourly temporal resolution

### 3. Process ERA5

ERA5 is on a regular lat/lon grid. Interpolate to the same UTM grid:

```python
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Load ERA5 variable
ds = xr.open_dataset('ERA5_raw.nc')

# Interpolate to UTM grid matching CONUS404
# The target grid should match exactly what CONUS404 was interpolated to
```

Key considerations:
- ERA5 is much coarser than CONUS404; interpolation will produce a smooth field
- Use bilinear interpolation for continuous variables
- Ensure time coordinates match between ERA5 and CONUS404 (both hourly UTC)

### 4. Output format

Each processed NetCDF file should have:

- **Dimensions**: `time`, `y`, `x` (or `latitude`, `longitude`)
- **Coordinates**: `time` (datetime64), `latitude` (2D), `longitude` (2D), optionally `x` and `y` (1D UTM meters)
- **Data variable**: One variable per file (e.g., `eastward_wind`, `dew_point_temperature`)

Example structure:
```
<xarray.Dataset>
Dimensions:    (time: 17544, y: 25, x: 20)
Coordinates:
    time       (time) datetime64[ns]
    latitude   (y, x) float64
    longitude  (y, x) float64
    x          (x) float64
    y          (y) float64
Data variables:
    eastward_wind (time, y, x) float32
```

### 5. File naming convention

Place files in `case_studies/<name>/data/raw/` following the naming convention used in the preprocessing config:

```
<SOURCE>_<DOMAIN>_<RESOLUTION>_<VARIABLE>_<YEAR_RANGE>_<CRS>.nc
```

Examples for SF Bay:
```
CONUS404_SFbay_4km_eastward_wind_2020_2021_UTM10.nc
ERA5_eastward_wind_2020_2021_UTM.nc
```

### 6. Verify alignment

Before running preprocessing, verify that all files share the same spatial grid:

```python
import xarray as xr

files = ['file1.nc', 'file2.nc', ...]
for f in files:
    ds = xr.open_dataset(f'case_studies/my_study/data/raw/{f}')
    print(f"{f}: dims={dict(ds.dims)}, time={len(ds.time)}")
    ds.close()
```

All files should have identical `y` and `x` dimensions, and overlapping time coordinates.

## Next Steps

After preparing the data, run the full pipeline:

```bash
python scripts/run_training_pipeline.py \
    --case-study case_studies/my_study \
    --run-name first_run \
    --gpus 4
```

Or configure individual steps:

1. Edit `configs/preprocessing.yaml` with your file mappings
2. Edit `configs/training.yaml` with variable pairs and hyperparameters
3. Edit `configs/inference_preprocessing.yaml` with source file mappings for inference

See [adding_case_study.md](adding_case_study.md) for the full walkthrough.

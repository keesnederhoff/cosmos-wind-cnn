"""
General-purpose regridder for interpolating coarse climate/weather data onto
a high-resolution target grid.

Designed for any coarse product (ERA5, CMIP6, etc.) that needs to be
interpolated to the training target grid (e.g. CONUS404) before being used
as CNN input -- both during training preprocessing and at inference time.

Typical workflow
----------------

**During training preprocessing** (run once, saves reference grid)::

    regridder = Regridder.from_target_dataset(conus404_ds)
    regridder.save_reference_grid(output_path)
    regridded_era5 = regridder.regrid(era5_ds, var_map, physical_bounds)

**During inference** (loads saved reference grid, no target data needed)::

    regridder = Regridder.from_reference_grid(reference_grid_path)
    regridded = regridder.regrid(era5_ds, var_map, physical_bounds)
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Optional


class Regridder:
    """
    Interpolate coarse-resolution data onto a target grid.

    The target grid is defined by 1-D ``y`` and ``x`` coordinate arrays
    (typically UTM easting/northing, but any projected or geographic CRS
    works as long as the coarse source uses the same CRS).

    Parameters
    ----------
    target_y : np.ndarray
        1-D array of target grid y-coordinates (e.g. UTM northing in metres).
    target_x : np.ndarray
        1-D array of target grid x-coordinates (e.g. UTM easting in metres).
    method : str, optional
        Interpolation method passed to :func:`xarray.DataArray.interp`.
        One of ``'linear'`` (default), ``'nearest'``, ``'cubic'``, etc.
    target_attrs : dict, optional
        Coordinate attributes (units, CRS, etc.) carried from the target
        dataset.  Stored so they can be written into output files.
    """

    def __init__(
        self,
        target_y: np.ndarray,
        target_x: np.ndarray,
        method: str = 'linear',
        target_attrs: Optional[Dict] = None,
    ):
        self.target_y = np.asarray(target_y)
        self.target_x = np.asarray(target_x)
        self.method = method
        self.target_attrs = target_attrs or {}

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_target_dataset(cls, ds: xr.Dataset, method: str = 'linear') -> 'Regridder':
        """
        Create a :class:`Regridder` from an xarray Dataset that lives on the
        target grid (e.g. a CONUS404 file).

        The dataset must contain ``y`` and ``x`` coordinates (or will be
        renamed automatically from lat/lon variants).

        Parameters
        ----------
        ds : xr.Dataset
            Any dataset on the target grid.  Only the ``y`` / ``x``
            coordinate arrays are used.
        method : str, optional
            Interpolation method (default ``'linear'``).
        """
        y, x = cls._extract_yx(ds)
        attrs = {}
        for coord in ('y', 'x'):
            if coord in ds.coords:
                attrs[coord] = dict(ds[coord].attrs)
        return cls(y, x, method=method, target_attrs=attrs)

    @classmethod
    def from_reference_grid(cls, path: str, method: str = 'linear') -> 'Regridder':
        """
        Load a previously saved reference grid (tiny NetCDF with just
        ``y`` and ``x`` coordinate arrays).

        Parameters
        ----------
        path : str or Path
            Path to the reference grid NetCDF produced by
            :meth:`save_reference_grid`.
        method : str, optional
            Interpolation method (default ``'linear'``).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reference grid not found: {path}\n"
                f"Run preprocessing first to generate it, or use "
                f"Regridder.from_target_dataset() with a target-grid file."
            )
        ds = xr.open_dataset(path)
        y = ds['y'].values
        x = ds['x'].values
        attrs = {}
        for coord in ('y', 'x'):
            if coord in ds.coords:
                attrs[coord] = dict(ds[coord].attrs)
        ds.close()
        return cls(y, x, method=method, target_attrs=attrs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_reference_grid(self, path: str) -> None:
        """
        Save the target grid coordinates to a small NetCDF file (~100 KB).

        This file is all that is needed at inference time -- the original
        target dataset (e.g. CONUS404) is no longer required.

        Parameters
        ----------
        path : str or Path
            Output path for the reference grid NetCDF.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ds = xr.Dataset({
            'y': ('y', self.target_y, self.target_attrs.get('y', {})),
            'x': ('x', self.target_x, self.target_attrs.get('x', {})),
        })
        ds.attrs['description'] = (
            'Target grid reference for Regridder.  Contains only the y/x '
            'coordinate arrays needed to interpolate coarse data onto the '
            'high-resolution target grid.'
        )
        ds.to_netcdf(path)
        size_kb = path.stat().st_size / 1024
        print(f"  Saved reference grid to {path} ({size_kb:.0f} KB)")

    # ------------------------------------------------------------------
    # Core regridding
    # ------------------------------------------------------------------

    def regrid(
        self,
        ds: xr.Dataset,
        var_map: Dict[str, str],
        physical_bounds: Optional[Dict[str, Dict]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> xr.Dataset:
        """
        Interpolate one or more variables from a coarse dataset onto the
        target grid.

        Parameters
        ----------
        ds : xr.Dataset
            Coarse-resolution source dataset (e.g. ERA5 or CMIP6).
            Must have a ``time`` dimension and spatial coordinates that
            can be standardised to ``y`` / ``x``.
        var_map : dict
            Mapping of ``{output_name: source_var_name}`` so the caller
            controls the variable names in the returned dataset.
            Example::

                {'era5_u': 'u10', 'era5_v': 'v10'}

            *output_name* is the key in the returned dataset (matching
            what the CNN expects).  *source_var_name* is the variable
            name inside ``ds``.  If *source_var_name* is ``None``, the
            first data variable in ``ds`` is used (convenience shortcut
            for single-variable files).
        physical_bounds : dict, optional
            Per-variable bounds ``{output_name: {'min': lo, 'max': hi}}``.
            Values outside the range are replaced with NaN.
        start_date, end_date : str, optional
            ISO date strings to restrict the time axis before
            interpolation (reduces memory for large files).

        Returns
        -------
        xr.Dataset
            Dataset with variables named by *output_name*, on the target
            grid ``(time, y, x)``.  Coordinate attributes from the
            target grid are attached to ``y`` and ``x``.
        """
        physical_bounds = physical_bounds or {}

        data_vars = {}
        for output_name, source_var in var_map.items():
            # Resolve source variable name
            if source_var is None:
                source_var = list(ds.data_vars)[0]
            if source_var not in ds.data_vars:
                available = list(ds.data_vars)
                raise KeyError(
                    f"Variable '{source_var}' not found in dataset.  "
                    f"Available: {available}"
                )

            da = ds[source_var]
            da = self._standardize_coords(da)

            # Time slicing (before interpolation to save memory)
            if start_date is not None or end_date is not None:
                da = da.sel(time=slice(start_date, end_date))

            # Physical bounds masking
            bounds = physical_bounds.get(output_name)
            if bounds:
                da = self._mask_physical_bounds(da, output_name, bounds)

            # Spatial interpolation to target grid
            src_ny, src_nx = da.sizes.get('y', '?'), da.sizes.get('x', '?')
            tgt_ny, tgt_nx = len(self.target_y), len(self.target_x)
            print(f"  Interpolating {output_name} ({src_ny}x{src_nx}) "
                  f"-> target grid ({tgt_ny}x{tgt_nx}) [{self.method}]...")

            da = da.interp(
                y=self.target_y,
                x=self.target_x,
                method=self.method,
            )

            # Rename to the output key
            da.name = output_name
            data_vars[output_name] = da

        # Build output dataset
        result = xr.Dataset(data_vars)

        # Attach target grid coordinate attributes
        for coord, attrs in self.target_attrs.items():
            if coord in result.coords:
                result[coord].attrs.update(attrs)

        print(f"\n  Regridded dataset:")
        print(f"    Variables : {list(result.data_vars)}")
        print(f"    Grid      : {result.sizes.get('y', '?')} x {result.sizes.get('x', '?')}")
        print(f"    Timesteps : {result.sizes.get('time', '?')}")

        return result

    # ------------------------------------------------------------------
    # Helpers (static-ish)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_yx(ds: xr.Dataset):
        """Extract y and x coordinate arrays, renaming if necessary."""
        # Try standard names
        y_names = ['y', 'lat', 'latitude', 'Y', 'LAT']
        x_names = ['x', 'lon', 'longitude', 'X', 'LON']

        y = x = None
        for name in y_names:
            if name in ds.coords:
                y = ds[name].values
                break
        for name in x_names:
            if name in ds.coords:
                x = ds[name].values
                break

        if y is None or x is None:
            raise ValueError(
                f"Cannot find y/x coordinates in dataset.  "
                f"Available coords: {list(ds.coords)}"
            )
        return y, x

    @staticmethod
    def _standardize_coords(da: xr.DataArray) -> xr.DataArray:
        """Rename common coordinate variants to (time, y, x)."""
        rename = {}
        for name in ['TIME', 't', 'Time']:
            if name in da.coords and 'time' not in da.coords:
                rename[name] = 'time'
                break
        for name in ['lat', 'latitude', 'LAT', 'Y']:
            if name in da.coords and 'y' not in da.coords:
                rename[name] = 'y'
                break
        for name in ['lon', 'longitude', 'LON', 'X']:
            if name in da.coords and 'x' not in da.coords:
                rename[name] = 'x'
                break
        if rename:
            da = da.rename(rename)
        return da

    @staticmethod
    def _mask_physical_bounds(
        da: xr.DataArray,
        var_name: str,
        bounds: Dict,
    ) -> xr.DataArray:
        """Replace values outside [min, max] with NaN."""
        lo = bounds.get('min', -1e10)
        hi = bounds.get('max', 1e10)
        bad = da.isnull() | (da < lo) | (da > hi)
        n_bad = int(bad.sum().compute()) if hasattr(bad.sum(), 'compute') else int(bad.sum())
        if n_bad > 0:
            total = int(da.size)
            pct = 100.0 * n_bad / total
            print(f"    [{var_name}] {n_bad:,}/{total:,} ({pct:.2f}%) "
                  f"outside [{lo}, {hi}] -> NaN")
            da = da.where((da >= lo) & (da <= hi))
        return da

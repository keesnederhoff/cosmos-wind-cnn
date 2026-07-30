"""
Validate meteorological model predictions against observations.

Compares ERA5, HRRR, and ERA5-CNN downscaled model output (10 m winds and
air temperature) against local Whales Tale moorings and NDBC buoy observations.

Supports nearest-neighbour and bilinear spatial interpolation of model grids
(all rectilinear in UTM Zone 10N) to observation locations.

Outputs
-------
- Scatter plots, timeseries plots, wind-rose comparisons per model/station
- Taylor diagram and multi-model bar chart comparing all models
- CSV of validation statistics (bias, RMSE, MAE, correlation, scatter index)
"""

# Moduels => run with hydromt-sfincs-dev
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000   # faster long-line Agg rendering
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from pyproj import Transformer, CRS
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print("Warning: pyproj not found. Using approximate UTM conversion.")

try:
    from windrose import WindroseAxes
    HAS_WINDROSE = True
except ImportError:
    HAS_WINDROSE = False

import config
from config import *          # single source of truth: paths, MODELS, run options, colors, constants
_FALLBACK_COLORS = FALLBACK_COLORS       # engine uses the underscore-prefixed name internally
OUTPUT_DIR = config.OUTPUT_ROOT / "adhoc"  # the driver (run_validation.py) overrides V.OUTPUT_DIR per era


def audit_model_paths(models_to_run):
    """Print a reachability table for each product before a run. A missing
    file/dir is reported here; the model is skipped cleanly later."""
    print("\n=== Model data path audit ===")
    print(f"  {'model':<26} {'reachable':>9}   detail")
    for name in models_to_run:
        cfg = MODELS.get(name, {})
        targets = []
        for k in ('u_file', 'v_file', 'temp_file'):
            if cfg.get(k) is not None:
                targets.append(Path(cfg[k]))
        if 'data_dir' in cfg:
            targets.append(Path(cfg['data_dir']))
        ok = all(t.exists() for t in targets) if targets else False
        missing = [str(t) for t in targets if not t.exists()]
        detail = 'OK' if ok else ('NO TARGETS' if not targets else f"MISSING: {missing[0]}")
        print(f"  {name:<26} {str(ok):>9}   {detail}")
    print("=" * 60)


# ===========================================================================
# Configuration
# ===========================================================================
# All config DATA (paths, MODELS registry, run options, plot control, colors,
# constants) lives in config.py and is pulled in via `from config import *`
# above. Only the two helper FUNCTIONS below and the derived STATIONS table
# remain here.


def build_pws_stations():
    """Enumerate stations (those with any wind data) from the 3 archive NetCDFs."""
    out = {}
    for group, fp, height in PWS_SOURCES:
        if group not in config.STATION_GROUPS:
            continue
        if not fp.exists():
            print(f"  WARNING: pws archive not found: {fp}")
            continue
        # Guard against a transient 0-byte / mid-write file (the obs scraper
        # writes *.NEW.nc then renames) so a scrape in progress can't crash import.
        try:
            if fp.stat().st_size == 0:
                raise OSError("0-byte file (scrape in progress?)")
            ds = xr.open_dataset(str(fp))
        except Exception as e:
            print(f"  WARNING: skipping unreadable pws archive {fp.name}: {e}")
            continue
        ids = [str(s) for s in ds['station_id'].values]
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        has_w = ds['wind_speed'].notnull().sum('time').values
        has_t = (ds['temperature'].notnull().sum('time').values
                 if 'temperature' in ds else np.zeros(len(ids)))
        for i, fid in enumerate(ids):
            if has_w[i] <= 0 and has_t[i] <= 0:   # keep wind- OR temp-bearing stations
                continue
            key = fid.upper() if group == 'NDBC' else fid
            if key in out or key in STATIONS:
                key = f"{group}_{fid}"
            out[key] = {
                'source': 'pws', 'group': group,
                'file_path': fp, 'file_station_id': fid,
                'lat': float(lats[i]), 'lon': float(lons[i]),
                'anemometer_height_m': height,
            }
        ds.close()
    n = {g: sum(v['group'] == g for v in out.values()) for g in ('IEM', 'NDBC', 'CWOP')}
    print(f"  build_pws_stations: {len(out)} stations ({n})")
    return out


# --- Station table ---------------------------------------------------------
# USGS moorings (optional, off by default) seed the table; the pws archive
# stations are appended. build_pws_stations reads the STATIONS global for
# collision checks, so it must exist before the .update() call.
STATIONS = dict(config.USGS_MOORINGS) if config.INCLUDE_USGS_MOORINGS else {}
STATIONS.update(build_pws_stations())
STATIONS_TO_RUN = list(STATIONS.keys())


def model_color_map(models):
    """Return {model: color} with fixed colors for known models and unique,
    stable fallbacks for the rest. Pass the models present in a figure."""
    cmap = {}
    unknown = sorted(m for m in set(models) if m not in MODEL_COLORS)
    for i, m in enumerate(unknown):
        cmap[m] = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
    for m in models:
        if m in MODEL_COLORS:
            cmap[m] = MODEL_COLORS[m]
    return cmap

# HRRR  start 2014-10-10; CONUS404 end 2022-10-01. Physical constants
# (DEFAULT_ROUGHNESS, MODEL_HEIGHT, FIGURE_DPI, PEAK_WINDOW_DAYS) now live in
# config.py and arrive via `from config import *`.

# ===========================================================================
# Coordinate conversion
# ===========================================================================

def latlon_to_utm10(lat, lon):
    """Convert lat/lon (WGS84) to UTM Zone 10N (EPSG:32610)."""
    if HAS_PYPROJ:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return float(x), float(y)
    else:
        lat_c, lon_c = 37.8, -122.3
        x = (lon - lon_c) * 111320 * np.cos(np.radians(lat_c)) + 5.8e5
        y = (lat - lat_c) * 111320 + 4.19e6
        return float(x), float(y)


def lcc_to_utm10(x_lcc, y_lcc, lcc_attrs):
    """Convert Lambert Conformal Conic coordinates to UTM 10N.

    Parameters
    ----------
    x_lcc, y_lcc : float or array, LCC coordinates (m)
    lcc_attrs : dict, Lambert_Conformal variable attributes

    Returns
    -------
    x_utm, y_utm : float or array, UTM 10N coordinates (m)
    """
    if not HAS_PYPROJ:
        raise RuntimeError("pyproj is required for LCC coordinate conversion")

    std_par = lcc_attrs['standard_parallel']
    if hasattr(std_par, '__len__'):
        std_par1, std_par2 = float(std_par[0]), float(std_par[1])
    else:
        std_par1 = std_par2 = float(std_par)

    lcc_crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={std_par1} +lat_2={std_par2} "
        f"+lat_0={float(lcc_attrs['latitude_of_projection_origin'])} "
        f"+lon_0={float(lcc_attrs['longitude_of_central_meridian'])} "
        f"+R={float(lcc_attrs['earth_radius'])} +units=m +no_defs"
    )
    transformer = Transformer.from_crs(lcc_crs, "EPSG:32610", always_xy=True)
    x_utm, y_utm = transformer.transform(x_lcc, y_lcc)
    return x_utm, y_utm


def latlon_to_utm10_array(lat, lon):
    """Convert lat/lon arrays to UTM Zone 10N arrays.

    Parameters
    ----------
    lat, lon : array-like (any shape)

    Returns
    -------
    x_utm, y_utm : arrays of same shape, UTM 10N (m)
    """
    if HAS_PYPROJ:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
        x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
        return x, y
    else:
        lat_c, lon_c = 37.8, -122.3
        x = (np.asarray(lon) - lon_c) * 111320 * np.cos(np.radians(lat_c)) + 5.8e5
        y = (np.asarray(lat) - lat_c) * 111320 + 4.19e6
        return x, y


def utm10_to_lcc(x_utm, y_utm, lcc_attrs):
    """Convert UTM 10N coordinates to Lambert Conformal Conic.

    Parameters
    ----------
    x_utm, y_utm : float, UTM 10N easting/northing (m)
    lcc_attrs : dict, Lambert_Conformal variable attributes from the NetCDF
                (must contain standard_parallel, latitude_of_projection_origin,
                 longitude_of_central_meridian, earth_radius)

    Returns
    -------
    x_lcc, y_lcc : float, coordinates in the LCC projection (m)
    """
    if not HAS_PYPROJ:
        raise RuntimeError("pyproj is required for LCC coordinate conversion")

    std_par = lcc_attrs['standard_parallel']
    if hasattr(std_par, '__len__'):
        std_par1, std_par2 = float(std_par[0]), float(std_par[1])
    else:
        std_par1 = std_par2 = float(std_par)

    lcc_crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={std_par1} +lat_2={std_par2} "
        f"+lat_0={float(lcc_attrs['latitude_of_projection_origin'])} "
        f"+lon_0={float(lcc_attrs['longitude_of_central_meridian'])} "
        f"+R={float(lcc_attrs['earth_radius'])} +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:32610", lcc_crs, always_xy=True)
    x_lcc, y_lcc = transformer.transform(x_utm, y_utm)
    return float(x_lcc), float(y_lcc)


def get_station_coordinates(station_id, cfg, ds=None):
    """
    Get station coordinates with fallback chain:
      1. Read from NetCDF attributes/variables (if READ_COORDS_FROM_NETCDF and ds provided)
      2. Use KNOWN_STATION_COORDINATES lookup
      3. Fall back to cfg['lat'], cfg['lon']
    
    Parameters
    ----------
    station_id : str
        Station identifier (e.g., 'AAMC1', '46026')
    cfg : dict
        Station configuration dict with 'lat' and 'lon' keys
    ds : xarray.Dataset, optional
        Open NetCDF dataset to read coordinates from
    
    Returns
    -------
    tuple
        (lat, lon) coordinates
    """
    lat, lon = None, None
    source = None
    
    # 1. Try reading from NetCDF if enabled and dataset provided
    if READ_COORDS_FROM_NETCDF and ds is not None:
        # Check global attributes first
        for lat_attr in ['latitude', 'lat', 'station_latitude', 'station_lat']:
            if lat_attr in ds.attrs:
                lat = float(ds.attrs[lat_attr])
                break
        for lon_attr in ['longitude', 'lon', 'station_longitude', 'station_lon']:
            if lon_attr in ds.attrs:
                lon = float(ds.attrs[lon_attr])
                break
        
        # Check coordinate variables
        if lat is None or lon is None:
            for lat_var in ['latitude', 'lat', 'station_latitude']:
                if lat_var in ds.coords or lat_var in ds.data_vars:
                    val = ds[lat_var].values
                    lat = float(val.item() if val.ndim == 0 else val[0])
                    break
            for lon_var in ['longitude', 'lon', 'station_longitude']:
                if lon_var in ds.coords or lon_var in ds.data_vars:
                    val = ds[lon_var].values
                    lon = float(val.item() if val.ndim == 0 else val[0])
                    break
        
        if lat is not None and lon is not None:
            source = 'netcdf'
    
    # 2. Use KNOWN_STATION_COORDINATES lookup
    if (lat is None or lon is None) and station_id in KNOWN_STATION_COORDINATES:
        lat, lon = KNOWN_STATION_COORDINATES[station_id]
        source = 'known_coords'
    
    # 3. Fall back to config values
    if lat is None or lon is None:
        lat = cfg.get('lat')
        lon = cfg.get('lon')
        source = 'config'
    
    if lat is None or lon is None:
        raise ValueError(f"No coordinates found for station {station_id}")
    
    return lat, lon, source


# ===========================================================================
# Wind height conversion
# ===========================================================================

def convert_wind_height(u, z_from, z_to, z0=DEFAULT_ROUGHNESS):
    """Convert wind speed/component from height z_from to z_to via log profile."""
    u = np.asarray(u, dtype=float)
    if z0 <= 0 or z_from <= z0 or z_to <= z0:
        return u
    factor = np.log(z_to / z0) / np.log(z_from / z0)
    return u * factor


# ===========================================================================
# Observation loading
# ===========================================================================

# Default variable names match the Whales Tale / EMC schema; ERO20 and future
# USGS files override these via cfg['var_map'].
_USGS_MET_DEFAULT_VARS = {
    'time':        'time',
    'wind_speed':  'wind_speed',
    'wind_dir':    'wind_dir_from',        # ERO20: 'wind_dir'
    'air_temp':    'air_temp',             # already degrees C
    'pressure':    'rel_barometric_press', # ERO20: 'air_pressure'
    'rh':          None,                   # ERO20: 'rel_humidity'; WT/EMC have none
    'pressure_units': 'auto',              # 'auto' = detect Pa->hPa; 'hPa' = as-is
}


def load_usgs_met(station_id, cfg):
    """Load a USGS met-station observation file (Whales Tale / EMC / ERO20).

    Variable names default to the Whales Tale schema (`_USGS_MET_DEFAULT_VARS`);
    override per station via cfg['var_map'] (e.g. ERO20 supplies wind_dir /
    air_pressure / rel_humidity and a 'pressure_units' hint). Winds are left at
    the measured height (no log-profile correction), consistent with the USGS
    convention. Coordinates come from the file if present, else cfg lat/lon.
    """
    fp = cfg['file_path']
    if not fp.exists():
        print(f"  WARNING: file not found: {fp}")
        return None

    vm = dict(_USGS_MET_DEFAULT_VARS)
    vm.update(cfg.get('var_map', {}))

    print(f"  Loading USGS met: {fp.name}")
    ds = xr.open_dataset(str(fp))

    obs_time = pd.to_datetime(ds[vm['time']].values)
    n = len(obs_time)
    nan = np.full(n, np.nan)

    def _get(key):
        name = vm.get(key)
        if name and name in ds:
            return ds[name].values.flatten().astype(float)
        return nan.copy()

    speed_raw = _get('wind_speed')
    dir_raw   = _get('wind_dir')
    temp_raw  = _get('air_temp')          # already degrees C
    rh_pct    = _get('rh')                # % (ERO20 only; else NaN)

    # Pressure: 'auto' detects Pa (median > 2000) and converts to hPa; an
    # explicit 'hPa'/'mbar'/'millibar' hint means the values are already hPa
    # (ERO20 is millibar) and are used as-is (also skips the inHg mangling that
    # NaN's the Whales Tale inches-Hg pressure downstream).
    pressure_hpa = _get('pressure')
    punits = str(vm.get('pressure_units', 'auto')).lower()
    if punits == 'auto':
        if np.isfinite(pressure_hpa).any() and np.nanmedian(pressure_hpa) > 2000:
            pressure_hpa = pressure_hpa / 100.0     # Pa -> hPa

    # Derive u/v from speed + direction (met convention: "from").
    dir_rad = np.radians(dir_raw)
    u_raw = -speed_raw * np.sin(dir_rad)
    v_raw = -speed_raw * np.cos(dir_rad)

    # USGS winds left at measured height — NO log-profile correction to 10 m.
    u10, v10, speed10 = u_raw, v_raw, speed_raw

    lat, lon, coord_source = get_station_coordinates(station_id, cfg, ds)
    x_utm, y_utm = latlon_to_utm10(lat, lon)

    ds.close()
    print(f"    Time: {obs_time.min()} - {obs_time.max()}, N={n}")
    print(f"    Coords: ({lat:.4f}, {lon:.4f}) from {coord_source}")
    print(f"    UTM: ({x_utm:.0f}, {y_utm:.0f})")

    return {
        'time': obs_time, 'u10': u10, 'v10': v10,
        'speed10': speed10, 'dir_deg': dir_raw,
        'air_temp_C': temp_raw,
        'pressure_hPa': pressure_hpa, 'dewpoint_C': nan.copy(),
        'rh_pct': rh_pct, 'solar_wm2': nan.copy(), 'precip_mmhr': nan.copy(),
        'x_utm': x_utm, 'y_utm': y_utm,
        'lat': lat, 'lon': lon,
        'station_id': station_id,
    }


# Backward-compatible alias — existing 'whales_tale' entries keep working unchanged.
load_whales_tale = load_usgs_met


def load_ndbc(station_id, cfg):
    """Load NDBC buoy observation data."""
    fp = cfg['file_path']
    if not fp.exists():
        print(f"  WARNING: file not found: {fp}")
        return None

    print(f"  Loading NDBC: {fp.name}")
    ds = xr.open_dataset(str(fp))

    obs_time = pd.to_datetime(ds['datetime'].values)
    u10 = ds['u10_ms'].values.flatten().astype(float)
    v10 = ds['v10_ms'].values.flatten().astype(float)
    speed10 = ds['wind_speed_ms'].values.flatten().astype(float)
    dir_deg = ds['wind_direction_deg'].values.flatten().astype(float)
    temp_k = ds['air_temperature_k'].values.flatten().astype(float)
    temp_c = temp_k - 273.15

    # Get coordinates (try NetCDF first, then known coords, then config)
    lat, lon, coord_source = get_station_coordinates(station_id, cfg, ds)
    x_utm, y_utm = latlon_to_utm10(lat, lon)

    ds.close()
    print(f"    Time: {obs_time.min()} – {obs_time.max()}, N={len(obs_time)}")
    print(f"    Coords: ({lat:.4f}, {lon:.4f}) from {coord_source}")
    print(f"    UTM: ({x_utm:.0f}, {y_utm:.0f})")

    return {
        'time': obs_time, 'u10': u10, 'v10': v10,
        'speed10': speed10, 'dir_deg': dir_deg,
        'air_temp_C': temp_c,
        'x_utm': x_utm, 'y_utm': y_utm,
        'lat': lat, 'lon': lon,
        'station_id': station_id,
    }


_PWS_DS_CACHE = {}


def _open_pws(file_path):
    """Open (and cache) a pws_scraper archive NetCDF so each file opens once."""
    key = str(file_path)
    if key not in _PWS_DS_CACHE:
        _PWS_DS_CACHE[key] = xr.open_dataset(key)
    return _PWS_DS_CACHE[key]


def load_pws_archive(station_id, cfg):
    """Load one station out of a pws_scraper archive NetCDF (IEM / NDBC / CWOP).

    Files are (station, time) with wind_speed, wind_direction (met 'from'), and
    temperature (IEM/NDBC only). u/v derived from speed + direction, then
    height-corrected from cfg['anemometer_height_m'] to 10 m (no-op at 10 m).
    """
    fp = cfg['file_path']
    if not fp.exists():
        print(f"  WARNING: file not found: {fp}")
        return None
    ds = _open_pws(fp)
    fid = cfg.get('file_station_id', station_id)
    ids = [str(s) for s in ds['station_id'].values]
    if fid not in ids:
        print(f"  WARNING: station '{fid}' not in {fp.name}")
        return None
    i = ids.index(fid)
    sub = ds.isel(station=i)

    obs_time = pd.to_datetime(ds['time'].values)
    speed_raw = sub['wind_speed'].values.astype(float)
    dir_raw = sub['wind_direction'].values.astype(float)
    if 'temperature' in ds:
        temp_raw = sub['temperature'].values.astype(float)   # already °C
    else:
        temp_raw = np.full(len(obs_time), np.nan)

    dir_rad = np.radians(dir_raw)
    u_raw = -speed_raw * np.sin(dir_rad)
    v_raw = -speed_raw * np.cos(dir_rad)

    z_from = cfg['anemometer_height_m']
    u10 = convert_wind_height(u_raw, z_from, MODEL_HEIGHT)
    v10 = convert_wind_height(v_raw, z_from, MODEL_HEIGHT)
    speed10 = convert_wind_height(speed_raw, z_from, MODEL_HEIGHT)

    # --- Scalar obs (IEM/NDBC have these; CWOP wind-only -> NaN) ---
    nan = np.full(len(obs_time), np.nan)

    def _ovar(name):
        return sub[name].values.astype(float) if name in ds else nan.copy()

    pressure_hpa = _ovar('pressure')
    if np.isfinite(pressure_hpa).any() and np.nanmedian(pressure_hpa) > 2000:
        pressure_hpa = pressure_hpa / 100.0          # Pa -> hPa (auto-detect)
    dewpoint_c = _ovar('dew_point')                  # already °C
    rh_pct = _ovar('relative_humidity')              # %
    solar_wm2 = _ovar('solar_radiation')             # W/m2
    precip_acc = _ovar('precipitation')              # mm accumulated per interval
    if np.isfinite(precip_acc).any():
        dt_h = np.median(np.diff(obs_time.values).astype('timedelta64[s]').astype(float)) / 3600.0
        precip_mmhr = precip_acc / dt_h if dt_h and dt_h > 0 else precip_acc
    else:
        precip_mmhr = precip_acc

    lat = float(sub['latitude'].values)
    lon = float(sub['longitude'].values)
    x_utm, y_utm = latlon_to_utm10(lat, lon)

    n_valid = int(np.isfinite(speed_raw).sum())
    print(f"  Loading pws [{cfg['group']}]: {fp.name} -> {fid}  "
          f"(N={n_valid} valid, {lat:.4f},{lon:.4f})")

    return {
        'time': obs_time, 'u10': u10, 'v10': v10,
        'speed10': speed10, 'dir_deg': dir_raw,
        'air_temp_C': temp_raw,
        'pressure_hPa': pressure_hpa, 'dewpoint_C': dewpoint_c,
        'rh_pct': rh_pct, 'solar_wm2': solar_wm2, 'precip_mmhr': precip_mmhr,
        'x_utm': x_utm, 'y_utm': y_utm,
        'lat': lat, 'lon': lon,
        'station_id': station_id,
    }


def load_station(station_id, cfg):
    """Dispatch to the appropriate loader based on source type."""
    if cfg['source'] in ('whales_tale', 'usgs_met'):
        data = load_usgs_met(station_id, cfg)
    elif cfg['source'] == 'ndbc':
        data = load_ndbc(station_id, cfg)
    elif cfg['source'] == 'pws':
        data = load_pws_archive(station_id, cfg)
    else:
        print(f"  ERROR: unknown source '{cfg['source']}' for {station_id}")
        return None
    if data is None:
        return None
    # QC Tier 1: per-reading physical bounds on OBS (spurious spikes -> NaN,
    # e.g. the IEM archive's 50-75 m/s values). Legit 0 m/s calms are preserved.
    clip_keys = [('speed10', 'Wind Speed [m/s]'), ('u10', 'Wind U10 [m/s]'),
                 ('v10', 'Wind V10 [m/s]'), ('dir_deg', 'Wind Direction [deg]')]
    clip_keys += [(spec['obs_key'], spec['label']) for spec in SCALAR_VARS.values()]
    for key, label in clip_keys:
        if key in data and data[key] is not None:
            data[key] = _clip_to_physical_bounds(np.asarray(data[key], dtype=float), label)
    return data


# ===========================================================================
# Model loading and spatial interpolation
# ===========================================================================

def _decode_time(ds):
    """Decode time coordinate, handling 'seconds since' units."""
    time_raw = ds['time'].values
    time_units = ds['time'].attrs.get('units', '')
    if 'seconds since' in time_units:
        origin_str = time_units.replace('seconds since', '').strip()
        origin = pd.Timestamp(origin_str)
        return pd.to_datetime([origin + pd.Timedelta(seconds=float(t)) for t in time_raw])
    return pd.to_datetime(time_raw)


def _get_1d_coords(ds):
    """Extract 1D x and y coordinate arrays from a dataset.

    Handles both truly 1D arrays and 2D arrays where rows/cols are identical
    (HRRR stores x,y as 2D but the grid is rectilinear).
    """
    x = ds['x'].values
    y = ds['y'].values
    if x.ndim == 2:
        x = x[0, :]
    if y.ndim == 2:
        y = y[:, 0]
    return x.astype(float), y.astype(float)


def load_model(model_name, model_cfg):
    """Open model datasets lazily and extract grid info.

    Returns dict with keys: ds_u, ds_v, ds_temp, x1d, y1d, time, datasets.
    For single-file models (ERA5-CNN), ds_u/ds_v/ds_temp point to the same object.
    """
    print(f"\n  Loading model: {model_name}")

    # Check files exist (temp_file is optional)
    files_to_check = {model_cfg['u_file'], model_cfg['v_file']}
    if 'temp_file' in model_cfg and model_cfg['temp_file'] is not None:
        files_to_check.add(model_cfg['temp_file'])
    for fp in files_to_check:
        if not fp.exists():
            print(f"    WARNING: file not found: {fp}")
            return None

    if model_cfg['single_file']:
        ds = xr.open_dataset(str(model_cfg['u_file']), chunks={'time': 500})
        ds_u = ds_v = ds_temp = ds
        datasets = [ds]
    else:
        ds_u = xr.open_dataset(str(model_cfg['u_file']), chunks={'time': 500})
        ds_v = xr.open_dataset(str(model_cfg['v_file']), chunks={'time': 500})
        if 'temp_file' in model_cfg and model_cfg['temp_file'] is not None:
            ds_temp = xr.open_dataset(str(model_cfg['temp_file']), chunks={'time': 500})
            datasets = [ds_u, ds_v, ds_temp]
        else:
            ds_temp = None
            datasets = [ds_u, ds_v]

    x1d, y1d = _get_1d_coords(ds_u)
    time_pd = _decode_time(ds_u)

    # Perf: when a global TIME_RANGE is set, slice the model to the windowed
    # records BEFORE the stencil read. _read_stencils calls .values, which
    # otherwise materializes the model's ENTIRE time axis = a full-file read
    # (e.g. ~84 GB for CNN-RTMA) no matter how small the window. Slicing here
    # makes the read cost scale with the window, not the file size. Results are
    # unchanged: the obs are filtered to the same TIME_RANGE, so only the
    # overlapping timestamps ever match; a 1-day pad keeps boundary matching
    # clean. 'datasets' keeps the original open handles for cleanup; the
    # ds_u/ds_v/ds_temp returned below are sliced (lazy) views over them.
    _n_full = len(time_pd)
    if TIME_RANGE is not None and _n_full > 0:
        _t0 = pd.Timestamp(TIME_RANGE[0]) - pd.Timedelta(days=1)
        _t1 = pd.Timestamp(TIME_RANGE[1]) + pd.Timedelta(days=1)
        _idx = np.where((time_pd >= _t0) & (time_pd <= _t1))[0]
        if 0 < len(_idx) < _n_full:
            _tdim = ds_u[model_cfg['u_var']].dims[0]   # 'time'
            _sl = {_tdim: slice(int(_idx[0]), int(_idx[-1]) + 1)}  # contiguous (time monotonic)
            ds_u = ds_u.isel(_sl)
            if model_cfg['single_file']:
                ds_v = ds_temp = ds_u
            else:
                ds_v = ds_v.isel(_sl)
                if ds_temp is not None:
                    ds_temp = ds_temp.isel(_sl)
            time_pd = time_pd[int(_idx[0]):int(_idx[-1]) + 1]
            print(f"    Time-window read: {len(time_pd)} of {_n_full} records "
                  f"[{time_pd[0]} .. {time_pd[-1]}]")

    # Store CRS info for non-UTM grids
    crs_type = model_cfg.get('crs', 'utm10n')
    lcc_attrs = None
    if crs_type == 'lcc' and 'Lambert_Conformal' in ds_u:
        lcc_attrs = dict(ds_u['Lambert_Conformal'].attrs)
        print(f"    CRS: Lambert Conformal Conic")

    print(f"    Grid: {len(x1d)} x {len(y1d)}")
    print(f"    Time: {time_pd[0]} – {time_pd[-1]}, N={len(time_pd)}")

    ds_scalars = _load_scalar_datasets(model_cfg, sort_y=False)
    return {
        'ds_u': ds_u, 'ds_v': ds_v, 'ds_temp': ds_temp, 'ds_scalars': ds_scalars,
        'x1d': x1d, 'y1d': y1d, 'time': time_pd,
        'datasets': datasets + [v['ds'] for v in ds_scalars.values() if 'ds' in v],
        'crs': crs_type, 'lcc_attrs': lcc_attrs,
    }


def load_model_utm_multifile(model_name, model_cfg):
    """Load a multi-file UTM model via glob patterns (e.g. CONUS404-downscaled).

    Concatenates all matching u/v files along the time dimension.
    Temperature is optional — if no temp_pattern is given, air_temp_C will be NaN.
    """
    print(f"\n  Loading model: {model_name}")

    data_dir = model_cfg['data_dir']
    if not data_dir.exists():
        print(f"    WARNING: directory not found: {data_dir}")
        return None

    u_files = sorted(data_dir.glob(model_cfg['u_pattern']))
    v_files = sorted(data_dir.glob(model_cfg['v_pattern']))

    if not u_files:
        print(f"    WARNING: no u files found matching {model_cfg['u_pattern']}")
        return None
    if not v_files:
        print(f"    WARNING: no v files found matching {model_cfg['v_pattern']}")
        return None

    print(f"    u files: {len(u_files)}")
    print(f"    v files: {len(v_files)}")

    mf_kwargs = dict(concat_dim='time', combine='nested',
                     chunks={'time': 200}, data_vars='minimal',
                     coords='minimal', compat='override')

    def _ascending_y(d):
        # interpolate_to_point() uses searchsorted -> needs ascending y.
        # The RTMA-GEE grid is north-up (descending y); flip it. No-op otherwise.
        if 'y' in d.coords and d['y'].ndim == 1 and float(d['y'][0]) > float(d['y'][-1]):
            d = d.sortby('y')
        return d

    ds_u = _ascending_y(xr.open_mfdataset([str(f) for f in u_files], **mf_kwargs))
    ds_v = _ascending_y(xr.open_mfdataset([str(f) for f in v_files], **mf_kwargs))

    ds_temp = None
    datasets = [ds_u, ds_v]
    if 'temp_pattern' in model_cfg and model_cfg['temp_pattern']:
        temp_files = sorted(data_dir.glob(model_cfg['temp_pattern']))
        if temp_files:
            print(f"    temp files: {len(temp_files)}")
            ds_temp = _ascending_y(xr.open_mfdataset([str(f) for f in temp_files], **mf_kwargs))
            datasets.append(ds_temp)
        else:
            print(f"    No temp files found — temperature will be NaN")

    x1d, y1d = _get_1d_coords(ds_u)
    time_pd = _decode_time(ds_u)

    print(f"    CRS: UTM 10N (EPSG:32610)")
    print(f"    Grid: {len(x1d)} x {len(y1d)}")
    print(f"    Time: {time_pd[0]} – {time_pd[-1]}, N={len(time_pd)}")

    ds_scalars = _load_scalar_datasets(model_cfg, sort_y=True)   # GEE grids are north-up

    # Window the (lazy) multifile datasets to TIME_RANGE before any .values read.
    # Without this, RTMA / CONUS404-downscaled read their ENTIRE record (RTMA
    # 2011-2026, ~10 GB/var with a tiny-chunk + sortby-reindex read pathology) even
    # for a 1-week validation window — the same optimisation load_model() already
    # applies to single/standard multifile products. Obs are filtered to the same
    # TIME_RANGE so only overlapping timestamps ever match; a 1-day pad keeps
    # boundary matching clean; results are unchanged.
    _n_full = len(time_pd)
    if TIME_RANGE is not None and _n_full > 0:
        _t0 = pd.Timestamp(TIME_RANGE[0]) - pd.Timedelta(days=1)
        _t1 = pd.Timestamp(TIME_RANGE[1]) + pd.Timedelta(days=1)
        _idx = np.where((time_pd >= _t0) & (time_pd <= _t1))[0]
        if 0 < len(_idx) < _n_full:
            _sl = {'time': slice(int(_idx[0]), int(_idx[-1]) + 1)}  # contiguous (time monotonic)
            ds_u = ds_u.isel(_sl)
            ds_v = ds_v.isel(_sl)
            if ds_temp is not None:
                ds_temp = ds_temp.isel(_sl)
            for _sc in ds_scalars.values():
                if isinstance(_sc, dict) and _sc.get('ds') is not None:
                    try:
                        _sc['ds'] = _sc['ds'].isel(_sl)
                    except Exception:
                        pass
            time_pd = time_pd[int(_idx[0]):int(_idx[-1]) + 1]
            print(f"    Time-window read: {len(time_pd)} of {_n_full} records "
                  f"[{time_pd[0]} .. {time_pd[-1]}]")

    return {
        'ds_u': ds_u, 'ds_v': ds_v, 'ds_temp': ds_temp, 'ds_scalars': ds_scalars,
        'x1d': x1d, 'y1d': y1d, 'time': time_pd,
        'datasets': datasets + [v['ds'] for v in ds_scalars.values() if 'ds' in v],
        'crs': 'utm10n', 'lcc_attrs': None,
    }


def load_model_wrf_calnev(model_name, model_cfg):
    """Load WRF-CalNev multi-file model with curvilinear 2D lat/lon grid.

    Returns dict compatible with the standard model_data format, plus
    extra keys for 2D lat/lon and the speed-only flag.
    """
    print(f"\n  Loading model: {model_name}")

    data_dir = model_cfg['data_dir']
    if not data_dir.exists():
        print(f"    WARNING: directory not found: {data_dir}")
        return None

    speed_files = sorted(data_dir.glob(model_cfg['speed_pattern']))
    temp_files = sorted(data_dir.glob(model_cfg['temp_pattern']))

    if not speed_files:
        print(f"    WARNING: no wind speed files found matching {model_cfg['speed_pattern']}")
        return None
    if not temp_files:
        print(f"    WARNING: no temperature files found matching {model_cfg['temp_pattern']}")
        return None

    print(f"    Wind speed files: {len(speed_files)}")
    print(f"    Temperature files: {len(temp_files)}")

    ds_speed = xr.open_mfdataset(
        [str(f) for f in speed_files],
        concat_dim='time', combine='nested',
        chunks={'time': 500}, data_vars='minimal',
        coords='minimal', compat='override')
    ds_temp = xr.open_mfdataset(
        [str(f) for f in temp_files],
        concat_dim='time', combine='nested',
        chunks={'time': 500}, data_vars='minimal',
        coords='minimal', compat='override')

    time_pd = _decode_time(ds_speed)

    # Extract 2D lat/lon
    lat2d = ds_speed['latitude'].values.astype(float)
    lon2d = ds_speed['longitude'].values.astype(float)

    print(f"    CRS: curvilinear 2D lat/lon (Lambert Conformal, 1.5 km)")
    print(f"    Grid: {lat2d.shape[1]} x {lat2d.shape[0]}")
    print(f"    Lat range: {lat2d.min():.3f} – {lat2d.max():.3f}")
    print(f"    Lon range: {lon2d.min():.3f} – {lon2d.max():.3f}")
    print(f"    Time: {time_pd[0]} – {time_pd[-1]}, N={len(time_pd)}")

    ds_scalars = _load_scalar_datasets(model_cfg, sort_y=False)
    return {
        'ds_speed': ds_speed, 'ds_temp': ds_temp, 'ds_scalars': ds_scalars,
        'ds_u': None, 'ds_v': None,
        'lat2d': lat2d, 'lon2d': lon2d,
        'x1d': None, 'y1d': None,
        'time': time_pd,
        'datasets': [ds_speed, ds_temp] + [v['ds'] for v in ds_scalars.values() if 'ds' in v],
        'crs': 'latlon_2d', 'lcc_attrs': None,
        'has_uv': False,
    }


def load_model_aorc(model_name, model_cfg):
    """Load AORC v1.1: NetCDF(s) with u/v + scalars on a 1-D regular lat/lon grid.

    Two modes:
      * single file  -> cfg['u_file']
      * multi-year   -> cfg['data_dir'] + optional cfg['year_range']=(lo,hi), globbed
                        as AORC_SFbay_800m_<year>.nc and concatenated along time.

    AORC stores latitude/longitude as 1-D coordinate arrays; the latlon_2d extract
    path expects 2-D grids, so we meshgrid them here. All variables (wind + scalars)
    live in the same dataset on one grid, so a single stencil set serves both and
    ds_scalars just point back at the shared ds. The open Pacific is masked (NaN) ->
    offshore stations on masked cells return NaN and drop out cleanly.
    crs='latlon_2d', has_uv=True.
    """
    print(f"\n  Loading model: {model_name}")

    data_dir = model_cfg.get('data_dir')
    if data_dir is not None:
        files = sorted(Path(data_dir).glob('AORC_SFbay_800m_*.nc'))
        yr = model_cfg.get('year_range')
        if yr is not None:
            lo, hi = yr
            files = [f for f in files if lo <= int(f.stem.split('_')[-1]) <= hi]
        if not files:
            print(f"    WARNING: no AORC files in {data_dir} for {model_cfg.get('year_range')}")
            return None
        print(f"    {len(files)} files [{files[0].stem.split('_')[-1]}"
              f"..{files[-1].stem.split('_')[-1]}]")
        ds = xr.open_mfdataset([str(f) for f in files], concat_dim='time',
                               combine='nested', chunks={'time': 500},
                               data_vars='minimal', coords='minimal', compat='override')
    else:
        fp = Path(model_cfg['u_file'])
        if not fp.exists():
            print(f"    WARNING: file not found: {fp}")
            return None
        ds = xr.open_dataset(str(fp), chunks={'time': 500})

    time_pd = _decode_time(ds)
    lat1d = ds['latitude'].values.astype(float)
    lon1d = ds['longitude'].values.astype(float)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)     # (nlat, nlon) == (ny, nx)

    # Window to TIME_RANGE (with a 1-day pad) so stencil reads scale with the
    # requested window, not the full multi-year file set (mirrors load_model).
    _n_full = len(time_pd)
    if TIME_RANGE is not None and _n_full > 0:
        _t0 = pd.Timestamp(TIME_RANGE[0]) - pd.Timedelta(days=1)
        _t1 = pd.Timestamp(TIME_RANGE[1]) + pd.Timedelta(days=1)
        _idx = np.where((time_pd >= _t0) & (time_pd <= _t1))[0]
        if 0 < len(_idx) < _n_full:
            _tdim = ds[model_cfg['u_var']].dims[0]
            ds = ds.isel({_tdim: slice(int(_idx[0]), int(_idx[-1]) + 1)})
            time_pd = time_pd[int(_idx[0]):int(_idx[-1]) + 1]

    print(f"    CRS: 1-D regular lat/lon -> meshgridded to 2-D "
          f"({lat2d.shape[0]} x {lat2d.shape[1]})")
    print(f"    Time: {time_pd[0]} – {time_pd[-1]}, N={len(time_pd)}")

    # All AORC variables share this one (windowed) dataset -> point scalars at it
    # directly instead of re-opening files per scalar via _load_scalar_datasets.
    ds_scalars = {}
    for key, spec in model_cfg.get('scalars', {}).items():
        if spec.get('method') == 'from_dewpoint':
            ds_scalars[key] = {'method': 'from_dewpoint'}
        else:
            ds_scalars[key] = {'ds': ds, 'var': spec['var'],
                               'units': spec.get('units'), 'method': spec.get('method')}
    return {
        'ds_u': ds, 'ds_v': ds, 'ds_temp': ds, 'ds_scalars': ds_scalars,
        'lat2d': lat2d, 'lon2d': lon2d, 'x1d': None, 'y1d': None,
        'time': time_pd,
        'datasets': [ds],
        'crs': 'latlon_2d', 'lcc_attrs': None, 'has_uv': True,
    }


def load_model_ucla(model_name, model_cfg):
    """Load UCLA multi-file reanalysis on an LCC grid with 1D x/y coordinates.

    Uses open_mfdataset to concatenate yearly NetCDF files for u10, v10, and t2.
    The grid uses Lambert Conformal Conic projection with 1D x and y coordinates
    (3 km spacing, 101×100 grid).

    Returns dict compatible with the standard model_data format (same as load_model).
    """
    print(f"\n  Loading model: {model_name}")

    data_dir = model_cfg['data_dir']
    if not data_dir.exists():
        print(f"    WARNING: directory not found: {data_dir}")
        return None

    u_files = sorted(data_dir.glob(model_cfg['u_pattern']))
    v_files = sorted(data_dir.glob(model_cfg['v_pattern']))
    temp_files = sorted(data_dir.glob(model_cfg['temp_pattern']))

    if not u_files:
        print(f"    WARNING: no u10 files found matching {model_cfg['u_pattern']}")
        return None
    if not v_files:
        print(f"    WARNING: no v10 files found matching {model_cfg['v_pattern']}")
        return None
    if not temp_files:
        print(f"    WARNING: no t2 files found matching {model_cfg['temp_pattern']}")
        return None

    print(f"    u10 files: {len(u_files)}")
    print(f"    v10 files: {len(v_files)}")
    print(f"    t2 files:  {len(temp_files)}")

    mf_kwargs = dict(concat_dim='time', combine='nested',
                     chunks={'time': 500}, data_vars='minimal',
                     coords='minimal', compat='override')

    ds_u = xr.open_mfdataset([str(f) for f in u_files], **mf_kwargs)
    ds_v = xr.open_mfdataset([str(f) for f in v_files], **mf_kwargs)
    ds_temp = xr.open_mfdataset([str(f) for f in temp_files], **mf_kwargs)

    x1d, y1d = _get_1d_coords(ds_u)
    time_pd = _decode_time(ds_u)

    # Read LCC projection info
    lcc_attrs = None
    if 'Lambert_Conformal' in ds_u:
        lcc_attrs = dict(ds_u['Lambert_Conformal'].attrs)
        print(f"    CRS: Lambert Conformal Conic")

    print(f"    Grid: {len(x1d)} x {len(y1d)}")
    print(f"    Time: {time_pd[0]} – {time_pd[-1]}, N={len(time_pd)}")

    ds_scalars = _load_scalar_datasets(model_cfg, sort_y=False)
    return {
        'ds_u': ds_u, 'ds_v': ds_v, 'ds_temp': ds_temp, 'ds_scalars': ds_scalars,
        'x1d': x1d, 'y1d': y1d, 'time': time_pd,
        'datasets': [ds_u, ds_v, ds_temp] + [v['ds'] for v in ds_scalars.values() if 'ds' in v],
        'crs': 'lcc', 'lcc_attrs': lcc_attrs,
    }


def interpolate_to_point_latlon2d(lat2d, lon2d, ds, var_name, obs_lat, obs_lon,
                                   method='linear'):
    """Interpolate a 3D field on a curvilinear 2D lat/lon grid to a point.

    Parameters
    ----------
    lat2d, lon2d : 2D arrays (ny, nx) of grid lat/lon
    ds : xarray Dataset
    var_name : str
    obs_lat, obs_lon : float, target location in lat/lon
    method : 'nearest' or 'linear'

    Returns
    -------
    ts : 1D array (ntime,)
    """
    # Find nearest grid cell using Euclidean distance on lat/lon
    # (fine for small domains; no need for haversine at ~1.5 km res)
    dist2 = (lat2d - obs_lat)**2 + (lon2d - obs_lon)**2
    iy_near, ix_near = np.unravel_index(np.argmin(dist2), dist2.shape)
    dist_deg = np.sqrt(dist2[iy_near, ix_near])
    dist_km = dist_deg * 111.0  # approximate

    if method == 'nearest' or iy_near <= 0 or ix_near <= 0 \
       or iy_near >= lat2d.shape[0] - 1 or ix_near >= lat2d.shape[1] - 1:
        print(f"      nearest iy={iy_near}, ix={ix_near}, dist={dist_km:.2f} km")
        ts = ds[var_name][:, iy_near, ix_near].values.astype(float)

    elif method == 'linear':
        # Determine which 2x2 cell the point falls in relative to nearest
        # Use the 4 surrounding cells centered on the nearest point
        # Shift to get the lower-left corner of the enclosing cell
        iy0 = iy_near - 1 if obs_lat < lat2d[iy_near, ix_near] else iy_near
        ix0 = ix_near - 1 if obs_lon < lon2d[iy_near, ix_near] else ix_near

        # Clamp
        iy0 = int(np.clip(iy0, 0, lat2d.shape[0] - 2))
        ix0 = int(np.clip(ix0, 0, lat2d.shape[1] - 2))

        # 2x2 cell corner lat/lons
        lat_ll = lat2d[iy0, ix0]
        lat_ul = lat2d[iy0 + 1, ix0]
        lon_ll = lon2d[iy0, ix0]
        lon_lr = lon2d[iy0, ix0 + 1]

        # Bilinear weights
        dy = lat_ul - lat_ll if lat_ul != lat_ll else 1.0
        dx = lon_lr - lon_ll if lon_lr != lon_ll else 1.0
        ty = np.clip((obs_lat - lat_ll) / dy, 0.0, 1.0)
        tx = np.clip((obs_lon - lon_ll) / dx, 0.0, 1.0)

        w11 = (1 - tx) * (1 - ty)
        w21 = tx * (1 - ty)
        w12 = (1 - tx) * ty
        w22 = tx * ty

        print(f"      bilinear iy=[{iy0},{iy0+1}], ix=[{ix0},{ix0+1}], "
              f"weights=({w11:.3f},{w21:.3f},{w12:.3f},{w22:.3f}), dist={dist_km:.2f} km")

        slab = ds[var_name][:, iy0:iy0 + 2, ix0:ix0 + 2].values.astype(float)
        ts = (w11 * slab[:, 0, 0] + w21 * slab[:, 0, 1] +
              w12 * slab[:, 1, 0] + w22 * slab[:, 1, 1])
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    ts[np.abs(ts) > 1e30] = np.nan
    return ts


def interpolate_to_point(x1d, y1d, ds, var_name, obs_x, obs_y, method='linear'):
    """Interpolate a 3D field (time, y, x) to a single (obs_x, obs_y) point.

    Parameters
    ----------
    x1d, y1d : 1D arrays of grid coordinates
    ds : xarray Dataset containing the variable
    var_name : str, variable name in ds
    obs_x, obs_y : float, target location in UTM
    method : 'nearest' or 'linear'

    Returns
    -------
    ts : 1D array (ntime,)
    """
    nx, ny = len(x1d), len(y1d)

    if method == 'nearest':
        ix = int(np.argmin(np.abs(x1d - obs_x)))
        iy = int(np.argmin(np.abs(y1d - obs_y)))
        dist = np.sqrt((x1d[ix] - obs_x)**2 + (y1d[iy] - obs_y)**2)
        print(f"      nearest ix={ix}, iy={iy}, dist={dist/1000:.2f} km")
        ts = ds[var_name][:, iy, ix].values.astype(float)

    elif method == 'linear':
        ix_low = int(np.searchsorted(x1d, obs_x, side='right')) - 1
        iy_low = int(np.searchsorted(y1d, obs_y, side='right')) - 1

        can_bilinear = (0 <= ix_low < nx - 1) and (0 <= iy_low < ny - 1)

        if can_bilinear:
            x1, x2 = x1d[ix_low], x1d[ix_low + 1]
            y1, y2 = y1d[iy_low], y1d[iy_low + 1]
            tx = np.clip((obs_x - x1) / (x2 - x1), 0.0, 1.0)
            ty = np.clip((obs_y - y1) / (y2 - y1), 0.0, 1.0)

            w11 = (1 - tx) * (1 - ty)
            w21 = tx * (1 - ty)
            w12 = (1 - tx) * ty
            w22 = tx * ty

            print(f"      bilinear ix=[{ix_low},{ix_low+1}], iy=[{iy_low},{iy_low+1}], "
                  f"weights=({w11:.3f},{w21:.3f},{w12:.3f},{w22:.3f})")

            slab = ds[var_name][:, iy_low:iy_low + 2, ix_low:ix_low + 2].values.astype(float)
            ts = (w11 * slab[:, 0, 0] + w21 * slab[:, 0, 1] +
                  w12 * slab[:, 1, 0] + w22 * slab[:, 1, 1])
        else:
            # Fall back to nearest at domain edge
            ix = int(np.clip(ix_low, 0, nx - 1))
            iy = int(np.clip(iy_low, 0, ny - 1))
            print(f"      edge fallback to nearest ix={ix}, iy={iy}")
            ts = ds[var_name][:, iy, ix].values.astype(float)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Replace fill values (HRRR uses ~1e31)
    ts[np.abs(ts) > 1e30] = np.nan
    return ts


def load_model_box(model_name, model_cfg):
    """Load a per-year 'box' product for on-the-fly interpolation to any station.

    crs 'latlon_2d'   -> curvilinear 2-D latitude/longitude (Sup3rWind; has u/v),
                         reuses the latlon_2d extract path.
    crs 'unstructured'-> 1-D scattered nodes with latitude/longitude per point
                         (NOW-23), nearest-node in the extract path.
    """
    print(f"\n  Loading model: {model_name}")
    data_dir = model_cfg['data_dir']
    files = sorted(data_dir.glob(model_cfg['pattern']))
    if not files:
        print(f"    WARNING: no files match {model_cfg['pattern']} in {data_dir}")
        return None
    ds = xr.open_mfdataset([str(f) for f in files], concat_dim='time',
                           combine='nested', chunks={'time': 500},
                           data_vars='minimal', coords='minimal', compat='override')
    time_pd = _decode_time(ds)
    crs = model_cfg['crs']
    print(f"    {len(files)} files, crs={crs}, "
          f"time {time_pd[0]} – {time_pd[-1]}, N={len(time_pd)}")
    if crs == 'latlon_2d':
        return {
            'ds_u': ds, 'ds_v': ds, 'ds_speed': ds, 'ds_temp': None,
            'lat2d': ds['latitude'].values.astype(float),
            'lon2d': ds['longitude'].values.astype(float),
            'x1d': None, 'y1d': None, 'time': time_pd, 'datasets': [ds],
            'crs': 'latlon_2d', 'lcc_attrs': None, 'has_uv': model_cfg.get('has_uv', True),
        }
    elif crs == 'unstructured':
        return {
            'ds': ds, 'node_lat': ds['latitude'].values.astype(float),
            'node_lon': ds['longitude'].values.astype(float),
            'time': time_pd, 'datasets': [ds], 'crs': 'unstructured', 'lcc_attrs': None,
        }
    print(f"    ERROR: load_model_box unknown crs '{crs}'")
    ds.close()
    return None


def load_point_product(model_name, model_cfg):
    """Load a pre-extracted buoy-point product (NOW-23 / Sup3rWind / RTMA).

    These NetCDFs already hold the model timeseries at each buoy (dim 'station'),
    so no grid interpolation is needed. crs='point_product' makes the main loop
    skip the spatial-map step cleanly.
    """
    fp = model_cfg['point_file']
    if not fp.exists():
        print(f"    WARNING: point-product file not found: {fp}")
        return None
    ds = xr.open_dataset(str(fp))
    time_pd = pd.to_datetime(ds['time'].values)
    stns = [str(s) for s in ds['station'].values]
    print(f"    point product: {fp.name}; stations={stns}")
    print(f"    Time: {time_pd[0]} – {time_pd[-1]}, N={len(time_pd)}")
    return {'ds': ds, 'time': time_pd, 'datasets': [ds],
            'crs': 'point_product', 'lcc_attrs': None}


def extract_point_product(model_data, model_cfg, station_id):
    """Return a product's timeseries for an obs station via its station_map.

    u/v are derived from speed + meteorological 'from' direction (same convention
    as the observations). Temperature is NaN (these are wind-only products).
    """
    ds = model_data['ds']
    ntime = len(model_data['time'])
    nan = np.full(ntime, np.nan)
    smap = model_cfg.get('station_map', {})
    pt = smap.get(station_id, station_id)   # identity: product stations named by obs id
    stns = [str(s) for s in ds['station'].values]
    if pt is None or pt not in stns:
        print(f"    point product has no mapping for station '{station_id}' -> NaN")
        return {'time': model_data['time'], 'u10': nan, 'v10': nan,
                'speed10': nan, 'dir_deg': nan, 'air_temp_C': nan}
    i = stns.index(pt)
    speed = ds[model_cfg['speed_var']].isel(station=i).values.astype(float)
    direction = ds[model_cfg['dir_var']].isel(station=i).values.astype(float)
    dr = np.radians(direction)
    u = -speed * np.sin(dr)
    v = -speed * np.cos(dr)
    print(f"    point '{pt}' -> station '{station_id}' "
          f"({int(np.isfinite(speed).sum())} valid steps)")
    return {'time': model_data['time'], 'u10': u, 'v10': v,
            'speed10': speed, 'dir_deg': direction, 'air_temp_C': nan}


def extract_model_at_station(model_data, model_cfg, obs_x, obs_y,
                             obs_lat=None, obs_lon=None, method='linear',
                             station_id=None):
    """Extract u, v, temperature timeseries from model at an observation point.

    Returns dict with keys: time, u10, v10, speed10, dir_deg, air_temp_C.
    For speed-only models (has_uv=False), u10/v10/dir_deg are NaN arrays.
    """
    if model_cfg.get('kind') == 'point_product':
        return extract_point_product(model_data, model_cfg, station_id)

    crs_type = model_data.get('crs', 'utm10n')
    ntime = len(model_data['time'])

    # --- Unstructured scattered nodes (e.g. NOW-23): nearest node ---
    if crs_type == 'unstructured':
        if obs_lat is None or obs_lon is None:
            raise ValueError("obs_lat/obs_lon required for unstructured models")
        nlat = model_data['node_lat']; nlon = model_data['node_lon']
        d2 = (nlat - obs_lat) ** 2 + (nlon - obs_lon) ** 2
        inode = int(np.argmin(d2))
        dist_km = float(np.sqrt(d2[inode]) * 111.0)
        ds = model_data['ds']
        print(f"      unstructured nearest node {inode}, dist={dist_km:.2f} km")
        if model_cfg.get('has_uv') and model_cfg.get('u_var') in ds:
            u = ds[model_cfg['u_var']][:, inode].values.astype(float)
            v = ds[model_cfg['v_var']][:, inode].values.astype(float)
            speed = np.sqrt(u ** 2 + v ** 2)
            direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
        else:
            speed = ds[model_cfg['speed_var']][:, inode].values.astype(float)
            direction = ds[model_cfg['dir_var']][:, inode].values.astype(float)
            dr = np.radians(direction)
            u = -speed * np.sin(dr); v = -speed * np.cos(dr)
        speed[np.abs(speed) > 1e30] = np.nan
        return {'time': model_data['time'], 'u10': u, 'v10': v,
                'speed10': speed, 'dir_deg': direction,
                'air_temp_C': np.full(ntime, np.nan)}

    # --- Curvilinear 2D lat/lon grid (e.g. WRF_CalNev) ---
    if crs_type == 'latlon_2d':
        if obs_lat is None or obs_lon is None:
            raise ValueError("obs_lat/obs_lon required for latlon_2d models")

        has_uv = model_cfg.get('has_uv', True)

        if has_uv:
            print(f"    Interpolating u ({method}) ...")
            u = interpolate_to_point_latlon2d(
                model_data['lat2d'], model_data['lon2d'],
                model_data['ds_u'], model_cfg['u_var'],
                obs_lat, obs_lon, method)
            print(f"    Interpolating v ({method}) ...")
            v = interpolate_to_point_latlon2d(
                model_data['lat2d'], model_data['lon2d'],
                model_data['ds_v'], model_cfg['v_var'],
                obs_lat, obs_lon, method)
            speed = np.sqrt(u**2 + v**2)
            direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
        else:
            # Speed-only model (no u/v components)
            print(f"    Interpolating wind speed ({method}) ...")
            speed = interpolate_to_point_latlon2d(
                model_data['lat2d'], model_data['lon2d'],
                model_data['ds_speed'], model_cfg['speed_var'],
                obs_lat, obs_lon, method)
            u = np.full(ntime, np.nan)
            v = np.full(ntime, np.nan)
            direction = np.full(ntime, np.nan)

        if model_data.get('ds_temp') is not None and 'temp_var' in model_cfg:
            print(f"    Interpolating temperature ({method}) ...")
            temp = interpolate_to_point_latlon2d(
                model_data['lat2d'], model_data['lon2d'],
                model_data['ds_temp'], model_cfg['temp_var'],
                obs_lat, obs_lon, method)
            temp_c = temp - 273.15  # K → °C
        else:
            temp_c = np.full(ntime, np.nan)

        return {
            'time': model_data['time'],
            'u10': u, 'v10': v,
            'speed10': speed, 'dir_deg': direction,
            'air_temp_C': temp_c,
        }

    # --- Rectilinear grids (UTM or LCC with 1D x/y) ---
    x1d = model_data['x1d']
    y1d = model_data['y1d']

    # Convert station coords to model CRS if needed
    if crs_type == 'lcc' and model_data.get('lcc_attrs') is not None:
        obs_x, obs_y = utm10_to_lcc(obs_x, obs_y, model_data['lcc_attrs'])
        print(f"    Station in LCC: x={obs_x:.1f}, y={obs_y:.1f}")

    print(f"    Interpolating u ({method}) ...")
    u = interpolate_to_point(x1d, y1d, model_data['ds_u'], model_cfg['u_var'],
                             obs_x, obs_y, method)
    print(f"    Interpolating v ({method}) ...")
    v = interpolate_to_point(x1d, y1d, model_data['ds_v'], model_cfg['v_var'],
                             obs_x, obs_y, method)
    
    # Temperature is optional
    if model_data.get('ds_temp') is not None and 'temp_var' in model_cfg:
        print(f"    Interpolating temperature ({method}) ...")
        temp = interpolate_to_point(x1d, y1d, model_data['ds_temp'], model_cfg['temp_var'],
                                    obs_x, obs_y, method)
        temp_c = temp - 273.15  # K → °C
    else:
        temp_c = np.full(len(model_data['time']), np.nan)

    speed = np.sqrt(u**2 + v**2)
    direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

    return {
        'time': model_data['time'],
        'u10': u, 'v10': v,
        'speed10': speed, 'dir_deg': direction,
        'air_temp_C': temp_c,
    }


# ===========================================================================
# Batched extraction — compute every station's interpolation stencil ONCE,
# then read each model variable in a SINGLE vectorized pass (all stations'
# 2x2 corners pulled together). This collapses the ~(n_stations x n_vars)
# repeated multi-file reads of load_model_*'s lazy datasets into ~n_vars reads,
# the dominant cost in long-record runs.
# ===========================================================================

def _stencil_1d(x1d, y1d, obs_x, obs_y, method='linear'):
    """Bilinear stencil on a rectilinear (1-D x, 1-D y) grid.

    Returns (iy[4], ix[4], w[4]) where the time series is
    sum_c w[c] * field[:, iy[c], ix[c]] — matching interpolate_to_point exactly.
    Corner order: (iy0,ix0), (iy0,ix0+1), (iy0+1,ix0), (iy0+1,ix0+1).
    """
    nx, ny = len(x1d), len(y1d)
    if method == 'nearest':
        ix = int(np.argmin(np.abs(x1d - obs_x)))
        iy = int(np.argmin(np.abs(y1d - obs_y)))
        return (np.full(4, iy), np.full(4, ix), np.array([1.0, 0.0, 0.0, 0.0]))

    ix_low = int(np.searchsorted(x1d, obs_x, side='right')) - 1
    iy_low = int(np.searchsorted(y1d, obs_y, side='right')) - 1
    if (0 <= ix_low < nx - 1) and (0 <= iy_low < ny - 1):
        x1, x2 = x1d[ix_low], x1d[ix_low + 1]
        y1, y2 = y1d[iy_low], y1d[iy_low + 1]
        tx = float(np.clip((obs_x - x1) / (x2 - x1), 0.0, 1.0))
        ty = float(np.clip((obs_y - y1) / (y2 - y1), 0.0, 1.0))
        w = np.array([(1 - tx) * (1 - ty), tx * (1 - ty),
                      (1 - tx) * ty, tx * ty])
        iy_idx = np.array([iy_low, iy_low, iy_low + 1, iy_low + 1])
        ix_idx = np.array([ix_low, ix_low + 1, ix_low, ix_low + 1])
        return (iy_idx, ix_idx, w)
    # edge fallback to nearest
    ix = int(np.clip(ix_low, 0, nx - 1))
    iy = int(np.clip(iy_low, 0, ny - 1))
    return (np.full(4, iy), np.full(4, ix), np.array([1.0, 0.0, 0.0, 0.0]))


def _stencil_2d(lat2d, lon2d, obs_lat, obs_lon, method='linear'):
    """Bilinear stencil on a curvilinear 2-D lat/lon grid (matches
    interpolate_to_point_latlon2d). Returns (iy[4], ix[4], w[4])."""
    dist2 = (lat2d - obs_lat) ** 2 + (lon2d - obs_lon) ** 2
    iy_near, ix_near = np.unravel_index(np.argmin(dist2), dist2.shape)
    edge = (iy_near <= 0 or ix_near <= 0
            or iy_near >= lat2d.shape[0] - 1 or ix_near >= lat2d.shape[1] - 1)
    if method == 'nearest' or edge:
        return (np.full(4, iy_near), np.full(4, ix_near),
                np.array([1.0, 0.0, 0.0, 0.0]))

    iy0 = iy_near - 1 if obs_lat < lat2d[iy_near, ix_near] else iy_near
    ix0 = ix_near - 1 if obs_lon < lon2d[iy_near, ix_near] else ix_near
    iy0 = int(np.clip(iy0, 0, lat2d.shape[0] - 2))
    ix0 = int(np.clip(ix0, 0, lat2d.shape[1] - 2))

    lat_ll = lat2d[iy0, ix0]; lat_ul = lat2d[iy0 + 1, ix0]
    lon_ll = lon2d[iy0, ix0]; lon_lr = lon2d[iy0, ix0 + 1]
    dy = lat_ul - lat_ll if lat_ul != lat_ll else 1.0
    dx = lon_lr - lon_ll if lon_lr != lon_ll else 1.0
    ty = float(np.clip((obs_lat - lat_ll) / dy, 0.0, 1.0))
    tx = float(np.clip((obs_lon - lon_ll) / dx, 0.0, 1.0))
    w = np.array([(1 - tx) * (1 - ty), tx * (1 - ty),
                  (1 - tx) * ty, tx * ty])
    iy_idx = np.array([iy0, iy0, iy0 + 1, iy0 + 1])
    ix_idx = np.array([ix0, ix0 + 1, ix0, ix0 + 1])
    return (iy_idx, ix_idx, w)


def _read_stencils(ds, var_name, stencils):
    """One vectorized read of a (time, y, x) field at all stations' 2x2 corners.

    stencils: list of (iy[4], ix[4], w[4]). Returns (ntime, nstations) array
    with fill values (>1e30) set to NaN."""
    S = len(stencils)
    iy_flat = np.concatenate([s[0] for s in stencils])
    ix_flat = np.concatenate([s[1] for s in stencils])
    w_stack = np.stack([s[2] for s in stencils])            # (S, 4)
    vdims = ds[var_name].dims                                # (time, ydim, xdim)
    tdim, ydim, xdim = vdims[0], vdims[1], vdims[2]
    da = ds[var_name].isel({ydim: xr.DataArray(iy_flat, dims='pt'),
                            xdim: xr.DataArray(ix_flat, dims='pt')})
    arr = da.transpose(tdim, 'pt').values.astype(float)     # (ntime, S*4)
    arr = arr.reshape(arr.shape[0], S, 4)
    vals = np.einsum('tsc,sc->ts', arr, w_stack)            # (ntime, S)
    vals[np.abs(vals) > 1e30] = np.nan
    return vals


def _read_nodes(ds, var_name, inodes):
    """One vectorized read of a (time, node) field at all stations' nearest nodes."""
    vdims = ds[var_name].dims                                # (time, nodedim)
    tdim, ndim = vdims[0], vdims[1]
    da = ds[var_name].isel({ndim: xr.DataArray(np.asarray(inodes), dims='pt')})
    arr = da.transpose(tdim, 'pt').values.astype(float)     # (ntime, S)
    arr[np.abs(arr) > 1e30] = np.nan
    return arr


def _open_scalar_source(spec, data_dir, sort_y=False):
    """Open one scalar variable's dataset: single 'file' or 'pattern' glob in data_dir."""
    try:
        if 'file' in spec:
            ds = xr.open_dataset(str(spec['file']), chunks={'time': 500})
        elif 'pattern' in spec and data_dir is not None:
            files = sorted(data_dir.glob(spec['pattern']))
            if not files:
                print(f"      [scalar: no files match {spec['pattern']}]")
                return None
            ds = xr.open_mfdataset([str(f) for f in files], concat_dim='time',
                                   combine='nested', chunks={'time': 200},
                                   data_vars='minimal', coords='minimal', compat='override')
        else:
            return None
    except Exception as e:
        print(f"      [scalar source failed: {type(e).__name__}: {e}]")
        return None
    if sort_y and 'y' in ds.coords and ds['y'].ndim == 1 and float(ds['y'][0]) > float(ds['y'][-1]):
        ds = ds.sortby('y')
    return ds


def _load_scalar_datasets(model_cfg, sort_y=False):
    """Open every scalar variable declared in model_cfg['scalars'].
    Returns {var_key: {'ds','var','units','method'}}; 'from_dewpoint' has no ds."""
    scalars = model_cfg.get('scalars')
    if not scalars:
        return {}
    data_dir = model_cfg.get('data_dir')
    out = {}
    for key, spec in scalars.items():
        if spec.get('method') == 'from_dewpoint':
            out[key] = {'method': 'from_dewpoint'}
            continue
        ds = _open_scalar_source(spec, data_dir, sort_y=sort_y)
        if ds is not None:
            out[key] = {'ds': ds, 'var': spec['var'],
                        'units': spec.get('units'), 'method': spec.get('method')}
    return out


def _extract_scalars(model_data, stencils=None, inodes=None):
    """Read all loaded scalar datasets at the stencils (1d/2d) or nodes
    (unstructured). Returns {var_key: (vals (ntime_v, S), time)} incl. derived RH."""
    ds_scalars = model_data.get('ds_scalars') or {}
    raw = {}

    def _read(info):
        if stencils is not None:
            vals = _read_stencils(info['ds'], info['var'], stencils)
        else:
            vals = _read_nodes(info['ds'], info['var'], inodes)
        u = (info.get('units') or '').lower().replace(' ', '')
        if u in ('k', 'kelvin'):
            vals = vals - 273.15
        elif u == 'pa':
            vals = vals / 100.0                     # Pa -> hPa
        return vals, _decode_time(info['ds'])

    for key, info in ds_scalars.items():
        if 'ds' not in info:
            continue
        try:
            raw[key] = _read(info)
        except Exception as e:
            print(f"      [scalar '{key}' read failed: {type(e).__name__}: {e}]")

    rh_info = ds_scalars.get('rh')
    if rh_info is not None:
        m = rh_info.get('method')
        try:
            if m == 'from_dewpoint' and 'temperature' in raw and 'dewpoint' in raw:
                raw['rh'] = (_rh_from_dewpoint(raw['temperature'][0], raw['dewpoint'][0]),
                             raw['temperature'][1])
            elif m == 'from_q' and 'rh' in raw and 'temperature' in raw and 'pressure' in raw:
                raw['rh'] = (_rh_from_q(raw['rh'][0], raw['temperature'][0], raw['pressure'][0]),
                             raw['rh'][1])
            # 'direct': raw['rh'] already holds RH%
        except Exception as e:
            print(f"      [RH derive failed: {type(e).__name__}: {e}]")
            raw.pop('rh', None)
    return raw


def _apply_scalars(out, sids, scal):
    """Fill every SCALAR_VARS obs_key (+ per-variable time '<key>__t') into each
    station's dict; missing variables -> NaN."""
    for var_key, spec in SCALAR_VARS.items():
        ok = spec['obs_key']
        if var_key in scal:
            vals, t = scal[var_key]
            for k, sid in enumerate(sids):
                out[sid][ok] = vals[:, k]
                out[sid][ok + '__t'] = t
        else:
            for sid in sids:
                out[sid][ok] = np.full(len(out[sid]['time']), np.nan)


def extract_model_all_stations(model_data, model_cfg, station_data,
                               method='linear'):
    """Extract model timeseries for ALL stations at once.

    Returns {station_id: {time, u10, v10, speed10, dir_deg, air_temp_C, + scalars}},
    each entry compatible with extract_model_at_station(). The win: one read per
    variable instead of one per (station, variable).
    """
    crs_type = model_data.get('crs', 'utm10n')
    time = model_data['time']
    ntime = len(time)
    sids = list(station_data.keys())
    out = {}
    # Skip scalar (temp/pressure/dewpoint/rh/radiation/precip) extraction entirely
    # when only wind is requested — those reads are full-file and pure waste here.
    _want_scalars = any(v != 'wind' for v in VARIABLES)

    # Point products are already point timeseries in memory — per-station is cheap.
    if model_cfg.get('kind') == 'point_product':
        for sid in sids:
            out[sid] = extract_point_product(model_data, model_cfg, sid)
        return out

    # --- Unstructured scattered nodes (NOW-23): nearest node, batched ---
    if crs_type == 'unstructured':
        nlat = model_data['node_lat']; nlon = model_data['node_lon']
        ds = model_data['ds']
        inodes = []
        for sid in sids:
            o = station_data[sid]
            d2 = (nlat - o['lat']) ** 2 + (nlon - o['lon']) ** 2
            j = int(np.argmin(d2))
            inodes.append(j)
            print(f"      {sid}: node {j}, dist={float(np.sqrt(d2[j])) * 111.0:.2f} km")
        has_uv = model_cfg.get('has_uv') and model_cfg.get('u_var') in ds
        if has_uv:
            U = _read_nodes(ds, model_cfg['u_var'], inodes)
            Vv = _read_nodes(ds, model_cfg['v_var'], inodes)
            for k, sid in enumerate(sids):
                u, v = U[:, k], Vv[:, k]
                speed = np.sqrt(u ** 2 + v ** 2)
                direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
                out[sid] = {'time': time, 'u10': u, 'v10': v, 'speed10': speed,
                            'dir_deg': direction}
        else:
            SP = _read_nodes(ds, model_cfg['speed_var'], inodes)
            DR = _read_nodes(ds, model_cfg['dir_var'], inodes)
            for k, sid in enumerate(sids):
                speed = SP[:, k]; direction = DR[:, k]
                dr = np.radians(direction)
                u = -speed * np.sin(dr); v = -speed * np.cos(dr)
                out[sid] = {'time': time, 'u10': u, 'v10': v, 'speed10': speed,
                            'dir_deg': direction}
        if _want_scalars:
            _apply_scalars(out, sids, _extract_scalars(model_data, inodes=inodes))
        return out

    # --- Curvilinear 2-D lat/lon grid (WRF_CalNev / Sup3rWind) ---
    if crs_type == 'latlon_2d':
        lat2d = model_data['lat2d']; lon2d = model_data['lon2d']
        has_uv = model_cfg.get('has_uv', True)
        stencils = [_stencil_2d(lat2d, lon2d, station_data[s]['lat'],
                                station_data[s]['lon'], method) for s in sids]
        print(f"    Batched bilinear (latlon_2d): {len(sids)} stations, vectorized read/var")
        if has_uv:
            U = _read_stencils(model_data['ds_u'], model_cfg['u_var'], stencils)
            Vv = _read_stencils(model_data['ds_v'], model_cfg['v_var'], stencils)
        else:
            SP = _read_stencils(model_data['ds_speed'], model_cfg['speed_var'], stencils)
        for k, sid in enumerate(sids):
            if has_uv:
                u, v = U[:, k], Vv[:, k]
                speed = np.sqrt(u ** 2 + v ** 2)
                direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
            else:
                speed = SP[:, k]
                u = np.full(ntime, np.nan); v = np.full(ntime, np.nan)
                direction = np.full(ntime, np.nan)
            out[sid] = {'time': time, 'u10': u, 'v10': v, 'speed10': speed,
                        'dir_deg': direction}
        if _want_scalars:
            _apply_scalars(out, sids, _extract_scalars(model_data, stencils=stencils))
        return out

    # --- Rectilinear grids (UTM 1-D x/y, or LCC after coord conversion) ---
    x1d = model_data['x1d']; y1d = model_data['y1d']
    stencils = []
    for sid in sids:
        o = station_data[sid]
        ox, oy = o['x_utm'], o['y_utm']
        if crs_type == 'lcc' and model_data.get('lcc_attrs') is not None:
            ox, oy = utm10_to_lcc(ox, oy, model_data['lcc_attrs'])
        stencils.append(_stencil_1d(x1d, y1d, ox, oy, method))
    print(f"    Batched bilinear ({crs_type}): {len(sids)} stations, vectorized read/var")
    U = _read_stencils(model_data['ds_u'], model_cfg['u_var'], stencils)
    Vv = _read_stencils(model_data['ds_v'], model_cfg['v_var'], stencils)
    for k, sid in enumerate(sids):
        u, v = U[:, k], Vv[:, k]
        speed = np.sqrt(u ** 2 + v ** 2)
        direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
        out[sid] = {'time': time, 'u10': u, 'v10': v, 'speed10': speed,
                    'dir_deg': direction}
    _apply_scalars(out, sids, _extract_scalars(model_data, stencils=stencils))
    return out


# ===========================================================================
# Time matching
# ===========================================================================

def match_timeseries(model_vals, model_time, obs_vals, obs_time, tolerance='1h'):
    """Match model and obs by nearest time within tolerance."""
    model_time_pd = pd.to_datetime(model_time)
    obs_time_pd = pd.to_datetime(obs_time)

    overlap_start = max(model_time_pd.min(), obs_time_pd.min())
    overlap_end = min(model_time_pd.max(), obs_time_pd.max())

    if overlap_start >= overlap_end:
        print(f"      No time overlap.")
        return None, None, None

    df_model = (pd.DataFrame({'time': model_time_pd, 'value': model_vals})
                .set_index('time').sort_index()
                .loc[overlap_start:overlap_end])
    df_obs = (pd.DataFrame({'time': obs_time_pd, 'value': obs_vals})
              .set_index('time').sort_index()
              .loc[overlap_start:overlap_end])

    df_combined = pd.merge_asof(
        df_obs.reset_index(), df_model.reset_index(),
        on='time', direction='nearest',
        tolerance=pd.Timedelta(tolerance),
        suffixes=('_obs', '_model'),
    ).dropna()

    if len(df_combined) == 0:
        print(f"      No matching points within tolerance.")
        return None, None, None

    print(f"      Matched {len(df_combined)} points "
          f"({overlap_start.strftime('%Y-%m-%d')} – {overlap_end.strftime('%Y-%m-%d')})")

    return (df_combined['value_model'].values,
            df_combined['value_obs'].values,
            df_combined['time'].values)


# ===========================================================================
# Statistics
# ===========================================================================

def calculate_statistics(model, obs):
    """Standard validation statistics."""
    mask = ~(np.isnan(model) | np.isnan(obs))
    mc, oc = model[mask], obs[mask]
    if len(mc) < 2:
        return None

    bias = np.mean(mc - oc)
    rmse = np.sqrt(np.mean((mc - oc) ** 2))
    mae = np.mean(np.abs(mc - oc))
    corr, p_value = stats.pearsonr(mc, oc)
    obs_mean = np.mean(oc)
    obs_std = np.std(oc)
    obs_var = np.var(oc)
    mse = np.mean((mc - oc) ** 2)

    si = rmse / obs_mean if obs_mean != 0 else np.nan
    r2 = corr ** 2
    nrmse = rmse / obs_std if obs_std != 0 else np.nan
    rel_bias = bias / obs_mean if obs_mean != 0 else np.nan
    skill = 1.0 - mse / obs_var if obs_var != 0 else np.nan

    return {
        'n': int(len(mc)),
        'bias': float(bias), 'rmse': float(rmse), 'mae': float(mae),
        'corr': float(corr), 'p_value': float(p_value),
        'r2': float(r2), 'nrmse': float(nrmse),
        'scatter_index': float(si),
        'rel_bias': float(rel_bias), 'skill': float(skill),
        'model_mean': float(np.mean(mc)), 'obs_mean': float(obs_mean),
        'model_std': float(np.std(mc)), 'obs_std': float(obs_std),
    }


def calculate_circular_statistics(model_dir, obs_dir):
    """Circular statistics for wind direction."""
    mask = ~(np.isnan(model_dir) | np.isnan(obs_dir))
    md, od = model_dir[mask], obs_dir[mask]
    if len(md) < 2:
        return None

    diff = ((md - od + 180) % 360) - 180  # shortest angular distance
    circ_bias = np.mean(diff)
    circ_rmse = np.sqrt(np.mean(diff ** 2))
    circ_mae = np.mean(np.abs(diff))

    # Circular correlation (Jammalamadaka & SenGupta)
    sin_m = np.sin(np.radians(md - np.mean(md)))
    sin_o = np.sin(np.radians(od - np.mean(od)))
    circ_corr = np.sum(sin_m * sin_o) / np.sqrt(np.sum(sin_m**2) * np.sum(sin_o**2))

    return {
        'n': int(len(md)),
        'bias': float(circ_bias), 'rmse': float(circ_rmse), 'mae': float(circ_mae),
        'corr': float(circ_corr), 'p_value': np.nan,
        'scatter_index': np.nan,
        'model_mean': float(np.mean(md)), 'obs_mean': float(np.mean(od)),
        'model_std': float(np.std(diff)), 'obs_std': np.nan,
    }


# ===========================================================================
# Plotting helpers
# ===========================================================================

def _stats_text(st):
    return (f"N = {st['n']}\n"
            f"Bias = {st['bias']:.3f}\n"
            f"RMSE = {st['rmse']:.3f}\n"
            f"MAE = {st['mae']:.3f}\n"
            f"R = {st['corr']:.3f}\n"
            f"R² = {st.get('r2', st['corr']**2):.3f}\n"
            f"NRMSE = {st.get('nrmse', np.nan):.3f}\n"
            f"SI = {st['scatter_index']:.3f}\n"
            f"Rel Bias = {st.get('rel_bias', np.nan):.3f}\n"
            f"Skill = {st.get('skill', np.nan):.3f}")


def _safe_filename(s):
    return s.replace(' ', '_').replace('/', '_').replace('[', '').replace(']', '')


def plot_scatter(model_vals, obs_vals, var_name, station_name, model_name,
                 stats_dict, output_dir):
    """Scatter plot with 1:1 line, regression, and skill-score text box."""
    mask = ~(np.isnan(model_vals) | np.isnan(obs_vals))
    mc, oc = model_vals[mask], obs_vals[mask]
    if len(mc) < 5:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(oc, mc, c='black', s=15, alpha=0.5, edgecolors='none')

    lo = min(oc.min(), mc.min())
    hi = max(oc.max(), mc.max())
    if hi <= lo:                      # all values identical (e.g. all-zero precip) -> nothing to fit
        plt.close(fig); return
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='1:1 line')

    # Regression line only when both axes vary (linregress errors on constant x)
    if np.ptp(oc) > 0 and np.ptp(mc) > 0:
        slope, intercept, *_ = stats.linregress(oc, mc)
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                'b-', lw=1.5, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

    alpha_fit = np.sum(oc * mc) / np.sum(oc ** 2) if np.sum(oc ** 2) != 0 else 1.0
    ax.plot([lo, hi], [alpha_fit * lo, alpha_fit * hi],
            'g-', lw=1.5, label=f'Fit (origin): y={alpha_fit:.2f}x')

    ax.set_xlabel(f'Observed {var_name}', fontsize=11)
    ax.set_ylabel(f'{model_name} {var_name}', fontsize=11)
    ax.set_title(f'{station_name}: {var_name} ({model_name})', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, _stats_text(stats_dict), transform=ax.transAxes,
            fontsize=10, va='top', bbox=props)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_dir / f"{station_name}_{_safe_filename(var_name)}_scatter.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


def plot_timeseries(model_vals, obs_vals, time_vals, var_name, station_name,
                    model_name, stats_dict, output_dir, period_label=''):
    """Single-panel timeseries comparison (observations vs model)."""
    mask = ~(np.isnan(model_vals) | np.isnan(obs_vals))
    if mask.sum() < 5:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_vals, obs_vals, 'b-', label='Observations (10 m)', lw=1.0)
    ax.plot(time_vals, model_vals, 'r-', label=model_name, lw=1.0)
    ax.set_ylabel(var_name, fontsize=11)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_title(f'{station_name}: {var_name} ({model_name})  [{period_label}]', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.97, _stats_text(stats_dict), transform=ax.transAxes,
            fontsize=9, va='top', bbox=props)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    label_safe = period_label.replace(' ', '_').replace(':', '')
    fname = output_dir / f"{station_name}_{_safe_filename(var_name)}_{label_safe}_timeseries.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


def select_timeseries_window(time_vals, model_vals, obs_vals, mode='random', days=30):
    """Select sub-window: 'random' or 'peak'."""
    time_pd = pd.to_datetime(time_vals)
    total_days = (time_pd[-1] - time_pd[0]).days

    if mode == 'random':
        rng = np.random.default_rng(seed=42)
        start_offset = int(total_days * 0.33)
        end_offset = max(start_offset + days, int(total_days * 0.66))
        start_day = rng.integers(start_offset, max(start_offset + 1, end_offset - days))
        t_start = time_pd[0] + pd.Timedelta(days=int(start_day))
        t_end = t_start + pd.Timedelta(days=days)
    elif mode == 'peak':
        obs_clean = np.where(np.isnan(obs_vals), -np.inf, obs_vals)
        peak_idx = np.argmax(obs_clean)
        t_peak = time_pd[peak_idx]
        t_start = t_peak - pd.Timedelta(days=days / 2)
        t_end = t_peak + pd.Timedelta(days=days / 2)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mask = (time_pd >= t_start) & (time_pd <= t_end)
    return time_pd[mask], model_vals[mask], obs_vals[mask]


def plot_wind_rose(model_speed, model_dir, obs_speed, obs_dir,
                   station_name, model_name, output_dir):
    """Side-by-side wind roses for observations and model."""
    if not HAS_WINDROSE:
        print(f"      Skipping wind rose (windrose package not installed)")
        return

    # Clean data
    mask_obs = ~(np.isnan(obs_speed) | np.isnan(obs_dir))
    mask_mod = ~(np.isnan(model_speed) | np.isnan(model_dir))

    if mask_obs.sum() < 50 or mask_mod.sum() < 50:
        return

    # Fixed speed bins: 0-4, 4-8, 8-12, 12-16, 16-20, >20 m/s
    speed_bins = [0, 4, 8, 12, 16, 20]

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='windrose')
    ax1.bar(obs_dir[mask_obs], obs_speed[mask_obs],
            bins=speed_bins, normed=True, opening=0.8, edgecolor='white')
    ax1.set_title(f'Observed — {station_name}', fontsize=11, pad=20)
    ax1.set_legend(loc='lower left', fontsize=8)

    ax2 = fig.add_subplot(122, projection='windrose')
    ax2.bar(model_dir[mask_mod], model_speed[mask_mod],
            bins=speed_bins, normed=True, opening=0.8, edgecolor='white')
    ax2.set_title(f'{model_name} — {station_name}', fontsize=11, pad=20)
    ax2.set_legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    fname = output_dir / f"{station_name}_wind_rose.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


def _find_obs_peak_time(obs_speed, obs_time):
    """Find the time of peak observed wind speed. Returns None if no valid data."""
    if obs_speed is None or len(obs_speed) == 0:
        return None
    valid = ~np.isnan(obs_speed)
    if not np.any(valid):
        return None
    speed_clean = np.where(valid, obs_speed, -np.inf)
    peak_idx = np.argmax(speed_clean)
    return pd.Timestamp(obs_time[peak_idx])


def _find_obs_peak_temp_time(obs_temp, obs_time):
    """Find the time of peak observed air temperature. Returns None if no valid data."""
    if obs_temp is None or len(obs_temp) == 0:
        return None
    valid = ~np.isnan(obs_temp)
    if not np.any(valid):
        return None
    temp_clean = np.where(valid, obs_temp, -np.inf)
    peak_idx = np.argmax(temp_clean)
    return pd.Timestamp(obs_time[peak_idx])


# Bay Area fixed axis limits in UTM 10N (km)
BAY_AREA_XLIM = (520, 600)   # Easting [km]
BAY_AREA_YLIM = (4140, 4230)  # Northing [km]


def _load_landboundary(ldb_path):
    """Load Delft3D .ldb land boundary file.

    Returns list of (x_array, y_array) polylines, coordinates in UTM.
    Polylines are separated by 999.999 sentinel values.
    """
    polygons = []
    xs, ys = [], []

    with open(ldb_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('L'):
                continue
            # Header line (point count + columns) — skip lines with integers only
            parts = line.split()
            if len(parts) == 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                # Sentinel check
                if abs(x - 999.999) < 0.01 and abs(y - 999.999) < 0.01:
                    if len(xs) > 1:
                        polygons.append((np.array(xs), np.array(ys)))
                    xs, ys = [], []
                else:
                    xs.append(x)
                    ys.append(y)

    # Flush last segment
    if len(xs) > 1:
        polygons.append((np.array(xs), np.array(ys)))

    return polygons


def _plot_landboundary(ax, ldb_polygons):
    """Plot land boundary polylines on an axis (coordinates in km)."""
    for xp, yp in ldb_polygons:
        ax.plot(xp / 1000, yp / 1000, 'w-', lw=0.8, alpha=0.9)


def _bay_area_extent(x1d, y1d):
    """Return (x_min, x_max, y_min, y_max) clipped to Bay Area."""
    ba_xmin, ba_xmax = BAY_AREA_XLIM[0] * 1000, BAY_AREA_XLIM[1] * 1000
    ba_ymin, ba_ymax = BAY_AREA_YLIM[0] * 1000, BAY_AREA_YLIM[1] * 1000
    x_min = max(x1d.min(), ba_xmin)
    x_max = min(x1d.max(), ba_xmax)
    y_min = max(y1d.min(), ba_ymin)
    y_max = min(y1d.max(), ba_ymax)
    return x_min, x_max, y_min, y_max


def plot_spatial_peak_wind(model_data, model_cfg, obs, obs_peak_time,
                           station_id, model_name, output_dir,
                           ldb_polygons=None):
    """Pcolormesh of max wind speed over 24h around obs peak, with vectors."""
    t_start = obs_peak_time - pd.Timedelta(hours=12)
    t_end = obs_peak_time + pd.Timedelta(hours=12)

    model_time = model_data['time']
    time_mask = (model_time >= t_start) & (model_time <= t_end)
    if time_mask.sum() == 0:
        print(f"      No model data in 24h peak window for spatial wind plot.")
        return

    tidx = np.where(time_mask)[0]
    x1d = model_data['x1d']
    y1d = model_data['y1d']

    # Load u and v for the 24h window (full spatial grid)
    u_slab = model_data['ds_u'][model_cfg['u_var']][tidx, :, :].values.astype(float)
    v_slab = model_data['ds_v'][model_cfg['v_var']][tidx, :, :].values.astype(float)
    u_slab[np.abs(u_slab) > 1e30] = np.nan
    v_slab[np.abs(v_slab) > 1e30] = np.nan

    speed_slab = np.sqrt(u_slab**2 + v_slab**2)

    # Max speed per grid cell across the 24h window
    max_speed = np.nanmax(speed_slab, axis=0)

    # u/v at the time of max speed per grid cell
    tmax_idx = np.nanargmax(speed_slab, axis=0)
    ny, nx = max_speed.shape
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    u_at_max = u_slab[tmax_idx, jj, ii]
    v_at_max = v_slab[tmax_idx, jj, ii]

    # Clip to Bay Area extent
    xmin, xmax, ymin, ymax = _bay_area_extent(x1d, y1d)
    ix_mask = (x1d >= xmin) & (x1d <= xmax)
    iy_mask = (y1d >= ymin) & (y1d <= ymax)

    if ix_mask.sum() < 2 or iy_mask.sum() < 2:
        ix_mask = np.ones(len(x1d), dtype=bool)
        iy_mask = np.ones(len(y1d), dtype=bool)

    x_sub = x1d[ix_mask]
    y_sub = y1d[iy_mask]
    iy_sl = np.where(iy_mask)[0]
    ix_sl = np.where(ix_mask)[0]
    max_speed_sub = max_speed[np.ix_(iy_sl, ix_sl)]
    u_sub = u_at_max[np.ix_(iy_sl, ix_sl)]
    v_sub = v_at_max[np.ix_(iy_sl, ix_sl)]

    X, Y = np.meshgrid(x_sub, y_sub)

    # Interpolate max speed at station for color-coding
    obs_spd = float(np.interp(obs['x_utm'], x_sub, max_speed_sub[
        np.argmin(np.abs(y_sub - obs['y_utm'])), :]))

    fig, ax = plt.subplots(figsize=(10, 10))
    pcm = ax.pcolormesh(X / 1000, Y / 1000, max_speed_sub, shading='auto',
                        cmap='viridis', vmin=0, vmax=25)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Max Wind Speed [m/s]', fontsize=11)

    # Land boundary
    if ldb_polygons is not None:
        _plot_landboundary(ax, ldb_polygons)

    # Quiver vectors (subsample for readability)
    n_arrows_target = 18
    step_x = max(1, len(x_sub) // n_arrows_target)
    step_y = max(1, len(y_sub) // n_arrows_target)
    qx = X[::step_y, ::step_x] / 1000
    qy = Y[::step_y, ::step_x] / 1000
    qu = u_sub[::step_y, ::step_x]
    qv = v_sub[::step_y, ::step_x]
    ax.quiver(qx, qy, qu, qv, color='k', alpha=0.7, scale=None,
              width=0.003, headwidth=3.5)

    # Station location (color-coded by local max speed)
    sc = ax.scatter(obs['x_utm'] / 1000, obs['y_utm'] / 1000,
                    s=150, c=obs_spd, cmap='viridis', vmin=0, vmax=25,
                    edgecolors='white', linewidths=2.5, zorder=5,
                    label=station_id)
    ax.legend(loc='upper right', fontsize=10)

    ax.set_xlabel('Easting [km]', fontsize=11)
    ax.set_ylabel('Northing [km]', fontsize=11)
    ax.set_title(f'{model_name} — Max Wind Speed (24h around obs peak)\n'
                 f'{station_id}: {t_start:%Y-%m-%d %H:%M} – {t_end:%Y-%m-%d %H:%M}',
                 fontsize=12)
    ax.set_xlim(BAY_AREA_XLIM)
    ax.set_ylim(BAY_AREA_YLIM)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_dir / f"{station_id}_spatial_peak_wind_{model_name}.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


def plot_spatial_peak_temperature(model_data, model_cfg, obs, obs_peak_time,
                                  station_id, model_name, output_dir,
                                  ldb_polygons=None):
    """Pcolormesh of air temperature at the moment of peak obs temperature."""
    model_time = model_data['time']
    x1d = model_data['x1d']
    y1d = model_data['y1d']

    # Find closest model timestep to the observed peak temperature moment
    time_diffs = np.abs(model_time - obs_peak_time)
    nearest_idx = int(np.argmin(time_diffs))
    snap_time = model_time[nearest_idx]

    if abs((snap_time - obs_peak_time).total_seconds()) > 3 * 3600:
        print(f"      Nearest model timestep is >3h from obs peak temp, skipping.")
        return

    # Load temperature at the single peak timestep (full spatial grid)
    temp_snap = model_data['ds_temp'][model_cfg['temp_var']][nearest_idx, :, :].values.astype(float)
    temp_snap[np.abs(temp_snap) > 1e30] = np.nan
    temp_snap_c = temp_snap - 273.15  # K → °C

    # Clip to Bay Area extent
    xmin, xmax, ymin, ymax = _bay_area_extent(x1d, y1d)
    ix_mask = (x1d >= xmin) & (x1d <= xmax)
    iy_mask = (y1d >= ymin) & (y1d <= ymax)

    if ix_mask.sum() < 2 or iy_mask.sum() < 2:
        ix_mask = np.ones(len(x1d), dtype=bool)
        iy_mask = np.ones(len(y1d), dtype=bool)

    x_sub = x1d[ix_mask]
    y_sub = y1d[iy_mask]
    iy_sl = np.where(iy_mask)[0]
    ix_sl = np.where(ix_mask)[0]
    temp_sub = temp_snap_c[np.ix_(iy_sl, ix_sl)]

    X, Y = np.meshgrid(x_sub, y_sub)

    # Interpolate temperature at station for color-coding
    obs_temp = float(np.interp(obs['x_utm'], x_sub, temp_sub[
        np.argmin(np.abs(y_sub - obs['y_utm'])), :]))

    fig, ax = plt.subplots(figsize=(10, 10))
    pcm = ax.pcolormesh(X / 1000, Y / 1000, temp_sub, shading='auto',
                        cmap='RdYlBu_r', vmin=0, vmax=40)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Air Temperature [°C]', fontsize=11)

    # Land boundary
    if ldb_polygons is not None:
        _plot_landboundary(ax, ldb_polygons)

    # Station location (color-coded by local temperature)
    sc = ax.scatter(obs['x_utm'] / 1000, obs['y_utm'] / 1000,
                    s=150, c=obs_temp, cmap='RdYlBu_r', vmin=0, vmax=40,
                    edgecolors='white', linewidths=2.5, zorder=5,
                    label=station_id)
    ax.legend(loc='upper right', fontsize=10)

    ax.set_xlabel('Easting [km]', fontsize=11)
    ax.set_ylabel('Northing [km]', fontsize=11)
    ax.set_title(f'{model_name} — Air Temperature at obs peak\n'
                 f'{station_id}: {snap_time:%Y-%m-%d %H:%M}',
                 fontsize=12)
    ax.set_xlim(BAY_AREA_XLIM)
    ax.set_ylim(BAY_AREA_YLIM)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_dir / f"{station_id}_spatial_peak_temperature_{model_name}.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


# Per-variable display for the generic UTM scalar spatial-peak maps.
# vlims=None -> robust 2nd/98th-percentile stretch; (lo, None) -> lo fixed, hi=98th pct.
SPATIAL_SCALAR_DISPLAY = {
    'temperature':   dict(cmap='RdYlBu_r', clabel='Air Temperature [°C]',  vlims=(0, 40)),
    'pressure':      dict(cmap='cividis',  clabel='Air Pressure [hPa]',         vlims=None),
    'dewpoint':      dict(cmap='YlGnBu',   clabel='Dew Point [°C]',         vlims=None),
    'rh':            dict(cmap='BrBG',     clabel='Relative Humidity [%]',       vlims=(0, 100)),
    'precipitation': dict(cmap='Blues',    clabel='Precipitation [mm/hr]',       vlims=(0, None)),
    'radiation':     dict(cmap='inferno',  clabel='Solar Radiation [W/m2]',      vlims=(0, None)),
}


def _scalar_field_snapshot(model_data, var_key, obs_peak_time):
    """(2-D field, snap_time) for var_key at the model step nearest obs_peak_time,
    read from ds_scalars using each scalar's OWN time axis (scalars are not windowed
    to the wind time axis). Applies K->C / Pa->hPa; derives RH from T + Td fields.
    Returns (None, None) if unavailable or >3 h from the peak."""
    ds_scalars = model_data.get('ds_scalars') or {}

    def _snap(info):
        t = _decode_time(info['ds'])
        idx = int(np.argmin(np.abs(t - obs_peak_time)))
        if abs((pd.Timestamp(t[idx]) - obs_peak_time).total_seconds()) > 3 * 3600:
            return None, None
        fld = info['ds'][info['var']][idx, :, :].values.astype(float)
        fld[np.abs(fld) > 1e30] = np.nan
        u = (info.get('units') or '').lower().replace(' ', '')
        if u in ('k', 'kelvin'):
            fld = fld - 273.15
        elif u == 'pa':
            fld = fld / 100.0
        return fld, pd.Timestamp(t[idx])

    if var_key == 'rh':
        t_info, d_info = ds_scalars.get('temperature'), ds_scalars.get('dewpoint')
        if not (t_info and 'ds' in t_info and d_info and 'ds' in d_info):
            return None, None
        tf, ts = _snap(t_info)
        df, _ = _snap(d_info)
        if tf is None or df is None or tf.shape != df.shape:
            return None, None
        return _rh_from_dewpoint(tf, df), ts
    info = ds_scalars.get(var_key)
    if not (info and 'ds' in info):
        return None, None
    return _snap(info)


def plot_spatial_peak_scalar(model_data, model_cfg, obs, obs_peak_time,
                             station_id, model_name, output_dir, var_key,
                             ldb_polygons=None):
    """Pcolormesh of a scalar field (temperature/pressure/dewpoint/rh/precip/radiation)
    at the model timestep nearest the observed peak of that variable. UTM-10N grids."""
    x1d, y1d = model_data.get('x1d'), model_data.get('y1d')
    if x1d is None or y1d is None:
        return
    field, snap_time = _scalar_field_snapshot(model_data, var_key, obs_peak_time)
    if field is None or field.shape != (len(y1d), len(x1d)) or not np.isfinite(field).any():
        return

    xmin, xmax, ymin, ymax = _bay_area_extent(x1d, y1d)
    ix = (x1d >= xmin) & (x1d <= xmax)
    iy = (y1d >= ymin) & (y1d <= ymax)
    if ix.sum() < 2 or iy.sum() < 2:
        ix = np.ones(len(x1d), dtype=bool)
        iy = np.ones(len(y1d), dtype=bool)
    x_sub, y_sub = x1d[ix], y1d[iy]
    fsub = field[np.ix_(np.where(iy)[0], np.where(ix)[0])]
    X, Y = np.meshgrid(x_sub, y_sub)

    disp = SPATIAL_SCALAR_DISPLAY.get(var_key,
                                      dict(cmap='viridis', clabel=var_key, vlims=None))
    finite = fsub[np.isfinite(fsub)]
    if disp['vlims'] is None:
        vmin, vmax = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) \
            if finite.size else (None, None)
    else:
        vmin, vmax = disp['vlims']
        if vmax is None:
            vmax = float(np.percentile(finite, 98)) if finite.size else None

    fig, ax = plt.subplots(figsize=(10, 10))
    pcm = ax.pcolormesh(X / 1000, Y / 1000, fsub, shading='auto',
                        cmap=disp['cmap'], vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(disp['clabel'], fontsize=11)
    if ldb_polygons is not None:
        _plot_landboundary(ax, ldb_polygons)

    row = int(np.argmin(np.abs(y_sub - obs['y_utm'])))
    local = float(np.interp(obs['x_utm'], x_sub, fsub[row, :]))
    ax.scatter(obs['x_utm'] / 1000, obs['y_utm'] / 1000, s=150, c=local,
               cmap=disp['cmap'], vmin=vmin, vmax=vmax, edgecolors='white',
               linewidths=2.5, zorder=5, label=station_id)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('Easting [km]', fontsize=11)
    ax.set_ylabel('Northing [km]', fontsize=11)
    ax.set_title(f'{model_name} — {disp["clabel"]} at obs peak\n'
                 f'{station_id}: {snap_time:%Y-%m-%d %H:%M}', fontsize=12)
    ax.set_xlim(BAY_AREA_XLIM)
    ax.set_ylim(BAY_AREA_YLIM)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_dir / f"{station_id}_spatial_peak_{var_key}_{model_name}.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


def plot_spatial_peak_wind_lcc(model_data, model_cfg, obs, obs_peak_time,
                               station_id, model_name, output_dir,
                               ldb_polygons=None):
    """Spatial max wind speed for LCC grid models (e.g. UCLA).

    Converts LCC coordinates to UTM 10N for plotting on same Bay Area extent.
    """
    t_start = obs_peak_time - pd.Timedelta(hours=12)
    t_end = obs_peak_time + pd.Timedelta(hours=12)

    model_time = model_data['time']
    time_mask = (model_time >= t_start) & (model_time <= t_end)
    if time_mask.sum() == 0:
        print(f"      No model data in 24h peak window for spatial wind plot.")
        return

    tidx = np.where(time_mask)[0]
    x1d_lcc = model_data['x1d']
    y1d_lcc = model_data['y1d']

    # Convert LCC grid to UTM 10N
    X_lcc, Y_lcc = np.meshgrid(x1d_lcc, y1d_lcc)
    X_utm, Y_utm = lcc_to_utm10(X_lcc, Y_lcc, model_data['lcc_attrs'])

    # Load u and v for the 24h window
    u_slab = model_data['ds_u'][model_cfg['u_var']][tidx, :, :].values.astype(float)
    v_slab = model_data['ds_v'][model_cfg['v_var']][tidx, :, :].values.astype(float)
    u_slab[np.abs(u_slab) > 1e30] = np.nan
    v_slab[np.abs(v_slab) > 1e30] = np.nan

    speed_slab = np.sqrt(u_slab**2 + v_slab**2)
    max_speed = np.nanmax(speed_slab, axis=0)

    # u/v at time of max speed per cell
    tmax_idx = np.nanargmax(speed_slab, axis=0)
    ny, nx = max_speed.shape
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    u_at_max = u_slab[tmax_idx, jj, ii]
    v_at_max = v_slab[tmax_idx, jj, ii]

    # Clip to Bay Area in UTM (km)
    ba_xmin, ba_xmax = BAY_AREA_XLIM[0] * 1000, BAY_AREA_XLIM[1] * 1000
    ba_ymin, ba_ymax = BAY_AREA_YLIM[0] * 1000, BAY_AREA_YLIM[1] * 1000
    in_view = ((X_utm >= ba_xmin) & (X_utm <= ba_xmax) &
               (Y_utm >= ba_ymin) & (Y_utm <= ba_ymax))

    if in_view.sum() < 4:
        print(f"      Model grid does not cover Bay Area extent, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    pcm = ax.pcolormesh(X_utm / 1000, Y_utm / 1000, max_speed, shading='auto',
                        cmap='viridis', vmin=0, vmax=25)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Max Wind Speed [m/s]', fontsize=11)

    if ldb_polygons is not None:
        _plot_landboundary(ax, ldb_polygons)

    # Quiver vectors (subsample)
    n_arrows_target = 18
    step_x = max(1, nx // n_arrows_target)
    step_y = max(1, ny // n_arrows_target)
    ax.quiver(X_utm[::step_y, ::step_x] / 1000, Y_utm[::step_y, ::step_x] / 1000,
              u_at_max[::step_y, ::step_x], v_at_max[::step_y, ::step_x],
              color='k', alpha=0.7, scale=None, width=0.003, headwidth=3.5)

    # Station marker
    obs_spd_val = max_speed[
        np.argmin(np.abs(y1d_lcc - model_data['y1d'].mean())),
        np.argmin(np.abs(x1d_lcc - model_data['x1d'].mean()))]
    ax.scatter(obs['x_utm'] / 1000, obs['y_utm'] / 1000,
               s=150, c='red', edgecolors='white', linewidths=2.5, zorder=5,
               label=station_id)
    ax.legend(loc='upper right', fontsize=10)

    ax.set_xlabel('Easting [km]', fontsize=11)
    ax.set_ylabel('Northing [km]', fontsize=11)
    ax.set_title(f'{model_name} — Max Wind Speed (24h around obs peak)\n'
                 f'{station_id}: {t_start:%Y-%m-%d %H:%M} – {t_end:%Y-%m-%d %H:%M}',
                 fontsize=12)
    ax.set_xlim(BAY_AREA_XLIM)
    ax.set_ylim(BAY_AREA_YLIM)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_dir / f"{station_id}_spatial_peak_wind_{model_name}.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


def plot_spatial_peak_temperature_lcc(model_data, model_cfg, obs, obs_peak_time,
                                      station_id, model_name, output_dir,
                                      ldb_polygons=None):
    """Spatial temperature at obs peak moment for LCC grid models (e.g. UCLA)."""
    model_time = model_data['time']
    x1d_lcc = model_data['x1d']
    y1d_lcc = model_data['y1d']

    # Find closest model timestep
    time_diffs = np.abs(model_time - obs_peak_time)
    nearest_idx = int(np.argmin(time_diffs))
    snap_time = model_time[nearest_idx]

    if abs((snap_time - obs_peak_time).total_seconds()) > 3 * 3600:
        print(f"      Nearest model timestep is >3h from obs peak temp, skipping.")
        return

    # Convert LCC grid to UTM
    X_lcc, Y_lcc = np.meshgrid(x1d_lcc, y1d_lcc)
    X_utm, Y_utm = lcc_to_utm10(X_lcc, Y_lcc, model_data['lcc_attrs'])

    temp_snap = model_data['ds_temp'][model_cfg['temp_var']][nearest_idx, :, :].values.astype(float)
    temp_snap[np.abs(temp_snap) > 1e30] = np.nan
    temp_snap_c = temp_snap - 273.15

    fig, ax = plt.subplots(figsize=(10, 10))
    pcm = ax.pcolormesh(X_utm / 1000, Y_utm / 1000, temp_snap_c, shading='auto',
                        cmap='RdYlBu_r', vmin=0, vmax=40)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Air Temperature [°C]', fontsize=11)

    if ldb_polygons is not None:
        _plot_landboundary(ax, ldb_polygons)

    ax.scatter(obs['x_utm'] / 1000, obs['y_utm'] / 1000,
               s=150, c='red', edgecolors='white', linewidths=2.5, zorder=5,
               label=station_id)
    ax.legend(loc='upper right', fontsize=10)

    ax.set_xlabel('Easting [km]', fontsize=11)
    ax.set_ylabel('Northing [km]', fontsize=11)
    ax.set_title(f'{model_name} — Air Temperature at obs peak\n'
                 f'{station_id}: {snap_time:%Y-%m-%d %H:%M}',
                 fontsize=12)
    ax.set_xlim(BAY_AREA_XLIM)
    ax.set_ylim(BAY_AREA_YLIM)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_dir / f"{station_id}_spatial_peak_temperature_{model_name}.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


def plot_spatial_peak_wind_latlon2d(model_data, model_cfg, obs, obs_peak_time,
                                    station_id, model_name, output_dir,
                                    ldb_polygons=None):
    """Spatial max wind speed for curvilinear 2D lat/lon grid (e.g. WRF_CalNev).

    Speed-only model: pcolormesh without vectors.
    Converts lat/lon to UTM for same Bay Area plotting extent.
    """
    t_start = obs_peak_time - pd.Timedelta(hours=12)
    t_end = obs_peak_time + pd.Timedelta(hours=12)

    model_time = model_data['time']
    time_mask = (model_time >= t_start) & (model_time <= t_end)
    if time_mask.sum() == 0:
        print(f"      No model data in 24h peak window for spatial wind plot.")
        return

    tidx = np.where(time_mask)[0]
    lat2d = model_data['lat2d']
    lon2d = model_data['lon2d']

    # Convert to UTM
    X_utm, Y_utm = latlon_to_utm10_array(lat2d, lon2d)

    has_uv = model_cfg.get('has_uv', True)
    if has_uv and model_data.get('ds_u') is not None:
        u_slab = model_data['ds_u'][model_cfg['u_var']][tidx, :, :].values.astype(float)
        v_slab = model_data['ds_v'][model_cfg['v_var']][tidx, :, :].values.astype(float)
        u_slab[np.abs(u_slab) > 1e30] = np.nan
        v_slab[np.abs(v_slab) > 1e30] = np.nan
        speed_slab = np.sqrt(u_slab**2 + v_slab**2)
    else:
        # Speed-only model
        speed_slab = model_data['ds_speed'][model_cfg['speed_var']][tidx, :, :].values.astype(float)
        speed_slab[np.abs(speed_slab) > 1e30] = np.nan

    max_speed = np.nanmax(speed_slab, axis=0)

    # Check coverage
    ba_xmin, ba_xmax = BAY_AREA_XLIM[0] * 1000, BAY_AREA_XLIM[1] * 1000
    ba_ymin, ba_ymax = BAY_AREA_YLIM[0] * 1000, BAY_AREA_YLIM[1] * 1000
    in_view = ((X_utm >= ba_xmin) & (X_utm <= ba_xmax) &
               (Y_utm >= ba_ymin) & (Y_utm <= ba_ymax))
    if in_view.sum() < 4:
        print(f"      Model grid does not cover Bay Area extent, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    pcm = ax.pcolormesh(X_utm / 1000, Y_utm / 1000, max_speed, shading='auto',
                        cmap='viridis', vmin=0, vmax=25)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Max Wind Speed [m/s]', fontsize=11)

    if ldb_polygons is not None:
        _plot_landboundary(ax, ldb_polygons)

    # Quiver vectors only if u/v available
    if has_uv and model_data.get('ds_u') is not None:
        tmax_idx = np.nanargmax(speed_slab, axis=0)
        ny, nx = max_speed.shape
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        u_at_max = u_slab[tmax_idx, jj, ii]
        v_at_max = v_slab[tmax_idx, jj, ii]

        n_arrows = 18
        sx = max(1, nx // n_arrows)
        sy = max(1, ny // n_arrows)
        ax.quiver(X_utm[::sy, ::sx] / 1000, Y_utm[::sy, ::sx] / 1000,
                  u_at_max[::sy, ::sx], v_at_max[::sy, ::sx],
                  color='k', alpha=0.7, scale=None, width=0.003, headwidth=3.5)

    ax.scatter(obs['x_utm'] / 1000, obs['y_utm'] / 1000,
               s=150, c='red', edgecolors='white', linewidths=2.5, zorder=5,
               label=station_id)
    ax.legend(loc='upper right', fontsize=10)

    ax.set_xlabel('Easting [km]', fontsize=11)
    ax.set_ylabel('Northing [km]', fontsize=11)
    ax.set_title(f'{model_name} — Max Wind Speed (24h around obs peak)\n'
                 f'{station_id}: {t_start:%Y-%m-%d %H:%M} – {t_end:%Y-%m-%d %H:%M}',
                 fontsize=12)
    ax.set_xlim(BAY_AREA_XLIM)
    ax.set_ylim(BAY_AREA_YLIM)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_dir / f"{station_id}_spatial_peak_wind_{model_name}.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


def plot_spatial_peak_temperature_latlon2d(model_data, model_cfg, obs, obs_peak_time,
                                           station_id, model_name, output_dir,
                                           ldb_polygons=None):
    """Spatial temperature at obs peak for curvilinear 2D lat/lon grid (e.g. WRF_CalNev)."""
    model_time = model_data['time']
    lat2d = model_data['lat2d']
    lon2d = model_data['lon2d']

    # Find closest model timestep
    time_diffs = np.abs(model_time - obs_peak_time)
    nearest_idx = int(np.argmin(time_diffs))
    snap_time = model_time[nearest_idx]

    if abs((snap_time - obs_peak_time).total_seconds()) > 3 * 3600:
        print(f"      Nearest model timestep is >3h from obs peak temp, skipping.")
        return

    X_utm, Y_utm = latlon_to_utm10_array(lat2d, lon2d)

    temp_snap = model_data['ds_temp'][model_cfg['temp_var']][nearest_idx, :, :].values.astype(float)
    temp_snap[np.abs(temp_snap) > 1e30] = np.nan
    temp_snap_c = temp_snap - 273.15

    # Check coverage
    ba_xmin, ba_xmax = BAY_AREA_XLIM[0] * 1000, BAY_AREA_XLIM[1] * 1000
    ba_ymin, ba_ymax = BAY_AREA_YLIM[0] * 1000, BAY_AREA_YLIM[1] * 1000
    in_view = ((X_utm >= ba_xmin) & (X_utm <= ba_xmax) &
               (Y_utm >= ba_ymin) & (Y_utm <= ba_ymax))
    if in_view.sum() < 4:
        print(f"      Model grid does not cover Bay Area extent, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    pcm = ax.pcolormesh(X_utm / 1000, Y_utm / 1000, temp_snap_c, shading='auto',
                        cmap='RdYlBu_r', vmin=0, vmax=40)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Air Temperature [°C]', fontsize=11)

    if ldb_polygons is not None:
        _plot_landboundary(ax, ldb_polygons)

    ax.scatter(obs['x_utm'] / 1000, obs['y_utm'] / 1000,
               s=150, c='red', edgecolors='white', linewidths=2.5, zorder=5,
               label=station_id)
    ax.legend(loc='upper right', fontsize=10)

    ax.set_xlabel('Easting [km]', fontsize=11)
    ax.set_ylabel('Northing [km]', fontsize=11)
    ax.set_title(f'{model_name} — Air Temperature at obs peak\n'
                 f'{station_id}: {snap_time:%Y-%m-%d %H:%M}',
                 fontsize=12)
    ax.set_xlim(BAY_AREA_XLIM)
    ax.set_ylim(BAY_AREA_YLIM)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_dir / f"{station_id}_spatial_peak_temperature_{model_name}.png"
    plt.savefig(fname, dpi=FIGURE_DPI, pil_kwargs={'compress_level': 1})
    plt.close('all')
    print(f"      Saved: {fname.name}")


def plot_taylor_diagram(all_records, output_dir):
    """Taylor diagram per variable — one plot per unique variable in all_records."""
    records = [r for r in all_records if not np.isnan(r.get('corr', np.nan))
               and not np.isnan(r.get('obs_std', np.nan))
               and r.get('obs_std', 0) > 0]
    if not records:
        print("  No valid records for Taylor diagram.")
        return

    model_colors = model_color_map({r['model'] for r in records})
    station_markers = ['o', 's', '^', 'D', 'P', 'v', '<', '>', 'h', '*']

    variables = sorted({r['variable'] for r in records})

    for var in variables:
        var_recs = [r for r in records if r['variable'] == var]
        if not var_recs:
            continue

        stations = sorted({r['station'] for r in var_recs})
        stn_marker = {s: station_markers[i % len(station_markers)]
                      for i, s in enumerate(stations)}

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)

        for rec in var_recs:
            corr = rec['corr']
            norm_std = rec['model_std'] / rec['obs_std']
            theta = np.arccos(np.clip(corr, -1, 1))
            color = model_colors.get(rec['model'], 'gray')
            marker = stn_marker.get(rec['station'], 'o')
            ax.plot(theta, norm_std, marker=marker, color=color, ms=9, alpha=0.85,
                    mec='white', mew=0.5)

        # Reference point
        ax.plot(0, 1.0, 'k*', ms=15, zorder=5)
        ax.annotate('perfect', xy=(0, 1.0), xytext=(0.05, 1.08), fontsize=8)

        # Build legend: models by colour, stations by marker
        handles = [Line2D([0], [0], marker='*', color='k', ms=12, label='Perfect',
                          linestyle='None')]
        for m, c in model_colors.items():
            if any(r['model'] == m for r in var_recs):
                handles.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=c, ms=10, label=m,
                                      linestyle='None'))
        handles.append(Line2D([], [], color='none'))  # spacer
        for s in stations:
            mk = stn_marker[s]
            handles.append(Line2D([0], [0], marker=mk, color='w',
                                  markerfacecolor='gray', ms=9, label=s,
                                  linestyle='None'))

        ax.legend(handles=handles, loc='upper right',
                  bbox_to_anchor=(1.45, 1.05), fontsize=9, framealpha=0.9)
        ax.set_title(f'Taylor Diagram — {var}', fontsize=13, pad=30)

        plt.tight_layout()
        var_safe = _safe_filename(var)
        fname = output_dir / f"taylor_diagram_{var_safe}.png"
        plt.savefig(fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close('all')
        print(f"  Saved: {fname.name}")


def plot_multi_model_comparison(all_records, output_dir):
    """Per-variable figure: RMSE/bias by station + mean skill heatmap across stations."""
    df = pd.DataFrame(all_records)
    if df.empty:
        return

    model_colors = model_color_map(df['model'].unique())

    # Metrics shown in the heatmap, with display label and whether lower=better
    HEATMAP_METRICS = [
        ('rmse',          'RMSE',      True),
        ('mae',           'MAE',       True),
        ('bias',          'Bias',      False),
        ('corr',          'R',         False),
        ('r2',            'R²',        False),
        ('nrmse',         'NRMSE',     True),
        ('scatter_index', 'SI',        True),
        ('rel_bias',      'Rel Bias',  False),
        ('skill',         'Skill',     False),
    ]

    variables = df['variable'].unique()

    for var in variables:
        dfv = df[df['variable'] == var]
        if dfv.empty:
            continue

        # Skip "(top 10%)" entries for the per-station panels — keep them for heatmap
        dfv_full = dfv[~dfv['variable'].str.contains(r'\(top', na=False)]
        stations = sorted(dfv_full['station'].unique())
        models = sorted(dfv_full['model'].unique())
        n_stations = len(stations)
        n_models = len(models)
        bar_w = 0.8 / max(n_models, 1)

        fig = plt.figure(figsize=(max(12, n_stations * 1.8), 14))
        gs = fig.add_gridspec(3, 1, hspace=0.45, height_ratios=[1, 1, 1.2])
        ax_rmse = fig.add_subplot(gs[0])
        ax_bias = fig.add_subplot(gs[1], sharex=ax_rmse)
        ax_heat = fig.add_subplot(gs[2])

        # --- RMSE panel ---
        for j, model in enumerate(models):
            vals = [dfv_full[(dfv_full['station'] == s) & (dfv_full['model'] == model)]['rmse'].values
                    for s in stations]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            x = np.arange(n_stations) + j * bar_w
            ax_rmse.bar(x, vals, bar_w, label=model,
                        color=model_colors.get(model, 'gray'), alpha=0.85)
        ax_rmse.set_ylabel('RMSE', fontsize=11)
        ax_rmse.set_xticks(np.arange(n_stations) + bar_w * (n_models - 1) / 2)
        ax_rmse.set_xticklabels(stations, rotation=45, ha='right', fontsize=9)
        ax_rmse.legend(fontsize=9, ncol=max(1, min(n_models, 4)))
        ax_rmse.grid(True, alpha=0.3, axis='y')

        # --- Bias panel ---
        for j, model in enumerate(models):
            vals = [dfv_full[(dfv_full['station'] == s) & (dfv_full['model'] == model)]['bias'].values
                    for s in stations]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            x = np.arange(n_stations) + j * bar_w
            ax_bias.bar(x, vals, bar_w, label=model,
                        color=model_colors.get(model, 'gray'), alpha=0.85)
        ax_bias.set_ylabel('Bias', fontsize=11)
        ax_bias.set_xticks(np.arange(n_stations) + bar_w * (n_models - 1) / 2)
        ax_bias.set_xticklabels(stations, rotation=45, ha='right', fontsize=9)
        ax_bias.axhline(0, color='black', ls='--', lw=0.8)
        ax_bias.grid(True, alpha=0.3, axis='y')

        # --- Mean metrics heatmap (models × metrics, averaged across stations) ---
        metric_keys = [m[0] for m in HEATMAP_METRICS]
        metric_labels = [m[1] for m in HEATMAP_METRICS]
        lower_better = [m[2] for m in HEATMAP_METRICS]

        heat_data = np.full((len(models), len(metric_keys)), np.nan)
        for i, model in enumerate(models):
            dm = dfv_full[dfv_full['model'] == model]
            for k, key in enumerate(metric_keys):
                if key in dm.columns:
                    vals = dm[key].dropna().values
                    if len(vals) > 0:
                        heat_data[i, k] = float(np.mean(vals))

        # Normalise each metric column to [0,1] for colour — best=yellow, worst=purple
        heat_norm = np.full_like(heat_data, np.nan)
        for k in range(len(metric_keys)):
            col = heat_data[:, k]
            finite = col[np.isfinite(col)]
            if len(finite) < 2:
                heat_norm[:, k] = 0.5
                continue
            cmin, cmax = finite.min(), finite.max()
            if cmax == cmin:
                heat_norm[:, k] = 0.5
                continue
            normed = (col - cmin) / (cmax - cmin)
            # For lower-is-better metrics, invert so yellow = best
            heat_norm[:, k] = (1 - normed) if lower_better[k] else normed

        im = ax_heat.imshow(heat_norm, aspect='auto', cmap='RdYlGn',
                            vmin=0, vmax=1, interpolation='nearest')
        ax_heat.set_xticks(range(len(metric_keys)))
        ax_heat.set_xticklabels(metric_labels, fontsize=10)
        ax_heat.set_yticks(range(len(models)))
        ax_heat.set_yticklabels(models, fontsize=10)
        ax_heat.set_title('Mean across stations  (green = best)', fontsize=10)

        # Annotate cells with actual values
        for i in range(len(models)):
            for k in range(len(metric_keys)):
                val = heat_data[i, k]
                if np.isfinite(val):
                    txt = f'{val:.3f}' if abs(val) < 100 else f'{val:.1f}'
                    brightness = heat_norm[i, k] if np.isfinite(heat_norm[i, k]) else 0.5
                    txt_color = 'black' if 0.25 < brightness < 0.85 else 'white'
                    ax_heat.text(k, i, txt, ha='center', va='center',
                                 fontsize=8, color=txt_color)

        fig.suptitle(f'Model Comparison: {var}', fontsize=13, y=1.01)
        var_safe = _safe_filename(var)
        fname = output_dir / f"multi_model_{var_safe}.png"
        plt.savefig(fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close('all')
        print(f"  Saved: {fname.name}")


# ===========================================================================
# Main validation workflow
# ===========================================================================

# Physical value bounds for QC (from preprocessing.yaml, converted to plot units)
# Wind u/v: -100 to 100 m/s  →  speed: 0 to 100 m/s
# Temperature: 220-330 K  →  -53.15 to 56.85 °C
# === QC CONFIGURATION ===
# Per-reading physical bounds: out-of-range wind values -> NaN (sensor spikes;
# the IEM archive carries spurious values up to ~75 m/s). SF-Bay realistic max
# is ~25-30 m/s, so 50 m/s is a safe "obviously bad" ceiling.
WIND_SPEED_MAX = 50.0
PHYSICAL_BOUNDS = {
    'Wind Speed [m/s]':        (0, WIND_SPEED_MAX),
    'Wind U10 [m/s]':          (-WIND_SPEED_MAX, WIND_SPEED_MAX),
    'Wind V10 [m/s]':          (-WIND_SPEED_MAX, WIND_SPEED_MAX),
    'Wind Direction [deg]':    (0, 360),
    'Air Temperature [C]':     (-53.15, 56.85),   # 220-330 K in °C
    'Air Pressure [hPa]':      (900, 1100),
    'Dew Point [C]':           (-53.15, 56.85),
    'Relative Humidity [%]':   (0, 100),
    'Solar Radiation [W/m2]':  (0, 1361),          # up to solar constant
    'Precipitation [mm/hr]':   (0, 200),
}

# --- Scalar-variable registry ----------------------------------------------
# Drives the generalized (non-wind) variable handling. Each entry maps a
# VARIABLES key -> the per-station obs dict key, the CSV/figure label, and the
# physical-bounds label. Wind is handled separately (u/v components + circular
# direction). `is_circular` is False for all of these (linear stats).
SCALAR_VARS = {
    'temperature':   {'obs_key': 'air_temp_C',  'label': 'Air Temperature [C]'},
    'pressure':      {'obs_key': 'pressure_hPa', 'label': 'Air Pressure [hPa]'},
    'dewpoint':      {'obs_key': 'dewpoint_C',   'label': 'Dew Point [C]'},
    'rh':            {'obs_key': 'rh_pct',        'label': 'Relative Humidity [%]'},
    'radiation':     {'obs_key': 'solar_wm2',     'label': 'Solar Radiation [W/m2]'},
    'precipitation': {'obs_key': 'precip_mmhr',   'label': 'Precipitation [mm/hr]'},
}


def _rh_from_dewpoint(temp_c, dew_c):
    """Relative humidity (%) from temperature & dew point (°C), Magnus formula."""
    a, b = 17.625, 243.04
    es = np.exp(a * temp_c / (b + temp_c))
    ed = np.exp(a * dew_c / (b + dew_c))
    rh = 100.0 * ed / es
    return np.clip(rh, 0.0, 100.0)


def _rh_from_q(q, temp_c, pres_hpa):
    """Relative humidity (%) from specific humidity (kg/kg), T (°C), pressure (hPa)."""
    # vapor pressure e = q*p/(0.622 + 0.378 q); saturation es via Magnus (hPa)
    e = q * pres_hpa / (0.622 + 0.378 * q)
    es = 6.112 * np.exp(17.625 * temp_c / (243.04 + temp_c))
    rh = 100.0 * e / es
    return np.clip(rh, 0.0, 100.0)

# Per-station acceptance thresholds. A station is REJECTED (dropped from the
# validation, logged in qc_report.csv) if its QC'd wind speed over the window
# fails ANY of these. Tuned so every IEM/NDBC/USGS station passes (min observed
# mean 1.9 m/s) while the stalled/sheltered CWOP citizen sensors are removed.
QC_STATION_ENABLE   = True
QC_MEAN_MIN         = 0.5    # m/s; mean below this = stalled/sheltered
QC_STD_MIN          = 0.3    # m/s; std below this = flatlined
QC_ZEROFRAC_MAX     = 0.5    # fraction of EXACT-0 readings above this = stuck at zero
QC_MAX_MIN          = 3.0    # m/s; never reaches this = never registers real wind
QC_NMIN             = 100    # minimum valid samples


def qc_station_accept(obs):
    """Decide whether an observation station is trustworthy enough to validate.

    Evaluates QC'd `speed10` over the (already time-filtered) record. Returns
    (ok: bool, reason: str, stats: dict). reason='' when accepted.
    """
    s = np.asarray(obs.get('speed10'), dtype=float)
    s = s[np.isfinite(s)]
    n = int(s.size)
    stats_d = {'n': n, 'mean': np.nan, 'std': np.nan, 'zero_frac': np.nan, 'max': np.nan}
    if n < QC_NMIN:
        return False, f'n<{QC_NMIN} ({n})', stats_d
    mean = float(np.mean(s)); std = float(np.std(s))
    zero_frac = float(np.mean(s == 0.0)); mx = float(np.max(s))
    stats_d.update(mean=mean, std=std, zero_frac=zero_frac, max=mx)
    if not QC_STATION_ENABLE:
        return True, '', stats_d
    if mean < QC_MEAN_MIN:
        return False, f'mean<{QC_MEAN_MIN} ({mean:.2f})', stats_d
    if std < QC_STD_MIN:
        return False, f'std<{QC_STD_MIN} ({std:.2f})', stats_d
    if zero_frac > QC_ZEROFRAC_MAX:
        return False, f'zero_frac>{QC_ZEROFRAC_MAX} ({zero_frac:.2f})', stats_d
    if mx < QC_MAX_MIN:
        return False, f'max<{QC_MAX_MIN} ({mx:.2f})', stats_d
    return True, '', stats_d


def _clip_to_physical_bounds(arr, var_name):
    """Replace values outside physical bounds with NaN."""
    bounds = PHYSICAL_BOUNDS.get(var_name)
    if bounds is None:
        return arr
    lo, hi = bounds
    out = arr.copy()
    bad = (out < lo) | (out > hi)
    n_bad = np.nansum(bad)
    if n_bad > 0:
        out[bad] = np.nan
        print(f"      QC: {int(n_bad)} values outside [{lo}, {hi}] set to NaN for {var_name}")
    return out


def validate_variable(model_arr, obs_arr, model_time, obs_time, var_name,
                      station_id, model_name, output_dir, is_direction=False,
                      make_plots=True):
    """Validate one variable for one model/station pair. Returns stats dict or None.

    make_plots=False -> compute stats only (no figures); used for CWOP stations
    not in the plot sample, so they still populate the statistics CSV."""
    if obs_arr is None or len(obs_arr) == 0 or np.all(np.isnan(obs_arr)):
        return None
    if model_arr is None or len(model_arr) == 0 or np.all(np.isnan(model_arr)):
        return None

    # Apply physical bounds QC before matching
    model_arr = _clip_to_physical_bounds(model_arr, var_name)
    obs_arr = _clip_to_physical_bounds(obs_arr, var_name)

    model_matched, obs_matched, common_time = match_timeseries(
        model_arr, model_time, obs_arr, obs_time)

    if model_matched is None or len(model_matched) < 10:
        return None

    if is_direction:
        st = calculate_circular_statistics(model_matched, obs_matched)
    else:
        st = calculate_statistics(model_matched, obs_matched)
    if st is None:
        return None

    print(f"      RMSE={st['rmse']:.3f}  bias={st['bias']:.3f}  R={st['corr']:.3f}")

    if make_plots:
        # Scatter plot
        plot_scatter(model_matched, obs_matched, var_name, station_id, model_name,
                     st, output_dir)

        # Whole-period timeseries
        if len(common_time) > 5:
            plot_timeseries(model_matched, obs_matched, common_time, var_name,
                            station_id, model_name, st, output_dir,
                            period_label='whole_period')

        # Peak-event timeseries (peak identified from observations)
        if len(common_time) > 50:
            t_p, m_p, o_p = select_timeseries_window(
                common_time, model_matched, obs_matched, mode='peak', days=PEAK_WINDOW_DAYS)
            if len(t_p) > 5:
                st_p = calculate_statistics(m_p, o_p) or st
                plot_timeseries(m_p, o_p, t_p, var_name, station_id, model_name,
                                st_p, output_dir, period_label='peak_event')

    # Top-percentile analysis for wind speed (stats always; plot gated inside)
    st_top = None
    if not is_direction and 'Wind Speed' in var_name:
        st_top = _validate_top_percentile(
            model_matched, obs_matched, common_time,
            var_name, station_id, model_name, output_dir,
            percentile=90, make_plots=make_plots)

    return st, st_top


TOP_PERCENTILE_LABEL = 'top10pct'


def _validate_top_percentile(model_matched, obs_matched, common_time,
                              var_name, station_id, model_name, output_dir,
                              percentile=90, make_plots=True):
    """Separate scatter + metrics for the top percentile of observed values."""
    mask_valid = ~(np.isnan(model_matched) | np.isnan(obs_matched))
    if mask_valid.sum() < 20:
        return

    obs_clean = np.where(mask_valid, obs_matched, np.nan)
    threshold = np.nanpercentile(obs_clean, percentile)
    top_mask = mask_valid & (obs_matched >= threshold)
    n_top = top_mask.sum()

    if n_top < 10:
        print(f"      Top {100 - percentile}%: only {n_top} points, skipping.")
        return None

    m_top = model_matched[top_mask]
    o_top = obs_matched[top_mask]

    st_top = calculate_statistics(m_top, o_top)
    if st_top is None:
        return None

    pct_label = f"top {100 - percentile}%"
    print(f"      {pct_label} (>= {threshold:.2f}): N={st_top['n']}  "
          f"RMSE={st_top['rmse']:.3f}  bias={st_top['bias']:.3f}  "
          f"R={st_top['corr']:.3f}  skill={st_top['skill']:.3f}")

    # Scatter plot for top percentile
    if make_plots:
        var_top = f"{var_name} ({pct_label})"
        plot_scatter(m_top, o_top, var_top, station_id, model_name,
                     st_top, output_dir)

    return st_top


def plot_cwop_summary(records, output_dir):
    """Distribution of skill/bias/RMSE across the many CWOP stations, per model.

    One boxplot per model (over all CWOP stations) for Wind Speed — readable
    where a 251-station Taylor diagram would not be."""
    df = pd.DataFrame([r for r in records if r.get('variable') == 'Wind Speed [m/s]'])
    if df.empty:
        print("    CWOP summary: no Wind Speed records.")
        return
    models = sorted(df['model'].unique())
    metrics = [('skill', 'Skill (1=perfect)'), ('bias', 'Bias [m/s]'), ('rmse', 'RMSE [m/s]')]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    for ax, (key, label) in zip(np.atleast_1d(axes), metrics):
        data = [df[df['model'] == m][key].dropna().values for m in models]
        ax.boxplot(data, labels=models, showfliers=False)
        ax.set_title(f"CWOP Wind Speed: {label}\n(over {df['station'].nunique()} stations)")
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        if key == 'bias':
            ax.axhline(0, color='k', lw=0.8, ls='--')
    fig.tight_layout()
    out = output_dir / 'cwop_windspeed_distribution.png'
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    CWOP summary saved: {out}")


def main():
    print("=" * 70)
    print("Multi-Model Meteorological Validation")
    print("=" * 70)

    # Apply optional time filter
    time_slice = None
    if TIME_RANGE is not None:
        time_slice = (pd.Timestamp(TIME_RANGE[0]), pd.Timestamp(TIME_RANGE[1]))
        print(f"\nTime filter: {time_slice[0]} – {time_slice[1]}")

    # --- Load stations -----------------------------------------------------
    print("\n--- Loading observation stations ---")
    station_data = {}
    qc_rows = []
    for sid in STATIONS_TO_RUN:
        cfg = STATIONS.get(sid)
        if cfg is None:
            print(f"  WARNING: station '{sid}' not in STATIONS config, skipping.")
            continue
        data = load_station(sid, cfg)
        if data is None:
            continue
        if time_slice is not None:
            mask = (data['time'] >= time_slice[0]) & (data['time'] <= time_slice[1])
            tf_keys = ['time', 'u10', 'v10', 'speed10', 'dir_deg', 'air_temp_C']
            tf_keys += [spec['obs_key'] for spec in SCALAR_VARS.values()
                        if spec['obs_key'] != 'air_temp_C']   # pressure/dewpoint/rh/solar/precip
            for key in tf_keys:
                if key in data and data[key] is not None:
                    data[key] = data[key][mask]   # DatetimeIndex + ndarray both accept bool mask
        # QC Tier 2: per-station acceptance (on the QC'd, time-filtered record)
        grp = cfg.get('group', 'OTHER')
        ok, reason, qst = qc_station_accept(data)
        qc_rows.append({'station': sid, 'source': grp, 'accepted': ok,
                        'reason': reason, **qst})
        if not ok:
            continue
        station_data[sid] = data

    # QC report + per-source kept/dropped summary
    if qc_rows:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        qc_df = pd.DataFrame(qc_rows)
        qc_df.to_csv(OUTPUT_DIR / 'qc_report.csv', index=False)
        print("\n  --- Obs QA/QC (per-station acceptance) ---")
        for src, g in qc_df.groupby('source'):
            kept = int(g['accepted'].sum())
            print(f"    {src:<5}: {kept}/{len(g)} kept, {len(g) - kept} dropped")
        nrej = int((~qc_df['accepted']).sum())
        print(f"    -> {nrej} stations dropped; report: {OUTPUT_DIR / 'qc_report.csv'}")

    if not station_data:
        print("ERROR: no stations loaded. Exiting.")
        return

    # CWOP plot sample: the N CWOP stations with the most wind obs get full
    # figures; the rest are stats-only (still appended to the CSV).
    cwop_loaded = [s for s in station_data if STATIONS.get(s, {}).get('group') == 'CWOP']
    cwop_sample = set(sorted(
        cwop_loaded,
        key=lambda s: int(np.isfinite(station_data[s]['speed10']).sum()),
        reverse=True)[:CWOP_PLOT_SAMPLE_N])
    if cwop_loaded:
        print(f"\n  CWOP: {len(cwop_loaded)} loaded; figure sample = {sorted(cwop_sample)}")

    # --- Load land boundary ------------------------------------------------
    ldb_polygons = None
    if LDB_FILE.exists():
        print(f"\n  Loading land boundary: {LDB_FILE}")
        ldb_polygons = _load_landboundary(LDB_FILE)
        print(f"    Loaded {len(ldb_polygons)} polylines")
    else:
        print(f"\n  WARNING: land boundary not found: {LDB_FILE}")

    # --- Validate each model -----------------------------------------------
    audit_model_paths(MODELS_TO_RUN)
    all_records = []

    for model_name in MODELS_TO_RUN:
        model_cfg = MODELS.get(model_name)
        if model_cfg is None:
            print(f"\n  WARNING: model '{model_name}' not in MODELS config, skipping.")
            continue

        if model_cfg.get('kind') == 'point_product':
            model_data = load_point_product(model_name, model_cfg)
        elif model_cfg.get('kind') == 'box':
            model_data = load_model_box(model_name, model_cfg)
        elif model_cfg.get('kind') == 'aorc':
            model_data = load_model_aorc(model_name, model_cfg)
        elif model_cfg.get('crs') == 'latlon_2d':
            model_data = load_model_wrf_calnev(model_name, model_cfg)
        elif 'data_dir' in model_cfg and model_cfg.get('crs') == 'lcc':
            model_data = load_model_ucla(model_name, model_cfg)
        elif 'data_dir' in model_cfg and model_cfg.get('crs', 'utm10n') == 'utm10n':
            model_data = load_model_utm_multifile(model_name, model_cfg)
        else:
            model_data = load_model(model_name, model_cfg)
        if model_data is None:
            print(f"  Skipping {model_name} (could not load).")
            continue

        # Compute every station's interpolation stencil once and read each model
        # variable in a single vectorized pass (all stations together).
        print(f"\n  Extracting {model_name} at {len(station_data)} stations "
              f"(batched, one read per variable) ...")
        all_ts = extract_model_all_stations(
            model_data, model_cfg, station_data, method=INTERPOLATION_METHOD)

        for sid, obs in station_data.items():
            grp = STATIONS.get(sid, {}).get('group', 'OTHER')
            make_plots = (grp in PLOT_FIGURES_FOR_GROUPS) or (sid in cwop_sample)
            # Outputs foldered by source group, then model
            model_output = OUTPUT_DIR / grp / model_name
            model_output.mkdir(parents=True, exist_ok=True)
            tag = '' if make_plots else ', stats-only'
            print(f"\n  === {model_name} × {sid} [{grp}{tag}] ===")

            # Precomputed batched extraction (see extract_model_all_stations)
            model_ts = all_ts[sid]

            # --- Wind variables ---
            if 'wind' in VARIABLES:
                for var_key, var_label, is_dir in [
                    ('u10', 'Wind U10 [m/s]', False),
                    ('v10', 'Wind V10 [m/s]', False),
                    ('speed10', 'Wind Speed [m/s]', False),
                    ('dir_deg', 'Wind Direction [deg]', True),
                ]:
                    print(f"    Variable: {var_label}")
                    result = validate_variable(
                        model_ts[var_key], obs[var_key],
                        model_ts['time'], obs['time'],
                        var_label, sid, model_name, model_output,
                        is_direction=is_dir, make_plots=make_plots)
                    st, st_top = result if isinstance(result, tuple) else (result, None)
                    if st is not None:
                        all_records.append({
                            'model': model_name, 'station': sid, 'source': grp,
                            'variable': var_label, **st,
                        })
                    if st_top is not None:
                        all_records.append({
                            'model': model_name, 'station': sid, 'source': grp,
                            'variable': f"{var_label} (top 10%)", **st_top,
                        })

                # Wind rose + spatial peak map — figures only (gated)
                if make_plots:
                    # match speed & direction on the SAME time base; QC first
                    spd_m_qc = _clip_to_physical_bounds(model_ts['speed10'], 'Wind Speed [m/s]')
                    spd_o_qc = _clip_to_physical_bounds(obs['speed10'], 'Wind Speed [m/s]')
                    dir_m_qc = _clip_to_physical_bounds(model_ts['dir_deg'], 'Wind Direction [deg]')
                    dir_o_qc = _clip_to_physical_bounds(obs['dir_deg'], 'Wind Direction [deg]')

                    spd_m_all, spd_o_all, t_spd = match_timeseries(
                        spd_m_qc, model_ts['time'], spd_o_qc, obs['time'])
                    dir_m_all, dir_o_all, t_dir = match_timeseries(
                        dir_m_qc, model_ts['time'], dir_o_qc, obs['time'])
                    if spd_m_all is not None and dir_m_all is not None:
                        t_common = np.intersect1d(t_spd, t_dir)
                        if len(t_common) > 50:
                            mask_spd = np.isin(t_spd, t_common)
                            mask_dir = np.isin(t_dir, t_common)
                            plot_wind_rose(
                                spd_m_all[mask_spd], dir_m_all[mask_dir],
                                spd_o_all[mask_spd], dir_o_all[mask_dir],
                                sid, model_name, model_output)

                    # Spatial peak wind map (based on obs peak)
                    obs_peak_t = _find_obs_peak_time(obs['speed10'], obs['time'])
                    if not MAKE_SPATIAL_MAPS:
                        pass  # skip heavy spatial maps for fast overview runs
                    elif obs_peak_t is None:
                        print(f"    Skipping spatial wind map (no valid obs speed data)")
                    else:
                        crs_type = model_data.get('crs', 'utm10n')
                        print(f"    Spatial wind map (obs peak: {obs_peak_t})")
                        try:
                            if crs_type == 'utm10n':
                                plot_spatial_peak_wind(model_data, model_cfg, obs,
                                                      obs_peak_t, sid, model_name, model_output,
                                                      ldb_polygons=ldb_polygons)
                            elif crs_type == 'lcc':
                                plot_spatial_peak_wind_lcc(model_data, model_cfg, obs,
                                                          obs_peak_t, sid, model_name, model_output,
                                                          ldb_polygons=ldb_polygons)
                            elif crs_type == 'latlon_2d':
                                plot_spatial_peak_wind_latlon2d(model_data, model_cfg, obs,
                                                               obs_peak_t, sid, model_name, model_output,
                                                               ldb_polygons=ldb_polygons)
                            else:
                                print(f"    [spatial wind map skipped: {model_name} crs '{crs_type}' has no map renderer]")
                        except Exception as _e:
                            print(f"    [spatial wind map skipped: {type(_e).__name__}: {_e}]")

            # --- Scalar variables (temperature/pressure/dewpoint/rh/radiation/precip) ---
            for var_key, spec in SCALAR_VARS.items():
                if var_key not in VARIABLES:
                    continue
                obs_key, label = spec['obs_key'], spec['label']
                if obs_key not in obs or obs_key not in model_ts:
                    continue
                # skip cleanly if obs OR model has no usable data for this var/station
                if not (np.isfinite(obs[obs_key]).any() and np.isfinite(model_ts[obs_key]).any()):
                    continue
                print(f"    Variable: {label}")
                mtime = model_ts.get(obs_key + '__t', model_ts['time'])
                result = validate_variable(
                    model_ts[obs_key], obs[obs_key],
                    mtime, obs['time'],
                    label, sid, model_name, model_output,
                    make_plots=make_plots)
                st, _ = result if isinstance(result, tuple) else (result, None)
                if st is not None:
                    all_records.append({
                        'model': model_name, 'station': sid, 'source': grp,
                        'variable': label, **st,
                    })

                # Spatial peak map for this scalar variable (utm10n grids; obs-peak time)
                if make_plots and MAKE_SPATIAL_MAPS \
                        and model_data.get('crs', 'utm10n') == 'utm10n':
                    pk = _find_obs_peak_time(obs[obs_key], obs['time'])
                    if pk is not None:
                        try:
                            plot_spatial_peak_scalar(model_data, model_cfg, obs, pk, sid,
                                                     model_name, model_output, var_key,
                                                     ldb_polygons=ldb_polygons)
                        except Exception as _e:
                            print(f"    [spatial {var_key} map skipped: {type(_e).__name__}: {_e}]")

            # --- Temperature ---
            if 'temperature' in VARIABLES:
                # Spatial peak temperature map — figures only (gated)
                if make_plots and MAKE_SPATIAL_MAPS:
                    obs_peak_temp_t = _find_obs_peak_temp_time(
                        obs['air_temp_C'], obs['time'])
                    if obs_peak_temp_t is None:
                        print(f"    Skipping spatial temp map (no valid obs temperature data)")
                    else:
                        crs_type = model_data.get('crs', 'utm10n')
                        print(f"    Spatial temperature map (obs peak temp: {obs_peak_temp_t})")
                        if crs_type == 'utm10n':
                            pass  # utm10n temperature handled by the generic scalar spatial map
                        elif crs_type == 'lcc':
                            plot_spatial_peak_temperature_lcc(model_data, model_cfg, obs,
                                                             obs_peak_temp_t, sid, model_name,
                                                             model_output,
                                                             ldb_polygons=ldb_polygons)
                        elif crs_type == 'latlon_2d':
                            plot_spatial_peak_temperature_latlon2d(model_data, model_cfg, obs,
                                                                  obs_peak_temp_t, sid, model_name,
                                                                  model_output,
                                                                  ldb_polygons=ldb_polygons)
                        else:
                            print(f"    [spatial temp map skipped: {model_name} crs '{crs_type}' has no map renderer]")

        # Close datasets
        for ds in model_data['datasets']:
            ds.close()

    # --- Export results -----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if all_records:
        df = pd.DataFrame(all_records)

        # Aggregated mean rows: overall + per source group, per (model, variable)
        numeric_cols = ['n', 'bias', 'rmse', 'mae', 'corr', 'r2', 'nrmse',
                        'scatter_index', 'rel_bias', 'skill',
                        'model_mean', 'obs_mean', 'model_std', 'obs_std']

        def _mean_row(sub, label, **extra):
            row = {'station': label, **extra}
            for col in numeric_cols:
                if col in sub.columns:
                    vals = sub[col].dropna()
                    row[col] = float(vals.mean()) if len(vals) else np.nan
            return row

        agg_rows = []
        for (model, var), g in df.groupby(['model', 'variable']):
            agg_rows.append(_mean_row(g, 'ALL_STATIONS_MEAN', model=model, variable=var, source='ALL'))
        if 'source' in df.columns:
            for (model, var, src), g in df.groupby(['model', 'variable', 'source']):
                agg_rows.append(_mean_row(g, f'{src}_MEAN', model=model, variable=var, source=src))

        df_agg = pd.DataFrame(agg_rows)
        df_full = pd.concat([df, df_agg], ignore_index=True)

        csv_path = OUTPUT_DIR / 'validation_statistics.csv'
        df_full.to_csv(csv_path, index=False)
        print(f"\n  Statistics saved: {csv_path}")

        # Print summary table (overall mean rows only)
        display_cols = ['model', 'variable', 'source', 'n', 'bias', 'rmse', 'mae',
                        'corr', 'r2', 'nrmse', 'scatter_index', 'rel_bias', 'skill']
        display_cols = [c for c in display_cols if c in df_agg.columns]
        print("\n  === Mean skill metrics across ALL stations ===")
        print(df_agg[df_agg['source'] == 'ALL'][display_cols]
              .sort_values(['variable', 'model']).to_string(index=False))

        # Summary plots — Taylor + multi-model bar PER source group (≤24 stations
        # each, readable); CWOP (251) gets a distribution summary instead.
        for g in GROUPS_FOR_SUMMARY_PLOTS:
            recs_g = [r for r in all_records if r.get('source') == g]
            if not recs_g:
                continue
            gdir = OUTPUT_DIR / g
            gdir.mkdir(parents=True, exist_ok=True)
            try:
                plot_taylor_diagram(recs_g, gdir)
                plot_multi_model_comparison(recs_g, gdir)
            except Exception as exc:
                print(f"    WARNING: summary plots for {g} failed: {exc}")
        cwop_recs = [r for r in all_records if r.get('source') == 'CWOP']
        if cwop_recs:
            (OUTPUT_DIR / 'CWOP').mkdir(parents=True, exist_ok=True)
            plot_cwop_summary(cwop_recs, OUTPUT_DIR / 'CWOP')
    else:
        print("\n  No validation results to export.")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

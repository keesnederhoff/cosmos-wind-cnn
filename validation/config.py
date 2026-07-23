"""
Centralized configuration for SF Bay meteorological product validation.

Single source of truth for data roots, obs archive, model/product registry,
station scope, run options, plot control and physical constants. The engine
(validate_met_models.py), the driver (run_validation.py) and the analysis
scripts all import from here.

Paths resolve from two env vars so the same code runs on Windows and Caldera:
    COSMOS_VALIDATION_DATA_ROOT   -> canonical data bundle (see docs/…-plan.md)
    COSMOS_VALIDATION_OUTPUT_ROOT -> per-era run outputs
"""
import os
from pathlib import Path


def _root(var: str) -> Path:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"{var} is not set. Point it at the validation data/output base.\n"
            f"  Windows:  set {var}=G:\\03-downscaling_meteo_cnn\\validation\n"
            f"  Caldera:  export {var}=/caldera/.../validation"
        )
    return Path(val)


# ===========================================================================
# Roots (env-var driven — Windows + Caldera/HPC)
# ===========================================================================
DATA_ROOT   = _root("COSMOS_VALIDATION_DATA_ROOT")
OUTPUT_ROOT = _root("COSMOS_VALIDATION_OUTPUT_ROOT")   # per-era run dirs land here

# ===========================================================================
# Observation archive  (pws_scraper: IEM / NDBC / CWOP)
# ===========================================================================
READ_COORDS_FROM_NETCDF = True

# Authoritative NDBC/CO-OPS positions (override archive coords where present).
KNOWN_STATION_COORDINATES = {
    '46026': (37.750, -122.838),   # SF Offshore, 18NM W of SF
    '46237': (37.788, -122.634),   # SF Bar — Golden Gate approach
    '46012': (37.356, -122.881),   # Half Moon Bay, 24NM SSW of SF
    '46013': (38.235, -123.317),   # Bodega Bay, 48NM NW of SF
    'RTYC1': (37.507, -122.212),   # Redwood City — South Bay
    'AAMC1': (37.772, -122.300),   # Alameda — Central Bay
    'PCOC1': (38.056, -122.039),   # Port Chicago — North Bay / Delta
    'FTPC1': (37.806, -122.466),   # Fort Point — Golden Gate (SF side)
    'TIBC1': (37.892, -122.447),   # Tiburon Pier — Central Bay (Marin side)
    'PXSC1': (37.803, -122.397),   # Pier 17 — SF Central Bay
}

PWS_DIR      = DATA_ROOT / "obs"
MOORINGS_DIR = DATA_ROOT / "moorings"
REFERENCE_DIR = DATA_ROOT / "reference"
LDB_FILE     = REFERENCE_DIR / "deltabay.ldb"   # land boundary for cartopy spatial maps
PWS_SOURCES = [   # (group, archive NetCDF, anemometer height m)
    ('IEM',  PWS_DIR / 'pws_sfbay_waterfront_iem.nc',        10.0),
    ('NDBC', PWS_DIR / 'pws_sfbay_waterfront_ndbc.nc',       10.0),
    ('CWOP', PWS_DIR / 'pws_sfbay_waterfront_cwop_madis.nc', 10.0),
]

# ---- Optional USGS moorings (Whales Tale / Emeryville Crescent) ------------
# OFF by default: chosen scope is the full IEM+NDBC+CWOP archive. Flip to True
# to carry these Emeryville-project waterfront moorings (1.2 m anemometers,
# kept at measured height — compared directly to 10 m model output).
INCLUDE_USGS_MOORINGS = True
USGS_MOORINGS = {
    'WT_MW101': {
        'source': 'whales_tale', 'group': 'USGS',
        'file_path': MOORINGS_DIR / "DMP23MW101met.nc",
        'lat': 37.576294, 'lon': -122.208054, 'anemometer_height_m': 1.2,
    },
    'WT_MW201': {
        'source': 'whales_tale', 'group': 'USGS',
        'file_path': MOORINGS_DIR / "DMP23MW201met.nc",
        'lat': 37.576397, 'lon': -122.208038, 'anemometer_height_m': 1.2,
    },
    'EMC_MW101': {
        'source': 'whales_tale', 'group': 'USGS',
        'file_path': MOORINGS_DIR / "EMC26MW101met.nc",
        'lat': 37.840395, 'lon': -122.342093, 'anemometer_height_m': 1.2,
    },
    'ERO20_GRZ': {
        # USGS ERO20 Grizzly Bay (Suisun Bay, North Bay). Vaisala WXT530,
        # 2020-01-22..2020-06-23 -> Era 2. Variable names differ from the
        # whales_tale schema (var_map); air_pressure is millibar = hPa.
        'source': 'usgs_met', 'group': 'USGS',
        'file_path': PWS_DIR / "ERO20_GrizzlyBay_meteorological.nc",
        'lat': 38.11725, 'lon': -122.039833, 'anemometer_height_m': 4.93,
        'var_map': {'wind_dir': 'wind_dir', 'pressure': 'air_pressure',
                    'rh': 'rel_humidity', 'pressure_units': 'hPa'},
    },
}

# ===========================================================================
# Model / product registry
# ===========================================================================
# cosmos-wind-cnn data was moved OUT of the repo on 2026-06-26 into external G: storage,
# under per-case dirs (sf_bay_conus404 / sf_bay_rtma). ERA5 baseline + the CNN/CNN-RTMA
# inference outputs live there now — NOT the old d:\Git\...\case_studies\sf_bay\ paths.
ERA5_DIR      = DATA_ROOT / "era5"
HRRR_DIR      = DATA_ROOT / "hrrr"
C404_DIR      = DATA_ROOT / "conus404"
RTMA_DIR      = DATA_ROOT / "rtma"
PROD_DIR      = DATA_ROOT                 # NOW-23 -> now23/, Sup3rWind -> sup3rwind/
PROD_DATA_DIR = DATA_ROOT                 # UCLA -> ucla_reanalysis/, WRF_CalNev -> wrf_calnev/
DOWNSCALED_DIR = DATA_ROOT / "conus404_downscaled"
CNN_DIR       = DATA_ROOT / "cnn"
CNN_FILE      = CNN_DIR / "cnn_conus404.nc"
CNN_RTMA_FILE = CNN_DIR / "cnn_rtma.nc"

MODELS = {
    'ERA5': {
        'u_file': ERA5_DIR / "ERA5_eastward_wind_1940_2026_UTM.nc",
        'v_file': ERA5_DIR / "ERA5_northward_wind_1940_2026_UTM.nc",
        'temp_file': ERA5_DIR / "ERA5_air_temperature_1940_2026_UTM.nc",
        'u_var': 'eastward_wind', 'v_var': 'northward_wind',
        'temp_var': 'air_temperature', 'single_file': False,
    },
    'HRRR': {
        'u_file': HRRR_DIR / "HRRR_WY2015-WY2026_u10_eastward_wind.nc",
        'v_file': HRRR_DIR / "HRRR_WY2015-WY2026_v10_northward_wind.nc",
        'temp_file': HRRR_DIR / "HRRR_WY2015-WY2026_air_temp.nc",
        'u_var': 'eastward_wind', 'v_var': 'northward_wind',
        'temp_var': 'air_temperature', 'single_file': False,
    },
    'CONUS404': {
        'u_file': C404_DIR / "CONUS404_SFbay_4km_eastward_wind_1979_2021_UTM10.nc",
        'v_file': C404_DIR / "CONUS404_SFbay_4km_northward_wind_1979_2021_UTM10.nc",
        'temp_file': C404_DIR / "CONUS404_SFbay_4km_air_temperature_1979_2021_UTM10.nc",
        'u_var': 'eastward_wind', 'v_var': 'northward_wind',
        'temp_var': 'air_temperature', 'single_file': False,
    },
    'CONUS404-downscaled': {
        'data_dir': DOWNSCALED_DIR,
        'u_pattern': '*500m_full*eastward_wind.nc',
        'v_pattern': '*500m_full*northward_wind.nc',
        'u_var': 'eastward_wind', 'v_var': 'northward_wind', 'crs': 'utm10n',
    },
    'CONUS404-downscaled-100m': {
        'data_dir': DOWNSCALED_DIR,
        'u_pattern': '*100m_bay*eastward_wind.nc',
        'v_pattern': '*100m_bay*northward_wind.nc',
        'u_var': 'eastward_wind', 'v_var': 'northward_wind', 'crs': 'utm10n',
    },
    'CNN': {
        'u_file': CNN_FILE, 'v_file': CNN_FILE, 'temp_file': CNN_FILE,
        'u_var': 'conus404_u', 'v_var': 'conus404_v',
        'temp_var': 'conus404_air_temp', 'single_file': True,
    },
    'CNN-RTMA-20260625': {
        'u_file': CNN_RTMA_FILE, 'v_file': CNN_RTMA_FILE,
        'u_var': 'conus404_u', 'v_var': 'conus404_v', 'single_file': True,
    },
    'UCLA': {
        'data_dir': PROD_DATA_DIR / 'ucla_reanalysis',
        'u_pattern': 'era5_reanalysis_1hr_u10_*.nc',
        'v_pattern': 'era5_reanalysis_1hr_v10_*.nc',
        'temp_pattern': 'era5_reanalysis_1hr_t2_*.nc',
        'u_var': 'u10', 'v_var': 'v10', 'temp_var': 't2', 'crs': 'lcc',
    },
    'WRF_CalNev': {
        'data_dir': PROD_DATA_DIR / 'wrf_calnev',
        'speed_pattern': 'wrfout_d02_V1_WS_*_bayarea.nc',
        'temp_pattern': 'wrfout_d02_V1_TA_*_bayarea.nc',
        'speed_var': 'ws', 'temp_var': 'ta', 'has_uv': False, 'crs': 'latlon_2d',
    },
}
# NOTE: UCLA / WRF_CalNev data_dir nest under '...\other_meteo_data\data\' (verified
# on disk 2026-07-12) -> use PROD_DATA_DIR. NOW-23 / Sup3rWind sit at the top level
# ('...\other_meteo_data\') -> use PROD_DIR. Resolved paths match the source engine.

MODELS['NOW-23'] = {
    'kind': 'box', 'crs': 'unstructured',
    'data_dir': PROD_DIR / 'now23', 'pattern': 'now23_ca_bayarea_box_*.nc',
    'speed_var': 'windspeed_10m', 'dir_var': 'winddirection_10m',
}
MODELS['Sup3rWind'] = {
    'kind': 'box', 'crs': 'latlon_2d', 'has_uv': True,
    'data_dir': PROD_DIR / 'sup3rwind', 'pattern': 'sup3rwind_bayarea_box_*.nc',
    'u_var': 'u_10m', 'v_var': 'v_10m', 'speed_var': 'windspeed_10m',
}
MODELS['RTMA'] = {
    'data_dir': RTMA_DIR,
    'u_pattern': 'RTMA_grid_2p5km_2*.nc', 'v_pattern': 'RTMA_grid_2p5km_2*.nc',
    'u_var': 'eastward_wind', 'v_var': 'northward_wind', 'crs': 'utm10n',
}

# ---- Scalar (non-wind) variable sources per model -------------------------
MODELS['ERA5']['scalars'] = {
    'temperature':   {'file': ERA5_DIR / "ERA5_air_temperature_1940_2026_UTM.nc", 'var': 'air_temperature', 'units': 'K'},
    'pressure':      {'file': ERA5_DIR / "ERA5_air_pressure_fixed_height_1940_2026_UTM.nc", 'var': 'air_pressure_fixed_height', 'units': 'Pa'},
    'dewpoint':      {'file': ERA5_DIR / "ERA5_dew_point_temperature_1940_2026_UTM.nc", 'var': 'dew_point_temperature', 'units': 'K'},
    'rh':            {'method': 'from_dewpoint'},
    'radiation':     {'file': ERA5_DIR / "ERA5_surface_solar_radiation_1940_2026_UTM.nc", 'var': 'surface_solar_radiation', 'units': 'W/m2'},
    'precipitation': {'file': ERA5_DIR / "ERA5_precipitation_1940_2026_UTM.nc", 'var': 'precipitation', 'units': 'mm/hr'},
}
MODELS['CONUS404']['scalars'] = {
    'temperature':   {'file': C404_DIR / "CONUS404_SFbay_4km_air_temperature_1979_2021_UTM10.nc", 'var': 'air_temperature', 'units': 'K'},
    'pressure':      {'file': C404_DIR / "CONUS404_SFbay_4km_air_pressure_fixed_height_1979_2021_UTM10.nc", 'var': 'air_pressure_fixed_height', 'units': 'Pa'},
    'dewpoint':      {'file': C404_DIR / "CONUS404_SFbay_4km_dew_point_temperature_1979_2021_UTM10.nc", 'var': 'dew_point_temperature', 'units': 'K'},
    'rh':            {'method': 'from_dewpoint'},
    'radiation':     {'file': C404_DIR / "CONUS404_SFbay_4km_surface_solar_radiation_1979_2021_UTM10.nc", 'var': 'surface_solar_radiation', 'units': 'W/m2'},
    'precipitation': {'file': C404_DIR / "CONUS404_SFbay_4km_rainfall_1979_2021_UTM10.nc", 'var': 'rainfall', 'units': 'mm/hr'},
}
MODELS['CNN']['scalars'] = {
    'temperature':   {'file': CNN_FILE, 'var': 'conus404_air_temp', 'units': 'K'},
    'pressure':      {'file': CNN_FILE, 'var': 'conus404_pressure', 'units': 'Pa'},
    'dewpoint':      {'file': CNN_FILE, 'var': 'conus404_dew_temp', 'units': 'K'},
    'rh':            {'method': 'from_dewpoint'},
    'radiation':     {'file': CNN_FILE, 'var': 'conus404_solar', 'units': 'W/m2'},
    'precipitation': {'file': CNN_FILE, 'var': 'conus404_rain', 'units': 'mm/hr'},
}
MODELS['RTMA']['scalars'] = {   # GEE export already in Celsius (NOT K)
    'temperature':   {'pattern': 'RTMA_grid_2p5km_2*.nc', 'var': 'air_temperature', 'units': 'C'},
    'pressure':      {'pattern': 'RTMA_grid_2p5km_2*.nc', 'var': 'air_pressure', 'units': 'Pa'},
    'dewpoint':      {'pattern': 'RTMA_grid_2p5km_2*.nc', 'var': 'dew_point_temperature', 'units': 'C'},
    'rh':            {'method': 'from_dewpoint'},
    'precipitation': {'pattern': 'RTMA_grid_2p5km_precip_2*.nc', 'var': 'precipitation', 'units': 'mm/hr'},
}
MODELS['HRRR']['scalars'] = {
    'temperature':   {'file': HRRR_DIR / "HRRR_WY2015-WY2026_air_temp.nc", 'var': 'air_temperature', 'units': 'K'},
    'dewpoint':      {'file': HRRR_DIR / "HRRR_WY2015-WY2026_dew_point.nc", 'var': 'dew_point_temperature', 'units': 'K'},
    'rh':            {'method': 'from_dewpoint'},
}
MODELS['UCLA']['scalars'] = {
    'temperature':   {'pattern': 'era5_reanalysis_1hr_t2_*.nc', 'var': 't2', 'units': 'K'},
    'pressure':      {'pattern': 'era5_reanalysis_1hr_psfc_*.nc', 'var': 'psfc', 'units': 'Pa'},
    'dewpoint':      {'pattern': 'era5_reanalysis_1hr_dew_point_*.nc', 'var': 'dew_point', 'units': 'K'},
    'rh':            {'method': 'from_q', 'pattern': 'era5_reanalysis_1hr_q2_*.nc', 'var': 'q2', 'units': 'kg/kg'},
    'radiation':     {'pattern': 'era5_reanalysis_1hr_swdnb_*.nc', 'var': 'swdnb', 'units': 'W/m2'},
    'precipitation': {'pattern': 'era5_reanalysis_1hr_prec_*.nc', 'var': 'prec', 'units': 'mm/hr'},
}
MODELS['WRF_CalNev']['scalars'] = {
    'temperature':   {'pattern': 'wrfout_d02_V1_TA_*_bayarea.nc', 'var': 'ta', 'units': 'K'},
    'rh':            {'method': 'direct', 'pattern': 'wrfout_d02_V1_RH_*_bayarea.nc', 'var': 'hur', 'units': '%'},
}
# NOW-23, Sup3rWind: wind-only (no 'scalars').

# ===========================================================================
# Run options  (defaults; run_validation.py overrides per era)
# ===========================================================================
MODELS_TO_RUN        = ['ERA5', 'HRRR', 'CONUS404', 'CNN', 'CNN-RTMA-20260625',
                        'RTMA', 'NOW-23', 'Sup3rWind', 'UCLA', 'WRF_CalNev',
                        'CONUS404-downscaled', 'CONUS404-downscaled-100m']
STATION_GROUPS       = ['IEM', 'NDBC']   # CWOP skipped for now; USGS via INCLUDE_USGS_MOORINGS
VARIABLES            = ['wind', 'temperature', 'pressure', 'dewpoint',
                        'rh', 'radiation', 'precipitation']
TIME_RANGE           = None          # per-era driver sets this
INTERPOLATION_METHOD = 'linear'      # 'nearest' or 'linear'

# ===========================================================================
# Plot control
# ===========================================================================
PLOT_FIGURES_FOR_GROUPS  = {'IEM', 'NDBC', 'USGS'}   # CWOP stats-only
CWOP_PLOT_SAMPLE_N       = 6
MAKE_SPATIAL_MAPS        = True
GROUPS_FOR_SUMMARY_PLOTS = ['IEM', 'NDBC', 'USGS']

MODEL_COLORS = {
    'ERA5': 'tab:blue', 'HRRR': 'tab:red', 'CONUS404': 'tab:orange',
    'CNN': 'tab:green', 'UCLA': 'tab:purple', 'WRF_CalNev': 'tab:brown',
    'NOW-23': 'tab:pink', 'Sup3rWind': 'tab:olive', 'RTMA': 'tab:cyan',
    'CONUS404-downscaled': 'tab:gray', 'CONUS404-downscaled-100m': 'gold',
}
FALLBACK_COLORS = ['black', 'magenta', 'teal', 'navy', 'crimson', 'darkgreen',
                   'goldenrod', 'slateblue', 'darkorange', 'deeppink']

# ===========================================================================
# Physical / plotting constants
# ===========================================================================
DEFAULT_ROUGHNESS = 2e-4    # m, open water (log wind profile)
MODEL_HEIGHT      = 10.0    # m, all models output at 10 m
FIGURE_DPI        = 100    # 100 vs 150: ~2x faster render+encode for diagnostic figures
PEAK_WINDOW_DAYS  = 7

# ===========================================================================
# 2026-07-21 study: two new ERA5->RTMA CNN runs (wind validation 2011-2026)
#   CNN-allvars  = os_av_bc24_terr_res_s2 (all-vars loss + terrain + residual, bc24)
#   CNN-windonly = os_wo_bc24_base_res_s2 (wind-only loss + residual, no terrain, bc24)
# Both single-file on the RTMA 2.5 km UTM-10N grid; vars hr_u/hr_v. Mirrors the
# working CNN-RTMA-20260625 entry (no crs key -> engine defaults to UTM10).
# ===========================================================================
_CNN_STUDY_DIR = CNN_DIR
MODELS['CNN-allvars'] = {
    'u_file': CNN_DIR / 'cnn_allvars.nc',
    'v_file': CNN_DIR / 'cnn_allvars.nc',
    'u_var': 'hr_u', 'v_var': 'hr_v', 'single_file': True,
}
MODELS['CNN-windonly'] = {
    'u_file': CNN_DIR / 'cnn_windonly.nc',
    'v_file': CNN_DIR / 'cnn_windonly.nc',
    'u_var': 'hr_u', 'v_var': 'hr_v', 'single_file': True,
}
MODEL_COLORS['CNN-allvars']  = 'tab:green'
MODEL_COLORS['CNN-windonly'] = 'darkgreen'

# CNN-allvars trained ALL variables -> give it scalar sources so temp/dew/pressure/
# precip validate against obs. Units verified on disk: hr_air_temp/hr_dew_temp = K,
# hr_pressure = Pa, hr_rain = mm/hr. NO radiation variable in the file (omitted).
# RH derived from dewpoint. CNN-windonly is deliberately left wind-only (its scalar
# outputs are untrained -> engine skips it for non-wind variables).
_AV_FILE = MODELS['CNN-allvars']['u_file']
MODELS['CNN-allvars']['temp_file'] = _AV_FILE
MODELS['CNN-allvars']['temp_var']  = 'hr_air_temp'
MODELS['CNN-allvars']['scalars'] = {
    'temperature':   {'file': _AV_FILE, 'var': 'hr_air_temp', 'units': 'K'},
    'pressure':      {'file': _AV_FILE, 'var': 'hr_pressure', 'units': 'Pa'},
    'dewpoint':      {'file': _AV_FILE, 'var': 'hr_dew_temp', 'units': 'K'},
    'rh':            {'method': 'from_dewpoint'},
    'precipitation': {'file': _AV_FILE, 'var': 'hr_rain', 'units': 'mm/hr'},
}

# ===========================================================================
# 2026-07-22: NOAA AORC v1.1 (~800 m, hourly, 1979-present) — SF Bay subset
#   Single NetCDF per year, 1-D regular lat/lon grid, u/v + full scalar set in
#   one file (kind='aorc' loader meshgrids the coords to 2-D, crs=latlon_2d).
#   Downloaded via download_aorc.py -> data\aorc\AORC_SFbay_800m_<year>.nc.
#   NB AORC wind = URMA-interpolated post-2018 / NLDAS-2 pre-2018; the OPEN
#   PACIFIC IS MASKED (NaN) -> offshore NDBC buoys drop out of AORC's stats.
#   No dewpoint variable in AORC (has specific humidity) -> RH via from_q.
# ===========================================================================
AORC_DIR  = DATA_ROOT / "aorc"
AORC_FILE = AORC_DIR / "AORC_SFbay_800m_2020.nc"   # 2020 test year
MODELS['AORC'] = {
    'kind': 'aorc', 'crs': 'latlon_2d', 'has_uv': True,
    'u_file': AORC_FILE, 'v_file': AORC_FILE, 'temp_file': AORC_FILE,
    'u_var': 'UGRD_10maboveground', 'v_var': 'VGRD_10maboveground',
    'temp_var': 'TMP_2maboveground',
    'scalars': {
        'temperature':   {'file': AORC_FILE, 'var': 'TMP_2maboveground', 'units': 'K'},
        'pressure':      {'file': AORC_FILE, 'var': 'PRES_surface',      'units': 'Pa'},
        'rh':            {'method': 'from_q', 'file': AORC_FILE, 'var': 'SPFH_2maboveground', 'units': 'kg/kg'},
        'radiation':     {'file': AORC_FILE, 'var': 'DSWRF_surface',     'units': 'W/m2'},
        'precipitation': {'file': AORC_FILE, 'var': 'APCP_surface',      'units': 'mm/hr'},
    },
}
MODEL_COLORS['AORC'] = 'black'

# --- AORC split at the 2018 wind-provenance break (multi-year, kind='aorc') ---
# 10 m wind switches source at 2018: NLDAS-2 (~12 km, unassimilated) for 1979-2017
# vs URMA (RTMA's 2.5 km sibling) for 2018+. Registering the two eras as separate
# products makes the ~20% wind discontinuity measurable in skill terms. Both read
# the same data\aorc\AORC_SFbay_800m_<year>.nc set, filtered by year_range; the
# loader concatenates + windows to TIME_RANGE. No dewpoint var -> RH via from_q.
_AORC_SCALARS = {
    'temperature':   {'var': 'TMP_2maboveground', 'units': 'K'},
    'pressure':      {'var': 'PRES_surface',      'units': 'Pa'},
    'rh':            {'method': 'from_q', 'var': 'SPFH_2maboveground', 'units': 'kg/kg'},
    'radiation':     {'var': 'DSWRF_surface',     'units': 'W/m2'},
    'precipitation': {'var': 'APCP_surface',      'units': 'mm/hr'},
}
for _name, _yr in [('AORC-pre2018', (1979, 2017)), ('AORC-post2018', (2018, 2025))]:
    MODELS[_name] = {
        'kind': 'aorc', 'crs': 'latlon_2d', 'has_uv': True,
        'data_dir': AORC_DIR, 'year_range': _yr,
        'u_var': 'UGRD_10maboveground', 'v_var': 'VGRD_10maboveground',
        'temp_var': 'TMP_2maboveground',
        'scalars': dict(_AORC_SCALARS),
    }
MODEL_COLORS['AORC-pre2018']  = 'black'
MODEL_COLORS['AORC-post2018'] = 'dimgray'

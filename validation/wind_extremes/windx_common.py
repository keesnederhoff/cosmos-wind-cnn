# === CONFIGURATION ===
from pathlib import Path
ROOT       = Path(r"g:\03-downscaling_meteo_cnn\wind_extremes")
CACHE_DIR  = ROOT / "cache"
RESULT_DIR = ROOT / "results"
FIG_DIR    = ROOT / "figures"

CNN_DIR    = Path(r"g:\03-downscaling_meteo_cnn\sf_bay_rtma\results")
CNN_RAW    = CNN_DIR / "CNN_RTMA_v3_r1b_do010_s3_speed_quantiles_raw_19400101_20260810.nc"
CNN_BC     = CNN_DIR / "CNN_RTMA_v3_r1b_do010_s3_speed_quantiles_BC_19400101_20260810.nc"
BC_MAP     = CNN_DIR / "BC_map_r1b_do010_s3_valfit.nc"
CNN_Q_INDEX = 1                     # quantile axis index of tau = 0.5 (P50)

MET_ROOT   = Path(r"g:\01_meteorlogical_analysis_sfbay")
ERA5_U     = MET_ROOT / "modeled_data" / "era5" / "fm_netcdf" / "ERA5_eastward_wind_1940_2025_UTM.nc"
ERA5_V     = MET_ROOT / "modeled_data" / "era5" / "fm_netcdf" / "ERA5_northward_wind_1940_2025_UTM.nc"
RTMA_WS    = MET_ROOT / "modeled_data" / "rtma" / "fm_netcdf" / "RTMA_SFbay_2p5km_wind_speed_2011_2026_UTM10.nc"
OBS_DIR    = MET_ROOT / "observed_data"
IEM_FILE   = OBS_DIR / "pws_sfbay_waterfront_iem.nc"
NDBC_FILE  = OBS_DIR / "pws_sfbay_waterfront_ndbc.nc"
USGS_FILES = {                      # id: (file, lat, lon, anemometer height m)
    "WT_MW101":  (OBS_DIR / "DMP23MW101met.nc", 37.576294, -122.208054, 1.2),
    "WT_MW201":  (OBS_DIR / "DMP23MW201met.nc", 37.576397, -122.208038, 1.2),
    "EMC_MW101": (OBS_DIR / "EMC26MW101met.nc", 37.840395, -122.342093, 1.2),
    "ERO20_GRZ": (OBS_DIR / "ERO20_GrizzlyBay_meteorological.nc", 38.11725, -122.039833, 4.93),
}
LDB_FILE   = MET_ROOT / "reference" / "deltabay.ldb"   # coastline, UTM10 metres

PRODUCTS   = ["era5", "rtma", "cnn_raw", "cnn_bc"]     # fixed order everywhere
PRODUCT_LABEL = {"era5": "ERA5", "rtma": "RTMA", "cnn_raw": "CNN-RTMA raw", "cnn_bc": "CNN-RTMA BC"}

# Extra CNN predictive-distribution quantiles (raw file only), added after the
# original 4-product comparison. Kept out of PRODUCTS so the six 1x4 maps and
# the 4-product scatters are unaffected; appended in ALL_PRODUCTS for the
# station skill table and the new 3-quantile scatter.
EXTRA_PRODUCTS = ["cnn_q82", "cnn_q92", "cnn_ba"]
ALL_PRODUCTS   = PRODUCTS + EXTRA_PRODUCTS
PRODUCT_LABEL.update({"cnn_q82": "CNN-RTMA τ0.82", "cnn_q92": "CNN-RTMA τ0.92", "cnn_ba": "CNN-RTMA BA"})

# USGS observation-anchored BA product (2026-10-02): P50 x one factor per cell (obs/CNN at hourly P90,
# 39 stations, 2020-2026). The file has no quantile axis (index None). The RP maps show it in panel (d)
# in place of the clamped quantile-map BC.
CNN_BA_DIR    = Path(r"g:\03-downscaling_meteo_cnn\cnn_fullrecord")
CNN_BA        = CNN_BA_DIR / "CNN_RTMA_v3_r1b_do010_s3_P50_uvs_BA_19400101_20260810.nc"
BA_FACTOR_MAP = CNN_BA_DIR / "BA_factor_map_r1b_do010_s3_E3q90.nc"
MAP_PRODUCTS  = ["era5", "rtma", "cnn_raw", "cnn_ba"]   # panel order of the 1x4 RP maps and main scatters

# Quantile-axis index into each product's CNN source file. cnn_raw/cnn_bc keep
# their original CNN_Q_INDEX (P50); cnn_q82/cnn_q92 read other quantiles, both
# from CNN_RAW (quantile axis = [0.1316, 0.5, 0.8158, 0.9211, 0.9737]).
CNN_Q_INDEX_BY_PRODUCT = {"cnn_raw": CNN_Q_INDEX, "cnn_bc": CNN_Q_INDEX, "cnn_q82": 2, "cnn_q92": 3, "cnn_ba": None}
CNN_FILE_BY_PRODUCT    = {"cnn_raw": CNN_RAW, "cnn_bc": CNN_BC, "cnn_q82": CNN_RAW, "cnn_q92": CNN_RAW, "cnn_ba": CNN_BA}

WY_START_MONTH   = 10
FULL_WY          = {"era5": (1941, 2025), "rtma": (2012, 2025), "cnn_raw": (1941, 2025), "cnn_bc": (1941, 2025)}
FULL_WY.update({"cnn_q82": (1941, 2025), "cnn_q92": (1941, 2025), "cnn_ba": (1941, 2025)})
COMMON_WY        = (2012, 2025)
OBS_MAX_WY       = 2025
MIN_HOURS_PER_DAY = 12              # valid hours for a valid day
MIN_DAY_FRAC_WY   = 0.80            # valid-day fraction for a valid water year
MIN_YEARS        = 6                # below this: status dropped_<6yr, no fit

POT_LAMBDA       = 3.0              # peaks per year
SEP_HOURS        = 48.0             # declustering min separation
GPD_MODE         = "free"           # L-moment GPD; "exp" written as reference
RETURN_PERIODS   = [1, 10, 100]
N_BOOT           = 1000
CI_ALPHA         = 0.10
RNG_SEED         = 20261002

# Obs QC
WS_MAX_RAW       = 40.0             # m/s, readings above -> NaN (engine uses 50; 55-75 m/s spikes exist)
SPIKE_FACTOR     = 3.0              # hourly value > 3x mean of both neighbours AND > SPIKE_MIN -> NaN
SPIKE_MIN        = 15.0
REVIEW_RATIO     = 2.5              # top peak flagged for review if obs > 2.5 x ERA5 at that hour and obs > 20 m/s
PERSIST_MIN      = 15.0             # m/s, persistence rule applies above this
PERSIST_FRAC     = 0.6              # hour -> NaN if its higher neighbour < PERSIST_FRAC x value
# Two-hour bursts surrounded by calm, ERA5/CNN 2-5 m/s; reviewed with Kees 2026-10-02
QC_EXCLUDE: dict[str, list[str]] = {
    "HWD": ["1989-09-23 21", "1989-09-23 22"],
    "SQL": ["2017-04-18 17", "2017-04-18 18"],
    "RHV": ["2024-12-15 19", "2024-12-15 20"],
    "LVK": ["1997-06-06 16", "1997-06-06 17"],
}   # station_id -> list of 'YYYY-MM-DD HH' hours set to NaN after review (Task 7)
Z0_WATER         = 2e-4             # m, log-law roughness for the USGS height correction

# === IMPORTS ===
import warnings

import numpy as np
import pandas as pd
import pyproj
import netCDF4
import rainlib.extremes as rx

_UTM10_TRANSFORMER = pyproj.Transformer.from_crs(4326, 32610, always_xy=True)


# === TIME / CALENDAR HELPERS ===

def water_year(times: pd.DatetimeIndex) -> np.ndarray:
    """WY = year + 1 if month >= WY_START_MONTH, else year."""
    times = pd.DatetimeIndex(times)
    years = times.year.to_numpy()
    months = times.month.to_numpy()
    return np.where(months >= WY_START_MONTH, years + 1, years)


def daily_max(hourly: np.ndarray, times: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Daily maxima of an exact hourly UTC series, NaN where nvalid < MIN_HOURS_PER_DAY.

    `hourly` has shape (T, ...); `times` must cover whole UTC days starting at 00:00.
    """
    times = pd.DatetimeIndex(times)
    if len(times) == 0:
        raise ValueError("daily_max needs a non-empty time axis")
    if times[0].hour != 0 or times[0].minute != 0 or times[0].second != 0:
        raise ValueError("daily_max needs whole UTC days starting at 00:00")
    if len(times) % 24 != 0:
        raise ValueError("daily_max needs a whole number of UTC days (length not a multiple of 24)")
    if len(times) > 1:
        step_s = np.diff(times.to_numpy()).astype("timedelta64[s]").astype(float)
        if not np.allclose(step_s, 3600.0):
            raise ValueError("daily_max needs an exact hourly time axis")

    hourly = np.asarray(hourly, dtype=float)
    n_days = len(times) // 24
    tail_shape = hourly.shape[1:]
    resh = hourly.reshape(n_days, 24, *tail_shape)

    finite = np.isfinite(resh)
    nvalid = finite.sum(axis=1).astype(np.uint8)
    masked = np.where(finite, resh, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dmax = np.nanmax(masked, axis=1)
    dmax = np.where(nvalid < MIN_HOURS_PER_DAY, np.nan, dmax)

    days = times[::24].normalize()
    return dmax, days, nvalid


def valid_water_years(days: pd.DatetimeIndex, dmax: np.ndarray) -> dict[int, "np.ndarray | bool"]:
    """True for a WY whose valid-day fraction >= MIN_DAY_FRAC_WY.

    The denominator is the number of calendar days in that water year (not the
    number of days present in `days`). `dmax` may be 1-D (one station/cell) or
    N-D (e.g. a grid), in which case the result is an array per WY.
    """
    days = pd.DatetimeIndex(days)
    wy = water_year(days)
    dmax = np.asarray(dmax)
    valid = np.isfinite(dmax)

    out: dict[int, "np.ndarray | bool"] = {}
    for y in np.unique(wy):
        mask = wy == y
        wy_start = pd.Timestamp(year=int(y) - 1, month=WY_START_MONTH, day=1)
        wy_end = pd.Timestamp(year=int(y), month=WY_START_MONTH, day=1)
        n_cal_days = (wy_end - wy_start).days
        n_valid = valid[mask].sum(axis=0)
        frac = n_valid / n_cal_days
        result = frac >= MIN_DAY_FRAC_WY
        out[int(y)] = bool(result) if np.ndim(result) == 0 else result
    return out


# === SPATIAL HELPERS ===

def nearest_cell(x_grid_1d: np.ndarray, y_grid_1d: np.ndarray, x: float, y: float) -> tuple[int, int, float]:
    """Nearest (iy, ix) on a regular grid defined by 1-D x/y coordinate axes."""
    x_grid_1d = np.asarray(x_grid_1d, dtype=float)
    y_grid_1d = np.asarray(y_grid_1d, dtype=float)
    ix = int(np.argmin(np.abs(x_grid_1d - x)))
    iy = int(np.argmin(np.abs(y_grid_1d - y)))
    dist = float(np.hypot(x_grid_1d[ix] - x, y_grid_1d[iy] - y))
    return iy, ix, dist


def latlon_to_utm10(lat, lon):
    """lat/lon (WGS84) -> (x, y) in UTM10N metres. Scalars or array-likes."""
    x, y = _UTM10_TRANSFORMER.transform(lon, lat)
    return x, y


def read_ldb(path: Path) -> list[np.ndarray]:
    """Parse a Tekal .ldb file into a list of (n, 2) polyline arrays.

    Each block is a name line, then an "nrow ncol" line, then nrow data rows.
    Comment lines starting with "*" before a block are skipped. A block is
    further split into separate polylines wherever a dummy row appears (the
    Tekal break convention; this file uses "999.999 999.999", some tools use
    "-999 -999" -- both are caught by the >900-magnitude test below, which
    never collides with real UTM10 coordinates in this domain).
    """
    lines = Path(path).read_text().splitlines()
    n = len(lines)
    i = 0
    polylines: list[np.ndarray] = []

    while i < n:
        line = lines[i].strip()
        if not line or line.startswith("*"):
            i += 1
            continue
        i += 1  # consume the block-name line
        while i < n and (not lines[i].strip() or lines[i].strip().startswith("*")):
            i += 1
        dims = lines[i].split()
        nrow = int(dims[0])
        i += 1
        rows = []
        for _ in range(nrow):
            vals = lines[i].split()
            rows.append((float(vals[0]), float(vals[1])))
            i += 1
        arr = np.asarray(rows, dtype=float)

        is_dummy = (np.abs(np.abs(arr[:, 0]) - 999.0) < 50.0) & (np.abs(np.abs(arr[:, 1]) - 999.0) < 50.0)
        if is_dummy.any():
            start = 0
            for stop in np.where(is_dummy)[0]:
                if stop > start:
                    polylines.append(arr[start:stop])
                start = stop + 1
            if start < len(arr):
                polylines.append(arr[start:])
        else:
            polylines.append(arr)

    return polylines


# === EXTREME VALUE FIT ===

def pot_fit(daily: pd.Series, n_years: float, rng: np.random.Generator, return_peaks: bool = False) -> dict:
    """Declustered POT/GPD fit of a daily-maximum series.

    `daily` must be time-indexed (DatetimeIndex); NaN days are dropped before
    declustering. Returns a dict with status "ok", "dropped_<6yr" (n_years <
    MIN_YEARS, nothing fitted) or "insufficient_peaks" (fewer than k+1
    declustered peaks are available at lambda=POT_LAMBDA).
    """
    if n_years < MIN_YEARS:
        return {"status": "dropped_<6yr", "n_years": n_years}

    series = daily[np.isfinite(daily.to_numpy(dtype=float))]
    k = int(round(POT_LAMBDA * n_years))
    max_peaks = k + 1

    try:
        peaks = rx.decluster_peaks(series, SEP_HOURS, max_peaks=max_peaks)
    except ValueError:
        return {"status": "insufficient_peaks", "n_years": n_years}

    peaks_desc = np.sort(peaks.to_numpy(dtype=float))[::-1]
    if peaks_desc.size < max_peaks:
        return {"status": "insufficient_peaks", "n_years": n_years}

    threshold, excess, rate, n_peaks = rx.pot_sample(peaks_desc, n_years, POT_LAMBDA)
    sigma, xi = rx.gpd_fit_variant(excess, GPD_MODE)

    T = np.asarray(RETURN_PERIODS, dtype=float)
    rl = np.atleast_1d(rx.gpd_quantile(threshold, sigma, xi, rate, T))

    boot = rx.pot_bootstrap_quantiles(excess, threshold, rate, T, mode=GPD_MODE, rng=rng, n_boot=N_BOOT)
    lo = np.nanpercentile(boot, 100.0 * CI_ALPHA / 2.0, axis=0)
    hi = np.nanpercentile(boot, 100.0 * (1.0 - CI_ALPHA / 2.0), axis=0)

    sigma_exp, xi_exp = rx.gpd_fit_variant(excess, "exp")
    rl_exp = np.atleast_1d(rx.gpd_quantile(threshold, sigma_exp, xi_exp, rate, T))

    return {
        "status": "ok",
        "threshold": threshold,
        "sigma": sigma,
        "xi": xi,
        "rate": rate,
        "n_peaks": n_peaks,
        "n_years": n_years,
        "rl_1": rl[0], "rl_10": rl[1], "rl_100": rl[2],
        "lo_1": lo[0], "lo_10": lo[1], "lo_100": lo[2],
        "hi_1": hi[0], "hi_10": hi[1], "hi_100": hi[2],
        "rl_exp_1": rl_exp[0], "rl_exp_10": rl_exp[1], "rl_exp_100": rl_exp[2],
        "method": "POT-GPD-Lmom lam3 sep48h dailymax",
        # s6 station plots: the k fitted peaks (descending) and their excesses over the threshold
        **({"peaks": peaks_desc[:n_peaks].copy(), "excess": excess} if return_peaks else {}),
    }


# === OBS / PRODUCT HELPERS (Task 3 / Task 4) ===

def hourly_mean_centred(times: pd.DatetimeIndex, values: np.ndarray) -> pd.Series:
    """Mean of readings in [h-30min, h+30min), stamped at h, on an hourly index."""
    times = pd.DatetimeIndex(times)
    hour_bin = (times + pd.Timedelta(minutes=30)).floor("h")
    s = pd.Series(np.asarray(values, dtype=float), index=hour_bin)
    return s.groupby(level=0).mean().sort_index()


def bc_clamped(raw: np.ndarray, cnn_top: float) -> np.ndarray:
    """True where a raw CNN value sits above the bias-correction fit-window max."""
    return np.asarray(raw, dtype=float) > cnn_top


def _decode_station_ids(raw) -> list[str]:
    """Decode a station_id array that may hold bytes or str entries."""
    out = []
    for v in np.asarray(raw).tolist():
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
        out.append(str(v).strip())
    return out


def station_locations() -> pd.DataFrame:
    """Station table: station_id, network, lat, lon, x, y.

    IEM: all stations in the archive. NDBC: only stations with any finite
    wind_speed reading. USGS: the 4 moorings in USGS_FILES.
    """
    rows: list[dict] = []

    with netCDF4.Dataset(IEM_FILE) as ds:
        station_id = _decode_station_ids(ds.variables["station_id"][:])
        latitude = np.asarray(ds.variables["latitude"][:], dtype=float)
        longitude = np.asarray(ds.variables["longitude"][:], dtype=float)
        for sid, lat, lon in zip(station_id, latitude, longitude):
            rows.append({"station_id": sid, "network": "IEM", "lat": lat, "lon": lon})

    with netCDF4.Dataset(NDBC_FILE) as ds:
        station_id = _decode_station_ids(ds.variables["station_id"][:])
        latitude = np.asarray(ds.variables["latitude"][:], dtype=float)
        longitude = np.asarray(ds.variables["longitude"][:], dtype=float)
        ws = ds.variables["wind_speed"][:, :]
        ws = np.ma.filled(ws, np.nan) if np.ma.isMaskedArray(ws) else np.asarray(ws, dtype=float)
        has_data = np.any(np.isfinite(ws), axis=1)
        for sid, lat, lon, ok in zip(station_id, latitude, longitude, has_data):
            if ok:
                rows.append({"station_id": sid, "network": "NDBC", "lat": lat, "lon": lon})

    for sid, (_fpath, lat, lon, _height_m) in USGS_FILES.items():
        rows.append({"station_id": sid, "network": "USGS", "lat": lat, "lon": lon})

    df = pd.DataFrame(rows)
    x, y = latlon_to_utm10(df["lat"].to_numpy(), df["lon"].to_numpy())
    df["x"] = x
    df["y"] = y
    return df[["station_id", "network", "lat", "lon", "x", "y"]].reset_index(drop=True)

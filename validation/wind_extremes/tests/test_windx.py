"""Unit tests for windx_common.py helpers (Task 3 of the wind-return-levels plan).

NOTE on test 5 (pot_fit on a synthetic record): the plan's original version
(80 WY background noise + 240 storms at lambda=3) is statistically unsound,
because background noise can set the POT threshold instead of the storm
population. The corrected construction used here is 200 water years of daily
background U(0, 5) plus 1000 storm days placed >= 3 days apart, storm peak
= 10 + GPD(sigma=2, xi=-0.1) (fixed seed), i.e. a storm rate of 5/yr -- well
above the lambda=3 fit rate, so every one of the ~600 fitted peaks is a storm,
never background noise. pot_fit is called with n_years=200 at lambda=3, so it
picks the top 600 peaks (600 = round(3*200)). The analytic RP100 is
10 + (2/-0.1) * ((5*100)**-0.1 - 1) (rate = 5/yr, the true storm occurrence
rate, not the lambda=3 fit rate).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import windx_common as w


# --- test 1: daily_max known peak + NaN day ---

def test_daily_max_known_peak_and_nan_day():
    times = pd.date_range("2020-01-01", periods=48, freq="h")
    values = np.zeros(48)
    values[5] = 12.3          # known peak, day 0
    values[24:48] = 1.0       # day 1 baseline
    # day 1: knock out 13 hours -> only 11 valid -> below MIN_HOURS_PER_DAY (12) -> NaN
    values[24:37] = np.nan
    dmax, days, nvalid = w.daily_max(values, times)
    assert dmax.shape == (2,)
    assert dmax[0] == pytest.approx(12.3)
    assert nvalid[1] == 11
    assert np.isnan(dmax[1])


# --- test 2: daily_max raises on a non-midnight start ---

def test_daily_max_raises_on_bad_start_hour():
    times = pd.date_range("2020-01-01 05:00", periods=24, freq="h")
    values = np.arange(24, dtype=float)
    with pytest.raises(ValueError):
        w.daily_max(values, times)


# --- test 3: water_year boundary ---

def test_water_year_boundary():
    times = pd.DatetimeIndex(["2010-09-30", "2010-10-01"])
    wy = w.water_year(times)
    assert wy[0] == 2010
    assert wy[1] == 2011


# --- test 4: valid_water_years 79% vs 80% ---

def test_valid_water_years_threshold():
    # WY2001: Oct 1 2000 .. Sep 30 2001 -> 365 calendar days (2001 not a leap WY here)
    days = pd.date_range("2000-10-01", "2001-09-30", freq="D")
    n_cal_days = len(days)

    n_valid_79 = int(np.floor(0.79 * n_cal_days))
    dmax_79 = np.full(n_cal_days, np.nan)
    dmax_79[:n_valid_79] = 1.0
    frac_79 = n_valid_79 / n_cal_days
    assert frac_79 < 0.80
    out_79 = w.valid_water_years(days, dmax_79)
    assert out_79[2001] is False

    n_valid_80 = int(np.ceil(0.80 * n_cal_days))
    dmax_80 = np.full(n_cal_days, np.nan)
    dmax_80[:n_valid_80] = 1.0
    frac_80 = n_valid_80 / n_cal_days
    assert frac_80 >= 0.80
    out_80 = w.valid_water_years(days, dmax_80)
    assert out_80[2001] is True


# --- test 5: pot_fit on a synthetic record (corrected construction, see module docstring) ---

def test_pot_fit_synthetic_recovers_shape_and_rl100():
    rng_data = np.random.default_rng(12345)
    n_years = 200
    n_days = n_years * 365
    start = pd.Timestamp("1800-10-01")
    days = pd.date_range(start, periods=n_days, freq="D")

    background = rng_data.uniform(0.0, 5.0, size=n_days)

    # place 1000 storm days, each >= 3 days apart, within the record
    n_storms = 1000
    min_gap = 3
    storm_idx: list[int] = []
    candidates = rng_data.permutation(n_days)
    taken = np.zeros(n_days, dtype=bool)
    for idx in candidates:
        if len(storm_idx) >= n_storms:
            break
        lo = max(0, idx - min_gap)
        hi = min(n_days, idx + min_gap + 1)
        if not taken[lo:hi].any():
            storm_idx.append(idx)
            taken[lo:hi] = True
    storm_idx = np.sort(np.asarray(storm_idx))
    assert len(storm_idx) == n_storms

    sigma_true, xi_true = 2.0, -0.1
    u_unif = rng_data.uniform(0.0, 1.0, size=n_storms)
    # Inverse-CDF GPD sampling: with U the (uniform) exceedance probability,
    # X = (sigma/xi) * (U**(-xi) - 1); note the exponent is -xi, not xi.
    gpd_excess = (sigma_true / xi_true) * (u_unif ** (-xi_true) - 1.0)
    storm_peaks = 10.0 + gpd_excess

    values = background.copy()
    values[storm_idx] = storm_peaks

    series = pd.Series(values, index=days)
    rng_fit = np.random.default_rng(999)
    result = w.pot_fit(series, n_years=float(n_years), rng=rng_fit)

    assert result["status"] == "ok"
    assert abs(result["xi"] - xi_true) < 0.1

    true_storm_rate = n_storms / n_years  # 5 /yr
    analytic_rl100 = 10.0 + (sigma_true / xi_true) * ((true_storm_rate * 100.0) ** xi_true - 1.0)
    assert abs(result["rl_100"] - analytic_rl100) / analytic_rl100 < 0.05


# --- test 6: pot_fit monotonicity ---

def test_pot_fit_monotonicity():
    rng_data = np.random.default_rng(42)
    n_years = 80
    n_days = n_years * 365
    days = pd.date_range("1900-10-01", periods=n_days, freq="D")
    background = rng_data.uniform(0.0, 5.0, size=n_days)

    n_storms = 240
    min_gap = 3
    storm_idx: list[int] = []
    taken = np.zeros(n_days, dtype=bool)
    for idx in rng_data.permutation(n_days):
        if len(storm_idx) >= n_storms:
            break
        lo = max(0, idx - min_gap)
        hi = min(n_days, idx + min_gap + 1)
        if not taken[lo:hi].any():
            storm_idx.append(idx)
            taken[lo:hi] = True
    storm_idx = np.sort(np.asarray(storm_idx))

    u_unif = rng_data.uniform(0.0, 1.0, size=n_storms)
    values = background.copy()
    values[storm_idx] = 10.0 + (2.0 / -0.1) * (u_unif ** 0.1 - 1.0)

    series = pd.Series(values, index=days)
    result = w.pot_fit(series, n_years=float(n_years), rng=np.random.default_rng(7))

    assert result["status"] == "ok"
    assert result["threshold"] < result["rl_1"] < result["rl_10"] < result["rl_100"]
    assert result["lo_100"] < result["rl_100"] < result["hi_100"]


# --- test 7: pot_fit n_years < MIN_YEARS ---

def test_pot_fit_dropped_short_record():
    days = pd.date_range("2020-10-01", periods=5 * 365, freq="D")
    values = np.random.default_rng(1).uniform(0, 5, size=len(days))
    series = pd.Series(values, index=days)
    result = w.pot_fit(series, n_years=5, rng=np.random.default_rng(1))
    assert result["status"] == "dropped_<6yr"


# --- test 8: nearest_cell exact match ---

def test_nearest_cell_exact():
    x_grid = np.array([0.0, 2500.0, 5000.0, 7500.0])
    y_grid = np.array([1000.0, 3000.0, 5000.0])
    iy, ix, dist = w.nearest_cell(x_grid, y_grid, 5000.0, 3000.0)
    assert ix == 2
    assert iy == 1
    assert dist == 0.0


# --- test 9: hourly_mean_centred ---

def test_hourly_mean_centred():
    times = pd.DatetimeIndex(["2020-01-01 00:53", "2020-01-01 01:05", "2020-01-01 01:31"])
    values = np.array([1.0, 2.0, 3.0])
    s = w.hourly_mean_centred(times, values)
    t1 = pd.Timestamp("2020-01-01 01:00")
    t2 = pd.Timestamp("2020-01-01 02:00")
    assert s.loc[t1] == pytest.approx(1.5)   # mean of the 00:53 and 01:05 readings
    assert s.loc[t2] == pytest.approx(3.0)


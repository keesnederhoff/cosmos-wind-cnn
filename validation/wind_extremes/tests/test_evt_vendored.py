"""evt.py is a verbatim copy of rainlib objects: where rainlib is importable, results must be identical.

Skips in the cosmos_wind_cnn env (no rainlib). Run it in the acfcd_rain env.
Note: rainlib HEAD no longer has pooled_gpd_shape (removed 2026-10-08, commit 275d237),
so that function is checked only when rainlib still provides it.
"""
import numpy as np
import pytest

import evt

rainlib_rx = pytest.importorskip("rainlib.extremes")
rainlib_rt = pytest.importorskip("rainlib.trends")


def _assert_same(a, b) -> None:
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_same(a[k], b[k])
    elif hasattr(a, "to_numpy"):
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    else:
        np.testing.assert_array_equal(np.asarray(a, dtype=object if a is None else float),
                                      np.asarray(b, dtype=object if b is None else float))


def test_vendored_matches_rainlib():
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 20 * 365 * 24
    idx = pd.date_range("2000-01-01", periods=n, freq="h")
    series = pd.Series(rng.gamma(2.0, 3.0, n), index=idx)

    pk_a, pk_b = evt.decluster_peaks(series, 48.0), rainlib_rx.decluster_peaks(series, 48.0)
    _assert_same(pk_a, pk_b)
    peaks = np.sort(pk_a.to_numpy(dtype=float))[::-1]

    s_a, s_b = evt.pot_sample(peaks, 20.0, 3.0), rainlib_rx.pot_sample(peaks, 20.0, 3.0)
    _assert_same(tuple(s_a), tuple(s_b))
    excess, u, rate = s_a.excess, s_a.threshold, s_a.rate

    for mode in ("free", "exp"):
        _assert_same(evt.gpd_fit_variant(excess, mode), rainlib_rx.gpd_fit_variant(excess, mode))
    sigma, xi = evt.gpd_fit_variant(excess, "free")
    T = np.array([1.0, 10.0, 100.0])
    for t in T:
        _assert_same(evt.gpd_quantile(u, sigma, xi, rate, t), rainlib_rx.gpd_quantile(u, sigma, xi, rate, t))

    qa = evt.pot_bootstrap_quantiles(excess, u, rate, T, mode="free", rng=np.random.default_rng(1), n_boot=200)
    qb = rainlib_rx.pot_bootstrap_quantiles(excess, u, rate, T, mode="free", rng=np.random.default_rng(1), n_boot=200)
    _assert_same(qa, qb)

    annual = series.groupby(series.index.year).max().to_numpy()
    _assert_same(evt.pettitt_test(annual, np.arange(annual.size)), rainlib_rt.pettitt_test(annual, np.arange(annual.size)))

    if hasattr(rainlib_rx, "pooled_gpd_shape"):
        lst = [excess, excess[::2], excess[1::2]]
        _assert_same(evt.pooled_gpd_shape(lst), rainlib_rx.pooled_gpd_shape(lst))

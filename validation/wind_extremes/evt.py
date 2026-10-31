"""Vendored subset of the ACFCD `rainlib` extreme-value code, used by the wind return-level pipeline.

Source repo : acfcd_rain (private ACFCD repo), commit 902ae61 (2026-10-03).
Source files: rainlib/extremes.py and rainlib/trends.py in that repo (local clone: 01_meteorlogical_analysis_sfbay/acfcd_rain).
Objects      : GPD_SHAPE_TOL, _GPD_MODES, _clean_1d, _lmom_rows, decluster_peaks, _gpa_shape_lmom,
               gpd_fit_lmom_excess, gpd_quantile, PotSample, pot_sample, gpd_refit_rows, gpd_quantile_rows,
               gpd_fit_variant, pot_bootstrap_quantiles, pooled_gpd_shape (extremes.py) and pettitt_test (trends.py).
Bodies and docstrings are verbatim. Only the header, imports and N_BOOT are new.
Do not edit; re-copy if rainlib changes.

pooled_gpd_shape is needed by s6_station_eva.py. It was removed from rainlib HEAD in commit 275d237
(2026-10-08, "Remove dead code"), so it is taken from 902ae61 like the rest.
"""
from __future__ import annotations

from typing import Iterable, NamedTuple, Sequence  # noqa: F401  (annotations as in the source)

import numpy as np
import pandas as pd

N_BOOT = 2000  # rainlib.config default; windx always passes n_boot explicitly


GPD_SHAPE_TOL = 1e-6         # |xi| below this is treated as the exponential limit


_GPD_MODES = ("free", "fixed", "exp")


def _clean_1d(x, name: str = "x") -> np.ndarray:
    """Return `x` as a float 1-D array, raising if it holds any non-finite value."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        n_bad = int(np.sum(~np.isfinite(a)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s); refusing to drop them "
            "silently -- fix the record upstream"
        )
    return a


def _lmom_rows(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First three sample L-moments for each ROW of a 2-D array (delta_lib:78).

    Unbiased probability-weighted moments (Landwehr/Hosking b0, b1, b2).
    Returns (l1, l2, t3) as 1-D arrays of length a.shape[0].
    """
    n = a.shape[1]
    if n < 4:
        raise ValueError(f"need at least 4 values for L-moments, got {n}")
    xs = np.sort(a, axis=1)
    j = np.arange(1, n + 1, dtype=float)
    w1 = (j - 1.0) / (n - 1.0)
    w2 = w1 * (j - 2.0) / (n - 2.0)
    b0 = xs.mean(axis=1)
    b1 = (xs * w1).sum(axis=1) / n
    b2 = (xs * w2).sum(axis=1) / n
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    with np.errstate(divide="ignore", invalid="ignore"):
        t3 = np.where(l2 != 0.0, l3 / l2, 0.0)
    return l1, l2, t3


def decluster_peaks(series: pd.Series, min_separation_hours: float,
                    max_peaks: int | None = None) -> pd.Series:
    """Greedy declustering: keep peaks at least `min_separation_hours` apart.

    delta_lib:181. Candidates are taken in descending order of value; a
    candidate is accepted only if it is at least `min_separation_hours` from
    every already-accepted peak. This makes the many overlapping rolling-
    accumulation windows produced by one storm contribute a single peak.

    `series` must be time-indexed. Non-finite values are dropped here (they are
    the un-fillable head of a rolling window, not missing data); the returned
    peaks are all finite. Separation is enforced on the actual timestamps.

    `max_peaks` stops the pass once that many peaks are accepted. Because the
    pass runs in strictly descending order of value, stopping early returns
    exactly the `max_peaks` largest peaks of the full greedy result.

    Rejection is implemented by blanking the time window around each accepted
    peak (via searchsorted on the timestamps, so an absent 29 February in a
    no-leap model is handled exactly), which is equivalent to re-testing every
    accepted peak but costs O(1) per candidate.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("decluster_peaks needs a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("decluster_peaks needs a DatetimeIndex")
    s = series[np.isfinite(series.to_numpy(dtype=float))]
    if s.empty:
        raise ValueError("no finite values to decluster")

    v = s.to_numpy(dtype=float)
    t_ns = s.index.to_numpy().astype("datetime64[ns]").astype("int64")
    if np.any(np.diff(t_ns) <= 0):
        raise ValueError("decluster_peaks needs a strictly increasing time index")
    sep_ns = int(round(min_separation_hours * 3600.0 * 1e9))
    target = v.size if max_peaks is None else int(max_peaks)

    blocked = np.zeros(v.size, dtype=bool)
    kept_i: list[int] = []
    for idx in np.argsort(-v, kind="stable"):
        if blocked[idx]:
            continue
        kept_i.append(int(idx))
        if len(kept_i) >= target:
            break
        ti = t_ns[idx]
        lo = int(np.searchsorted(t_ns, ti - sep_ns + 1, side="left"))
        hi = int(np.searchsorted(t_ns, ti + sep_ns - 1, side="right"))
        blocked[lo:hi] = True
    if not kept_i:
        raise ValueError("declustering produced no peaks")
    keep = np.sort(np.asarray(kept_i, dtype=int))
    return s.iloc[keep]


def _gpa_shape_lmom(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Hosking GPA (threshold 0) L-moment shape: k = l1/l2 - 2.

    Shared by `gpd_fit_lmom_excess` and `gpd_refit_rows(mode="free")`
    (delta_lib:236 / pot_gpd_check.gpd_refit_rows:187). Non-positive `l2` is
    replaced with 1.0 before dividing so a vectorised caller can mask the
    result afterwards instead of raising mid-array; callers that need a single
    record to raise on a non-positive L-scale check that themselves first.
    """
    l2_safe = np.where(np.asarray(l2, dtype=float) > 0.0, l2, 1.0)
    return np.asarray(l1, dtype=float) / l2_safe - 2.0


def gpd_fit_lmom_excess(y) -> tuple[float, float]:
    """GPD (threshold 0) L-moment fit of a set of excesses. Returns (sigma, xi).

    delta_lib:236. Hosking's GPA with known lower bound: k = l1/l2 - 2 and
    alpha = (1+k) l1, converted to the standard convention xi = -k, sigma = alpha.
    """
    a = _clean_1d(y, "excesses")
    l1, l2, _ = _lmom_rows(a[None, :])
    if l2[0] <= 0.0:
        raise ValueError("L-scale of the excesses is non-positive")
    k = float(_gpa_shape_lmom(l1, l2)[0])
    sigma = (1.0 + k) * l1[0]
    if sigma <= 0.0:
        raise ValueError("fitted GPD scale is non-positive")
    return float(sigma), float(-k)


def gpd_quantile(threshold: float, sigma: float, xi: float, rate: float,
                 return_period: float | np.ndarray) -> float | np.ndarray:
    """Depth at return period T (years) from a POT/GPD fit, PDS convention.

    delta_lib:297. With `rate` independent peaks per year above `threshold`, the
    depth exceeded once per T years satisfies a per-peak exceedance probability
    of 1/(rate*T):

        x(T) = u + (sigma/xi) * ((rate*T)**xi - 1)        xi != 0
        x(T) = u + sigma * ln(rate*T)                     xi == 0

    Accepts scalar or array-like T. Requires rate*T >= 1 (above the threshold).
    """
    Tf = np.asarray(return_period, dtype=float)
    if np.any(Tf <= 0.0):
        raise ValueError("return period T must be positive")
    lam_t = float(rate) * Tf
    if np.any(lam_t < 1.0):
        raise ValueError("rate*T < 1 puts the quantile below the POT threshold")
    if abs(xi) < GPD_SHAPE_TOL:
        out = threshold + sigma * np.log(lam_t)
    else:
        out = threshold + (sigma / xi) * (lam_t ** xi - 1.0)
    return float(out) if np.ndim(return_period) == 0 else np.asarray(out, dtype=float)


class PotSample(NamedTuple):
    threshold: float           # (k+1)-th largest peak
    excess: np.ndarray         # the k excesses above it
    rate: float                # k / n_years, peaks per year
    k: int                     # number of excesses


def pot_sample(peaks_desc: np.ndarray, n_years: float, lam: float) -> PotSample:
    """Threshold, excesses and rate at `lam` exceedances/year (count rule).

    pot_gpd_check.pot_sample:174, unchanged: within each member x slice the
    threshold is the (k+1)-th largest independent peak with
    k = round(lam * n_years), so the exceedance rate is k / n_years on every
    slice and the PDS -> annual-RP mapping is identical on both sides of a
    change-factor ratio. run_loca2_potexp's RATE_TOL assert stays in the script.
    """
    peaks_desc = np.asarray(peaks_desc, dtype=float)
    k = int(round(lam * n_years))
    if peaks_desc.size < k + 1:
        raise ValueError(f"only {peaks_desc.size} peaks, need {k + 1}")
    u = float(peaks_desc[k])                    # (k+1)-th largest
    excess = peaks_desc[:k] - u
    rate = k / n_years
    return PotSample(u, excess, rate, k)


def gpd_refit_rows(sim: np.ndarray, mode: str, xi0: float | None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised GPD refit of each ROW of resampled excesses.

    pot_gpd_check.gpd_refit_rows:187. mode: 'free' (2-par L-moments), 'fixed'
    (xi frozen at xi0, sigma from the mean excess), 'exp' (xi = 0, sigma = mean
    excess). Returns (sigma, xi, ok).
    """
    l1, l2, _ = _lmom_rows(sim)
    if mode == "free":
        k = _gpa_shape_lmom(l1, l2)
        sigma = (1.0 + k) * l1
        xi = -k
        ok = (l2 > 0.0) & (sigma > 0.0)
    elif mode == "fixed":
        xi = np.full_like(l1, xi0)
        sigma = l1 * (1.0 - xi0)
        ok = sigma > 0.0
    elif mode == "exp":
        xi = np.zeros_like(l1)
        sigma = l1.copy()
        ok = sigma > 0.0
    else:
        raise ValueError(f"mode must be one of {_GPD_MODES}, got {mode!r}")
    return sigma, xi, ok


def gpd_quantile_rows(u: float, sigma: np.ndarray, xi: np.ndarray, rate: float, return_period
                      ) -> np.ndarray:
    """`gpd_quantile` broadcast over rows of (sigma, xi) and columns of T.

    pot_gpd_check.gpd_quantile_rows:213, unchanged (no rate*T >= 1 check; the
    caller's T is the published RETURN_PERIODS with rate ~ 2.5).
    """
    lam_t = rate * np.asarray(return_period, dtype=float)[None, :]
    xic = xi[:, None]
    sic = sigma[:, None]
    near = np.abs(xic) < GPD_SHAPE_TOL
    xis = np.where(near, 1.0, xic)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        q = np.where(near, u + sic * np.log(lam_t),
                     u + (sic / xis) * (lam_t ** xis - 1.0))
    return q


def gpd_fit_variant(excess: np.ndarray, mode: str = "exp",
                    xi0: float | None = None) -> tuple[float, float]:
    """Point GPD fit of one excess sample under one tail variant. Returns (sigma, xi).

    pot_gpd_check.fit_variant:226. 'exp' (xi = 0, sigma = mean excess) is the
    only production setting; 'free' and 'fixed' (sigma = l1 (1 - xi0)) are
    retained for the pot_gpd_check head-to-head.
    """
    excess = _clean_1d(excess, "excesses")
    if mode == "free":
        return gpd_fit_lmom_excess(excess)
    if mode not in _GPD_MODES:
        raise ValueError(f"mode must be one of {_GPD_MODES}, got {mode!r}")
    l1 = float(np.mean(excess))
    if mode == "fixed":
        if xi0 is None:
            raise ValueError("mode='fixed' needs xi0")
        return l1 * (1.0 - xi0), float(xi0)
    return l1, 0.0                              # exp


def pot_bootstrap_quantiles(excess: np.ndarray, u: float, rate: float, return_period, *,
                            mode: str = "exp", xi0: float | None = None, rng,
                            n_boot: int = N_BOOT) -> np.ndarray:
    """(n_boot, len(T)) NONparametric bootstrap quantile draws for one POT sample.

    pot_gpd_check.pot_boot_quantiles:236, byte-identical: the declustered
    excesses are resampled with replacement (`rng.choice(excess, (n_boot, n))`),
    the fit is redone under the variant's constraint, threshold and rate are
    held fixed (design choices of the POT construction).
    Draws whose refit fails or whose quantile is non-positive are NaN.
    """
    excess = np.asarray(excess, dtype=float)
    T = np.asarray(return_period, dtype=float)
    n = excess.size
    sim = rng.choice(excess, size=(n_boot, n), replace=True)
    sigma, xi, ok = gpd_refit_rows(sim, mode, xi0)
    q = gpd_quantile_rows(u, sigma, xi, rate, T)
    return np.where(ok[:, None] & np.isfinite(q) & (q > 0.0), q, np.nan)


def pooled_gpd_shape(excess_list: Iterable[np.ndarray], normalise: bool = True) -> float:
    """GPD shape of several members' pooled excesses, each mean-normalised.

    pot_gpd_check.pooled_hist_shape:246 without the peaks_store lookup: pass the
    per-member excess arrays (from `pot_sample`); each is divided by its own
    mean before pooling (`normalise=False` if already done), then one L-moment
    GPD fit on the pool. This is the xi0 of the 'fixed' variant.
    """
    pool = []
    for ex in excess_list:
        ex = np.asarray(ex, dtype=float)
        pool.append(ex / ex.mean() if normalise else ex)
    _, xi = gpd_fit_lmom_excess(np.concatenate(pool))
    return float(xi)


def pettitt_test(x, years=None) -> dict:
    """Non-parametric Pettitt single change-point test, with the change YEAR.

    rainfall_lib.pettitt_test:231 plus run_homogeneity.pettitt_change_year:50
    folded in. Returns {n, k_stat, change_index, p, mu_before, mu_after,
    change_year}; change_index is the 0-based index of the last point of the
    first segment IN THE NaN-STRIPPED series.

    `change_year`: NaN is dropped before testing, so change_index addresses the
    COMPRESSED series while `years` is the full one. Indexing `years` directly
    would shift the reported year by the number of NaN preceding the break -- a
    no-op on the gap-free long-term record, wrong on the short-term network with
    its wet-season outages. The surviving years are indexed instead:
    `change_year = years[~isnan(x)][change_index]`; do NOT "simplify" this back
    to years[change_index]. None when `years` is omitted, the test is degenerate
    (n < 4) or the index is out of range. normalisation_check:736-742 already
    dropna()s before calling, so its break year is unchanged.
    """
    x_full = np.asarray(x, dtype=float)
    valid = ~np.isnan(x_full)
    x = x_full[valid]
    n = x.size
    if n < 4:
        return {"n": n, "k_stat": float("nan"), "change_index": -1,
                "p": float("nan"), "mu_before": float("nan"), "mu_after": float("nan"),
                "change_year": None}
    ranks = pd.Series(x).rank().to_numpy()
    u = 2.0 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    t = int(np.argmax(np.abs(u)))
    k = float(np.abs(u[t]))
    p = min(1.0, 2.0 * np.exp(-6.0 * k * k / (n ** 3 + n ** 2)))
    mu_before = float(np.mean(x[:t + 1]))
    mu_after = float(np.mean(x[t + 1:])) if t + 1 < n else float("nan")
    change_year = None
    if years is not None:
        valid_years = np.asarray(years)[valid]
        if 0 <= t < valid_years.size:
            change_year = int(valid_years[t])
    return {"n": n, "k_stat": k, "change_index": t, "p": p,
            "mu_before": mu_before, "mu_after": mu_after, "change_year": change_year}

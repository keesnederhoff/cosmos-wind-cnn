"""Compare a fresh s2-s6 run with the golden snapshot of results (2026-10-03 / s6 2026-10-08).

Every *.csv and *.nc in GOLDEN is compared with the file of the same name in NEW, at rtol RTOL.
The s6 table lives under figures/ in the output root, so it is listed in FIGURE_CSVS and compared
with NEW.parent / "figures" / <name>. Prints one line per file (OK or DIFF with the largest
relative error and where it is) and exits 1 on any DIFF. A file missing in NEW counts as a DIFF.
Do not loosen RTOL: a difference means the vendored evt.py or the environment changed a number.
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

# === CONFIGURATION ===
GOLDEN = Path(r"g:\03-downscaling_meteo_cnn\archive\v3\wind_extremes\results_golden_20261003")
NEW = Path(r"g:\03-downscaling_meteo_cnn\wind_extremes_repo_check\results")
RTOL = 1e-9
FIGURE_CSVS = ["station_eva_table.csv"]   # golden copies in GOLDEN/figures/, new files in NEW.parent/figures/
# === END CONFIGURATION ===


def max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Largest |a-b| / max(|b|, tiny) over finite pairs; inf if the NaN patterns differ."""
    if a.shape != b.shape:
        return float("inf")
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    ok = ~np.isnan(a)
    if not ok.any():
        return 0.0
    d = np.abs(a[ok] - b[ok])
    return float(np.max(d / np.maximum(np.abs(b[ok]), np.finfo(float).tiny)))


def compare_csv(gold: Path, new: Path) -> str:
    g, n = pd.read_csv(gold), pd.read_csv(new)
    if list(g.columns) != list(n.columns):
        return f"DIFF columns {list(g.columns)} vs {list(n.columns)}"
    if len(g) != len(n):
        return f"DIFF rows {len(g)} vs {len(n)}"
    keys = [c for c in g.columns if g[c].dtype.kind not in "f"]
    if keys:
        g = g.sort_values(keys, kind="stable").reset_index(drop=True)
        n = n.sort_values(keys, kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(g, n, check_exact=False, rtol=RTOL)
        return "OK"
    except AssertionError:
        worst = ("", 0.0)
        for c in g.columns:
            if g[c].dtype.kind in "fiu":
                e = max_rel_err(g[c].to_numpy(float), n[c].to_numpy(float))
                if e > worst[1]:
                    worst = (c, e)
            elif not g[c].equals(n[c]):
                worst = (c, float("inf"))
                break
        return f"DIFF {worst[1]:.3e} (column {worst[0]})"


def _filled(v) -> np.ndarray:
    a = v[:]
    if np.ma.isMaskedArray(a):
        a = a.astype(float).filled(np.nan) if a.dtype.kind in "fiu" else a.filled("")
    return np.asarray(a)


def compare_nc(gold: Path, new: Path) -> str:
    worst = ("", 0.0)
    with netCDF4.Dataset(gold) as g, netCDF4.Dataset(new) as n:
        if set(g.variables) != set(n.variables):
            return f"DIFF variables {sorted(set(g.variables) ^ set(n.variables))}"
        for name in g.variables:
            a, b = _filled(g.variables[name]), _filled(n.variables[name])
            if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
                a, b = a.astype(float), b.astype(float)
                try:
                    np.testing.assert_allclose(a, b, rtol=RTOL, atol=0.0, equal_nan=True)
                except AssertionError:
                    e = max_rel_err(a, b)
                    if e >= worst[1]:
                        worst = (name, e)
            elif not np.array_equal(a, b):
                worst = (name, float("inf"))
    return "OK" if worst[0] == "" else f"DIFF {worst[1]:.3e} (variable {worst[0]})"


def main() -> int:
    jobs = []
    for gold in sorted(GOLDEN.glob("*.csv")) + sorted(GOLDEN.glob("*.nc")):
        jobs.append((gold, NEW / gold.name))
    for name in FIGURE_CSVS:
        jobs.append((GOLDEN / "figures" / name, NEW.parent / "figures" / name))
    n_ok = n_diff = 0
    for gold, new in jobs:
        label = f"{gold.parent.name}/{gold.name}" if gold.parent.name == "figures" else gold.name
        if not new.exists():
            status = "DIFF missing in NEW"
        elif gold.suffix == ".csv":
            status = compare_csv(gold, new)
        else:
            status = compare_nc(gold, new)
        print(f"{status:60s} {label}")
        n_ok += status == "OK"
        n_diff += status != "OK"
    print(f"\n{n_ok} OK, {n_diff} DIFF of {len(jobs)} files (rtol {RTOL:g})")
    return 1 if n_diff else 0


if __name__ == "__main__":
    sys.exit(main())

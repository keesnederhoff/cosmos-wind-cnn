#!/usr/bin/env python
"""Collect the v3 predictor-block campaign and apply the decision rule.

This exists because the single most repeated failure in this project has been
ranking configurations on differences smaller than the seed noise. It happened
with model size (a "perfectly monotonic bc8 > bc16 > bc24" trend spanning 0.032
against noise of 0.040, retracted), and again with the wave cells. So the rule
is enforced here in code rather than left to judgement:

    a block wins only if it clears the runner-up by more than 2 x the pooled
    standard error; otherwise the result is reported as a TIE.

Selection metric is validation twCRPS @ 10 m/s (LOWER is better) -- the quantity
the checkpoint was actually chosen on. Read from the checkpoint's own
'selection_metric' key where possible, since the archived training.yaml holds
file defaults rather than the per-run env overrides and is not trustworthy
provenance.

Usage:
    python scripts/qhead_collect.py [--results-root DIR] [--metric twcrps|crps]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import defaultdict

DEFAULT_ROOT = ("/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/"
                "sf_bay_rtma_v3/results")
LOG_DIR = "/home/cnederhoff/cosmos/cosmos-wind-cnn/sweep_logs"


def block_of(run_name):
    """qh_P3_s2 -> P3 ; c1_det_P0_s1 -> C1(det)"""
    if run_name.startswith("c1_"):
        return "C1(det)"
    m = re.match(r"qh_(P\d)_s\d+", run_name)
    return m.group(1) if m else run_name


def from_checkpoint(run_dir):
    """(selection_metric, epoch, extras) from best_model.pth, or None."""
    p = os.path.join(run_dir, "checkpoint", "best_model.pth")
    if not os.path.exists(p):
        return None
    try:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:                                    # noqa: BLE001
        return ("ERR", str(e)[:60], {})
    vm = ck.get("val_metrics", {}) or {}
    return (ck.get("selection_metric"), ck.get("epoch"),
            {"name": ck.get("selection_metric_name", "?"),
             "crps": vm.get("crps"), "p50_rmse": vm.get("rmse"),
             "std_ratio": vm.get("std_ratio"), "p50_bias": vm.get("p50_bias")})


def from_log(run_name):
    """Fallback: last 'Saved best model' value in the arm's log."""
    best, epochs, done = None, 0, False
    for f in glob.glob(os.path.join(LOG_DIR, "qhead_*.log")):
        with open(f, errors="ignore") as fh:
            txt = fh.read()
        if f"run={run_name} " not in txt and f"run={run_name}\n" not in txt:
            continue
        m = re.findall(r"Saved best model \((?:twCRPS|val_loss): ([0-9.]+)\)", txt)
        if m:
            best = float(m[-1])
        epochs = len(re.findall(r"^Epoch \d+/\d+", txt, re.M))
        done = "exit: 0" in txt or bool(re.search(r"arm \S+ exit:", txt))
        break
    return best, epochs, done


def mean_std(v):
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(v) / n
    if n == 1:
        return mu, float("nan"), 1
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))
    return mu, sd, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    runs = sorted(d for d in os.listdir(args.results_root)
                  if d.startswith(("qh_", "c1_")))
    if not runs:
        print("No campaign runs found under " + args.results_root)
        return

    print(f"{'run':<16} {'block':<9} {'twCRPS':>9} {'ep':>4} {'CRPS':>8} "
          f"{'P50rmse':>8} {'bias':>8} {'std_r':>7}  source")
    print("-" * 88)

    by_block = defaultdict(list)
    rows = []
    for rn in runs:
        rd = os.path.join(args.results_root, rn)
        ck = from_checkpoint(rd)
        src = "ckpt"
        extras = {}
        if ck is None or ck[0] is None or ck[0] == "ERR":
            val, ep, _done = from_log(rn)
            src = "log"
        else:
            val, ep, extras = ck[0], ck[1], ck[2]
        if val is None:
            print(f"{rn:<16} {block_of(rn):<9} {'--':>9} {'--':>4}"
                  f"{'':>26}  (not started)")
            continue
        by_block[block_of(rn)].append(float(val))
        rows.append(rn)
        print(f"{rn:<16} {block_of(rn):<9} {float(val):>9.4f} "
              f"{(ep if ep is not None else -1):>4} "
              f"{_f(extras.get('crps')):>8} {_f(extras.get('p50_rmse')):>8} "
              f"{_f(extras.get('p50_bias')):>8} {_f(extras.get('std_ratio')):>7}  {src}")

    print()
    print("=" * 88)
    print("BLOCK MEANS  (validation twCRPS @ 10 m/s -- LOWER IS BETTER)")
    print("=" * 88)
    print(f"{'block':<10} {'n':>3} {'mean':>9} {'std':>9} {'se':>9}   seeds")
    stats = {}
    for blk in sorted(by_block):
        v = by_block[blk]
        mu, sd, n = mean_std(v)
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        stats[blk] = (mu, sd, se, n)
        print(f"{blk:<10} {n:>3} {mu:>9.4f} {sd:>9.4f} {se:>9.4f}   "
              + " ".join(f"{x:.4f}" for x in sorted(v)))

    ranked = sorted((b for b in stats if stats[b][3] > 0), key=lambda b: stats[b][0])
    if len(ranked) < 2:
        print("\nNot enough completed blocks to rank yet.")
        return

    print()
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    best, second = ranked[0], ranked[1]
    mu_b, _, se_b, n_b = stats[best]
    mu_s, _, se_s, n_s = stats[second]
    gap = mu_s - mu_b
    pooled_se = math.sqrt((se_b ** 2 if se_b == se_b else 0)
                          + (se_s ** 2 if se_s == se_s else 0))
    print(f"  best      : {best}  {mu_b:.4f}  (n={n_b})")
    print(f"  runner-up : {second}  {mu_s:.4f}  (n={n_s})")
    print(f"  gap       : {gap:.4f}   2 x pooled SE = {2*pooled_se:.4f}")
    if n_b < 2 or n_s < 2:
        print("\n  INCOMPLETE -- need >=2 seeds per block before ranking means anything.")
    elif gap > 2 * pooled_se:
        print(f"\n  WINNER: {best} clears the runner-up by more than 2 x SE.")
    else:
        print(f"\n  TIE: the gap is INSIDE 2 x SE. Do NOT rank these blocks -- this is"
              f"\n  exactly the trap that forced a retraction in v2, where a perfectly"
              f"\n  monotonic size trend spanning 0.032 sat inside seed noise of 0.040.")
        near = [b for b in ranked if stats[b][3] >= 2
                and stats[b][0] - mu_b <= 2 * pooled_se]
        print(f"  Statistically indistinguishable: {', '.join(near)}")

    if "C1(det)" in stats and stats["C1(det)"][3] >= 2:
        qb = [b for b in ranked if b != "C1(det)"]
        if qb:
            mu_q = stats[qb[0]][0]
            mu_c = stats["C1(det)"][0]
            print(f"\n  HEAD CONTROL: best quantile block {qb[0]} {mu_q:.4f} vs "
                  f"deterministic C1 {mu_c:.4f}")
            print("  NOTE twCRPS is only comparable here if BOTH were scored the same"
                  "\n  way; the deterministic arm selects on val_loss, so compare these"
                  "\n  on the Phase 5 grid-point metrics, not on this column.")


def _f(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else "--"


if __name__ == "__main__":
    main()

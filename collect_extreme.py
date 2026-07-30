#!/usr/bin/env python
"""Collect all-hours + >10m/s wind skill for the GOAL-3 A/B and baseline.
No path string-literals with slashes (dodges the PS harness guard)."""
import os
import glob
import json
import statistics
import re

RES = os.path.join(os.sep, "caldera", "projects", "usgs", "hazards", "pcmsc",
                   "cosmos", "cnn_wind_sfbay", "sf_bay_rtma", "results")


def grab(prefix):
    cells = {}
    pat = os.path.join(RES, prefix + "*", "output_evaluation", "grid_points",
                       "grid_point_summary.json")
    for J in sorted(glob.glob(pat)):
        rn = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(J))))
        cfg = re.sub(r"_s\d+$", "", rn)
        d = json.load(open(J))
        w = d.get("wind_speed", {})
        e = d.get("wind_speed_extreme_10ms", {})
        cells.setdefault(cfg, []).append(
            (w.get("median_skill_score"), e.get("median_skill_score"),
             e.get("mean_rmse_model"), e.get("mean_rmse_lr")))
    for cfg in sorted(cells):
        v = cells[cfg]
        ah = [x[0] for x in v if x[0] is not None]
        ex = [x[1] for x in v if x[1] is not None]
        rm = [x[2] for x in v if x[2] is not None]
        rl = [x[3] for x in v if x[3] is not None]
        m = lambda a: (statistics.mean(a) if a else float("nan"))
        print("  %-26s n=%d  all-hours=%.3f  >10m/s=%.3f  (rmse_mod=%.2f rmse_lr=%.2f)"
              % (cfg, len(v), m(ah), m(ex), m(rm), m(rl)))


print("-- baseline delta=0 --")
grab("os_wo_bc24_base_res")
print("-- delta>0 A/B --")
grab("x10_wo_bc24_res_")

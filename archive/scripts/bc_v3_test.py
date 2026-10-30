#!/usr/bin/env python3
'''Phase 6b: does the v3 quantile product still need bias correction?

The v3 premise was that peak under-prediction is a consequence of training a
CONDITIONAL MEAN, and that predicting a distribution fixes it at the source. If
that holds, the post-hoc quantile-mapping correction that bought v2 about +0.06
skill should now buy close to nothing -- and that is a falsifiable check on the
whole exercise, not a tuning knob.

Two deliberate departures from scripts/bias_correct.py:

1. THE MAP IS FITTED ON VAL, NOT TRAIN. The stock script fits on the train
   period. For v2 that was fine. Here it is not: every quantile arm selected its
   best checkpoint at epoch 0-1, so the weights have barely fitted the train
   period and their train-period error distribution is not representative of
   out-of-sample behaviour. VAL is out-of-sample for the weights and disjoint
   from TEST.

2. FILES ARE NAMED EXPLICITLY. The stock script globs full_record_*.nc and takes
   the LARGEST match. This run directory now holds two same-sized full_record
   files (val window and test window), so that heuristic would pick between them
   essentially at random.

Reuses fit_quantile_maps / apply_maps / _skill_block from bias_correct.py so the
correction and the scoring are identical to what produced the v2 numbers -- the
comparison is only meaningful if the method is unchanged.
'''
import argparse
import json
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bias_correct import fit_quantile_maps, apply_maps, _skill_block  # noqa: E402

ROOT = "/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay"
CASE = "sf_bay_rtma_v3"
SHARED = os.path.join(ROOT, CASE, "results", "v3data", "data_processed")
VAL_FILE = "full_record_ERA5_20240314_20250206.nc"
TEST_FILE = "full_record_ERA5_20250206_20260101.nc"


def _speed(ds, u="hr_u", v="hr_v"):
    return np.sqrt(ds[u].astype("float32") ** 2 + ds[v].astype("float32") ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--nquant", type=int, default=200)
    # Inference filename prefix. Empty reproduces the Phase 6b behaviour exactly.
    # The wrap-up arms hold several checkpoints in one run dir, distinguished only
    # by this prefix (speed_ for best_speed.pth), so without it the fit would
    # silently pick up whichever checkpoint left an unprefixed file -- a
    # different model from the one being bias-corrected.
    ap.add_argument("--prefix", default="",
                    help="inference filename prefix, e.g. speed_")
    args = ap.parse_args()

    val_file = args.prefix + VAL_FILE
    test_file = args.prefix + TEST_FILE
    inf_dir = os.path.join(ROOT, CASE, "results", args.run, "output_inference")
    vp, tp = os.path.join(inf_dir, val_file), os.path.join(inf_dir, test_file)
    for p in (vp, tp):
        if not os.path.exists(p):
            sys.exit("ABORT missing %s" % p)

    inf_val, inf_test = xr.open_dataset(vp), xr.open_dataset(tp)
    val, test = (xr.open_dataset(os.path.join(SHARED, s + ".nc"))
                 for s in ("val", "test"))

    # ---- fit on VAL -------------------------------------------------------
    cnn_v, rtma_v = xr.align(_speed(inf_val), _speed(val), join="inner")
    n_fit = cnn_v.sizes["time"]
    print("fit on VAL: %d samples (%s -> %s)"
          % (n_fit, str(cnn_v.time.values[0])[:13], str(cnn_v.time.values[-1])[:13]))
    # An empty inner join is the exact failure the stock script hides: it would
    # produce all-NaN maps and write a garbage file. Refuse instead.
    if n_fit < 1000:
        sys.exit("ABORT only %d overlapping fit samples -- refusing to fit" % n_fit)
    cnn_q, rtma_q = fit_quantile_maps(cnn_v.values, rtma_v.values, args.nquant)

    # ---- apply to TEST ----------------------------------------------------
    u = inf_test["hr_u"].values.astype("float32")
    v = inf_test["hr_v"].values.astype("float32")
    s = np.sqrt(u ** 2 + v ** 2)
    s_bc = apply_maps(s, cnn_q, rtma_q)
    _EPS = 1e-6            # keeps direction bit-exact; see bias_correct.py
    ratio = np.where(s > _EPS, np.maximum(s_bc, _EPS) / np.maximum(s, _EPS), 1.0)
    u_bc, v_bc = (u * ratio).astype("float32"), (v * ratio).astype("float32")

    out = xr.Dataset(
        {"hr_u": (("time", "y", "x"), u_bc), "hr_v": (("time", "y", "x"), v_bc)},
        coords={c: inf_test[c] for c in ("time", "y", "x") if c in inf_test.coords},
    )
    if "crs" in inf_test:
        out["crs"] = inf_test["crs"]
    for vv in ("hr_u", "hr_v"):
        out[vv].attrs = dict(inf_test[vv].attrs,
                             bias_corrected="per-cell quantile map vs RTMA, FITTED ON VAL")
    op = os.path.join(inf_dir, "BCVAL_%s" % test_file)
    out.to_netcdf(op, encoding={vv: {"zlib": True, "complevel": 4,
                                     "_FillValue": np.float32(np.nan)}
                                for vv in ("hr_u", "hr_v")})
    print("wrote %s" % op)

    # ---- score raw vs BC on TEST -----------------------------------------
    tru, e5 = _speed(test), _speed(test, "lr_u", "lr_v")
    raw = _speed(inf_test)
    bc = np.sqrt(out["hr_u"] ** 2 + out["hr_v"] ** 2)
    tru, e5, raw, bc = xr.align(tru, e5, raw, bc, join="inner")
    print("score on TEST: %d samples" % tru.sizes["time"])
    res = {"run": args.run, "fit_period": "val", "n_fit": int(n_fit),
           "n_test": int(tru.sizes["time"]),
           "raw": _skill_block(raw.values, tru.values, e5.values),
           "bc_val_fitted": _skill_block(bc.values, tru.values, e5.values)}
    res["delta_bc_minus_raw"] = {
        k: res["bc_val_fitted"][k] - res["raw"][k] for k in res["raw"]}
    ed = os.path.join(ROOT, CASE, "results", args.run, "output_evaluation")
    os.makedirs(ed, exist_ok=True)
    with open(os.path.join(ed, "bc_v3_val_fitted.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

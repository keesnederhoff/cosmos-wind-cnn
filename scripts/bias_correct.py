#!/usr/bin/env python3
"""
Post-hoc bias-correction for the CNN wind product via per-grid-cell empirical
quantile mapping (CDF matching) of CNN speed -> RTMA target speed.

Motivation: obs-validation showed the CNN systematically under-predicts wind
speed (~0.77 m/s low bias) and badly under-predicts peaks (negative top-decile
skill) because MSE/MAE training regresses toward the mean. Quantile mapping is a
monotone transform that repairs the marginal distribution (including the tail)
without touching the model.

Method (no leakage):
  * Fit one monotone quantile map per grid cell on the TRAIN period only, from
    CNN speed (inference `hr_u/hr_v`) vs RTMA truth speed (train.nc `hr_u/hr_v`).
  * Apply the fitted maps to the CNN speed over the WHOLE record; preserve wind
    direction by rescaling u,v by corrected_speed/original_speed.
  * Write `BC_<...>.nc` (uppercase BC prefix so it does NOT match the pipeline's
    `full_record_*.nc` eval glob and cannot hijack a normal eval).

Optional --evaluate scores RAW vs BC vs ERA5 wind-speed skill on the held-out
TEST period (all-hours, >10 m/s, energy-weighted q2/q3), referenced to ERA5, so
you can confirm the correction helps the tail without hurting all-hours.
"""
import argparse
import json
import os

import numpy as np
import xarray as xr

CAL_DEFAULT = "/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay"


def _speed(ds, u="hr_u", v="hr_v"):
    return np.sqrt(ds[u].astype("float32") ** 2 + ds[v].astype("float32") ** 2)


def fit_quantile_maps(cnn_tr, rtma_tr, nquant):
    """Return (cnn_q, rtma_q) each shape (Q, y, x), monotone in the Q axis."""
    q = np.linspace(0.0, 1.0, nquant)
    cnn_q = np.nanquantile(cnn_tr, q, axis=0).astype("float32")   # (Q,y,x)
    rtma_q = np.nanquantile(rtma_tr, q, axis=0).astype("float32")
    # Enforce strictly-increasing xp for np.interp (flat CDF regions -> ties).
    cnn_q = np.maximum.accumulate(cnn_q, axis=0)
    cnn_q += (np.arange(nquant, dtype="float32")[:, None, None]) * 1e-5
    rtma_q = np.maximum.accumulate(rtma_q, axis=0)
    return cnn_q, rtma_q


def apply_maps(cnn_speed, cnn_q, rtma_q):
    """Map cnn_speed (T,y,x) through per-cell maps; NaNs preserved."""
    T, Y, X = cnn_speed.shape
    out = np.full_like(cnn_speed, np.nan, dtype="float32")
    for iy in range(Y):
        for ix in range(X):
            col = cnn_speed[:, iy, ix]
            m = np.isfinite(col)
            if not m.any():
                continue
            out[m, iy, ix] = np.interp(col[m], cnn_q[:, iy, ix], rtma_q[:, iy, ix])
    return out


def _skill_block(mod_s, tru_s, e5_s):
    """Median-across-cells skill vs ERA5: all-hours, >10 m/s, energy-wt q2/q3."""
    T, Y, X = mod_s.shape
    ss, ss_ext, ss_ew2, ss_ew3, biases = [], [], [], [], []
    for iy in range(Y):
        for ix in range(X):
            t = tru_s[:, iy, ix]
            mo = mod_s[:, iy, ix]
            e5 = e5_s[:, iy, ix]
            mask = np.isfinite(t) & np.isfinite(mo) & np.isfinite(e5)
            if mask.sum() < 10:
                continue
            t, mo, e5 = t[mask], mo[mask], e5[mask]
            rm = np.sqrt(np.mean((mo - t) ** 2))
            re = np.sqrt(np.mean((e5 - t) ** 2))
            if re > 0:
                ss.append(1.0 - rm / re)
            biases.append(np.mean(mo - t))
            ext = t > 10.0
            if ext.sum() >= 10:
                rme = np.sqrt(np.mean((mo[ext] - t[ext]) ** 2))
                ree = np.sqrt(np.mean((e5[ext] - t[ext]) ** 2))
                if ree > 0:
                    ss_ext.append(1.0 - rme / ree)
            for q, acc in ((2, ss_ew2), (3, ss_ew3)):
                w = t ** q
                wsum = w.sum()
                if wsum > 0:
                    wrm = np.sqrt((w * (mo - t) ** 2).sum() / wsum)
                    wre = np.sqrt((w * (e5 - t) ** 2).sum() / wsum)
                    if wre > 0:
                        acc.append(1.0 - wrm / wre)
    med = lambda a: float(np.nanmedian(a)) if a else float("nan")
    return {
        "skill_allhours": med(ss),
        "skill_ext10": med(ss_ext),
        "skill_ew_q2": med(ss_ew2),
        "skill_ew_q3": med(ss_ew3),
        "mean_bias": float(np.nanmean(biases)) if biases else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="os_wo_bc24_base_res_s2")
    ap.add_argument("--case", default="sf_bay_rtma")
    ap.add_argument("--results-root", default=CAL_DEFAULT)
    ap.add_argument("--nquant", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="debug: cap #timesteps applied/written (0 = full)")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.join(args.results_root, args.case, "results", args.run)
    dp = os.path.join(run_dir, "data_processed")
    inf_dir = os.path.join(run_dir, "output_inference")
    import glob
    inf_files = sorted(glob.glob(os.path.join(inf_dir, "full_record_*.nc")),
                       key=os.path.getsize)
    assert inf_files, "no full_record_*.nc in %s" % inf_dir
    inf_path = inf_files[-1]
    print("Inference : %s" % inf_path)

    inf = xr.open_dataset(inf_path)
    train = xr.open_dataset(os.path.join(dp, "train.nc"))

    # --- Fit on the TRAIN period only (inner-join times to align exactly) ---
    cnn_tr, rtma_tr = xr.align(
        _speed(inf).sel(time=slice(str(train.time.values[0]), str(train.time.values[-1]))),
        _speed(train), join="inner")
    print("Fit samples: %d (train %s -> %s)" % (
        cnn_tr.sizes["time"], str(cnn_tr.time.values[0])[:10],
        str(cnn_tr.time.values[-1])[:10]))
    cnn_q, rtma_q = fit_quantile_maps(cnn_tr.values, rtma_tr.values, args.nquant)
    train.close()

    # --- Apply to the whole record ---
    if args.max_steps:
        inf = inf.isel(time=slice(0, args.max_steps))
    u = inf["hr_u"].values.astype("float32")
    v = inf["hr_v"].values.astype("float32")
    s = np.sqrt(u ** 2 + v ** 2)
    s_bc = apply_maps(s, cnn_q, rtma_q)
    # Direction MUST be exactly invariant here: u, v are rescaled by a positive
    # scalar, which preserves atan2. Two degenerate cases break that and inject
    # spurious 0-degree (due north) headings:
    #   s    ~ 0  -> the input vector has no direction to preserve.
    #   s_bc == 0 -> the quantile map sent a small speed to exactly zero, so
    #                u_bc = v_bc = 0 and atan2(0, 0) = 0.
    # Flooring the corrected speed at _EPS keeps the output collinear with the
    # input: magnitude error <= 1e-6 m/s, direction bit-exact. This was measured
    # as a ~0.2-0.3 deg direction DEGRADATION for every flavour after BC.
    _EPS = 1e-6
    s_bc_safe = np.maximum(s_bc, _EPS)
    ratio = np.where(s > _EPS, s_bc_safe / np.maximum(s, _EPS), 1.0)
    u_bc = (u * ratio).astype("float32")
    v_bc = (v * ratio).astype("float32")

    out = xr.Dataset(
        {"hr_u": (("time", "y", "x"), u_bc), "hr_v": (("time", "y", "x"), v_bc)},
        coords={c: inf[c] for c in ("time", "y", "x") if c in inf.coords},
    )
    if "crs" in inf:
        out["crs"] = inf["crs"]
    out["hr_u"].attrs = dict(inf["hr_u"].attrs, bias_corrected="per-cell quantile map vs RTMA (train period)")
    out["hr_v"].attrs = dict(inf["hr_v"].attrs, bias_corrected="per-cell quantile map vs RTMA (train period)")
    base = os.path.basename(inf_path).replace("full_record_", "").replace(".nc", "")
    out_path = os.path.join(inf_dir, "BC_%s.nc" % base)
    enc = {vv: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
           for vv in ("hr_u", "hr_v")}
    out.to_netcdf(out_path, encoding=enc)
    print("Wrote     : %s" % out_path)

    # --- Optional held-out TEST evaluation ---
    if args.evaluate:
        test = xr.open_dataset(os.path.join(dp, "test.nc"))
        t0, t1 = str(test.time.values[0]), str(test.time.values[-1])
        tru = _speed(test)                       # RTMA truth
        e5 = _speed(test, "lr_u", "lr_v")        # ERA5 baseline
        raw = _speed(inf.sel(time=slice(t0, t1)))
        bc = np.sqrt(out["hr_u"].sel(time=slice(t0, t1)) ** 2 +
                     out["hr_v"].sel(time=slice(t0, t1)) ** 2)
        tru, e5, raw, bc = xr.align(tru, e5, raw, bc, join="inner")
        print("Eval samples (test): %d" % tru.sizes["time"])
        res = {
            "period": "%s .. %s" % (t0[:10], t1[:10]),
            "n_time": int(tru.sizes["time"]),
            "raw_cnn": _skill_block(raw.values, tru.values, e5.values),
            "bias_corrected": _skill_block(bc.values, tru.values, e5.values),
        }
        eval_dir = os.path.join(run_dir, "output_evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "bias_correction_test.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        test.close()

    inf.close()
    print("DONE")


if __name__ == "__main__":
    main()

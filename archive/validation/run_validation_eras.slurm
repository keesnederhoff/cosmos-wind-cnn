#!/bin/bash
#SBATCH --job-name=val_eras
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=val_eras_%a_%A.log
set -euo pipefail

# Three-era OBSERVATION validation of the v3 shipping recipe.
#
# This is the only track in the project where the reference is not the training
# target. RTMA is what the CNN was trained to reproduce, so beating RTMA on a
# grid means nothing; stations are independent. E1 is the sharpest test in the
# campaign -- 2000-2010 predates RTMA entirely, so no gridded truth exists there
# and the model is extrapolating two decades outside its training window.
#
# ONE ERA PER ARRAY TASK, not one job. The comparable v3b run did 31 models x
# 0.9 yr in 2h18m (~5 min per model-year); this matrix is ~148 model-years, so a
# single job would sit near the 24 h wall with three eras' results hostage to it.

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export KMP_DUPLICATE_LIB_OK=TRUE

BUNDLE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_DATA_ROOT="$BUNDLE"
export COSMOS_VALIDATION_OUTPUT_ROOT="$BUNDLE/results"
mkdir -p "$COSMOS_VALIDATION_OUTPUT_ROOT"

cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation

# --- Fail loudly before burning a day on a partial leaderboard --------------
# The era arms are MULTIFILE products, so the usual u_file existence check does
# not apply -- audit the glob instead. A pattern resolving to zero files is
# dropped silently by the engine; one resolving to too many has picked up the
# overlapping day-3/day-5 leftovers and would corrupt the time axis.
echo "=== era inference audit ==="
python - <<'PY'
import sys; sys.path.insert(0, '.')
import config, pandas as pd, xarray as xr
bad = []
for m in config.V3_ERA_MODELS:
    c = config.MODELS[m]
    fs = sorted(c['data_dir'].glob(c['u_pattern']))
    if len(fs) != 27:
        bad.append(f"{m}: glob resolved {len(fs)} files, expected 27")
        continue
    ds = xr.open_mfdataset([str(f) for f in fs], concat_dim='time', combine='nested',
                           chunks={'time': 200}, data_vars='minimal',
                           coords='minimal', compat='override')
    ti = pd.Index(pd.to_datetime(ds['time'].values))
    tk = ti[~ti.duplicated(keep='first')]
    exp = int((tk[-1] - tk[0]).total_seconds() // 3600) + 1
    print(f"  {m}: {len(fs)} files, {len(tk)} unique hours, "
          f"{tk[0]} .. {tk[-1]}, gaps={exp - len(tk)}", flush=True)
    if not tk.is_monotonic_increasing:
        bad.append(f"{m}: time axis not monotonic after dedupe")
    if exp != len(tk):
        bad.append(f"{m}: {exp - len(tk)} missing hours in the record")
for m in ('ERA5', 'CONUS404', 'RTMA-SFbay'):
    if m not in config.MODELS:
        bad.append(f"reference product {m} not registered")
if bad:
    for b in bad:
        print("  FAIL", b)
    raise SystemExit("ABORT: era inputs are not sound.")
print("  all era arms sound", flush=True)
PY

export PYTHONUNBUFFERED=1

# VAL_RUN_TAG appends to every era's output directory so a re-run cannot
# overwrite an already-published set. Unset reproduces the original directory
# names exactly (obsE1_2000-2010, ...), so this is backward compatible.
# The 2026-08-27 re-run uses VAL_RUN_TAG=v20260810, which keeps the 2026-08-14
# baseline intact beside it for a before/after comparison.
TAG=${VAL_RUN_TAG:-}
echo "VAL_RUN_TAG='$TAG'  VAL_ERA_END='${VAL_ERA_END:-<default 2026-08-11>}'"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    echo "======== obs era E1 (2000-2010, pre-RTMA): IEM + NDBC pooled ========"
    VAL_ERA=E1 VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_eras \
      VAL_OUTDIR_SUFFIX="$TAG" python run_validation.py
    ;;
  1)
    echo "======== obs era E2 (2011-2019, unseen): IEM + NDBC pooled ========"
    VAL_ERA=E2 VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_eras \
      VAL_OUTDIR_SUFFIX="$TAG" python run_validation.py
    ;;
  2)
    echo "======== obs era E3 (2020-2026, training era): IEM + NDBC pooled ========"
    VAL_ERA=E3 VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_eras \
      VAL_OUTDIR_SUFFIX="$TAG" python run_validation.py
    ;;
  3)
    # USGS moorings: E3 only. Every mooring starts after 2020-01-22, and the
    # anemometers sit at 1.2-4.9 m rather than 10 m, so these are reported as
    # their own group and never pooled into the headline above.
    echo "======== obs era E3: USGS moorings (separate group) ========"
    VAL_ERA=E3 VAL_USGS=1 VAL_ONLY_GROUPS=USGS \
      VAL_OUTDIR_SUFFIX="${TAG}_usgs" python run_validation.py
    ;;
esac

echo "DONE task $SLURM_ARRAY_TASK_ID"

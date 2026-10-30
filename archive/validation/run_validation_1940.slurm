#!/bin/bash
#SBATCH --job-name=val_1940
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=val_1940_%a_%A.log
set -euo pipefail

# Six-era OBSERVATION validation of the 1940-2026 product (seed r1b_do010_s3).
# Em2 1940-59 | Em1 1960-79 | E0 1980-99 | E1 2000-10 | E2 2011-19 | E3 2020-26
# One era per array task (~5 min per model-year). Output dirs ALWAYS carry the
# _v1940 suffix so the published 2026-08 era results cannot be overwritten.

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1

BUNDLE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_DATA_ROOT="$BUNDLE"
export COSMOS_VALIDATION_OUTPUT_ROOT="$BUNDLE/results"
cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation

echo "=== 1940 product audit ==="
python - <<'PY'
import sys; sys.path.insert(0, '.')
import config, pandas as pd, xarray as xr
bad = []
for m in config.V3_1940_MODELS:
    c = config.MODELS[m]
    fs = sorted(c['data_dir'].glob(c['u_pattern']))
    if len(fs) != 87:
        bad.append(f"{m}: glob resolved {len(fs)} files, expected 87")
        continue
    ts = []
    for f in fs:
        with xr.open_dataset(f) as d:
            ts.append(pd.to_datetime(d['time'].values))
    ti = pd.Index(ts[0].append(ts[1:]))
    tk = ti[~ti.duplicated(keep='first')]
    exp = int((tk[-1] - tk[0]).total_seconds() // 3600) + 1
    print(f"  {m}: {len(fs)} files, {len(tk)} unique hours, {tk[0]} .. {tk[-1]}, "
          f"gaps={exp - len(tk)}", flush=True)
    if not tk.is_monotonic_increasing:
        bad.append(f"{m}: time axis not monotonic after dedupe")
    if exp != len(tk):
        bad.append(f"{m}: {exp - len(tk)} missing hours")
if bad:
    for b in bad:
        print("  FAIL", b)
    raise SystemExit("ABORT: 1940 product is not sound.")
print("  1940 product sound", flush=True)
PY

ERAS=(Em2 Em1 E0 E1 E2 E3)
E=${ERAS[$SLURM_ARRAY_TASK_ID]}
echo "======== obs era $E: IEM + NDBC pooled, 1940 product ========"
VAL_ERA=$E VAL_ERA_SET=1940 VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_1940 \
  VAL_OUTDIR_SUFFIX=v1940 python run_validation.py
echo "DONE task $SLURM_ARRAY_TASK_ID ($E)"

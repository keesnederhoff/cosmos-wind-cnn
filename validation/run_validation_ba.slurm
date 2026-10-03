#!/bin/bash
#SBATCH --job-name=val_ba
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=val_ba_%a_%A.log
set -euo pipefail

# Six-era OBSERVATION validation of the P50 products: CNN-RTMA (raw) and
# CNN-RTMA-BA (observation-based factor, scripts/ba_fit_factor.py), against
# ERA5 and RTMA-SFbay (E2/E3). Em2 1940-59 | Em1 1960-79 | E0 1980-99 |
# E1 2000-10 | E2 2011-19 | E3 2020-26, one era per array task.
#   VAL_STATION_HEIGHTS=1 : real anemometer heights (the BA fit used them)
#   VAL_USGS=1            : moorings loaded as their own group (weight 0 in the
#                           IEM+NDBC headline ranking; they only have data in E3)
#   VAL_OUTDIR_SUFFIX     : dirs end in _ba_q90 so nothing published is overwritten
# E3 is IN-SAMPLE for CNN-RTMA-BA at the stations; E2 is the honest era with RTMA.

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1

BUNDLE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_DATA_ROOT="$BUNDLE"
export COSMOS_VALIDATION_OUTPUT_ROOT="$BUNDLE/results"
export VAL_STATION_HEIGHTS=1 VAL_USGS=1
cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation

echo "=== P50 product audit ==="
python - <<'PY'
import sys; sys.path.insert(0, '.')
import config, numpy as np, xarray as xr
bad = []
for m in config.P50_MODELS:
    c = config.MODELS[m]
    f = c['u_file']
    if not f.exists():
        bad.append(f"{m}: missing {f}"); continue
    with xr.open_dataset(f) as d:
        t = d['time'].values
        dd = np.diff(t).astype('timedelta64[h]').astype(int)
        exp = int((t[-1] - t[0]).astype('timedelta64[h]').astype(int)) + 1
        print(f"  {m}: {t.size} hours {str(t[0])[:13]} .. {str(t[-1])[:13]}, gaps={exp - t.size}, "
              f"steps!=1h={int((dd != 1).sum())}, vars={list(d.data_vars)}", flush=True)
        if t.size != exp or (dd != 1).any():
            bad.append(f"{m}: time axis not hourly-contiguous")
        for v in ('u10', 'v10', 'wind_speed'):
            if v not in d:
                bad.append(f"{m}: missing variable {v}")
if bad:
    for b in bad:
        print("  FAIL", b)
    raise SystemExit("ABORT: P50 products are not sound.")
print("  P50 products sound", flush=True)
PY

ERAS=(Em2 Em1 E0 E1 E2 E3)
E=${ERAS[$SLURM_ARRAY_TASK_ID]}
echo "======== obs era $E: IEM + NDBC pooled (USGS own group), P50 raw + BA ========"
VAL_ERA=$E VAL_ERA_SET=ba VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_ba \
  VAL_OUTDIR_SUFFIX=ba_q90 python run_validation.py
echo "DONE task $SLURM_ARRAY_TASK_ID ($E)"

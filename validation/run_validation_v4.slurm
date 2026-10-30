#!/bin/bash
#SBATCH --job-name=v4_val
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/v4_logs/%x_%A_%a.log

# Observation validation of the v4 P50 products (V4-RTMA raw, V4-RTMA-BA) beside the v3
# P50 products (CNN-RTMA, CNN-RTMA-BA), ERA5 and RTMA-SFbay. One era per array task:
#   Em2 1940-59 | Em1 1960-79 | E0 1980-99 | E1 2000-10 | E2 2011-19 | E3 2020-26 | V4T WY2026
# V4T = the v4 HOLD-OUT year (2025-10-01 -> 2026-08-10, ends where the v3 products end so
# both are scored on the SAME window/stations; v3 never trained on any of it).
# E3 is IN-SAMPLE for the BA product at the stations; E2 is the honest era with RTMA.
# Output dirs end _v4 so nothing published is overwritten.
set -euo pipefail
source /home/cnederhoff/cosmos/cosmos-wind-cnn/scripts/v4_env.sh
ERAS=(Em2 Em1 E0 E1 E2 E3 V4T)
E=${ERAS[$SLURM_ARRAY_TASK_ID]}
v4_begin "val_${E}"
export VAL_STATION_HEIGHTS=1 VAL_USGS=1 VAL_ERA_SET=v4
cd "$REPO/validation"

echo "=== v4 + v3 P50 product audit ==="
python - <<'PY'
import sys; sys.path.insert(0, '.')
import config, numpy as np, xarray as xr
bad = []
for m in list(config.V4_P50_MODELS) + list(config.P50_MODELS):
    f = config.MODELS[m]['u_file']
    if not f.exists():
        bad.append(f'{m}: missing {f}'); continue
    with xr.open_dataset(f) as d:
        t = d['time'].values
        dd = np.diff(t).astype('timedelta64[h]').astype(int)
        print(f'  {m}: {t.size} h {str(t[0])[:13]} .. {str(t[-1])[:13]} steps!=1h={int((dd != 1).sum())} {list(d.data_vars)}', flush=True)
        if (dd != 1).any():
            bad.append(f'{m}: time axis not hourly-contiguous')
        for v in ('u10', 'v10', 'wind_speed'):
            if v not in d:
                bad.append(f'{m}: missing variable {v}')
if bad:
    print('\n'.join('  FAIL ' + b for b in bad)); raise SystemExit('ABORT: P50 products are not sound.')
print('  P50 products sound', flush=True)
PY

echo "======== obs era $E: IEM + NDBC pooled (USGS its own group), v4 vs v3 ========"
VAL_ERA=$E VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_v4 VAL_OUTDIR_SUFFIX=v4 python run_validation.py
echo "DONE $E"

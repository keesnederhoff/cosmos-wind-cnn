#!/bin/bash
#SBATCH --job-name=cnn_rot
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=06:00:00
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/logs/%x_%j.log

# Stage 1: rotate the RTMA u/v from NCEP Lambert grid-relative to earth-relative
# (true east/north) and PROVE it against the v3 files on the overlap. The gate lives
# inside scripts/rotate_rtma_earthrel.py (exit 1 = chain stops).
# Resume-safe: if both *_earthrel.nc files already exist they are re-verified, not rewritten.
set -euo pipefail
source /home/cnederhoff/cosmos/cosmos-wind-cnn/scripts/env.sh
stage_begin rotate
quota_guard 0.1
RAW=$CASE_DIR/raw_data
U=$RAW/RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10_earthrel.nc
V=$RAW/RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10_earthrel.nc
if [ -s "$U" ] && [ -s "$V" ]; then
  echo "earthrel files exist -> verify only"
  python scripts/rotate_rtma_earthrel.py --verify-only --n-gate 400
else
  python scripts/rotate_rtma_earthrel.py --n-gate 400
fi
ls -la "$U" "$V"

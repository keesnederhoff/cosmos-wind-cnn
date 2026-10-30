#!/bin/bash
#SBATCH --job-name=v4_rot
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=06:00:00
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/v4_logs/%x_%j.log

# Stage 1: rotate the v4 RTMA u/v from NCEP Lambert grid-relative to earth-relative
# (true east/north) and PROVE it against the v3 files on the overlap. The gate lives
# inside scripts/rotate_rtma_earthrel.py (exit 1 = chain stops).
# Resume-safe: if both *_earthrel.nc files already exist they are re-verified, not rewritten.
set -euo pipefail
source /home/cnederhoff/cosmos/cosmos-wind-cnn/scripts/v4_env.sh
v4_begin rotate
v4_quota_guard 0.1
RAW=$V4/raw_data
U=$RAW/RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10_earthrel.nc
V=$RAW/RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10_earthrel.nc
if [ -s "$U" ] && [ -s "$V" ]; then
  echo "earthrel files exist -> verify only"
  python scripts/rotate_rtma_earthrel.py --verify-only --n-gate 400
else
  python scripts/rotate_rtma_earthrel.py --n-gate 400
fi
ls -la "$U" "$V"

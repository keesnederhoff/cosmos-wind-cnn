#!/bin/bash
#SBATCH --job-name=stage_raw_caldera
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=pcmsc
#SBATCH --time=02:00:00
#SBATCH --output=stage_raw_%j.log

set -uo pipefail
BASE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn/data
REPO=/home/cnederhoff/cosmos/cosmos-wind-cnn
echo "host=$(hostname) START=$(date)"
mkdir -p "$BASE/sf_bay_conus404/raw" "$BASE/sf_bay_rtma/raw"
lfs setstripe -c 7 "$BASE/sf_bay_conus404/raw" 2>/dev/null || true
lfs setstripe -c 7 "$BASE/sf_bay_rtma/raw" 2>/dev/null || true
echo "--- sf_bay_conus404 raw ---"
rsync -aL --ignore-existing "$REPO/case_studies/sf_bay_conus404/data/raw/" "$BASE/sf_bay_conus404/raw/"; echo "sf_bay_conus404 rc=$?"
echo "--- sf_bay_rtma raw ---"
rsync -aL "$REPO/case_studies/sf_bay_rtma/data/raw/" "$BASE/sf_bay_rtma/raw/"; echo "rtma rc=$?"
echo "DONE=$(date)"
du -sh "$BASE/sf_bay_conus404/raw" "$BASE/sf_bay_rtma/raw"

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

# Raw inputs live OUTSIDE the repo at  $COSMOS_DATA_ROOT/<case>/raw_data/  (the
# pipeline reads them via utils/config.get_data_dir). This job copies the raw
# NetCDFs onto caldera fast (lustre) storage.
DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn

# SOURCE: where the raw data is staged before this copy. It is NO LONGER in the
# repo -- set SRC to wherever you rsync'd the raw NetCDFs (e.g. a /home staging
# area). Expected layout: $SRC/<case>/raw_data/.
SRC=/home/cnederhoff/cosmos/raw_data_staging

echo "host=$(hostname) START=$(date)  SRC=$SRC  DATA_ROOT=$DATA_ROOT"
for case in sf_bay_conus404 sf_bay_rtma; do
    dest="$DATA_ROOT/$case/raw_data"
    mkdir -p "$dest"
    lfs setstripe -c 7 "$dest" 2>/dev/null || true
    echo "--- $case raw_data ---"
    rsync -aL --ignore-existing "$SRC/$case/raw_data/" "$dest/"; echo "$case rc=$?"
done
echo "DONE=$(date)"
du -sh "$DATA_ROOT/sf_bay_conus404/raw_data" "$DATA_ROOT/sf_bay_rtma/raw_data"

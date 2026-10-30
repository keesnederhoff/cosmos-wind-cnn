#!/bin/bash
#SBATCH --job-name=cnn_yearskill
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/logs/%x_%A_%a.log

# NON-GATING side job: per-calendar-year skill of the REFERENCES (ERA5, RTMA-SFbay) against
# IEM+NDBC stations, 2011..2026. Locates the RTMA quality break the user believes is at 2020.
# If it is not, restart the chain from the preprocess stage with a different start_date
# in case_studies/sf_bay_rtma/configs/preprocessing.yaml. Never blocks anything:
# `|| true` keeps a failure here out of the chain.
set -uo pipefail
source /home/cnederhoff/cosmos/cosmos-wind-cnn/scripts/env.sh
Y=$((2011 + SLURM_ARRAY_TASK_ID))
stage_begin "yearskill_${Y}"
cd "$REPO/validation"
VAL_ERA=EY VAL_Y0=$Y VAL_MODELS=ERA5,RTMA-SFbay VAL_USGS=0 VAL_STATION_HEIGHTS=1 \
  VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_peryear \
  python run_validation.py || true
exit 0

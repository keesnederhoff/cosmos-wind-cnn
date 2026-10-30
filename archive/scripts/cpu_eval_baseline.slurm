#!/bin/bash
#SBATCH --job-name=bl_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=02:00:00
#SBATCH --output=bl_eval_%j.log

# Re-score the weight=0 baseline (Goal-2 winner) with the NEW energy-weighted
# grid metric, on the same 100-point / 2011-2026 basis as the wave arms, so the
# wave cells have an apples-to-apples reference. Eval-only: reads the existing
# full_record inference on disk (no model, no GPU).
set -uo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export SWEEP_BASE_CHANNELS=24
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_RESIDUAL=1
export SWEEP_NONWIND_WEIGHT=0.0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name os_wo_bc24_base_res_s2 \
    --skip-preprocess --skip-train --skip-inference \
    --inference-start 2011-01-01 --inference-end 2026-01-01
echo "bl_eval exit: $?"

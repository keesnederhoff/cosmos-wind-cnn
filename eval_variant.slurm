#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --account=pcmsc
#SBATCH --time=6:00:00
#SBATCH --output=sweep_logs/%x_%A.log

set -euo pipefail
: "${RN:?need RN}"; : "${BC:?need BC}"; : "${SL:?need SL}"; : "${DO:?need DO}"
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export SWEEP_BASE_CHANNELS=$BC
export SWEEP_SEQ_LEN=$SL
export SWEEP_DROPOUT=$DO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=18

# eval needs the processed splits (sweep only symlinked memmap/stats/gridref)
DP=$ROOT/sf_bay_rtma/results/$RN/data_processed
mkdir -p "$DP"
for s in train val test; do
  ln -sfn "$ROOT/sf_bay_rtma/results/3737874/data_processed/$s.nc" "$DP/$s.nc"
done

echo "VARIANT EVAL: $RN  bc=$BC sl=$SL do=$DO  | inference 2011-01-01..2026-01-01 + eval"
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name "$RN" \
    --skip-preprocess --skip-train \
    --inference-start 2011-01-01 --inference-end 2026-01-01 \
    --gpus 1 --num-workers 8

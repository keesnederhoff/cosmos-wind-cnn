#!/bin/bash
#SBATCH --job-name=bf_wp3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --account=pcmsc
#SBATCH --time=4:00:00
#SBATCH --array=0-5%2
#SBATCH --output=sweep_logs/%x_%a_%A.log

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/backfill_wavep3_manifest.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN BC START END <<< "$LINE"
echo "BACKFILL(wave-p3) idx=$SLURM_ARRAY_TASK_ID run=$RN bc=$BC  $START .. $END"

# The archived training.yaml carries FILE defaults, not what this arm actually
# trained with -- the real settings came through SWEEP_* env at launch. Replay
# them explicitly, read off the checkpoint's own config dict:
#   base_channels 24, sequence_length 6, dropout 0.0, residual_learning True,
#   additional_inputs [lr_cloud, static_terrain]
# Residual is the dangerous one: residual-off on a residual-trained model does
# not error, it just produces silently wrong output.
# NOTE the wave-loss settings (loss_wave_weight/wave_exp/wave_scale) are NOT set
# here on purpose -- step_inference only replays ARCHITECTURE parameters, and
# loss weights do not affect the built model.
export SWEEP_BASE_CHANNELS=$BC
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_RESIDUAL=1
export SWEEP_ADD_INPUTS=lr_cloud,static_terrain
export SWEEP_NONWIND_WEIGHT=0.0
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name "$RN" \
    --skip-preprocess --skip-train --skip-eval \
    --inference-start "$START" --inference-end "$END" \
    --gpus 1 --num-workers 8
echo "backfill(wave-p3) $RN $START..$END exit: $?"

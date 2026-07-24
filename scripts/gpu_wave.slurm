#!/bin/bash
#SBATCH --job-name=wave
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11%2
#SBATCH --output=sweep_logs/%x_%a_%A.log

# Wave-energy weighting sweep on the Goal-2 wind-only winner arch
# (bc24, residual ON, no derived terrain, nonwind_weight=0, sl6/do0.0/lr3e-4).
# The ONLY per-arm knobs are the smooth wave term: WAVE_WEIGHT and WAVE_EXP.
# Coupled: train (2-GPU DDP) -> inference 2011-2026 -> skill eval in one job, so
# each arm self-produces grid_point_summary.json incl. the new
# wind_speed_energy_weighted block.

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
SHARED=$ROOT/sf_bay_rtma/results/v2data/data_processed
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/wave_manifest.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN WW WE <<< "$LINE"
echo "WAVE idx=$SLURM_ARRAY_TASK_ID run=$RN wave_weight=$WW wave_exp=$WE"

# Fixed Goal-2 winner architecture
export SWEEP_BASE_CHANNELS=24
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_LR=0.0003
export SWEEP_RESIDUAL=1
export SWEEP_NONWIND_WEIGHT=0.0
export SWEEP_EXTREME_WEIGHT=0.0          # hard-threshold delta OFF; using wave term
export SWEEP_EXTREME_THRESH=10.0
# Smooth wave-energy weighting (this experiment's variable)
export SWEEP_WAVE_WEIGHT=$WW
export SWEEP_WAVE_EXP=$WE
export SWEEP_WAVE_SCALE=10.0
# NOTE: SWEEP_ADD_INPUTS intentionally unset -> config default 8-ch base
# (lr_cloud + static_terrain), i.e. NO 5 derived-terrain statics.

export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((30000 + SLURM_ARRAY_TASK_ID))
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RD=$ROOT/sf_bay_rtma/results/$RN
mkdir -p "$RD/data_processed"
for f in memmap normalization_stats.pkl target_grid_reference.nc train.nc val.nc test.nc; do
  ln -sfn "$SHARED/$f" "$RD/data_processed/$f"
done
echo "$RN bc=24 res=1 nw=0 wave_weight=$WW wave_exp=$WE wave_scale=10 sl6 do0.0 lr3e-4" > "$RD/arm_params.txt"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name $RN \
    --gpus 2 --num-workers 8 \
    --skip-preprocess \
    --inference-start 2011-01-01 --inference-end 2026-01-01
echo "wave $RN exit: $?"

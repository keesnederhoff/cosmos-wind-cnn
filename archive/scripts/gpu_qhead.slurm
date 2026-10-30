#!/bin/bash
#SBATCH --job-name=qhead
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=1-00:00:00
#SBATCH --array=0-20%3
#SBATCH --output=sweep_logs/%x_%a_%A.log

# v3 predictor-block campaign: 6 blocks x 3 seeds + 3 deterministic controls.
# Modelled on the proven gpu_opt_search.slurm pattern (recovered from e900bc2~1).
#
# TWO NON-NEGOTIABLES, both learned the hard way:
#
#  --exclusive --mem=0 : the 2026-07 sweep ran gpu:2 NON-exclusive backfill and
#      9 of 21 arms died with torch.OutOfMemoryError at the first forward pass,
#      because a foreign process held ~10.7 GB of the 16 GB V100. Not a config
#      bug. Whole-node reservation is the only reliable fix.
#
#  TRAIN-ONLY : the 2026-07 factorial had every arm write a ~38 GB inference
#      file, ~1.6 TB total, and the last 6 arms died on EDQUOT mid-run. Skill
#      evaluation happens in Phase 5 on winners only.

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
SHARED=$ROOT/sf_bay_rtma_v3/results/v3data/data_processed
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/qhead_manifest.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN HEAD QW GUSTW ADD <<< "$LINE"
echo "QHEAD idx=$SLURM_ARRAY_TASK_ID run=$RN head=$HEAD qw_exp=$QW gust_w=$GUSTW"
echo "QHEAD add_inputs=$ADD"

export SWEEP_HEAD=$HEAD
export SWEEP_QW_EXP=$QW
export SWEEP_GUST_WEIGHT=$GUSTW
export SWEEP_ADD_INPUTS=$ADD
export SWEEP_BASE_CHANNELS=24
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_LR=0.0003
# Wind-only: under the deterministic control this stops the gust channel taking
# gradient, keeping C1 a like-for-like wind control. Ignored by the quantile head,
# which weights gust explicitly via SWEEP_GUST_WEIGHT.
export SWEEP_NONWIND_WEIGHT=0.0
# NOTE: SWEEP_RESIDUAL is deliberately NOT set. The quantile head already anchors
# speed and direction on the interpolated ERA5 field, and build_wind3dunet RAISES
# if both are enabled -- the anchor would be applied twice.

export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((30500 + SLURM_ARRAY_TASK_ID))
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Share the ONE preprocessed store so every arm trains on bit-identical data --
# that is what keeps the block comparison controlled. train/val/test.nc are
# symlinked too so a later standalone eval does not silently skip (it checks for
# them and prints "No processed splits found").
RD=$ROOT/sf_bay_rtma_v3/results/$RN
mkdir -p "$RD/data_processed"
for f in memmap normalization_stats.pkl target_grid_reference.nc train.nc val.nc test.nc; do
  ln -sfn "$SHARED/$f" "$RD/data_processed/$f"
done
echo "$RN head=$HEAD qw_exp=$QW gust_w=$GUSTW bc=24 seq_len=6 do=0.0 lr=3e-4 add=$ADD" \
  > "$RD/arm_params.txt"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

# Train only. Inference + evaluation are Phase 5, on winners only.
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma_v3 \
    --run-name "$RN" \
    --gpus 2 --num-workers 8 \
    --skip-preprocess \
    --skip-inference \
    --skip-eval
echo "arm $RN exit: $?"

#!/bin/bash
#SBATCH --job-name=wrapup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=1-12:00:00
#SBATCH --array=0-7%3
#SBATCH --output=sweep_logs/%x_%a_%A.log

# 7-day wrap-up sweep: 5 quantile arms (qw_exp=1 + regularisation screen) and
# the C2a data-volume control (deterministic, 2011+ store, 3 seeds).
#
# Derived from gpu_qhead.slurm. Two differences beyond the manifest columns:
#   STORE selects the preprocessed store, so one harness drives both the 2020+
#         and 2011+ arms; and
#   CASE  selects the case study, because C2a needs a config without the gust
#         pair (v2data has no hr_gust/lr_gust).
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
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/wrapup_manifest.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN HEAD QW GUSTW DO WD STORE CASE ADD <<< "$LINE"
if [ -z "$ADD" ]; then
  echo "FATAL: manifest line $((SLURM_ARRAY_TASK_ID+1)) parsed short -- got '$LINE'"
  exit 1
fi

case "$STORE" in
  v3data) SHARED=$ROOT/sf_bay_rtma_v3/results/v3data/data_processed ;;
  v2data) SHARED=$ROOT/sf_bay_rtma/results/v2data/data_processed ;;
  longdata) SHARED=$ROOT/sf_bay_rtma_long/results/longdata/data_processed ;;
  *)      echo "FATAL: unknown STORE '$STORE'"; exit 1 ;;
esac
[ -d "$SHARED/memmap" ] || { echo "FATAL: no memmap under $SHARED"; exit 1; }
[ -d "/home/cnederhoff/cosmos/cosmos-wind-cnn/case_studies/$CASE" ] \
  || { echo "FATAL: no case study $CASE"; exit 1; }

echo "WRAPUP idx=$SLURM_ARRAY_TASK_ID run=$RN head=$HEAD qw_exp=$QW gust_w=$GUSTW"
echo "WRAPUP dropout=$DO weight_decay=$WD store=$STORE case=$CASE"
echo "WRAPUP shared=$SHARED"
echo "WRAPUP T=$(python -c "import json,sys;print(json.load(open('$SHARED/memmap/train/meta.json'))['T'])" 2>/dev/null || echo '?')"
echo "WRAPUP add_inputs=$ADD"

export SWEEP_HEAD=$HEAD
export SWEEP_QW_EXP=$QW
export SWEEP_GUST_WEIGHT=$GUSTW
export SWEEP_ADD_INPUTS=$ADD
export SWEEP_BASE_CHANNELS=24
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=$DO
export SWEEP_WEIGHT_DECAY=$WD
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
RD=$ROOT/$CASE/results/$RN
mkdir -p "$RD/data_processed"
for f in memmap normalization_stats.pkl target_grid_reference.nc train.nc val.nc test.nc; do
  ln -sfn "$SHARED/$f" "$RD/data_processed/$f"
done
echo "$RN head=$HEAD qw_exp=$QW gust_w=$GUSTW bc=24 seq_len=6 do=$DO wd=$WD lr=3e-4 store=$STORE case=$CASE add=$ADD" \
  > "$RD/arm_params.txt"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

# Train only. Inference + evaluation are Phase 5, on winners only.
srun python scripts/run_training_pipeline.py \
    --case-study "case_studies/$CASE" \
    --run-name "$RN" \
    --gpus 2 --num-workers 8 \
    --skip-preprocess \
    --skip-inference \
    --skip-eval
echo "arm $RN exit: $?"

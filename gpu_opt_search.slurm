#!/bin/bash
#SBATCH --job-name=opt_srch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=1-00:00:00
#SBATCH --array=0-47%3
#SBATCH --output=sweep_logs/%x_%a_%A.log

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
SHARED=$ROOT/sf_bay_rtma/results/v2data/data_processed
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/opt_search_manifest.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN BC RES NW ADD <<< "$LINE"
echo "OPTSRCH idx=$SLURM_ARRAY_TASK_ID run=$RN bc=$BC residual=$RES nonwind_weight=$NW add_inputs=$ADD"

# All hyperparams fixed; only goal(nw) x size(bc) x terrain(add) x residual(res) vary.
export SWEEP_BASE_CHANNELS=$BC
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_LR=0.0003
export SWEEP_RESIDUAL=$RES
export SWEEP_NONWIND_WEIGHT=$NW
export SWEEP_ADD_INPUTS=$ADD
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29000 + SLURM_ARRAY_TASK_ID))
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Share the ONE preprocessed dataset (no dup). train/val/test.nc symlinked so
# step_evaluate does not silently skip.
RD=$ROOT/sf_bay_rtma/results/$RN
mkdir -p "$RD/data_processed"
for f in memmap normalization_stats.pkl target_grid_reference.nc train.nc val.nc test.nc; do
  ln -sfn "$SHARED/$f" "$RD/data_processed/$f"
done
echo "$RN bc=$BC res=$RES nw=$NW add=$ADD seq_len=6 do=0.0 lr=0.0003" > "$RD/arm_params.txt"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

# Coupled: train (2-GPU DDP) -> inference 2011-2026 (bounded-span, no leak) -> skill eval.
# Exclusive+mem=0 gives the whole node (~377G) so the nondeterministic eval-leak has headroom.
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name $RN \
    --gpus 2 --num-workers 8 \
    --skip-preprocess \
    --inference-start 2011-01-01 --inference-end 2026-01-01
echo "arm $RN exit: $?"

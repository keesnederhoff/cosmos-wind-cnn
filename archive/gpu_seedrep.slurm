#!/bin/bash
#SBATCH --job-name=seedrep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7%2
#SBATCH --output=sweep_logs/%x_%a_%A.log
set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
SHARED=$ROOT/sf_bay_rtma/results/v2data/data_processed
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/seedrep_manifest.txt
LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN NWW <<< "$LINE"
echo "SEEDREP idx=$SLURM_ARRAY_TASK_ID run=$RN nonwind_weight=$NWW (no manual seed -> fresh init)"

export SWEEP_BASE_CHANNELS=8
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_LR=0.0003
export SWEEP_RESIDUAL=0
export SWEEP_ADD_INPUTS=lr_cloud,static_terrain
export SWEEP_NONWIND_WEIGHT=$NWW
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((30100 + SLURM_ARRAY_TASK_ID))
export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RD=$ROOT/sf_bay_rtma/results/$RN
mkdir -p "$RD/data_processed"
for f in memmap normalization_stats.pkl target_grid_reference.nc train.nc val.nc test.nc; do
  ln -sfn "$SHARED/$f" "$RD/data_processed/$f"
done
echo "$RN nonwind_weight=$NWW bc8 sl6 do0.0 lr3e-4 residual=0" > "$RD/seedrep_params.txt"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma --run-name $RN \
    --gpus 2 --num-workers 8 --skip-preprocess --skip-inference --skip-eval
echo "seedrep $RN exit: $?"

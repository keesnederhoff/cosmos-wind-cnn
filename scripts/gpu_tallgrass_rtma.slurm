#!/bin/bash
#SBATCH --job-name=cosmos_wind_rtma
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=kees.nederhoff@deltares-usa.us
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --account=pcmsc
#SBATCH --time=2-00:00:00
#SBATCH --output=gpu_pipeline_rtma_%j.log

set -euo pipefail

# ── Storage location (OFF /home, on caldera PCMSC project space) ─────────────
# Raw inputs are read from   $COSMOS_DATA_ROOT/<case_name>/raw
# Run outputs are written to $COSMOS_RESULTS_ROOT/<case_name>/<job_id>
# To relocate, edit these two lines, or pass --data-root/--results-root
# to scripts/run_training_pipeline.py.
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn/data
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn/results

CASE_STUDY=case_studies/sf_bay_rtma
RUN_NAME=$SLURM_JOB_ID
LOG_DIR="${COSMOS_RESULTS_ROOT}/sf_bay_rtma/${RUN_NAME}/logs"

trap 'mkdir -p "$LOG_DIR" && cp "${SLURM_SUBMIT_DIR}/gpu_pipeline_rtma_${SLURM_JOB_ID}.log" "$LOG_DIR/" 2>/dev/null || true' EXIT

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29501
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=18

echo "Master node : $MASTER_ADDR"
echo "Case study  : $CASE_STUDY"
echo "Data root   : $COSMOS_DATA_ROOT"
echo "Results root: $COSMOS_RESULTS_ROOT"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn

cd /home/cnederhoff/cosmos/cosmos-wind-cnn

which python
python -c "import cosmos_wind_cnn; print('import ok:', cosmos_wind_cnn.__file__)"

srun python scripts/run_training_pipeline.py \
    --case-study $CASE_STUDY \
    --run-name $RUN_NAME \
    --gpus 4 \
    --num-workers 8

#!/bin/bash
#SBATCH --job-name=cosmos_wind_cnn
#SBATCH --mail-type=begin,end,fail         # Mail events (BEGIN, END, FAIL)
#SBATCH --mail-user=kees.nederhoff@deltares-usa.us
#SBATCH --partition=gpu
#SBATCH --nodes=1                          # Single node (4 V100 GPUs)
#SBATCH --ntasks-per-node=1               # One launcher per node
#SBATCH --gres=gpu:4                       # 4 V100 GPUs per node (24 GPUs total)
#SBATCH --cpus-per-task=72               # All logical cores per node (2x 18-core Skylake + HT = 72)
#SBATCH --mem=0                            # Use all available RAM (384 GB per node)
#SBATCH --exclusive                        # Exclusive node access, no sharing
#SBATCH --account=pcmsc                    # Account to charge the job to
#SBATCH --time=2-00:00:00               # Max wall time for gpu partition (D-HH:MM:SS)
#SBATCH --output=gpu_pipeline_%j.log       # Temporary location (moved into results/ at end)

set -euo pipefail

# -- Storage location (OFF /home, on caldera PCMSC project space) --------------
# Raw inputs are read from   $COSMOS_DATA_ROOT/<case_name>/raw_data
# Run outputs are written to $COSMOS_RESULTS_ROOT/<case_name>/results/<run_name>
# These two vars are read by src/cosmos_wind_cnn/utils/config.py
# (get_data_dir / get_run_dirs). Edit the paths to relocate storage.
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn

# Override the case study at submit time without editing this file, e.g.:
#   sbatch --export=ALL,CASE_STUDY=case_studies/sf_bay_rtma scripts/gpu_tallgrass.slurm
CASE_STUDY="${CASE_STUDY:-case_studies/sf_bay_conus404}"
RUN_NAME=$SLURM_JOB_ID
# Logs live under $COSMOS_RESULTS_ROOT/<case_name>/results/<job_id>/logs (mirrors get_run_dirs)
LOG_DIR="${COSMOS_RESULTS_ROOT}/$(basename "$CASE_STUDY")/results/${RUN_NAME}/logs"

# --- always copy SLURM log into results dir, even if the pipeline fails ---
# $SLURM_SUBMIT_DIR is where sbatch was called from (log is written there regardless of cd below)
trap 'mkdir -p "$LOG_DIR" && cp "${SLURM_SUBMIT_DIR}/gpu_pipeline_${SLURM_JOB_ID}.log" "$LOG_DIR/" 2>/dev/null || true' EXIT

# --- distributed training setup ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN                     # Only log NCCL warnings/errors
export OMP_NUM_THREADS=18                  # 72 logical cores / 4 GPU workers per node

echo "Master node : $MASTER_ADDR"
echo "Nodes       : $SLURM_NNODES"
echo "GPUs / node : 4"
echo "Total GPUs  : $((SLURM_NNODES * 4))"
echo "Data root   : $COSMOS_DATA_ROOT"
echo "Results root: $COSMOS_RESULTS_ROOT"

# --- initialize conda for non-interactive shells ---
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn

# --- go to repo root ---
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

# --- sanity checks ---
which python
python -c "import sys; print(sys.executable)"
python -c "import cosmos_wind_cnn; print('import ok:', cosmos_wind_cnn.__file__)"

# --- full pipeline (4x V100 DDP) ---
# Steps: preprocess -> train (DDP) -> archive configs -> inference -> evaluate
# Skip individual steps by uncommenting:
#   --skip-preprocess  (reuse existing train/val/test splits)
#   --skip-train       (reuse existing checkpoint)
#   --skip-inference   (reuse existing inference output)
#   --skip-eval        (skip evaluation vs CONUS404)
srun python scripts/run_training_pipeline.py \
    --case-study $CASE_STUDY \
    --run-name $RUN_NAME \
    --gpus 4 \
    --num-workers 8

# Note: SLURM log is copied to $LOG_DIR automatically via the EXIT trap above.

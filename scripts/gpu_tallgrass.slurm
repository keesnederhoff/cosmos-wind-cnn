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
#SBATCH --output=gpu_pipeline_%j.log       # Standard output and error log

set -euo pipefail

# --- distributed training setup ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN                     # Only log NCCL warnings/errors
export OMP_NUM_THREADS=18                  # 72 logical cores / 4 GPU workers per node

echo "Master node : $MASTER_ADDR"
echo "Nodes       : $SLURM_NNODES"
echo "GPUs / node : 4"
echo "Total GPUs  : $((SLURM_NNODES * 4))"

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
    --case-study case_studies/sf_bay \
    --run-name $SLURM_JOB_ID \
    --gpus 4 \
    --num-workers 8

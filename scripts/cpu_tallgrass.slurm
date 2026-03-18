#!/bin/bash
#SBATCH --job-name=cosmos_wind_cnn
#SBATCH --mail-type=begin,end,fail         # Mail events (BEGIN, END, FAIL)
#SBATCH --mail-user=kees.nederhoff@deltares-usa.us
#SBATCH --partition=cpu                    # ml-* CPU nodes (15 available)
#SBATCH --nodes=1                          # Single node — CPU DL doesn't scale multi-node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72               # All logical cores per node (2x 18-core Skylake + HT = 72)
#SBATCH --mem=0                            # Use all available RAM on the node
#SBATCH --exclusive                        # Dedicated node access, no sharing
#SBATCH --account=pcmsc                    # Account to charge the job to
#SBATCH --time=2-00:00:00               # Max wall time for cpu partition (D-HH:MM:SS)
#SBATCH --output=cpu_pipeline_%j.log       # Standard output and error log

set -euo pipefail

# --- threading: tell PyTorch/OpenMP/MKL to use all allocated cores ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "CPUs allocated : $SLURM_CPUS_PER_TASK"
echo "Node           : $SLURM_NODELIST"

# --- initialize conda for non-interactive shells ---
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn

# --- go to repo root ---
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

# --- sanity checks ---
which python
python -c "import sys; print(sys.executable)"
python -c "import cosmos_wind_cnn; print('import ok:', cosmos_wind_cnn.__file__)"

# --- full pipeline (CPU, single process) ---
# Steps: preprocess -> train -> archive configs -> inference -> evaluate
# Skip individual steps by uncommenting:
#   --skip-preprocess  (reuse existing train/val/test splits)
#   --skip-train       (reuse existing checkpoint)
#   --skip-inference   (reuse existing inference output)
#   --skip-eval        (skip evaluation vs CONUS404)
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay \
    --run-name $SLURM_JOB_ID \
    --gpus 0 \
    --num-workers 8

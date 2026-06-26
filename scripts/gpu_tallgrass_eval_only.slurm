#!/bin/bash
#SBATCH --job-name=cosmos_eval
#SBATCH --mail-type=begin,end,fail         # Mail events (BEGIN, END, FAIL)
#SBATCH --mail-user=kees.nederhoff@deltares-usa.us
#SBATCH --partition=gpu
#SBATCH --nodes=1                          # Single node (4 V100 GPUs)
#SBATCH --ntasks-per-node=1               # One torchrun launcher per node
#SBATCH --gres=gpu:4                       # 4 V100 GPUs per node (24 GPUs total)
#SBATCH --cpus-per-task=72               # All logical cores per node (2x 18-core Skylake + HT = 72)
#SBATCH --mem=0                            # Use all available RAM
#SBATCH --exclusive                        # Exclusive node access
#SBATCH --account=pcmsc                    # Account to charge the job to
#SBATCH --time=0-04:00:00               # Max wall time (D-HH:MM:SS)
#SBATCH --output=gpu_eval_%j.log           # Standard output and error log

set -euo pipefail

# ── Storage location (OFF /home, on caldera PCMSC project space) ─────────────
# Raw inputs:  $COSMOS_DATA_ROOT/<case>/raw_data ; run outputs:
# $COSMOS_RESULTS_ROOT/<case>/results/<job> (read by utils/config.get_data_dir /
# get_run_dirs). Edit these to relocate storage.
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn

mkdir -p logs

# -------------------------------------------------------
# Run name of the training job to evaluate
# -------------------------------------------------------
RUN_NAME=3659043

# --- initialize conda for non-interactive shells ---
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn

# --- go to repo root ---
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

# --- sanity checks ---
which python
python -c "import sys; print(sys.executable)"
python -c "import cosmos_wind_cnn; print('import ok:', cosmos_wind_cnn.__file__)"

echo "========================================"
echo "Evaluation"
echo "  run name   : ${RUN_NAME}"
echo "  checkpoint : case_studies/sf_bay_conus404/results/${RUN_NAME}/checkpoint/best_model.pth"
echo "  output     : case_studies/sf_bay_conus404/results/${RUN_NAME}/output_evaluation/"
echo "========================================"

srun python scripts/evaluate.py \
    --case-study case_studies/sf_bay_conus404 \
    --run-name $RUN_NAME

echo "========================================"
echo "EVALUATION COMPLETE — run name: ${RUN_NAME}"
echo "========================================"

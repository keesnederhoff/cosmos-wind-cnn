#!/bin/bash
#SBATCH --job-name=cosmos_rtma_mm2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --account=pcmsc
#SBATCH --time=2-00:00:00
#SBATCH --output=gpu_pipeline_rtma_mm2_%j.log

set -euo pipefail
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
CASE_STUDY=case_studies/sf_bay_rtma
RUN_NAME=3737874
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29503
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=10
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
which python
echo "MEMMAP 2-GPU resume (skip-preprocess) on $(hostname), reusing run dir $RUN_NAME"
srun python scripts/run_training_pipeline.py --case-study $CASE_STUDY --run-name $RUN_NAME --gpus 2 --num-workers 8 --skip-preprocess

#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/logs/%x_%j.log

# Generic CPU stage runner for the chain. Resources and --job-name come from the
# submitter (--cpus-per-task, --mem, --time):
#   sbatch ... cpu_run.slurm <stage-name> <command> [args...]
# The command runs in the repo with the environment; its exit status is the stage's
# (the next stage is `afterok` on this job).
set -euo pipefail
source /home/cnederhoff/cosmos/cosmos-wind-cnn/scripts/env.sh
STAGE_NAME=$1; shift
stage_begin "$STAGE_NAME"
"$@"

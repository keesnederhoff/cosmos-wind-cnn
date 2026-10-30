#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/v4_logs/%x_%j.log

# Generic CPU stage runner for the v4 chain. Resources and --job-name come from the
# submitter (--cpus-per-task, --mem, --time):
#   sbatch ... cpu_v4_run.slurm <stage-name> <command> [args...]
# The command runs in the repo with the v4 environment; its exit status is the stage's
# (the next stage is `afterok` on this job).
set -euo pipefail
source /home/cnederhoff/cosmos/cosmos-wind-cnn/scripts/v4_env.sh
STAGE_NAME=$1; shift
v4_begin "$STAGE_NAME"
"$@"

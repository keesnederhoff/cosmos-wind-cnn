#!/bin/bash
# Shared environment for every stage of the unattended v4 chain. SOURCE this; do not execute.
#   source "$(dirname "${BASH_SOURCE[0]}")/v4_env.sh"; v4_begin <stage-name>
#
# - exports the v4 case study switches the v3-era scripts read (CNN_CASE ... CNN_E3_END);
#   the defaults inside those scripts stay v3, so v3 remains reproducible
# - v4_begin installs an EXIT trap that appends "STAGE <name> OK|FAIL(rc) <time> <job>" to
#   results/chain_status.log. Node death / walltime kills skip the trap, which is why the
#   final report job also lists stages with NO line at all.
# - never trusts sacct State (CLAUDE.md): the gates read files and logs.

export ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export V4=$ROOT/sf_bay_rtma_v4
export V4RES=$V4/results
export REPO=/home/cnederhoff/cosmos/cosmos-wind-cnn
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export COSMOS_VALIDATION_DATA_ROOT=$ROOT/validation
export COSMOS_VALIDATION_OUTPUT_ROOT=$ROOT/validation/results

export CNN_CASE=sf_bay_rtma_v4
export CNN_PREFIX=CNN_RTMA_v4
export CNN_STORE=v4data
export CNN_NQ=51
export CNN_SEEDS=v4_s1,v4_s2,v4_s3
export CNN_SEED_REF=v4_s1
export CNN_VAL_WIN=20241001_20251001      # WY2025: BC-map fit
export CNN_TEST_WIN=20251001_20261002     # WY2026: held-out
export CNN_BC_PROOF=0                     # v4 has no archived BCVAL_ file to reproduce
export CNN_E3_END=2026-10-01T23:00
export VAL_ERA_END=2026-10-02
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

# the quantile levels shipped: nearest ACTUAL grid levels of the 51-level head
# (tau_k=(k-0.5)/51): P05 P10 P25 P50 P75 P90 P95 P99
export V4_QLEVELS=0.0490,0.1078,0.2451,0.5000,0.7549,0.8922,0.9510,0.9902

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd "$REPO"
mkdir -p "$V4RES"

v4_chosen() { cat "$V4RES/chosen_seed.txt"; }

v4_begin() {
  STAGE="$1"
  trap 'rc=$?; echo "STAGE ${STAGE} $([ $rc -eq 0 ] && echo OK || echo FAIL\(rc=$rc\)) $(date -Is) job=${SLURM_JOB_ID:-?}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}" >> "$V4RES/chain_status.log"' EXIT
  echo "=== v4 stage ${STAGE} job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-} node=$(hostname) $(date -Is)"
}

# (defined first so `export -f` at the bottom makes both usable inside `bash -c` stages)
# exit 1 if the pcmsc group has < $1 TB (decimal) below its SOFT quota
v4_quota_guard() {
  local need=${1:-1.0}
  local nums used soft
  nums=$(lfs quota -g pcmsc -q /caldera/projects/usgs/hazards/pcmsc | tr '\n' ' ' | awk '{n=0; for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) {n++; v[n]=$i}; print v[1], v[2]}')
  used=${nums% *}; soft=${nums#* }
  awk -v u="$used" -v s="$soft" -v need="$need" 'BEGIN{free=(s-u)/1e9; printf "quota: used %.1f TB, soft %.1f TB, free %.2f TB (need %.2f)\n", u/1e9, s/1e9, free, need; exit (free<need)}'
}

export -f v4_chosen v4_quota_guard v4_begin

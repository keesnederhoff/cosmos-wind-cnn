#!/bin/bash
#SBATCH --job-name=v4_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/v4_logs/%x_%A_%a.log

# Stage 3: train the three v4 seeds (v4_s1..s3). Train-only; inference happens in stages 4-6.
# Derived from gpu_wrapup.slurm. Fixed recipe: quantile head (51 speed / 19 gust levels),
# qw_exp 1.0, gust weight 0.2, dropout 0.1, wd 1e-3, lr 3e-4, bc24, sl6, inputs
# u/v/gust (pair) + cloud + terrain. --exclusive --mem=0 (the 2026-07 sweep lost 9 of 21 arms
# to a co-tenant on a shared GPU node). Selection metric = validation twCRPS@10 m/s;
# per-head checkpoints best_speed.pth / best_direction.pth are written by train.py.
set -euo pipefail
source /home/cnederhoff/cosmos/cosmos-wind-cnn/scripts/v4_env.sh
RN=v4_s$((SLURM_ARRAY_TASK_ID + 1))
v4_begin "train_${SLURM_ARRAY_TASK_ID}"
v4_quota_guard 0.5

SHARED=$V4RES/v4data/data_processed
[ -s "$SHARED/memmap/train/meta.json" ] || { echo "FATAL: no memmap store under $SHARED"; exit 1; }

# fail in seconds on a wiring slip: the YAML defaults ARE the recipe in v4
python - <<'PY'
import yaml
c = yaml.safe_load(open('case_studies/sf_bay_rtma_v4/configs/training.yaml'))
want = dict(head='quantile', n_quantiles=51, n_gust_quantiles=19, loss_qw_exp=1.0,
            loss_gust_weight=0.2, dropout_rate=0.1, weight_decay=0.001, learning_rate=0.0003,
            base_channels=24, sequence_length=6, residual_learning=False)
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
assert c['additional_inputs'] == ['lr_cloud', 'static_terrain'], c['additional_inputs']
assert not bad, f'training.yaml differs from the v4 recipe: {bad}'
print('training.yaml == v4 recipe')
PY

export SWEEP_HEAD=quantile SWEEP_N_QUANTILES=51 SWEEP_QW_EXP=1.0 SWEEP_GUST_WEIGHT=0.2
export SWEEP_ADD_INPUTS=lr_cloud,static_terrain SWEEP_BASE_CHANNELS=24 SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.1 SWEEP_WEIGHT_DECAY=0.001 SWEEP_LR=0.0003 SWEEP_NONWIND_WEIGHT=0.0
# SWEEP_RESIDUAL deliberately unset: the quantile head already anchors on ERA5 and
# build_wind3dunet raises if both are on.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((30500 + SLURM_ARRAY_TASK_ID))
export NCCL_DEBUG=WARN OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RD=$V4RES/$RN
mkdir -p "$RD/data_processed"
for f in memmap normalization_stats.pkl target_grid_reference.nc train.nc val.nc test.nc; do
  ln -sfn "$SHARED/$f" "$RD/data_processed/$f"
done
echo "$RN head=quantile Q=51/19 qw=1.0 gust_w=0.2 bc=24 sl=6 do=0.1 wd=1e-3 lr=3e-4 store=v4data" > "$RD/arm_params.txt"

srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma_v4 --run-name "$RN" \
    --gpus 2 --num-workers 8 \
    --skip-preprocess --skip-inference --skip-eval
echo "arm $RN finished"

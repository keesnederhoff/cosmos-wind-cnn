# Merge Tallgrass HPC Changes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the runtime/operational improvements made on USGS Tallgrass into `D:\Git\cosmos-wind-cnn` without regressing the newer case-study configs and docs that already exist locally.

**Architecture:** Overlay a small, well-bounded set of HPC files onto the local repo on a dedicated git branch. The merge is *bidirectional*: take HPC for runtime code (caldera storage roots + streaming inference + new run scripts), keep local for case-study configs/READMEs (which are newer/more general). A baseline commit isolates the pre-existing working tree so the actual Tallgrass overlay is one clean, reviewable diff.

**Tech Stack:** git (Bash / Git Bash on Windows), Python 3.9+ package `cosmos_wind_cnn` (editable install), pytest, SLURM bash scripts. `netCDF4>=1.6.0` is already a declared dependency.

---

## Source & Target

- **Target repo:** `D:\Git\cosmos-wind-cnn` (Git Bash: `/d/Git/cosmos-wind-cnn`), branch `main`, remote `origin = github.com/keesnederhoff/cosmos-wind-cnn`.
- **HPC dump (no `.git`, plain copy + SLURM logs):** `G:\03-downscaling_meteo_cnn\dump-cosmos-wind-cnn` (Git Bash: `/g/03-downscaling_meteo_cnn/dump-cosmos-wind-cnn`).

> All commands below run in the **Bash tool (Git Bash)**, which is what this environment provides. `git`, `cp`, `diff`, `bash -n`, `pytest` all work there. Line endings are handled automatically: `.gitattributes` has `* text=auto` and `core.autocrlf=true`, so LF files copied from the dump normalize to LF in the repo and check out as CRLF on Windows — no manual conversion needed.

## Merge decisions (locked in with Kees)

1. **Divergent files → KEEP LOCAL.** The 3 `sf_bay_rtma` configs and 3 case-study READMEs are newer locally (generalized multi-variable + `target_prefix`/`input_prefix`; richer docs). Do **not** overwrite with the older wind-only HPC versions.
2. **`gpu_tallgrass.slurm` → RECONCILE.** Adopt HPC's `COSMOS_DATA_ROOT`/`COSMOS_RESULTS_ROOT` caldera exports, but keep the local `CASE_STUDY` override and explanatory comments.
3. **New scripts → ADD + FIX READMEs.** Add `scripts/preprocess.py` and `scripts/inference.py`, and add a consistent reference to them in the case-study READMEs (additive; preserve existing content).

## File Structure (what changes)

| Path | Action | Why |
|------|--------|-----|
| `.gitignore` | **Modify** | Add `**/results/**` ignore — `results/` is 158 GB of per-run outputs and is currently untracked. **Must precede any `git add`.** |
| `src/cosmos_wind_cnn/utils/config.py` | **Overwrite ← HPC** | Adds `get_data_dir()` + `COSMOS_DATA_ROOT`/`COSMOS_RESULTS_ROOT` env support. Verified additive (no local-only function lost). |
| `scripts/run_training_pipeline.py` | **Overwrite ← HPC** | Streamed NetCDF inference (bounded RAM) + vectorized eval extraction + `get_data_dir`. Coupled with config.py. |
| `scripts/preprocess.py` | **Create ← HPC** | New simple standalone preprocessing entry point. |
| `scripts/inference.py` | **Create ← HPC** | New simple standalone inference entry point. |
| `scripts/cpu_rtma_eval.slurm` | **Create ← HPC** | RTMA evaluation job. |
| `scripts/gpu_tallgrass_eval_only.slurm` | **Create ← HPC** | Evaluation-only job. |
| `scripts/gpu_tallgrass_inference.slurm` | **Create ← HPC** | Inference-only job. |
| `scripts/gpu_tallgrass_rtma.slurm` | **Create ← HPC** | RTMA full pipeline job. |
| `scripts/gpu_tallgrass_rtma_fresh.slurm` | **Create ← HPC** | RTMA fresh-start job. |
| `scripts/gpu_tallgrass_rtma_infer.slurm` | **Create ← HPC** | RTMA inference job. |
| `scripts/stage_raw_to_caldera.slurm` | **Create ← HPC** | Stage raw inputs to caldera. |
| `scripts/gpu_tallgrass.slurm` | **Modify (reconcile)** | Caldera env exports + keep `CASE_STUDY` override & comments. |
| `case_studies/sf_bay/README.md` | **Modify (additive)** | Document new standalone scripts. |
| `case_studies/puget_sound/README.md` | **Modify (additive)** | Document new standalone scripts. |
| `case_studies/_template/README.md` | **Modify (additive)** | Document new standalone scripts. |
| `case_studies/sf_bay_rtma/configs/*.yaml` (3) | **KEEP LOCAL — do not touch** | Local is newer/more general. |
| **NOT copied** | — | All `*.bak*` files and `*.log` SLURM logs in the dump (junk). |

---

### Task 0: Branch, gitignore guard, and baseline commit

**Files:**
- Modify: `/d/Git/cosmos-wind-cnn/.gitignore`

- [ ] **Step 1: Confirm starting state and create the branch**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git status --short | head -5
git checkout -b merge/tallgrass-hpc
git branch --show-current
```
Expected: branch `merge/tallgrass-hpc` is checked out; the working-tree changes carry over (they are not lost by branching).

- [ ] **Step 2: Add the `results/` ignore BEFORE staging anything**

Append to `.gitignore` (use Edit tool, after the existing `# Output files (generated)` block):
```gitignore

# Per-run output trees (results/<run_id>/...) — large, regenerated, never tracked
**/results/**
!**/results/.gitkeep
```

- [ ] **Step 3: Verify the 158 GB `results/` tree will NOT be staged**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git add -A
git status --porcelain | grep -E "results/" | grep -v "\.gitkeep" | head
```
Expected: **no output** (only `results/.gitkeep` may appear elsewhere; no `.png`/`.json`/run artifacts staged). If anything else under `results/` is listed, STOP and fix `.gitignore`.

- [ ] **Step 4: Sanity-check the staged set is small and contains no large blobs**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
echo "staged files:"; git diff --cached --name-only | wc -l
echo "any staged file > 5 MB?"; git diff --cached --name-only -z | \
  xargs -0 -I{} bash -c 'f="{}"; [ -f "$f" ] && s=$(wc -c <"$f") && [ "$s" -gt 5000000 ] && echo "$s  $f"' ; echo "(none above = good)"
```
Expected: a few dozen files (configs, scripts, src, docs, tests, `results/.gitkeep`); **no** file larger than 5 MB. If a large blob appears (e.g. a stray `.nc`/`.png`), unstage it and add an ignore rule before continuing.

- [ ] **Step 5: Commit the baseline**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git commit -m "$(cat <<'EOF'
chore: snapshot working tree before Tallgrass HPC overlay

Baseline commit on merge/tallgrass-hpc capturing the already-synced local
state (preprocessing/training/eval updates, sf_bay_rtma case study, tests,
docs) plus a new .gitignore rule excluding the per-run results/ tree.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -3
```
Expected: a new baseline commit on top of `08c4d93`. `git status` is now clean.

- [ ] **Step 6 (review gate): show what the baseline added vs main**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git diff main --stat | tail -30
```
Expected: a readable summary of the pre-existing working-tree changes. (These predate this merge; review only — do not act on them here.)

---

### Task 1: Caldera storage roots + streaming inference (coupled `config.py` + `run_training_pipeline.py`)

**Files:**
- Overwrite: `/d/Git/cosmos-wind-cnn/src/cosmos_wind_cnn/utils/config.py` ← dump
- Overwrite: `/d/Git/cosmos-wind-cnn/scripts/run_training_pipeline.py` ← dump
- Test: `/d/Git/cosmos-wind-cnn/tests/test_config_helpers.py` (existing)

- [ ] **Step 1: Copy both HPC files over local**

Run:
```bash
HPC=/g/03-downscaling_meteo_cnn/dump-cosmos-wind-cnn; LOC=/d/Git/cosmos-wind-cnn
cp "$HPC/src/cosmos_wind_cnn/utils/config.py"        "$LOC/src/cosmos_wind_cnn/utils/config.py"
cp "$HPC/scripts/run_training_pipeline.py"           "$LOC/scripts/run_training_pipeline.py"
```

- [ ] **Step 2: Confirm the change is exactly the expected additive delta in config.py**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git --no-pager diff --stat src/cosmos_wind_cnn/utils/config.py
git --no-pager diff src/cosmos_wind_cnn/utils/config.py | grep -E "^\+" | grep -E "import os|COSMOS_RESULTS_ROOT|COSMOS_DATA_ROOT|def get_data_dir"
```
Expected: the added lines include `import os`, a `COSMOS_RESULTS_ROOT` branch in `get_run_dirs`, and a new `def get_data_dir`. No unrelated deletions of other helper functions.

- [ ] **Step 3: Verify the package imports and the new helpers exist**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
pip install -e . >/dev/null 2>&1 || pip install -e .
python -c "from cosmos_wind_cnn.utils.config import get_data_dir, get_run_dirs, parse_variable_config, var_units_for, wind_var_names; print('imports ok')"
```
Expected: `imports ok` (no ImportError). `get_data_dir` resolves.

- [ ] **Step 4: Verify env-var behavior of the new helpers (defaults unchanged when unset)**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
python - <<'PY'
import os
from pathlib import Path
from cosmos_wind_cnn.utils.config import get_data_dir, get_run_dirs
# Defaults (env unset) must match the old behavior:
assert get_data_dir('case_studies/sf_bay') == Path('case_studies/sf_bay/data/raw')
assert get_run_dirs('case_studies/sf_bay', '123')['run_root'] == Path('case_studies/sf_bay/results/123')
# Env override redirects to <root>/<case_name>/...:
os.environ['COSMOS_DATA_ROOT'] = '/caldera/x'
os.environ['COSMOS_RESULTS_ROOT'] = '/caldera/y'
assert get_data_dir('case_studies/sf_bay') == Path('/caldera/x/sf_bay/raw')
assert get_run_dirs('case_studies/sf_bay', '123')['run_root'] == Path('/caldera/y/sf_bay/123')
print('config env behavior ok')
PY
```
Expected: `config env behavior ok`.

- [ ] **Step 5: Syntax-check the pipeline script and run the existing config tests**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
python -m py_compile scripts/run_training_pipeline.py && echo "py_compile ok"
python -c "import netCDF4; print('netCDF4', netCDF4.__version__)"
pytest tests/test_config_helpers.py -q
```
Expected: `py_compile ok`, a netCDF4 version prints, and the config-helper tests PASS (they exercise the unchanged default code paths).

- [ ] **Step 6: Commit**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git add src/cosmos_wind_cnn/utils/config.py scripts/run_training_pipeline.py
git commit -m "$(cat <<'EOF'
feat: caldera storage roots + streaming inference (from Tallgrass)

- utils/config.py: add get_data_dir() and COSMOS_DATA_ROOT/COSMOS_RESULTS_ROOT
  env support so raw data and run outputs can live off /home on caldera.
- run_training_pipeline.py: stream inference output to NetCDF one time-chunk
  at a time (bounded RAM for the 1940-2027 ERA5 hindcast) and vectorize the
  evaluation point extraction over the overlap window.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: commit succeeds.

---

### Task 2: Add standalone `preprocess.py` and `inference.py`

**Files:**
- Create: `/d/Git/cosmos-wind-cnn/scripts/preprocess.py` ← dump
- Create: `/d/Git/cosmos-wind-cnn/scripts/inference.py` ← dump

- [ ] **Step 1: Copy the two new scripts**

Run:
```bash
HPC=/g/03-downscaling_meteo_cnn/dump-cosmos-wind-cnn; LOC=/d/Git/cosmos-wind-cnn
cp "$HPC/scripts/preprocess.py" "$LOC/scripts/preprocess.py"
cp "$HPC/scripts/inference.py"  "$LOC/scripts/inference.py"
```

- [ ] **Step 2: Verify no name collision and both compile**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
ls scripts/preprocess.py scripts/inference.py
python -m py_compile scripts/preprocess.py scripts/inference.py && echo "py_compile ok"
python scripts/preprocess.py --help >/dev/null 2>&1 && echo "preprocess --help ok" || echo "check: preprocess --help"
python scripts/inference.py --help  >/dev/null 2>&1 && echo "inference --help ok"  || echo "check: inference --help"
```
Expected: both files present, `py_compile ok`. `--help` should print usage (these are argparse scripts). If `--help` errors due to a missing import, record it for follow-up but it does not block the commit (syntax is valid).

- [ ] **Step 3: Commit**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git add scripts/preprocess.py scripts/inference.py
git commit -m "$(cat <<'EOF'
feat: add standalone preprocess.py and inference.py entry points (from Tallgrass)

Lightweight per-step scripts (argparse, --case-study) alongside the existing
run_training_pipeline.py / preprocess_training.py / run_inference.py path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: commit succeeds.

---

### Task 3: Add the new SLURM scripts (7 files)

**Files (all Create ← dump, in `/d/Git/cosmos-wind-cnn/scripts/`):**
`cpu_rtma_eval.slurm`, `gpu_tallgrass_eval_only.slurm`, `gpu_tallgrass_inference.slurm`, `gpu_tallgrass_rtma.slurm`, `gpu_tallgrass_rtma_fresh.slurm`, `gpu_tallgrass_rtma_infer.slurm`, `stage_raw_to_caldera.slurm`

- [ ] **Step 1: Copy the 7 SLURM files (explicit list — do NOT glob, to avoid `.bak`/`.log`)**

Run:
```bash
HPC=/g/03-downscaling_meteo_cnn/dump-cosmos-wind-cnn; LOC=/d/Git/cosmos-wind-cnn
for f in cpu_rtma_eval.slurm gpu_tallgrass_eval_only.slurm gpu_tallgrass_inference.slurm \
         gpu_tallgrass_rtma.slurm gpu_tallgrass_rtma_fresh.slurm gpu_tallgrass_rtma_infer.slurm \
         stage_raw_to_caldera.slurm; do
  cp "$HPC/scripts/$f" "$LOC/scripts/$f"
done
ls -1 "$LOC/scripts/"*.slurm
```
Expected: the 7 new `.slurm` files plus the existing `cpu_tallgrass.slurm` and `gpu_tallgrass.slurm` are listed (9 total).

- [ ] **Step 2: Syntax-check every new SLURM script with bash**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
for f in cpu_rtma_eval gpu_tallgrass_eval_only gpu_tallgrass_inference \
         gpu_tallgrass_rtma gpu_tallgrass_rtma_fresh gpu_tallgrass_rtma_infer stage_raw_to_caldera; do
  bash -n "scripts/$f.slurm" && echo "ok: $f.slurm"
done
```
Expected: `ok:` for all 7 (no bash syntax errors).

- [ ] **Step 3: Confirm no junk (`.bak`/`.log`) sneaked in**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git status --porcelain scripts/ | grep -E "\.bak|\.log" || echo "no junk staged — good"
```
Expected: `no junk staged — good`.

- [ ] **Step 4: Commit**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git add scripts/cpu_rtma_eval.slurm scripts/gpu_tallgrass_eval_only.slurm \
        scripts/gpu_tallgrass_inference.slurm scripts/gpu_tallgrass_rtma.slurm \
        scripts/gpu_tallgrass_rtma_fresh.slurm scripts/gpu_tallgrass_rtma_infer.slurm \
        scripts/stage_raw_to_caldera.slurm
git commit -m "$(cat <<'EOF'
feat: add RTMA / caldera SLURM job scripts (from Tallgrass)

eval-only, inference-only, RTMA full/fresh/infer pipelines, and a
stage_raw_to_caldera helper for moving raw inputs to project space.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: commit succeeds.

---

### Task 4: Reconcile `gpu_tallgrass.slurm` (caldera exports + keep override & comments)

**Files:**
- Modify: `/d/Git/cosmos-wind-cnn/scripts/gpu_tallgrass.slurm`

- [ ] **Step 1: Overwrite the file with the reconciled version**

Write `/d/Git/cosmos-wind-cnn/scripts/gpu_tallgrass.slurm` with **exactly** this content (Write tool):
```bash
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
# Raw inputs are read from   $COSMOS_DATA_ROOT/<case_name>/raw
# Run outputs are written to $COSMOS_RESULTS_ROOT/<case_name>/<job_id>
# These two vars are read by src/cosmos_wind_cnn/utils/config.py
# (get_data_dir / get_run_dirs). Edit the paths to relocate storage.
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn/data
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn/results

# Override the case study at submit time without editing this file, e.g.:
#   sbatch --export=ALL,CASE_STUDY=case_studies/sf_bay_rtma scripts/gpu_tallgrass.slurm
CASE_STUDY="${CASE_STUDY:-case_studies/sf_bay}"
RUN_NAME=$SLURM_JOB_ID
# Logs live under $COSMOS_RESULTS_ROOT/<case_name>/<job_id>/logs (mirrors get_run_dirs)
LOG_DIR="${COSMOS_RESULTS_ROOT}/$(basename "$CASE_STUDY")/${RUN_NAME}/logs"

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
```

- [ ] **Step 2: Verify it has both the caldera exports AND the preserved override/comments, and is valid bash**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
bash -n scripts/gpu_tallgrass.slurm && echo "bash syntax ok"
grep -q "export COSMOS_DATA_ROOT="    scripts/gpu_tallgrass.slurm && echo "has DATA_ROOT export"
grep -q "export COSMOS_RESULTS_ROOT=" scripts/gpu_tallgrass.slurm && echo "has RESULTS_ROOT export"
grep -q 'CASE_STUDY="${CASE_STUDY:-' scripts/gpu_tallgrass.slurm && echo "kept CASE_STUDY override"
grep -q 'basename "$CASE_STUDY"'      scripts/gpu_tallgrass.slurm && echo "LOG_DIR is case-generic"
```
Expected: all five lines print (`bash syntax ok`, `has DATA_ROOT export`, `has RESULTS_ROOT export`, `kept CASE_STUDY override`, `LOG_DIR is case-generic`).

- [ ] **Step 3: Commit**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git add scripts/gpu_tallgrass.slurm
git commit -m "$(cat <<'EOF'
feat: route gpu_tallgrass.slurm outputs to caldera, keep CASE_STUDY override

Adopt the Tallgrass COSMOS_DATA_ROOT/COSMOS_RESULTS_ROOT exports while keeping
the submit-time CASE_STUDY override and the explanatory comments. LOG_DIR is
derived from basename(CASE_STUDY) so it stays case-study-generic.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: commit succeeds.

---

### Task 5: README consistency — document the new standalone scripts (additive)

**Files:**
- Modify: `/d/Git/cosmos-wind-cnn/case_studies/sf_bay/README.md`
- Modify: `/d/Git/cosmos-wind-cnn/case_studies/puget_sound/README.md`
- Modify: `/d/Git/cosmos-wind-cnn/case_studies/_template/README.md`

> Additive only: keep all existing content (the richer local docs we chose to preserve). Insert a short subsection that references the new `preprocess.py` / `inference.py` scripts.

- [ ] **Step 1: `sf_bay/README.md` — add a standalone-scripts note after the "Standalone inference" block**

Use the Edit tool to insert the following block immediately **after** line 46 (the closing ```` ``` ```` of the `run_inference.py` example, before `### HPC (SLURM on Tallgrass)`):
```markdown

### Quick per-step scripts (local experiments)

Lightweight standalone alternatives to the run-isolated pipeline above:

```bash
python scripts/preprocess.py --case-study case_studies/sf_bay
python scripts/train.py      --case-study case_studies/sf_bay
python scripts/evaluate.py   --case-study case_studies/sf_bay
python scripts/inference.py  --case-study case_studies/sf_bay
```

For reproducible runs prefer `run_training_pipeline.py`, which isolates every
artifact under `results/<run_name>/`.
```

- [ ] **Step 2: `puget_sound/README.md` — add `preprocess.py`/`inference.py` to the existing per-step list**

Use the Edit tool to replace the `# Or individual steps` block (lines 20-23) with:
```markdown
# Or individual steps (run-isolated)
python scripts/preprocess_training.py --case-study case_studies/puget_sound --run-name first_run
python scripts/train.py --case-study case_studies/puget_sound --run-name first_run
python scripts/evaluate.py --case-study case_studies/puget_sound --run-name first_run

# Or quick standalone scripts (local experiments)
python scripts/preprocess.py --case-study case_studies/puget_sound
python scripts/inference.py  --case-study case_studies/puget_sound
```

- [ ] **Step 3: `_template/README.md` — add the standalone scripts under the individual-steps block**

Use the Edit tool to insert the following immediately **after** line 33 (the closing ```` ``` ```` of the individual-steps block, before the final "All outputs are saved..." line):
```markdown

   Or quick standalone scripts (local experiments, not run-isolated):
   ```bash
   python scripts/preprocess.py --case-study case_studies/my_study
   python scripts/inference.py --case-study case_studies/my_study
   ```
```

- [ ] **Step 4: Verify the edits are additive (no content removed) and references resolve**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git --no-pager diff --stat case_studies/*/README.md
# Every script named in the READMEs must exist in scripts/:
for s in run_training_pipeline run_inference preprocess_training train evaluate preprocess inference; do
  test -f "scripts/$s.py" && echo "ok: scripts/$s.py" || echo "MISSING: scripts/$s.py"
done
```
Expected: the three READMEs show only additions (green lines, minimal deletions for the puget_sound block replacement). Every referenced script prints `ok:`.

- [ ] **Step 5: Commit**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git add case_studies/sf_bay/README.md case_studies/puget_sound/README.md case_studies/_template/README.md
git commit -m "$(cat <<'EOF'
docs: reference standalone preprocess.py/inference.py in case-study READMEs

Keep the existing richer docs; add a short "quick per-step scripts" note so
README script references match the scripts that now exist.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: commit succeeds.

---

### Task 6: Final verification and branch finalization

**Files:** none (verification only)

- [ ] **Step 1: Confirm the divergent files were left untouched (still differ from the dump on purpose)**

Run:
```bash
HPC=/g/03-downscaling_meteo_cnn/dump-cosmos-wind-cnn; LOC=/d/Git/cosmos-wind-cnn
for f in case_studies/sf_bay_rtma/configs/preprocessing.yaml \
         case_studies/sf_bay_rtma/configs/training.yaml \
         case_studies/sf_bay_rtma/configs/inference_preprocessing.yaml; do
  if diff -q --strip-trailing-cr "$LOC/$f" "$HPC/$f" >/dev/null; then
    echo "UNEXPECTED match (was overwritten?): $f"
  else
    echo "ok kept-local: $f"
  fi
done
```
Expected: `ok kept-local:` for all three (they intentionally still differ from the older HPC versions).

- [ ] **Step 2: Confirm no junk landed in the repo**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git ls-files | grep -E "\.bak[0-9]*$" && echo "FAIL: .bak tracked" || echo "ok: no .bak tracked"
git ls-files | grep -E "gpu_pipeline_.*\.log$|cpu_rtma_eval_.*\.log$|stage_raw_.*\.log$" && echo "FAIL: SLURM log tracked" || echo "ok: no SLURM logs tracked"
```
Expected: `ok: no .bak tracked` and `ok: no SLURM logs tracked`.

- [ ] **Step 3: Full validation sweep**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
pip install -e . >/dev/null
python -m py_compile scripts/*.py && echo "all scripts compile"
python -c "import cosmos_wind_cnn; from cosmos_wind_cnn.utils.config import get_data_dir; print('package import ok')"
pytest -q
```
Expected: `all scripts compile`, `package import ok`, and the full test suite PASSES (`tests/test_config_helpers.py`, `tests/test_preprocessing_reindex.py`).

- [ ] **Step 4: Review the complete overlay vs the baseline**

Run:
```bash
cd /d/Git/cosmos-wind-cnn
git log --oneline main..HEAD
git diff --stat $(git rev-parse main)..HEAD
```
Expected: 5 overlay commits (Tasks 1-5) on top of the baseline; the stat shows only the intended files (config.py, run_training_pipeline.py, 9 new scripts, gpu_tallgrass.slurm, 3 READMEs).

- [ ] **Step 5: Finalize — invoke the finishing-a-development-branch skill**

Use `superpowers:finishing-a-development-branch` to choose how to integrate `merge/tallgrass-hpc`:
- merge into `main` locally (`git checkout main && git merge --no-ff merge/tallgrass-hpc`), and/or
- push and open a PR (`git push -u origin merge/tallgrass-hpc`), and/or
- keep the branch for now.

Do **not** push or merge without Kees's go-ahead — the repo has a public GitHub remote.

---

## Rollback

Everything is on the `merge/tallgrass-hpc` branch; `main` is untouched until Step 5 of Task 6.
- Undo the last commit (keep changes): `git reset --soft HEAD~1`
- Abandon the whole branch: `git checkout main && git branch -D merge/tallgrass-hpc`
- The pre-existing working tree is preserved in the Task 0 baseline commit.

## Notes / follow-ups (out of scope for this merge)

- **Redundant entry points:** `preprocess.py` vs `preprocess_training.py`, and `inference.py` vs `run_inference.py`, now coexist. A future cleanup could converge on one set (the per-step scripts are not run-isolated; the pipeline path is). Flagged, not done here.
- **Launcher `CLAUDE.md`** (`C:\Users\keesn\.claude_projects\cosmos-wind-cnn\CLAUDE.md`, outside the repo) still says "Two SLURM files only" and predates the caldera `COSMOS_*` env vars. Worth a small update so future sessions have accurate context.
- **`sf_bay_rtma` divergence:** the HPC ran wind-only with the `conus404_` prefix workaround while local is generalized multi-variable. If the HPC operational runs are meant to become canonical, that's a separate decision — not part of "follow HPC runtime changes."
```

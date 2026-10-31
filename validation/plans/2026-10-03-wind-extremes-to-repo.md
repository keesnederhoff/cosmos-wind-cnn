# Wind return levels into cosmos-wind-cnn — Plan

**Goal:** Move the wind return-level pipeline from `g:\03-downscaling_meteo_cnn\wind_extremes\` into the `cosmos-wind-cnn` git repo. In the repo it must reproduce today's results exactly, and run on Windows and Caldera without editing paths.

**Approach:** The pipeline goes in as `validation/wind_extremes/`, next to the existing point-validation engine. Paths become environment-variable roots, following the pattern of `validation/config.py`. The six `rainlib` functions it uses are copied in verbatim, so the repo stops depending on the ACFCD rainfall env. Outputs stay outside the repo. The acceptance gate is a fresh s2→s5 run in the repo's `cosmos_wind_cnn` env, which must match a golden snapshot of today's results to rtol 1e-9.

**Depends on:**
- Repo `d:\Git\cosmos-wind-cnn` (remote `github.com/keesnederhoff/cosmos-wind-cnn`, branch `main` at `cb28805`).
- Conda env `cosmos_wind_cnn`: pandas 3.0.0, numpy 2.4.2, scipy 1.17.0, sklearn, pyproj and xarray present; **cmocean missing**.
- About 9 GB free on `g:` for the regression cache copy; 1.6 TB is free.
- No HPC. Running on Caldera is a follow-up and not part of this plan.

**Design:** Original pipeline plan: `g:\03-downscaling_meteo_cnn\plans\2026-10-02-wind-return-levels.md`. Method write-ups: `README.md` and `METHOD_bias_correction.md` in the source folder.

---

## Decisions taken (Kees can overrule before execution)

1. **Location.** The pipeline goes in `validation/wind_extremes/`, not under `scripts/` or `src/`. It is an evaluation of shipped products against observations, which is what `validation/` holds. `scripts/` holds the training and inference pipeline.
2. **Vendor `rainlib`.** The functions are copied, not installed: `acfcd_rain` is a separate private ACFCD repo and its env is not on Caldera. The copy is verbatim (281 lines) and stamped with the source commit. A test checks it against `rainlib` whenever `rainlib` is importable.
3. **Environment variables.**
   - `COSMOS_VALIDATION_DATA_ROOT` is reused for ERA5, RTMA, the observations and the coastline. The current `MET_ROOT`, `g:\01_meteorlogical_analysis_sfbay`, already *is* that bundle.
   - Two new variables are added:
     - `COSMOS_WINDX_CNN_ROOT`, the folder holding `sf_bay_rtma\results` and `cnn_fullrecord`; on this desktop that is `g:\03-downscaling_meteo_cnn`;
     - `COSMOS_WINDX_OUTPUT_ROOT`, for `cache\`, `results\` and `figures\`.
4. **Nothing generated is committed.** The repo `.gitignore` already drops `validation/**/*.nc` and `*.png`. CSV results live under the output root, outside the repo.
5. **Branch, not main.**
   - All work happens on `feature/wind-extremes` in a worktree.
   - `main` currently carries 58 uncommitted `#SBATCH --wckey=reimbursable` edits to slurm files; those are Kees's and are never staged here.
   - The branch is pushed but not merged.
6. **The old folder stays.** `g:\...\wind_extremes\` keeps its cache, results and scripts, plus a `MOVED.md` pointer. Kees deletes the scripts and the `cache\tmp_*` folders (16 GB) himself; the guard hook blocks recursive deletes on `g:`.

## Artifacts

| Path | Role |
|---|---|
| `validation/wind_extremes/windx_common.py` | Config and shared helpers; paths now come from env vars |
| `validation/wind_extremes/s1_extract_products.py` … `s5_obs_correction.py` | The five pipeline stages, logic unchanged |
| `validation/wind_extremes/evt.py` | Verbatim copy of 14 `rainlib.extremes` objects plus `rainlib.trends.pettitt_test` |
| `validation/wind_extremes/figstyle.py`, `skill_utils.py` | Copies already in the source folder |
| `validation/wind_extremes/checks/` | `check_s1_smoke.py`, `check_s1_q_smoke.py`, `check_s1_full.py`, new `compare_to_golden.py` |
| `validation/wind_extremes/tests/` | `test_windx.py` (existing 10 tests), new `conftest.py`, new `test_evt_vendored.py` |
| `validation/wind_extremes/README.md`, `METHOD_bias_correction.md` | Docs, with a new environment and run section |
| `validation/plans/2026-10-02-wind-return-levels.md`, `2026-10-03-wind-extremes-to-repo.md` | The design record, copied in |
| `validation/README.md` | One layout line added |
| `g:\03-downscaling_meteo_cnn\wind_extremes\results_golden_20261003\` | Golden snapshot of today's results (outside the repo) |
| `g:\03-downscaling_meteo_cnn\wind_extremes_repo_check\` | Output root for the regression run (outside the repo) |

Shell convention for all steps: Git Bash on the desktop, with the env activated by prepending to PATH:

```
E=/c/Users/keesn/anaconda3/envs/cosmos_wind_cnn; export PATH="$E:$E/Library/bin:$E/Scripts:$PATH"; export PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE
```

`SRC=/g/03-downscaling_meteo_cnn/wind_extremes` and `WT=/d/Git/cosmos-wind-cnn/.claude/worktrees/wind-extremes` are shorthand in the commands below; set them in each shell.

---

### Task 1: Branch and golden snapshot

**Files:**
- Create: worktree `d:\Git\cosmos-wind-cnn\.claude\worktrees\wind-extremes` on branch `feature/wind-extremes`
- Create: `g:\03-downscaling_meteo_cnn\wind_extremes\results_golden_20261003\`

- [ ] **Step 1: Create the worktree from origin/main**

```
git -C /d/Git/cosmos-wind-cnn fetch origin
git -C /d/Git/cosmos-wind-cnn worktree add .claude/worktrees/wind-extremes -b feature/wind-extremes origin/main
git -C $WT status -sb
```
Expected: `## feature/wind-extremes...origin/main` and a clean tree. The 58 modified slurm files on `main` must **not** appear here.

- [ ] **Step 2: Snapshot today's results**

```
cp -r $SRC/results $SRC/results_golden_20261003
ls $SRC/results_golden_20261003 | wc -l
```
Expected: the same file count as `ls $SRC/results | wc -l`. The snapshot must contain `station_rl.csv`, `skill_station_rl.csv`, `homogeneity.csv`, `station_table.csv`, `obs_correction_loo.csv`, `obs_correction_factors.csv`, 14 `rl_map_<product>_<window>.nc` files and 2 `rl_map_cnn_obscorr_*.nc` files.

### Task 2: Copy the files in, unchanged

**Files:**
- Create: everything under `$WT/validation/wind_extremes/` listed in Artifacts, plus the two plan files

- [ ] **Step 1: Copy the scripts, docs and tests**

```
D=$WT/validation/wind_extremes; mkdir -p $D/checks $D/tests
cp $SRC/{windx_common,s1_extract_products,s2_prepare_obs,s3_fit_return_levels,s4_figures,s5_obs_correction,figstyle,skill_utils}.py $D/
cp $SRC/{README,METHOD_bias_correction}.md $D/
cp $SRC/check_s1_{smoke,q_smoke,full}.py $D/checks/
cp $SRC/tests/test_windx.py $D/tests/
cp /g/03-downscaling_meteo_cnn/plans/2026-10-02-wind-return-levels.md /g/03-downscaling_meteo_cnn/plans/2026-10-03-wind-extremes-to-repo.md $WT/validation/plans/
```
Do **not** copy `__pycache__`, `cache\`, `results\`, `figures\` or `results_golden_20261003\`.

- [ ] **Step 2: Commit the verbatim copy**

```
git -C $WT add validation/wind_extremes validation/plans/2026-10-02-wind-return-levels.md validation/plans/2026-10-03-wind-extremes-to-repo.md
git -C $WT commit -m "wind_extremes: import pipeline verbatim from g:\\03-downscaling_meteo_cnn\\wind_extremes"
```
Expected: 16 files added (8 `.py`, 2 `.md`, 3 checks, 1 test, 2 plans). Committing the copy before any edit lets later diffs show exactly what changed in the move.

### Task 3: Vendor the rainlib functions into `evt.py`

**Files:**
- Create: `$WT/validation/wind_extremes/evt.py`
- Modify: `windx_common.py`, `s2_prepare_obs.py`, `s3_fit_return_levels.py`, `s4_figures.py` (import lines only)

- [ ] **Step 1: Build `evt.py` from the source, verbatim**

Source: `g:\01_meteorlogical_analysis_sfbay\acfcd_rain\rainlib\extremes.py` and `trends.py`, repo commit `902ae61`.

Copy these top-level objects **in source order**, with their bodies and docstrings unchanged:
- From `extremes.py`: `GPD_SHAPE_TOL`, `_GPD_MODES`, `_clean_1d`, `_lmom_rows`, `decluster_peaks`, `_gpa_shape_lmom`, `gpd_fit_lmom_excess`, `gpd_quantile`, `PotSample`, `pot_sample`, `gpd_refit_rows`, `gpd_quantile_rows`, `gpd_fit_variant`, `pot_bootstrap_quantiles`.
- From `trends.py`: `pettitt_test`.

That is the full dependency closure (checked with an AST walk). The only `rainlib.config` name it reaches is `N_BOOT`; define it as `N_BOOT = 2000  # rainlib.config default; windx always passes n_boot explicitly`.

The header docstring must state the source repo, the commit `902ae61`, both file paths, and "do not edit; re-copy if rainlib changes". The imports are `numpy`, `pandas`, `from typing import NamedTuple, Sequence`, and from scipy whatever `pettitt_test` uses.

Expected: `python -c "import evt"` succeeds when run from `$WT/validation/wind_extremes`, in the `cosmos_wind_cnn` env.

- [ ] **Step 2: Repoint the imports, keeping the aliases**

| File | Old line | New line |
|---|---|---|
| `windx_common.py` | `import rainlib.extremes as rx` | `import evt as rx` |
| `s2_prepare_obs.py` | `import rainlib.extremes as rx` | `import evt as rx` |
| `s3_fit_return_levels.py` | `import rainlib.trends as rt` | `import evt as rt` |
| `s4_figures.py` | `import rainlib.extremes as rx` | `import evt as rx` |

Expected: `grep -rn rainlib $WT/validation/wind_extremes --include=*.py` returns only comments and `evt.py`'s header.

- [ ] **Step 3: Add `tests/test_evt_vendored.py`**

The test calls `pytest.importorskip("rainlib")`, so it skips cleanly in `cosmos_wind_cnn`. For a fixed `np.random.default_rng(0)` synthetic hourly series of 20 years of gamma(2, 3) values, it asserts that `evt` and `rainlib` return identical results (`np.testing.assert_array_equal`) for:
- `decluster_peaks(series, 48.0)`;
- `pot_sample(peaks, 20.0, 3.0)`;
- `gpd_fit_variant(excess, "free")` and `gpd_fit_variant(excess, "exp")`;
- `gpd_quantile` at T = 1, 10, 100;
- `pot_bootstrap_quantiles(..., rng=np.random.default_rng(1), n_boot=200)`;
- `pettitt_test` on the annual maxima.

Expected: run in the `acfcd_rain` env with `python -m pytest tests/test_evt_vendored.py -q`, it reports `1 passed`. In `cosmos_wind_cnn` it reports `1 skipped`.

### Task 4: Environment-variable paths

**Files:**
- Modify: `windx_common.py` (CONFIGURATION block), `s5_obs_correction.py` (`STATIC_WATER_FILE`), `s2_prepare_obs.py` (docstring path only)

- [ ] **Step 1: Replace the hardcoded roots in `windx_common.py`**

Add `import os` to the configuration block, then replace the `ROOT`, `CNN_DIR`, `MET_ROOT` and `CNN_BA_DIR` lines with:

```python
def _root(var: str) -> Path:
    """Env-var root, as validation/config.py. Raises with set-commands for both platforms."""
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(f"{var} is not set.\n  Windows:  set {var}=...\n  Caldera:  export {var}=...\n"
                           f"See validation/wind_extremes/README.md, section Environment.")
    return Path(val)

ROOT       = _root("COSMOS_WINDX_OUTPUT_ROOT")      # cache/, results/, figures/ (outside the repo)
CNN_ROOT   = _root("COSMOS_WINDX_CNN_ROOT")         # holds sf_bay_rtma/results and cnn_fullrecord
MET_ROOT   = _root("COSMOS_VALIDATION_DATA_ROOT")   # validation bundle: modeled_data/, observed_data/, reference/
CNN_DIR    = CNN_ROOT / "sf_bay_rtma" / "results"
CNN_BA_DIR = CNN_ROOT / "cnn_fullrecord"
```
All other paths in the block are already built from these roots with `/`, so leave them unchanged.

Expected: `grep -n 'Path(r' windx_common.py` returns nothing.

- [ ] **Step 2: Fix `STATIC_WATER_FILE` in `s5_obs_correction.py`**

Replace the three-line `STATIC_WATER_FILE = Path(r"g:\...")` with

```python
STATIC_WATER_FILE = w.MET_ROOT / "modeled_data" / "rtma" / "fm_netcdf" / "RTMA_SFbay_2p5km_surface_height_static_UTM10.nc"
```
Move the line to just below `import windx_common as w`, because the configuration block runs before the import, and add a comment saying so.

Expected: `grep -rn 'g:\\\\\|Path(r' $WT/validation/wind_extremes --include=*.py` returns only the docstring in `s2_prepare_obs.py` line 7. Change that to `validation/plans/2026-10-02-wind-return-levels.md`, then the grep returns nothing.

- [ ] **Step 3: Drop the unused cmocean import**

In `s4_figures.py`, delete `import cmocean  # noqa: F401 ...`. No `cmo.*` colormap is used in s4; `figstyle.py` already imports cmocean only when it is present.

Expected: `python -c "import s4_figures"` succeeds when run with the env vars from Task 6 Step 1 set, in `cosmos_wind_cnn`, which has no cmocean.

- [ ] **Step 4: Make tests and checks importable**

- Create `tests/conftest.py`. Before any import, it inserts `Path(__file__).resolve().parents[1]` into `sys.path`. It then sets defaults with `os.environ.setdefault` for the three windx variables plus `COSMOS_VALIDATION_DATA_ROOT`, all pointing at a `tempfile.gettempdir()/cosmos_windx_test_root` folder, following the same pattern as `validation/tests/conftest.py`.
- Remove the now-redundant `sys.path.insert` from `test_windx.py`.
- In each `checks/*.py`, add `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` above `import windx_common as w`.

Expected: `cd $WT/validation/wind_extremes && python -m pytest tests -q` gives `10 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```
git -C $WT add -A validation/wind_extremes
git -C $WT commit -m "wind_extremes: env-var roots, vendored evt.py, drop rainlib/cmocean deps"
```

### Task 5: Docs

**Files:**
- Modify: `validation/wind_extremes/README.md`, `validation/README.md`

- [ ] **Step 1: Add an Environment section and a Run section to `validation/wind_extremes/README.md`**

These go just below the title. The Environment block mirrors `validation/README.md`:

```
conda activate cosmos_wind_cnn
set KMP_DUPLICATE_LIB_OK=TRUE
set COSMOS_VALIDATION_DATA_ROOT=g:\01_meteorlogical_analysis_sfbay
set COSMOS_WINDX_CNN_ROOT=g:\03-downscaling_meteo_cnn
set COSMOS_WINDX_OUTPUT_ROOT=g:\03-downscaling_meteo_cnn\wind_extremes
```
Run section:
- Run `python s1_extract_products.py` through `python s5_obs_correction.py` from this folder. s1 skips any product with a `finalize OK` line in `cache\s1_log.txt`.
- s1 needs the 127 GB CNN files and takes hours. s2 to s5 take about 25 min together.
- `evt.py` is vendored; record that and the re-copy rule.

Replace every remaining `g:\03-downscaling_meteo_cnn\wind_extremes\` file reference in the body with a path relative to the output root (`results\...`, `figures\...`).

- [ ] **Step 2: Add one layout line to `validation/README.md`**

Under `## Layout`, after the `analysis/` line, add:

`- wind_extremes/ — RP1/10/100 wind return levels (POT/GPD) for ERA5, RTMA, CNN-RTMA raw/BC/BA vs observations; own README.`

Expected: `git -C $WT diff --stat` shows only these two files changed.

- [ ] **Step 3: Commit**

```
git -C $WT commit -am "wind_extremes: README environment/run sections; validation layout entry"
```

### Task 6: Regression run against the golden snapshot

**Files:**
- Create: `g:\03-downscaling_meteo_cnn\wind_extremes_repo_check\cache\` (copied inputs)
- Create: `$WT/validation/wind_extremes/checks/compare_to_golden.py`

- [ ] **Step 1: Stage the cached inputs into a fresh output root**

```
N=/g/03-downscaling_meteo_cnn/wind_extremes_repo_check; mkdir -p $N/cache
cp $SRC/cache/*.nc $SRC/cache/s1_log.txt $N/cache/
du -sh $N/cache
```
Expected: about 8.9 GB, made up of 7 `dailymax_*.nc`, 7 `stations_hourly_*.nc` and `obs_hourly.nc`. No `tmp_*` folders are copied.

- [ ] **Step 2: Run s2→s5 from the repo in `cosmos_wind_cnn`**

```
export COSMOS_VALIDATION_DATA_ROOT='g:\01_meteorlogical_analysis_sfbay' COSMOS_WINDX_CNN_ROOT='g:\03-downscaling_meteo_cnn' COSMOS_WINDX_OUTPUT_ROOT='g:\03-downscaling_meteo_cnn\wind_extremes_repo_check'
cd $WT/validation/wind_extremes
for s in s2_prepare_obs s3_fit_return_levels s4_figures s5_obs_correction; do python $s.py > $COSMOS_WINDX_OUTPUT_ROOT/$s.log 2>&1; echo "$s exit $?"; done
```
Expected:
- all four stages exit 0;
- s3 prints `checks passed` and `0 degenerate`, and runs about 15 min;
- s4 writes 6 RP maps, 2 scatters, the quantile scatter and the BC-tail figure;
- s5 picks `V4_lw_kernel` as the winner.

s1 is not rerun, because its inputs are the 127 GB files. Its code path changed only in config lines, and Step 4 smoke-tests it.

- [ ] **Step 3: Write and run `checks/compare_to_golden.py`**

The script has a `# === CONFIGURATION ===` block with `GOLDEN = Path(r"g:\03-downscaling_meteo_cnn\wind_extremes\results_golden_20261003")`, `NEW = Path(r"g:\03-downscaling_meteo_cnn\wind_extremes_repo_check\results")` and `RTOL = 1e-9`. It then checks:
- every `*.csv` in GOLDEN, with `pd.testing.assert_frame_equal(check_exact=False, rtol=RTOL)`, after sorting both frames by all non-float columns;
- every `*.nc` in GOLDEN, opened with `netCDF4.Dataset`; for each variable, `np.testing.assert_allclose(rtol=RTOL, equal_nan=True)` on the filled arrays.

It prints one line per file, `OK` or `DIFF <max rel err>`, and exits 1 on any DIFF. A file in GOLDEN that is missing in NEW counts as a DIFF.

Expected: every line `OK` and exit 0. If anything differs:
- **stop**, and record which file and variable differs and by how much;
- do not loosen RTOL. A difference means the vendored `evt.py` or the env (pandas 3.0.0 against 3.0.5) changed a number, and Kees decides.

- [ ] **Step 4: Smoke-test s1 against the real files**

```
python checks/check_s1_smoke.py
```
Expected: exit 0, with the same output it gave in the source folder. It reads a small window from each product file via the new env-var paths.

- [ ] **Step 5: Look at the figures**

Open `wind_extremes_repo_check\figures\map_rl100_full.png` and `scatter_rl_matched.png`, and compare them with the same files in `$SRC/figures`. Expected: visually identical. The figures are turbo, 5–30 m/s, with BA as panel (d).

- [ ] **Step 6: Commit the check script**

```
git -C $WT add validation/wind_extremes/checks/compare_to_golden.py
git -C $WT commit -m "wind_extremes: golden-snapshot regression check (s2-s5 reproduce 2026-10-03 results to rtol 1e-9)"
```

### Task 7: Linux-readiness pass

**Files:**
- Check: all `.py` under `$WT/validation/wind_extremes`

- [ ] **Step 1: Grep for Windows-only constructs**

```
grep -rnE '\\\\|[a-zA-Z]:[\\/]|os\.path|\.exe' $WT/validation/wind_extremes --include=*.py | grep -v compare_to_golden
```
Expected: no hits outside comments and docstrings. The only absolute Windows paths allowed are in `compare_to_golden.py`'s CONFIGURATION block and in the README.

- [ ] **Step 2: Check the process pool**

`s3` and `s4` use `ProcessPoolExecutor`. Confirm that each entry point has `if __name__ == "__main__":`; Windows needs it (spawn), and it is harmless on Linux. Also confirm that workers receive only picklable arguments, which was already true on Windows.

Expected: all five stages already have the guard (verified 2026-10-03), so no edit is needed.

### Task 8: Push and hand off

**Files:**
- Create: `g:\03-downscaling_meteo_cnn\wind_extremes\MOVED.md`

- [ ] **Step 1: Push the branch**

```
git -C $WT log --oneline origin/main..HEAD
git -C $WT push -u origin feature/wind-extremes
```
Expected: 4 or 5 commits listed, and the push creates the remote branch. Do **not** merge into `main`, and do not open a PR unless Kees asks.

- [ ] **Step 2: Leave a pointer in the old folder**

Write `MOVED.md`. It says:
- the code now lives at `d:\Git\cosmos-wind-cnn\validation\wind_extremes` on branch `feature/wind-extremes`;
- this folder remains the output root (`COSMOS_WINDX_OUTPUT_ROOT`);
- the `.py` files here are frozen copies, and Kees can delete them together with `cache\tmp_*` (16 GB);
- `wind_extremes_repo_check\` can be deleted once the branch is merged.

Do not delete anything.

- [ ] **Step 3: Update memory and the vault**

- In memory, append one line to `project_cnn_wind_extremes.md` with the new code location, the branch name and the env vars.
- In the vault, add one entry to `ACFCD log.md` under today's heading, using the vault-note skill.

---

## Out of scope (follow-ups)

- **Running on Caldera.** This needs the two 127 GB CNN files and the RTMA/ERA5 bundle staged there, plus a SLURM launcher modelled on `validation/run_validation_v3.slurm`. Write that plan once Kees confirms where the CNN full-record files live on Caldera.
- **Merging to `main` and the PR.**
- **The open method decisions:** land vs water factor in s5, one factor per cell across RPs, and s5 on top of BA. They are unchanged by the move.

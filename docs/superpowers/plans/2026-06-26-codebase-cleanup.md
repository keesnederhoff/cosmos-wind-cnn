# Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove duplication and redundant entry points in `scripts/` — extract the copy-pasted inference core into the `src/` package, delete the two redundant quick-scripts, and complete deferred repo hygiene — without changing runtime behavior.

**Architecture:** The sliding-window inference dataset + bounded-RAM streamed-NetCDF inference loop is currently copy-pasted across `run_training_pipeline.py`, `run_inference.py`, and `inference_full_record.py`. Extract it into one new package module `src/cosmos_wind_cnn/inference.py` (`SlidingWindowDataset` + `run_streaming_inference`), covered by a real unit test, and make the three scripts thin callers that each prepare `full_ds`/`model`/`attrs` their own way. Then remove `preprocess.py`/`inference.py` (redundant), update doc/notebook references, port notebooks to `hr_`/`lr_`, and refresh the launcher `CLAUDE.md`.

**Tech Stack:** Python 3.11 package `cosmos_wind_cnn` (conda env `cosmos_wind_cnn`), PyTorch, xarray + netCDF4, pytest. Bash tool = Git Bash on Windows; use `conda run -n cosmos_wind_cnn` for python/pytest (base env is 3.8 and won't work). The env's python is also at `C:/Users/keesn/anaconda3/envs/cosmos_wind_cnn/python.exe` if `conda run` swallows stdout.

**Scope (set with Kees):** consolidate redundant scripts + de-duplicate pipeline logic + repo hygiene, at moderate risk. **Out of scope:** splitting `validate_inference.py`, SLURM consolidation.

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `src/cosmos_wind_cnn/inference.py` | **Create** | `SlidingWindowDataset` + `run_streaming_inference(model, full_ds, …, output_path, attrs)` — the shared inference core |
| `tests/test_inference.py` | **Create** | Unit tests for the new module (synthetic data + deterministic dummy model) |
| `scripts/run_training_pipeline.py` | **Modify** | `step_inference` builds `full_ds`/`model`/`attrs`, then calls `run_streaming_inference`; remove inlined `_SlidingWindowDataset` + stream loop |
| `scripts/run_inference.py` | **Modify** | Same: remove its duplicate `_SlidingWindowDataset` + loop; call the shared function |
| `scripts/inference_full_record.py` | **Modify** | Use the shared function for the write loop (keep its split-concat data prep) |
| `scripts/preprocess.py` | **Delete** | Redundant non-isolated dup of `preprocess_training.py` |
| `scripts/inference.py` | **Delete** | Redundant HPC quick-dup |
| `case_studies/{_template,puget_sound,sf_bay_conus404}/README.md` | **Modify** | Replace `preprocess.py`/`inference.py` refs with `preprocess_training.py`/`run_inference.py` |
| `notebooks/01_…`, `02_…`, `03_…`, `04_…` | **Modify** | Port variable keys to `hr_`/`lr_`; fix `preprocess.py` prose refs |
| `C:\Users\keesn\.claude_projects\cosmos-wind-cnn\CLAUDE.md` (launcher, outside repo) | **Modify** | Fix "two SLURM files only"; add `COSMOS_*`, `hr_`/`lr_`, `sf_bay_conus404` |

**Note on `end_date`:** investigated and left as-is — `2022-09-30` is a harmless loose upper bound; preprocessing's common-time intersection clips to the actual CONUS404 extent. No task.

---

### Task 0: Commit the pending rename and branch

**Files:** working tree (the uncommitted `sf_bay_conus404` rename + reference fixes)

- [ ] **Step 1: Confirm the working tree holds exactly the rename + reference fixes**
```bash
cd /d/Git/cosmos-wind-cnn
git status --short | head
conda run -n cosmos_wind_cnn pytest -q
```
Expected: the staged/unstaged changes are the `case_studies/sf_bay -> sf_bay_conus404` rename plus the reference updates from the prior task; tests pass.

- [ ] **Step 2: Commit the rename to main**
```bash
cd /d/Git/cosmos-wind-cnn
git add -A
git commit -m "$(cat <<'EOF'
Rename case_studies/sf_bay -> sf_bay_conus404 and update all references

Disambiguate the CONUS404-based SF Bay study from sf_bay_rtma. Updates SLURM
scripts, Python entry-point defaults, validate_inference, RTMA stage_data,
READMEs, and notebooks; fixes the inference/eval SLURM banners to echo the real
results/<run>/{checkpoint,output_inference,output_evaluation}/ layout.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -1
```

- [ ] **Step 3: Create the cleanup branch**
```bash
cd /d/Git/cosmos-wind-cnn
git checkout -b cleanup/dedup-scripts
git branch --show-current
```
Expected: on `cleanup/dedup-scripts`, clean tree.

---

### Task 1: Extract the shared inference core (TDD) and refactor `run_training_pipeline.py`

**Files:**
- Create: `src/cosmos_wind_cnn/inference.py`
- Create: `tests/test_inference.py`
- Modify: `scripts/run_training_pipeline.py` (remove `_SlidingWindowDataset` at lines 179-211 and the stream loop inside `step_inference`)

- [ ] **Step 1: Write the failing test for the new module**

Create `tests/test_inference.py`:
```python
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import netCDF4

from cosmos_wind_cnn.inference import SlidingWindowDataset, run_streaming_inference


def _make_ds(n_time=8, h=3, w=3, var_names=('lr_u', 'lr_v')):
    t0 = np.datetime64('2000-01-01T00')
    times = t0 + np.arange(n_time) * np.timedelta64(1, 'h')
    coords = {'time': times, 'y': np.arange(h, dtype='f8'), 'x': np.arange(w, dtype='f8')}
    data = {v: (('time', 'y', 'x'),
                (np.arange(n_time * h * w, dtype='f4').reshape(n_time, h, w) + i))
            for i, v in enumerate(var_names)}
    return xr.Dataset(data, coords=coords)


def _unit_stats(names):
    return {n: {'mean': 0.0, 'std': 1.0} for n in names}


def test_sliding_window_dataset_shapes_and_window_count():
    ds = _make_ds(n_time=8)
    d = SlidingWindowDataset(ds, ['lr_u', 'lr_v'], _unit_stats(['lr_u', 'lr_v']), sequence_length=3)
    assert len(d) == 8 - 3 + 1
    x, start = d[0]
    assert tuple(x.shape) == (3, 2, 3, 3)  # (seq, n_vars, y, x)
    assert int(start) == 0


def test_sliding_window_dataset_drops_nan_windows():
    ds = _make_ds(n_time=6)
    ds['lr_u'][2, :, :] = np.nan  # any window covering t=2 is invalid
    d = SlidingWindowDataset(ds, ['lr_u', 'lr_v'], _unit_stats(['lr_u', 'lr_v']), sequence_length=3)
    # windows starting at 0,1,2 all span t=2; only windows starting at 3 remain (n=6,seq=3 -> starts 0..3)
    assert len(d) == 1


class _LastFrameModel(nn.Module):
    """Deterministic stub: output channel c = last input timestep of channel c."""
    def __init__(self, n_out):
        super().__init__()
        self.n_out = n_out

    def forward(self, x):  # x: (B, seq, n_in, H, W)
        return x[:, -1, : self.n_out, :, :]


def test_run_streaming_inference_writes_expected_structure(tmp_path):
    ds = _make_ds(n_time=6)
    input_vars = ['lr_u', 'lr_v']
    output_vars = ['hr_u', 'hr_v']
    stats = _unit_stats(input_vars + output_vars)
    out = tmp_path / 'out.nc'
    model = _LastFrameModel(n_out=2).eval()

    n_pred, n_total = run_streaming_inference(
        model, ds, input_vars, output_vars, stats, sequence_length=3,
        output_path=out, device=torch.device('cpu'),
        batch_size=4, num_workers=0, time_chunk=100,
        attrs={'run_name': 'test', 'hr_source': 'CONUS404', 'lr_source': 'ERA5'},
    )

    assert n_total == 6
    assert n_pred > 0
    nc = netCDF4.Dataset(str(out))
    try:
        assert set(output_vars).issubset(set(nc.variables.keys()))
        assert nc.variables['hr_u'].shape == (6, 3, 3)
        assert nc.run_name == 'test'
        assert nc.lr_source == 'ERA5'
        # predictions land at t = window_start + (seq-1); first 2 rows stay fill (NaN)
        vals = nc.variables['hr_u'][:]
        assert np.isnan(vals[0]).all() and np.isnan(vals[1]).all()
        assert np.isfinite(vals[2]).all()
    finally:
        nc.close()
```

- [ ] **Step 2: Run the test to confirm it fails (module not yet created)**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn pytest tests/test_inference.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'cosmos_wind_cnn.inference'`.

- [ ] **Step 3: Create `src/cosmos_wind_cnn/inference.py`**

Move `SlidingWindowDataset` verbatim from `run_training_pipeline.py:179-211` (rename the class from `_SlidingWindowDataset` to `SlidingWindowDataset`, keep the body identical). Then write `run_streaming_inference` by lifting the streamed-write logic from `run_training_pipeline.py:282-394`, parameterizing the caller-specific bits (`output_path`, `attrs`). Use this exact signature and structure:

```python
"""Shared inference core: sliding-window dataset + bounded-RAM streamed NetCDF inference."""
import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from cosmos_wind_cnn.utils.config import var_units_for


class SlidingWindowDataset(Dataset):
    """In-memory sliding-window dataset for inference (normalizes inputs, drops NaN windows)."""

    def __init__(self, data, input_vars, stats, sequence_length):
        self.input_vars = input_vars
        self.sequence_length = sequence_length
        n_times = data.sizes['time']

        self.arrays = {}
        nan_at_time = np.zeros(n_times, dtype=bool)
        for var in input_vars:
            arr = data[var].values.astype(np.float32)
            nan_at_time |= np.isnan(arr).any(axis=(1, 2))
            mean, std = stats[var]['mean'], stats[var]['std']
            self.arrays[var] = (arr - mean) / (std + 1e-8)

        self.n_times = n_times
        self.valid_indices = [
            i for i in range(n_times - sequence_length + 1)
            if not nan_at_time[i:i + sequence_length].any()
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        slices = [self.arrays[v][start:start + self.sequence_length]
                  for v in self.input_vars]
        return torch.from_numpy(np.stack(slices, axis=1)), start


def run_streaming_inference(model, full_ds, input_vars, output_vars, stats,
                            sequence_length, output_path, *, device,
                            batch_size=64, num_workers=8, time_chunk=10000,
                            attrs=None):
    """Stream sliding-window inference over `full_ds`, writing predictions to a
    NetCDF at `output_path` one time-chunk at a time (bounded RAM).

    `full_ds` is an xarray Dataset of the `input_vars` on the target grid (may be
    lazy; loaded per chunk). `attrs` (dict) is written as NetCDF global attributes.
    Returns (n_predicted, n_total).
    """
    attrs = attrs or {}
    n_total = len(full_ds.time)
    time_coords = full_ds.time.values
    y_coords = full_ds.y.values if 'y' in full_ds.coords else None
    x_coords = full_ds.x.values if 'x' in full_ds.coords else None
    height = full_ds.sizes.get('y', full_ds.sizes.get('latitude'))
    width = full_ds.sizes.get('x', full_ds.sizes.get('longitude'))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    VAR_UNITS = var_units_for(output_vars)
    target_offset = sequence_length - 1

    epoch0 = np.datetime64('1900-01-01T00:00:00')
    time_hours = (time_coords.astype('datetime64[ns]') - epoch0) / np.timedelta64(1, 'h')

    nc = netCDF4.Dataset(str(output_path), 'w', format='NETCDF4')
    nc.createDimension('time', n_total)
    nc.createDimension('y', height)
    nc.createDimension('x', width)
    tv = nc.createVariable('time', 'f8', ('time',))
    tv.units = 'hours since 1900-01-01'
    tv.calendar = 'gregorian'
    tv[:] = time_hours
    if y_coords is not None:
        nc.createVariable('y', 'f8', ('y',))[:] = y_coords
    if x_coords is not None:
        nc.createVariable('x', 'f8', ('x',))[:] = x_coords
    t_chunk_nc = max(1, min(720, n_total))
    out_nc = {}
    for var in output_vars:
        v = nc.createVariable(var, 'f4', ('time', 'y', 'x'), zlib=True, complevel=1,
                              chunksizes=(t_chunk_nc, height, width),
                              fill_value=np.float32(np.nan))
        if var in VAR_UNITS:
            v.units = VAR_UNITS[var]
        out_nc[var] = v
    for key, value in attrs.items():
        setattr(nc, key, value)

    n_windows = max(0, n_total - sequence_length + 1)
    n_predicted = 0
    n_nan_outputs = 0
    with torch.no_grad():
        for s0 in tqdm(range(0, n_windows, time_chunk), desc='    Inference'):
            e0 = min(s0 + time_chunk, n_windows)
            in_hi = min(e0 + target_offset, n_total)
            block = full_ds.isel(time=slice(s0, in_hi)).load()
            ds_block = SlidingWindowDataset(block, input_vars, stats, sequence_length)
            pred_block = {var: np.full((e0 - s0, height, width), np.nan, dtype=np.float32)
                          for var in output_vars}
            if len(ds_block) > 0:
                loader = DataLoader(ds_block, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=torch.cuda.is_available())
                for batch_inputs, batch_starts in loader:
                    outputs = model(batch_inputs.to(device))
                    bnan = (~torch.isfinite(outputs)).sum().item()
                    if bnan > 0:
                        n_nan_outputs += bnan
                        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
                    outputs = outputs.cpu().numpy()
                    for b, local_start in enumerate(batch_starts.numpy()):
                        j = int(local_start)
                        for c, var in enumerate(output_vars):
                            mean, std = stats[var]['mean'], stats[var]['std']
                            pred_block[var][j] = outputs[b, c] * (std + 1e-8) + mean
                del loader
            t0 = s0 + target_offset
            t1 = e0 + target_offset
            for var in output_vars:
                out_nc[var][t0:t1, :, :] = pred_block[var]
            n_predicted += int(np.isfinite(
                next(iter(pred_block.values()))).any(axis=(1, 2)).sum())
            del block, ds_block, pred_block

    nc.close()
    if n_nan_outputs > 0:
        print(f"    WARNING: {n_nan_outputs:,} non-finite outputs replaced with 0.")
    return n_predicted, n_total
```

- [ ] **Step 4: Run the test to confirm it passes**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn pip install -e . >/dev/null
conda run -n cosmos_wind_cnn pytest tests/test_inference.py -q
```
Expected: PASS (3 tests).

- [ ] **Step 5: Refactor `run_training_pipeline.py` to use the shared core**

In `scripts/run_training_pipeline.py`:
1. Delete the `_SlidingWindowDataset` class (lines 179-211).
2. Add to the package imports block (near line 47): `from cosmos_wind_cnn.inference import run_streaming_inference`.
3. In `step_inference`, keep the data-prep (configs, stats, regrid → `full_ds`) and model-load unchanged (lines 217-302). Replace the output-file construction + streamed loop (lines 304-394) with: build the output path + attrs dict, call the shared function, then print/return. Use exactly:
```python
    # -- Output path + provenance attrs --
    tag_start = (start_date or str(common_times[0])[:10]).replace('-', '')
    tag_end = (end_date or str(common_times[-1])[:10]).replace('-', '')
    output_filename = f'full_record_ERA5_{tag_start}_{tag_end}.nc'
    output_path = run_dirs['output_inference'] / output_filename

    attrs = {
        'source_checkpoint': str(checkpoint_path),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'run_name': run_dirs['run_root'].name,
        'sequence_length': int(sequence_length),
        'hr_source': str(train_config.get('hr_source', 'HR')),
        'lr_source': str(train_config.get('lr_source', 'LR')),
    }
    if 'crs' in train_config:
        attrs['crs'] = str(train_config['crs'])

    n_predicted, n_total = run_streaming_inference(
        model, full_ds, input_vars, output_vars, stats, sequence_length,
        output_path, device=device, batch_size=batch_size,
        num_workers=num_workers, time_chunk=int(inf_config.get('inference_time_chunk', 10000)),
        attrs=attrs,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n    Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"    Predicted: {n_predicted:,} / {n_total:,} timesteps")
    return output_path
```
4. Remove now-unused imports if any (`netCDF4`, `Dataset`, `DataLoader`, `tqdm`) are still used elsewhere in the file — check with grep before deleting; only remove an import if it has zero remaining uses.

- [ ] **Step 6: Verify the pipeline still compiles and nothing else broke**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn python -m py_compile scripts/run_training_pipeline.py src/cosmos_wind_cnn/inference.py && echo "compile ok"
echo "--- _SlidingWindowDataset fully removed from the pipeline script? ---"
grep -n "_SlidingWindowDataset" scripts/run_training_pipeline.py && echo "FAIL still present" || echo "ok removed"
echo "--- shared fn wired? ---"
grep -n "run_streaming_inference" scripts/run_training_pipeline.py
conda run -n cosmos_wind_cnn pytest -q
```
Expected: `compile ok`; `ok removed`; the import + call present; full suite PASSES.

- [ ] **Step 7: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
git add src/cosmos_wind_cnn/inference.py tests/test_inference.py scripts/run_training_pipeline.py
git commit -m "$(cat <<'EOF'
refactor: extract shared streamed-inference core into package

Add cosmos_wind_cnn.inference (SlidingWindowDataset + run_streaming_inference)
with unit tests; run_training_pipeline.step_inference now calls it instead of
inlining the sliding-window dataset and stream loop.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Refactor `run_inference.py` onto the shared core

**Files:** Modify `scripts/run_inference.py` (remove its `_SlidingWindowDataset` at line 54 and the inline inference at ~line 281)

- [ ] **Step 1: Read the current inference section**
```bash
cd /d/Git/cosmos-wind-cnn
sed -n '54,90p;270,378p' scripts/run_inference.py
```
Note how it builds `full_ds`, `stats`, `output_path`, and which global attrs it writes — that data-prep stays; only the dataset class + streamed loop are replaced.

- [ ] **Step 2: Apply the refactor**

In `scripts/run_inference.py`:
1. Delete the local `_SlidingWindowDataset` class (starts line 54).
2. Add `from cosmos_wind_cnn.inference import run_streaming_inference` to the package imports (near line 47).
3. Replace the inline inference/output block (the part that builds the `_SlidingWindowDataset`, runs the DataLoader loop, and writes the NetCDF) with a call to `run_streaming_inference(model, full_ds, input_vars, output_vars, stats, sequence_length, output_path, device=device, batch_size=batch_size, num_workers=num_workers, attrs=attrs)`, constructing `attrs` from the same global attributes it previously wrote (source_checkpoint, checkpoint_epoch, run_name, sequence_length, crs, hr_source, lr_source — match whatever it currently sets). Keep its existing `output_path`/filename construction.
4. Remove now-unused imports (`netCDF4`, `Dataset`, `DataLoader`, `tqdm`) only if grep shows zero remaining uses.

- [ ] **Step 3: Verify**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn python -m py_compile scripts/run_inference.py && echo "compile ok"
grep -n "_SlidingWindowDataset" scripts/run_inference.py && echo "FAIL still present" || echo "ok removed"
grep -n "run_streaming_inference" scripts/run_inference.py
conda run -n cosmos_wind_cnn pytest -q
```
Expected: `compile ok`; `ok removed`; import + call present; tests PASS.

- [ ] **Step 4: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
git add scripts/run_inference.py
git commit -m "$(cat <<'EOF'
refactor: run_inference.py uses shared run_streaming_inference

Remove the duplicated _SlidingWindowDataset + stream loop (now in the package).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Refactor `inference_full_record.py` onto the shared core

**Files:** Modify `scripts/inference_full_record.py` (it has `ERA5InferenceDataset` at line 32; SLURM-invoked, so behavior must be preserved)

- [ ] **Step 1: Read its dataset + inference loop**
```bash
cd /d/Git/cosmos-wind-cnn
sed -n '32,120p' scripts/inference_full_record.py
grep -n "ERA5InferenceDataset\|DataLoader\|createVariable\|def main" scripts/inference_full_record.py
```
Determine whether `ERA5InferenceDataset` is equivalent to `SlidingWindowDataset` (same normalization + sliding-window + NaN-drop). It concatenates the train/val/test processed splits into one time-ordered `full_ds`, then slides windows.

- [ ] **Step 2: Apply the refactor**

In `scripts/inference_full_record.py`:
1. Keep the split-concatenation logic that builds the combined `full_ds` (this is its distinctive data-prep).
2. Add `from cosmos_wind_cnn.inference import run_streaming_inference`.
3. Replace the `ERA5InferenceDataset`-based inference + NetCDF write with a call to `run_streaming_inference(...)`, building `attrs` from the globals it currently writes. Delete `ERA5InferenceDataset` if it becomes unused.
4. If `ERA5InferenceDataset` has a genuinely different windowing contract than `SlidingWindowDataset` (e.g. it does NOT drop NaN windows, or indexes differently), do NOT force it — instead report DONE_WITH_CONCERNS describing the difference, and only share the NetCDF-write half (extract that separately is out of scope; in that case leave this script as-is and note it). Behavior preservation for this SLURM-invoked script takes priority over de-duplication.
5. Remove now-unused imports only if grep shows zero remaining uses.

- [ ] **Step 3: Verify**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn python -m py_compile scripts/inference_full_record.py && echo "compile ok"
conda run -n cosmos_wind_cnn pytest -q
```
Expected: `compile ok`; tests PASS. (No runtime data available; correctness rests on the equivalence review of the windowing contract — be explicit in the report about whether the contracts matched.)

- [ ] **Step 4: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
git add scripts/inference_full_record.py
git commit -m "$(cat <<'EOF'
refactor: inference_full_record.py uses shared run_streaming_inference

Reuse the package inference core for the streamed write; keep the split-concat
data preparation specific to the full-record path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Remove the redundant scripts and fix their references

**Files:** Delete `scripts/preprocess.py`, `scripts/inference.py`; modify `case_studies/{_template,puget_sound,sf_bay_conus404}/README.md`

- [ ] **Step 1: Confirm nothing executable references them**
```bash
cd /d/Git/cosmos-wind-cnn
git grep -nE "scripts/(preprocess|inference)\.py" -- scripts/*.slurm scripts/*.py src/ tests/ 2>/dev/null | grep -vE "preprocess_|inference_full|run_inference|preprocess_inference|validate_inference" || echo "no executable references — safe to delete"
```
Expected: `no executable references — safe to delete` (only docs/notebooks reference them).

- [ ] **Step 2: Delete the two scripts**
```bash
cd /d/Git/cosmos-wind-cnn
git rm scripts/preprocess.py scripts/inference.py
```

- [ ] **Step 3: Fix the README references**

In each of `case_studies/_template/README.md`, `case_studies/puget_sound/README.md`, `case_studies/sf_bay_conus404/README.md`, find the "quick per-step scripts" lines that call `scripts/preprocess.py` and `scripts/inference.py`:
```bash
cd /d/Git/cosmos-wind-cnn
grep -nE "scripts/(preprocess|inference)\.py" case_studies/*/README.md
```
Replace `python scripts/preprocess.py --case-study <X>` → `python scripts/preprocess_training.py --case-study <X> --run-name <run>` and `python scripts/inference.py --case-study <X>` → `python scripts/run_inference.py --case-study <X> --run-name <run> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>` (preserve each file's existing `<X>` case-study path). Keep surrounding prose; the point is the canonical run-isolated scripts.

- [ ] **Step 4: Verify no dangling refs + tests**
```bash
cd /d/Git/cosmos-wind-cnn
git grep -nE "scripts/(preprocess|inference)\.py([^_]|$)" -- case_studies/*/README.md ':!notebooks' 2>/dev/null | grep -vE "preprocess_training|preprocess_inference|run_inference|inference_full" || echo "READMEs clean"
ls scripts/preprocess.py scripts/inference.py 2>&1 | grep -c "No such file" | xargs -I{} echo "{} of 2 deleted"
conda run -n cosmos_wind_cnn pytest -q
```
Expected: `READMEs clean`; `2 of 2 deleted`; tests PASS. (Notebook prose refs are handled in Task 5.)

- [ ] **Step 5: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
git add -A
git commit -m "$(cat <<'EOF'
cleanup: remove redundant preprocess.py / inference.py quick-scripts

Both duplicated the run-isolated entry points (preprocess_training.py /
run_inference.py) and were referenced only in docs. Update case-study READMEs
to the canonical scripts.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Port notebooks to `hr_`/`lr_` and fix script prose

**Files:** `notebooks/01_data_exploration.ipynb`, `02_validate_raw_meteo.ipynb`, `03_validate_cnn_output.ipynb`, `04_compare_case_studies.ipynb`

- [ ] **Step 1: Inventory the stale references in notebooks**
```bash
cd /d/Git/cosmos-wind-cnn
grep -nE "conus404_[a-z]|era5_[a-z]|rtma_[a-z]|scripts/(preprocess|inference)\.py" notebooks/*.ipynb | head -40
```

- [ ] **Step 2: Update variable keys + script prose**

For each notebook, edit the JSON source cells: replace structural variable keys `conus404_<x>`→`hr_<x>`, `rtma_<x>`→`hr_<x>`, `era5_<x>`→`lr_<x>` (same suffix-preserving rule used in the repo refactor), and replace prose/code references to `scripts/preprocess.py`→`scripts/preprocess_training.py`. Keep factual provenance strings (dataset names like "ERA5"/"CONUS404" in markdown prose) intact. Use the Edit tool per occurrence (the `.ipynb` is JSON — keep it valid).

- [ ] **Step 3: Verify the notebooks are valid JSON and keys are ported**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn python - <<'PY'
import json, glob
for f in sorted(glob.glob('notebooks/*.ipynb')):
    json.load(open(f, encoding='utf-8'))  # raises if invalid
    print('valid:', f)
PY
grep -nE "conus404_[a-z]|era5_[a-z]|rtma_[a-z]" notebooks/*.ipynb && echo "FAIL: structural keys remain" || echo "ok: keys ported"
```
Expected: all notebooks `valid:`; `ok: keys ported`.

- [ ] **Step 4: Commit**
```bash
cd /d/Git/cosmos-wind-cnn
git add notebooks/
git commit -m "$(cat <<'EOF'
docs: port notebooks to hr_/lr_ keys and canonical scripts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Refresh the launcher `CLAUDE.md`

**Files:** Modify `C:\Users\keesn\.claude_projects\cosmos-wind-cnn\CLAUDE.md` (outside the git repo — edited in place, not committed)

- [ ] **Step 1: Apply targeted updates**

Read the file, then make these edits (Edit tool):
1. The HPC section says "Two SLURM files only: `cpu_tallgrass.slurm` and `gpu_tallgrass.slurm`". Replace with text noting there are now several SLURM jobs: the two full-pipeline launchers (`cpu_tallgrass.slurm`, `gpu_tallgrass.slurm`) plus standalone `gpu_tallgrass_inference.slurm`, `gpu_tallgrass_eval_only.slurm`, the RTMA variants (`gpu_tallgrass_rtma{,_fresh,_infer}.slurm`, `cpu_rtma_eval.slurm`), and `stage_raw_to_caldera.slurm`.
2. Add to the HPC/Behavior section: raw data and run outputs can live off `/home` via `COSMOS_DATA_ROOT` / `COSMOS_RESULTS_ROOT` (read by `utils/config.get_data_dir` / `get_run_dirs`); the Tallgrass SLURM jobs set these to caldera project space.
3. Add a "Naming convention" note: variable keys are model-agnostic `hr_*` (high-res target) / `lr_*` (low-res input); provenance is recorded in each `training.yaml`'s `hr_source`/`lr_source`.
4. Update the case-study list to `sf_bay_conus404`, `sf_bay_rtma`, `puget_sound` (the bare CONUS404 SF Bay study was renamed to `sf_bay_conus404`).

- [ ] **Step 2: Verify the key stale lines are gone**
```bash
grep -niE "two slurm files only" "C:/Users/keesn/.claude_projects/cosmos-wind-cnn/CLAUDE.md" && echo "FAIL still stale" || echo "ok updated"
grep -niE "COSMOS_DATA_ROOT|hr_|lr_|sf_bay_conus404" "C:/Users/keesn/.claude_projects/cosmos-wind-cnn/CLAUDE.md" | head
```
Expected: `ok updated`; the new terms present. (No commit — this file is outside the repo.)

---

### Task 7: Final verification and finalization

- [ ] **Step 1: Full sweep**
```bash
cd /d/Git/cosmos-wind-cnn
conda run -n cosmos_wind_cnn python -m py_compile scripts/*.py src/cosmos_wind_cnn/**/*.py && echo "all compile"
conda run -n cosmos_wind_cnn pytest -q
echo "=== duplication gone: _SlidingWindowDataset only defined once (in the package)? ==="
git grep -n "class SlidingWindowDataset\|_SlidingWindowDataset" -- src/ scripts/
echo "=== removed scripts gone, canonical set intact ==="
ls scripts/*.py
echo "=== config smoke-load still works ==="
conda run -n cosmos_wind_cnn python - <<'PY'
from pathlib import Path
from cosmos_wind_cnn.utils.config import load_config, parse_variable_config
for cs in ['sf_bay_conus404','sf_bay_rtma','puget_sound','_template']:
    tr = load_config(Path('case_studies')/cs/'configs'/'training.yaml')
    iv, ov, _ = parse_variable_config(tr)
    assert all(v.startswith('lr_') for v in iv if v not in (tr.get('additional_inputs') or []))
    assert all(v.startswith('hr_') for v in ov)
print('configs OK')
PY
git --no-pager diff --stat main..HEAD | tail -20
```
Expected: `all compile`; tests PASS; `SlidingWindowDataset` defined exactly once (in `src/cosmos_wind_cnn/inference.py`), with no `_SlidingWindowDataset` left in `scripts/`; `preprocess.py`/`inference.py` gone; `configs OK`.

- [ ] **Step 2: Finalize** — use `superpowers:finishing-a-development-branch` to integrate `cleanup/dedup-scripts` (offer merge-to-main-locally / push+PR / keep / discard). Do not push without Kees's go-ahead.

---

## Rollback
All refactor work is on `cleanup/dedup-scripts`; `main` (with the committed rename) is the integration point.
- Undo last commit (keep changes): `git reset --soft HEAD~1`
- Abandon the branch: `git checkout main && git branch -D cleanup/dedup-scripts`

## Notes / out of scope
- `validate_inference.py` (998 lines) split and SLURM consolidation were explicitly deferred.
- `run_inference.py` (date-range) and `inference_full_record.py` (full-record) stay as separate CLIs by design — they share the inference core but serve distinct entry points.
- No runtime equivalence test is possible without trained checkpoints + data + GPU; correctness of the inference extraction rests on the new unit tests plus the windowing-contract review in Tasks 2–3.

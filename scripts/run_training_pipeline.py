"""
Full training pipeline: preprocess, train, inference, and evaluate in one run.

Steps:
  1. Preprocess  — load raw data, align ERA5/CONUS404, split train/val/test
  2. Train       — full training loop with early stopping
  3. Archive     — copy all configs into checkpoint dir for reproducibility
  4. Inference   — regrid full ERA5 record onto target grid, run model
  5. Evaluate    — compare predictions vs CONUS404 at ~100 random grid points

Usage:
    # Single GPU
    python scripts/run_training_pipeline.py --case-study case_studies/sf_bay

    # Multi-GPU (DDP) — training step uses torchrun internally
    python scripts/run_training_pipeline.py --case-study case_studies/sf_bay --gpus 4

    # Custom run name and inference period
    python scripts/run_training_pipeline.py \\
        --case-study case_studies/sf_bay \\
        --run-name my_experiment \\
        --inference-start 1940-01-01 \\
        --inference-end   2026-12-31
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import pickle
import shutil
import subprocess
import netCDF4
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cosmos_wind_cnn.data.preprocessing import NetCDFPreprocessor
from cosmos_wind_cnn.data.regridder import Regridder
from cosmos_wind_cnn.models.unet3d import Wind3DUNET
from cosmos_wind_cnn.utils.config import (
    load_config, parse_variable_config, get_run_dirs, get_data_dir, var_units_for, wind_var_names,
)
from cosmos_wind_cnn.utils.visualization import plot_normalization_stats, plot_spatial_stats


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: Preprocess
# ═══════════════════════════════════════════════════════════════════════════

def step_preprocess(case_dir, run_dirs):
    """Load raw data, align, split, save stats and reference grid."""
    config = load_config(case_dir / 'configs' / 'preprocessing.yaml')
    data_dir = get_data_dir(case_dir)
    output_dir = run_dirs['data_processed']

    preprocessor = NetCDFPreprocessor({
        'data_dir': str(data_dir),
        'physical_bounds': config.get('physical_bounds', {}),
        'target_prefix': config.get('target_prefix', 'conus404_'),
        'input_prefix': config.get('input_prefix', 'era5_'),
        'regular_time_grid': config.get('regular_time_grid', False),
    })

    file_dict = config['file_dict']
    print("\nFiles to process:")
    for var, filename in file_dict.items():
        filepath = data_dir / filename
        status = "OK" if filepath.exists() else "NOT FOUND"
        print(f"  [{status}] {var}: {filename}")

    start_date = config.get('start_date')
    end_date = config.get('end_date')
    if start_date or end_date:
        print(f"  Time period: {start_date or 'start'} to {end_date or 'end'}")

    combined_ds = preprocessor.load_and_align_datasets(
        file_dict, start_date=start_date, end_date=end_date
    )

    # Save reference grid
    output_dir.mkdir(parents=True, exist_ok=True)
    regridder = Regridder.from_target_dataset(combined_ds)
    regridder.save_reference_grid(output_dir / 'target_grid_reference.nc')

    # Split
    train_ds, val_ds, test_ds = preprocessor.create_train_val_test_split(
        combined_ds,
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15),
    )

    # Save splits
    preprocessor.save_processed_data(train_ds, output_dir / 'train.nc')
    preprocessor.save_processed_data(val_ds, output_dir / 'val.nc')
    preprocessor.save_processed_data(test_ds, output_dir / 'test.nc')

    # Stats
    stats = preprocessor.calculate_and_save_stats(
        train_ds, output_dir / 'normalization_stats.pkl'
    )

    # Plots
    plot_normalization_stats(stats, output_dir)
    plot_spatial_stats(train_ds, output_dir)

    print(f"\n  Preprocessing complete:")
    print(f"    train.nc  : {len(train_ds.time)} timesteps")
    print(f"    val.nc    : {len(val_ds.time)} timesteps")
    print(f"    test.nc   : {len(test_ds.time)} timesteps")

    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: Train
# ═══════════════════════════════════════════════════════════════════════════

def step_train(case_dir, run_name, gpus):
    """Launch training via subprocess (supports DDP with --gpus > 1)."""
    script = Path(__file__).resolve().parent / 'train.py'

    if gpus > 1:
        cmd = [
            sys.executable, '-m', 'torch.distributed.run',
            '--nproc_per_node', str(gpus),
            str(script),
            '--case-study', str(case_dir),
            '--run-name', run_name,
        ]
    else:
        cmd = [
            sys.executable, str(script),
            '--case-study', str(case_dir),
            '--run-name', run_name,
        ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: Archive configs
# ═══════════════════════════════════════════════════════════════════════════

def step_archive_configs(case_dir, run_dirs):
    """Copy all config files into the checkpoint directory for reproducibility."""
    checkpoint_dir = run_dirs['checkpoint']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = case_dir / 'configs'
    archived = []
    for yaml_file in sorted(configs_dir.glob('*.yaml')):
        dest = checkpoint_dir / yaml_file.name
        shutil.copy2(yaml_file, dest)
        archived.append(yaml_file.name)

    print(f"  Archived {len(archived)} config(s) to {checkpoint_dir}/")
    for name in archived:
        print(f"    {name}")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: Inference (regrid + run model)
# ═══════════════════════════════════════════════════════════════════════════

class _SlidingWindowDataset(Dataset):
    """In-memory sliding-window dataset for inference."""

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
        n_dropped = (n_times - sequence_length + 1) - len(self.valid_indices)
        print(f"    {len(self.valid_indices):,} valid windows "
              f"({n_dropped:,} dropped -- NaN)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        slices = [self.arrays[v][start:start + self.sequence_length]
                  for v in self.input_vars]
        return torch.from_numpy(np.stack(slices, axis=1)), start


def step_inference(case_dir, run_dirs, start_date, end_date, batch_size,
                   num_workers):
    """Regrid ERA5 onto target grid and run trained model."""
    processed_dir = run_dirs['data_processed']
    data_dir = get_data_dir(case_dir)
    checkpoint_dir = run_dirs['checkpoint']

    # Load archived configs (from checkpoint dir for reproducibility)
    train_config = load_config(checkpoint_dir / 'training.yaml')
    inf_config = load_config(checkpoint_dir / 'inference_preprocessing.yaml')

    input_vars, output_vars, _ = parse_variable_config(train_config)
    sequence_length = train_config['sequence_length']

    # Stats from training
    with open(processed_dir / 'normalization_stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    # Pad start for sliding window
    if start_date:
        load_start = str(
            np.datetime64(start_date, 'ns')
            - np.timedelta64(sequence_length - 1, 'h')
        )[:19]
    else:
        load_start = None

    # -- Regrid --
    print("\n  Regridding ERA5 onto target grid...")
    ref_grid_path = processed_dir / 'target_grid_reference.nc'
    interp_method = inf_config.get('interpolation_method', 'linear')
    regridder = Regridder.from_reference_grid(ref_grid_path, method=interp_method)

    sources = inf_config['sources']
    physical_bounds = inf_config.get('physical_bounds', {})

    regridded_vars = {}
    for var_name, source_cfg in sources.items():
        filepath = data_dir / source_cfg['file']
        if not filepath.exists():
            raise FileNotFoundError(f"Source file not found: {filepath}")

        ds = xr.open_dataset(filepath, chunks='auto')
        source_var = source_cfg.get('source_var')
        if source_var is None:
            source_var = list(ds.data_vars)[0]

        var_map = {var_name: source_var}
        bounds = {var_name: physical_bounds[var_name]} if var_name in physical_bounds else {}
        regridded = regridder.regrid(ds, var_map=var_map,
                                     physical_bounds=bounds,
                                     start_date=load_start,
                                     end_date=end_date)
        regridded_vars[var_name] = regridded[var_name]
        ds.close()

    # Align times
    time_sets = [set(da.time.values) for da in regridded_vars.values()]
    common_times = sorted(time_sets[0].intersection(*time_sets[1:]))
    if not common_times:
        raise RuntimeError("No overlapping timesteps across source files.")
    print(f"\n    Common timesteps: {len(common_times)}")

    for var_name in regridded_vars:
        regridded_vars[var_name] = regridded_vars[var_name].sel(time=common_times)

    full_ds = xr.Dataset(regridded_vars)[input_vars]   # kept lazy; loaded per time-chunk

    n_total = len(full_ds.time)
    time_coords = full_ds.time.values
    y_coords = full_ds.y.values if 'y' in full_ds.coords else None
    x_coords = full_ds.x.values if 'x' in full_ds.coords else None
    height = full_ds.sizes.get('y', full_ds.sizes.get('latitude'))
    width = full_ds.sizes.get('x', full_ds.sizes.get('longitude'))

    # -- Load model --
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = checkpoint_dir / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = Wind3DUNET(
        in_channels=len(input_vars),
        out_channels=len(output_vars),
        base_channels=train_config['base_channels'],
        dropout_rate=train_config['dropout_rate'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"    Model loaded from epoch {checkpoint['epoch']}")

    # -- Prepare streamed output file (RAM stays bounded: one time-chunk at a time) --
    tag_start = (start_date or str(common_times[0])[:10]).replace('-', '')
    tag_end = (end_date or str(common_times[-1])[:10]).replace('-', '')
    output_filename = f'full_record_ERA5_{tag_start}_{tag_end}.nc'
    output_path = run_dirs['output_inference'] / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    VAR_UNITS = var_units_for(output_vars)
    target_offset = sequence_length - 1
    time_chunk = int(inf_config.get('inference_time_chunk', 10000))

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
    nc.source_checkpoint = str(checkpoint_path)
    nc.checkpoint_epoch = int(checkpoint['epoch'])
    nc.run_name = run_dirs['run_root'].name
    nc.sequence_length = int(sequence_length)
    if 'crs' in train_config:
        nc.crs = str(train_config['crs'])

    # -- Streamed inference: iterate over blocks of sliding-window start indices --
    n_windows = max(0, n_total - sequence_length + 1)
    n_predicted = 0
    n_nan_outputs = 0
    print(f"    Streaming inference over {n_windows:,} windows "
          f"in chunks of {time_chunk:,} (grid {height}x{width})...")
    with torch.no_grad():
        for s0 in tqdm(range(0, n_windows, time_chunk), desc='    Inference'):
            e0 = min(s0 + time_chunk, n_windows)          # window-starts [s0, e0)
            in_hi = min(e0 + target_offset, n_total)       # input rows [s0, in_hi)
            block = full_ds.isel(time=slice(s0, in_hi)).load()
            ds_block = _SlidingWindowDataset(block, input_vars, stats, sequence_length)
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
                        j = int(local_start)               # block-local window-start
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
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n    Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"    Predicted: {n_predicted:,} / {n_total:,} timesteps")

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5: Evaluate vs CONUS404 at random grid points
# ═══════════════════════════════════════════════════════════════════════════

def step_evaluate_grid_points(case_dir, run_dirs, inference_path,
                              n_points=100, seed=42):
    """Compare model vs ERA5 vs CONUS404 at random grid points."""
    processed_dir = run_dirs['data_processed']
    output_dir = run_dirs['output_evaluation'] / 'grid_points'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = load_config(run_dirs['checkpoint'] / 'training.yaml')
    names = wind_var_names(train_config['variable_pairs'])
    if names is None:
        print("    No wind pair in training config -- skipping evaluation.")
        return
    u_tgt, v_tgt, u_in, v_in = names

    # Load inference and processed data
    inference_ds = xr.open_dataset(inference_path, chunks='auto')

    splits = []
    for name in ('train', 'val', 'test'):
        p = processed_dir / f'{name}.nc'
        if p.exists():
            splits.append(xr.open_dataset(p, chunks='auto'))
    if not splits:
        print("    No processed splits found -- skipping evaluation.")
        return
    processed_ds = xr.concat(splits, dim='time').sortby('time')

    # Find temporal overlap (CONUS404 is only 1979-2021)
    inf_time = pd.DatetimeIndex(inference_ds.time.values)
    proc_time = pd.DatetimeIndex(processed_ds.time.values)
    common = inf_time.intersection(proc_time)

    if len(common) < 100:
        print(f"    Only {len(common)} common timesteps -- skipping evaluation.")
        return

    inf_idx = inf_time.get_indexer(common)
    proc_idx = proc_time.get_indexer(common)
    print(f"    Temporal overlap: {len(common)} timesteps "
          f"({common[0].date()} -- {common[-1].date()})")

    # Random grid points
    rng = np.random.default_rng(seed)
    ny, nx = len(inference_ds.y), len(inference_ds.x)
    iys = rng.integers(0, ny, n_points)
    ixs = rng.integers(0, nx, n_points)

    # Check required variables
    for var in [u_tgt, v_tgt]:
        if var not in inference_ds:
            print(f"    {var} not in inference output -- skipping evaluation.")
            return
    for var in [u_tgt, v_tgt, u_in, v_in]:
        if var not in processed_ds:
            print(f"    {var} not in processed data -- skipping evaluation.")
            return

    all_records = []
    running_ss = []

    # Single-pass vectorized extraction of all sampled points, restricted to the
    # evaluation overlap window. The model output spans the full ERA5 record
    # (1940-2027) but eval only needs the common period (where the high-res
    # target exists), so we read just that time-slice -- far less I/O.
    inf_lo = int(inf_idx.min())
    inf_hi = int(inf_idx.max()) + 1
    inf_idx_rel = inf_idx - inf_lo
    inf_sub = inference_ds.isel(time=slice(inf_lo, inf_hi))
    pts_y = xr.DataArray(iys, dims='points')
    pts_x = xr.DataArray(ixs, dims='points')
    print(f'    Extracting {n_points} points over {inf_hi - inf_lo} overlap steps (single pass)...')
    mod_u_all = inf_sub[u_tgt].isel(y=pts_y, x=pts_x).transpose('time', 'points').values
    mod_v_all = inf_sub[v_tgt].isel(y=pts_y, x=pts_x).transpose('time', 'points').values
    tru_u_all = processed_ds[u_tgt].isel(y=pts_y, x=pts_x).transpose('time', 'points').values
    tru_v_all = processed_ds[v_tgt].isel(y=pts_y, x=pts_x).transpose('time', 'points').values
    e5_u_all = processed_ds[u_in].isel(y=pts_y, x=pts_x).transpose('time', 'points').values
    e5_v_all = processed_ds[v_in].isel(y=pts_y, x=pts_x).transpose('time', 'points').values

    for pt, (iy, ix) in enumerate(tqdm(zip(iys, ixs),
                                        total=n_points,
                                        desc='    Grid points')):
        iy, ix = int(iy), int(ix)

        # Indexed from pre-extracted (time, points) arrays (single-pass read above)
        mod_u = mod_u_all[inf_idx_rel, pt].astype(float)
        mod_v = mod_v_all[inf_idx_rel, pt].astype(float)
        tru_u = tru_u_all[proc_idx, pt].astype(float)
        tru_v = tru_v_all[proc_idx, pt].astype(float)
        e5_u = e5_u_all[proc_idx, pt].astype(float)
        e5_v = e5_v_all[proc_idx, pt].astype(float)

        # Wind speed
        mod_ws = np.sqrt(mod_u**2 + mod_v**2)
        tru_ws = np.sqrt(tru_u**2 + tru_v**2)
        e5_ws = np.sqrt(e5_u**2 + e5_v**2)

        # RMSE
        mask = ~(np.isnan(mod_ws) | np.isnan(tru_ws) | np.isnan(e5_ws))
        if mask.sum() < 10:
            continue

        rmse_mod = float(np.sqrt(np.nanmean((mod_ws[mask] - tru_ws[mask])**2)))
        rmse_e5 = float(np.sqrt(np.nanmean((e5_ws[mask] - tru_ws[mask])**2)))
        ss = 1.0 - rmse_mod / rmse_e5 if rmse_e5 > 0 else np.nan

        rmse_mod_u = float(np.sqrt(np.nanmean((mod_u[mask] - tru_u[mask])**2)))
        rmse_e5_u = float(np.sqrt(np.nanmean((e5_u[mask] - tru_u[mask])**2)))
        rmse_mod_v = float(np.sqrt(np.nanmean((mod_v[mask] - tru_v[mask])**2)))
        rmse_e5_v = float(np.sqrt(np.nanmean((e5_v[mask] - tru_v[mask])**2)))

        all_records.append({
            'iy': iy, 'ix': ix, 'n_valid': int(mask.sum()),
            'rmse_model_ws': rmse_mod, 'rmse_era5_ws': rmse_e5,
            'skill_score_ws': ss,
            'rmse_model_u': rmse_mod_u, 'rmse_era5_u': rmse_e5_u,
            'rmse_model_v': rmse_mod_v, 'rmse_era5_v': rmse_e5_v,
        })
        running_ss.append(ss)

    if not all_records:
        print("    No valid grid points -- skipping.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(output_dir / 'grid_point_metrics.csv', index=False)

    # Summary
    med_ss = float(np.nanmedian(df['skill_score_ws']))
    mean_ss = float(np.nanmean(df['skill_score_ws']))
    mean_rmse_model = float(df['rmse_model_ws'].mean())
    mean_rmse_era5 = float(df['rmse_era5_ws'].mean())

    summary = {
        'n_points': len(df),
        'n_common_timesteps': len(common),
        'period': f"{common[0].date()} -- {common[-1].date()}",
        'wind_speed': {
            'median_skill_score': med_ss,
            'mean_skill_score': mean_ss,
            'mean_rmse_model': mean_rmse_model,
            'mean_rmse_era5': mean_rmse_era5,
        },
    }
    with open(output_dir / 'grid_point_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n    Results ({len(df)} grid points, {len(common)} timesteps):")
    print(f"      Wind speed RMSE  model: {mean_rmse_model:.3f} m/s")
    print(f"      Wind speed RMSE  ERA5:  {mean_rmse_era5:.3f} m/s")
    print(f"      Skill score (median):   {med_ss:.3f}")
    print(f"      Skill score (mean):     {mean_ss:.3f}")
    print(f"    Saved to: {output_dir}")

    # Close datasets
    inference_ds.close()
    processed_ds.close()
    for ds in splits:
        ds.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")

    parser = argparse.ArgumentParser(
        description='Full training pipeline: preprocess -> train -> inference -> evaluate'
    )
    parser.add_argument('--case-study', default='case_studies/sf_bay')
    parser.add_argument('--data-root', default=None,
                        help='Base dir for raw input data; reads <data-root>/<case_name>/raw. '
                             'Overrides COSMOS_DATA_ROOT. Default: <case-study>/data/raw')
    parser.add_argument('--results-root', default=None,
                        help='Base dir for run outputs; writes <results-root>/<case_name>/<run-name>. '
                             'Overrides COSMOS_RESULTS_ROOT. Default: <case-study>/results')
    parser.add_argument('--run-name', default='default',
                        help='Name for this run (used for checkpoint/output dirs)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs for training (default: 1)')
    parser.add_argument('--inference-start', default=None,
                        help='Inference start date (default: from config)')
    parser.add_argument('--inference-end', default=None,
                        help='Inference end date (default: from config)')
    parser.add_argument('--eval-points', type=int, default=100,
                        help='Number of random grid points for evaluation (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Inference batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader workers (default: 8)')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip preprocessing (use existing processed data)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training (use existing checkpoint)')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip inference')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip grid point evaluation')
    args = parser.parse_args()

    # Explicit path overrides (callable from CLI/Python). Set env vars so every
    # downstream helper (get_run_dirs / get_data_dir) picks up the same location.
    if args.data_root:
        os.environ['COSMOS_DATA_ROOT'] = args.data_root
    if args.results_root:
        os.environ['COSMOS_RESULTS_ROOT'] = args.results_root

    case_dir = Path(args.case_study)
    run_name = args.run_name
    run_dirs = get_run_dirs(case_dir, run_name)
    pipeline_start = time.time()

    print("=" * 70)
    print(f"TRAINING PIPELINE: {case_dir.name}")
    print("=" * 70)
    print(f"  Run name : {run_name}")
    print(f"  Run root : {run_dirs['run_root']}")
    print(f"  GPUs     : {args.gpus}")

    # ── Step 1: Preprocess ────────────────────────────────────────────────
    if not args.skip_preprocess:
        print("\n" + "=" * 70)
        print("STEP 1/5: Preprocessing")
        print("=" * 70)
        t0 = time.time()
        step_preprocess(case_dir, run_dirs)
        print(f"\n  Step 1 completed in {timedelta(seconds=int(time.time() - t0))}")
    else:
        print("\n  Step 1: Preprocessing -- SKIPPED")

    # ── Step 2: Train ─────────────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 70)
        print("STEP 2/5: Training")
        print("=" * 70)
        t0 = time.time()
        step_train(case_dir, run_name, args.gpus)
        print(f"\n  Step 2 completed in {timedelta(seconds=int(time.time() - t0))}")
    else:
        print("\n  Step 2: Training -- SKIPPED")

    # ── Step 3: Archive configs ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3/5: Archiving configs")
    print("=" * 70)
    step_archive_configs(case_dir, run_dirs)

    # ── Step 4: Inference ─────────────────────────────────────────────────
    inference_path = None
    if not args.skip_inference:
        print("\n" + "=" * 70)
        print("STEP 4/5: Inference (regrid + model)")
        print("=" * 70)

        # Get inference period from config if not specified on CLI
        # Prefer archived copy (step 3 just put it there); fall back to configs/
        inf_config_path = run_dirs['checkpoint'] / 'inference_preprocessing.yaml'
        if not inf_config_path.exists():
            inf_config_path = case_dir / 'configs' / 'inference_preprocessing.yaml'
        inf_config = load_config(inf_config_path)
        inf_start = args.inference_start or inf_config.get('start_date')
        inf_end = args.inference_end or inf_config.get('end_date')
        print(f"  Period: {inf_start or '(start)'} -> {inf_end or '(end)'}")

        t0 = time.time()
        inference_path = step_inference(
            case_dir, run_dirs, inf_start, inf_end,
            args.batch_size, args.num_workers,
        )
        print(f"\n  Step 4 completed in {timedelta(seconds=int(time.time() - t0))}")
    else:
        print("\n  Step 4: Inference -- SKIPPED")
        inf_dir = run_dirs['output_inference']
        if inf_dir.exists():
            cands = sorted(inf_dir.glob('full_record_*.nc'))
            if cands:
                inference_path = max(cands, key=lambda p: p.stat().st_size)
                print(f"    Using existing inference output: {inference_path}")

    # ── Step 5: Evaluate vs CONUS404 ──────────────────────────────────────
    if not args.skip_eval and inference_path is not None:
        print("\n" + "=" * 70)
        print("STEP 5/5: Evaluating vs CONUS404 at random grid points")
        print("=" * 70)
        t0 = time.time()
        step_evaluate_grid_points(
            case_dir, run_dirs, inference_path,
            n_points=args.eval_points,
        )
        print(f"\n  Step 5 completed in {timedelta(seconds=int(time.time() - t0))}")
    else:
        print("\n  Step 5: Evaluation -- SKIPPED")

    # ── Copy SLURM log into the run's logs directory ────────────────────
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        log_dir = run_dirs['logs']
        log_dir.mkdir(parents=True, exist_ok=True)
        # Common SLURM log patterns: gpu_pipeline_<id>.log, cpu_pipeline_<id>.log
        for pattern in (f'gpu_pipeline_{slurm_job_id}.log',
                        f'cpu_pipeline_{slurm_job_id}.log',
                        f'slurm-{slurm_job_id}.out'):
            src = Path(pattern)
            if src.exists():
                dest = log_dir / src.name
                shutil.copy2(src, dest)
                print(f"\n  Copied SLURM log: {src} -> {dest}")

    # ── Done ──────────────────────────────────────────────────────────────
    total = timedelta(seconds=int(time.time() - pipeline_start))
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE  ({total})")
    print("=" * 70)
    print(f"\n  Run root    : {run_dirs['run_root']}/")
    print(f"  Checkpoint  : {run_dirs['checkpoint']}/")
    if inference_path:
        print(f"  Inference   : {inference_path}")
    print(f"  Evaluation  : {run_dirs['output_evaluation']}/")


if __name__ == '__main__':
    main()

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
    python scripts/run_training_pipeline.py --case-study case_studies/sf_bay_conus404

    # Multi-GPU (DDP) — training step uses torchrun internally
    python scripts/run_training_pipeline.py --case-study case_studies/sf_bay_conus404 --gpus 4

    # Custom run name and inference period
    python scripts/run_training_pipeline.py \\
        --case-study case_studies/sf_bay_conus404 \\
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
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

from cosmos_wind_cnn.data.preprocessing import NetCDFPreprocessor
from cosmos_wind_cnn.inference import run_streaming_inference
from cosmos_wind_cnn.data.regridder import Regridder
from cosmos_wind_cnn.models.unet3d import Wind3DUNET, build_wind3dunet
from cosmos_wind_cnn.training.quantile_losses import (
    crps_numpy, twcrps_numpy, pit_values, interval_coverage)
from cosmos_wind_cnn.utils.config import (
    load_config, parse_variable_config, get_run_dirs, get_data_dir, var_units_for, wind_var_names,
    env_bool, env_list,
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
        'target_prefix': config.get('target_prefix', 'hr_'),
        'input_prefix': config.get('input_prefix', 'lr_'),
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
        split_dates=config.get('split_dates'),
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


def step_inference(case_dir, run_dirs, start_date, end_date, batch_size,
                   num_workers):
    """Regrid ERA5 onto target grid and run trained model."""
    processed_dir = run_dirs['data_processed']
    data_dir = get_data_dir(case_dir)
    checkpoint_dir = run_dirs['checkpoint']

    # Load archived configs (from checkpoint dir for reproducibility)
    train_config = load_config(checkpoint_dir / 'training.yaml')
    # Which inference-preprocessing config to use. Defaults to the archived copy
    # so every existing result reproduces. INFER_CONFIG points it elsewhere.
    #
    # WHY THIS EXISTS: the archived config is deliberately a SUPERSET of every
    # predictor block, and step_inference regrids ALL declared sources and
    # intersects their time axes BEFORE selecting the ones this arm needs. The
    # pressure-level files start 2015, so a 4-channel P0 arm -- whose own inputs
    # go back to 1940 -- still aborts with 'No overlapping timesteps' for any
    # year before 2015. A trimmed config restores the arm's true usable span.
    _icfg = os.environ.get('INFER_CONFIG')
    inf_config = load_config(Path(_icfg) if _icfg
                             else checkpoint_dir / 'inference_preprocessing.yaml')

    # Apply env overrides BEFORE parsing variables: SWEEP_ADD_INPUTS changes
    # additional_inputs, which parse_variable_config turns into input channels.
    # (Previously the overrides ran after the parse, so ADD_INPUTS would have been
    # silently ignored here and inference would build a mis-shaped model.)
    for _env, _key, _cast in [('SWEEP_BASE_CHANNELS', 'base_channels', int),
                              ('SWEEP_SEQ_LEN', 'sequence_length', int),
                              ('SWEEP_DROPOUT', 'dropout_rate', float),
                              ('SWEEP_RESIDUAL', 'residual_learning', env_bool),
                              ('SWEEP_ADD_INPUTS', 'additional_inputs', env_list),
                              # The head settings MUST be replayed here too. The
                              # archived training.yaml holds file defaults, not
                              # the per-run env overrides, so without these a
                              # quantile-trained checkpoint gets rebuilt as a
                              # deterministic model -- silently wrong output, the
                              # same trap as the old residual-blind backfill.
                              ('SWEEP_HEAD', 'head', str),
                              ('SWEEP_N_QUANTILES', 'n_quantiles', int),
                              ('SWEEP_GUST_WEIGHT', 'loss_gust_weight', float)]:
        if os.environ.get(_env):
            train_config[_key] = _cast(os.environ[_env])

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

    # Only regrid what this checkpoint actually consumes. Every declared source
    # used to be loaded and its time axis intersected with the rest, so an
    # UNUSED variable with a shorter record silently truncated the window: the
    # ERA5 pressure-level files start 2015-01-01 and collapsed a 2014 run to 24
    # hours while still exiting 0.
    _all_sources = inf_config['sources']
    _needed = set(input_vars)
    sources = {k: v for k, v in _all_sources.items() if k in _needed}
    _skipped = [k for k in _all_sources if k not in sources]
    if _skipped:
        print(f"    Skipping {len(_skipped)} source(s) unused by this checkpoint: "
              f"{', '.join(_skipped)}")
    _missing = [v for v in input_vars if v not in sources]
    if _missing:
        raise KeyError(
            f"inference_preprocessing.yaml declares no source for required "
            f"input(s): {_missing}")
    physical_bounds = inf_config.get('physical_bounds', {})

    regridded_vars = {}   # time-varying inputs -> (time, y, x)
    static_vars = {}      # static, no-time inputs -> (y, x), broadcast below
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

        # Static input-only fields (e.g. terrain) have no time dim: interp onto
        # the target grid now, then broadcast over the common time axis below
        # (mirrors training preprocessing).
        if 'time' not in ds[source_var].dims:
            regridded = regridder.regrid_static(ds, var_map=var_map,
                                                physical_bounds=bounds)
            static_vars[var_name] = regridded[var_name]
            ds.close()
            continue

        regridded = regridder.regrid(ds, var_map=var_map,
                                     physical_bounds=bounds,
                                     start_date=load_start,
                                     end_date=end_date)
        regridded_vars[var_name] = regridded[var_name]
        ds.close()

    # Align times across the time-varying inputs
    time_sets = [set(da.time.values) for da in regridded_vars.values()]
    common_times = sorted(time_sets[0].intersection(*time_sets[1:]))
    if not common_times:
        raise RuntimeError("No overlapping timesteps across source files.")
    print(f"\n    Common timesteps: {len(common_times)}")

    # Fail loudly on a short window. Silent truncation has produced a 12-hour
    # "12-year" file once already; partial coverage must be opted into.
    if start_date and end_date:
        _exp = int((np.datetime64(end_date, 'h') - np.datetime64(start_date, 'h'))
                   / np.timedelta64(1, 'h'))
        _cov = len(common_times) / max(_exp, 1)
        print(f"    Requested {start_date} -> {end_date} = {_exp} h; "
              f"sources cover {len(common_times)} ({_cov:.1%})")
        if _cov < 0.95 and not os.environ.get('ALLOW_PARTIAL_INFERENCE'):
            _spans = {k: (str(v.time.values[0])[:13], str(v.time.values[-1])[:13],
                          int(v.time.size)) for k, v in regridded_vars.items()}
            raise RuntimeError(
                f"Source coverage {_cov:.1%} of the requested window is below 95%. "
                f"Per-variable spans: {_spans}. Set ALLOW_PARTIAL_INFERENCE=1 to "
                f"proceed anyway.")

    for var_name in regridded_vars:
        regridded_vars[var_name] = regridded_vars[var_name].sel(time=common_times)

    # Broadcast static inputs onto the common time axis as constant channels
    # (kept lazy; materialized per time-chunk during streamed inference).
    for var_name, da in static_vars.items():
        regridded_vars[var_name] = da.expand_dims(
            time=common_times).transpose('time', 'y', 'x')

    full_ds = xr.Dataset(regridded_vars)[input_vars]   # kept lazy; loaded per time-chunk

    n_total = len(full_ds.time)
    time_coords = full_ds.time.values
    y_coords = full_ds.y.values if 'y' in full_ds.coords else None
    x_coords = full_ds.x.values if 'x' in full_ds.coords else None
    height = full_ds.sizes.get('y', full_ds.sizes.get('latitude'))
    width = full_ds.sizes.get('x', full_ds.sizes.get('longitude'))

    # -- Load model --
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Which checkpoint to run inference from. Defaults to best_model.pth so
    # every existing result is reproduced bit-for-bit; INFER_CHECKPOINT lets the
    # per-head artefacts (best_speed / best_direction / best_gust / best_smooth)
    # be scored without a second code path. The name is carried into the output
    # filename below so two checkpoints of the same run cannot overwrite one
    # another -- a silent-overwrite trap, since the arm name is otherwise the
    # only thing distinguishing them.
    _ckpt_name = os.environ.get('INFER_CHECKPOINT', 'best_model.pth')
    checkpoint_path = checkpoint_dir / _ckpt_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"INFER_CHECKPOINT={_ckpt_name} not found in {checkpoint_dir}. "
            f"Present: {sorted(p.name for p in checkpoint_dir.glob('*.pth'))}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_wind3dunet(train_config, stats, input_vars, output_vars).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"    Model loaded from epoch {checkpoint['epoch']}")

    # -- Output path + provenance attrs --
    tag_start = (start_date or str(common_times[0])[:10]).replace('-', '')
    tag_end = (end_date or str(common_times[-1])[:10]).replace('-', '')
    _ckpt_tag = ('' if _ckpt_name == 'best_model.pth'
                 else _ckpt_name[:-4].replace('best_', '') + '_')
    output_filename = f'{_ckpt_tag}full_record_ERA5_{tag_start}_{tag_end}.nc'
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

    # Describe the probabilistic head to the writer, or None for the
    # deterministic path (which stays bit-identical).
    head_spec = None
    if str(train_config.get('head', 'det')).lower() in ('quantile', 'q'):
        from cosmos_wind_cnn.models.quantile_head import make_tau_grid
        _n_speed = int(train_config.get('n_quantiles', 19))
        _gust_name = train_config.get('gust_target', 'hr_gust')
        _n_gust = (int(train_config.get('n_gust_quantiles', 9))
                   if _gust_name in output_vars else 0)
        _subset = os.environ.get('INFER_QUANTILES')
        head_spec = {
            'n_speed': _n_speed,
            'n_gust': _n_gust,
            'speed_taus': make_tau_grid(_n_speed),
            'gust_taus': make_tau_grid(_n_gust) if _n_gust else None,
            'quantile_subset': ([float(t) for t in _subset.split(',')]
                                if _subset else None),
        }
        print(f"    Probabilistic output: {_n_speed} speed + {_n_gust} gust "
              f"quantiles" + (f", writing subset {head_spec['quantile_subset']}"
                              if head_spec['quantile_subset'] else " (all levels)"))

    n_predicted, n_total = run_streaming_inference(
        model, full_ds, input_vars, output_vars, stats, sequence_length,
        output_path, device=device, batch_size=batch_size,
        num_workers=num_workers,
        time_chunk=int(inf_config.get('inference_time_chunk', 10000)),
        attrs=attrs, head_spec=head_spec,
    )
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

    # A probabilistic run carries hr_speed_q; a deterministic one does not, and
    # then every block below degrades to exactly the previous behaviour.
    has_q = 'hr_speed_q' in inference_ds
    if has_q:
        q_taus = np.asarray(inference_ds['quantile'].values, dtype=float)
        print(f"    Probabilistic output detected: {len(q_taus)} quantile levels "
              f"({q_taus[0]:.3f} .. {q_taus[-1]:.3f})")

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
    if has_q:
        # (time, quantile, points) -- ~400 MB at 52k steps x 19 levels x 100 points.
        mod_q_all = (inf_sub['hr_speed_q'].isel(y=pts_y, x=pts_x)
                     .transpose('time', 'quantile', 'points').values)
        pit_pool = []

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

        # GOAL-3 metric: skill conditioned on EXTREME true winds (>10 m/s)
        ext = mask & (tru_ws > 10.0)
        n_ext = int(ext.sum())
        if n_ext >= 10:
            rmse_mod_ext = float(np.sqrt(np.nanmean((mod_ws[ext] - tru_ws[ext])**2)))
            rmse_e5_ext = float(np.sqrt(np.nanmean((e5_ws[ext] - tru_ws[ext])**2)))
            ss_ext = 1.0 - rmse_mod_ext / rmse_e5_ext if rmse_e5_ext > 0 else np.nan
        else:
            rmse_mod_ext = rmse_e5_ext = ss_ext = np.nan

        # Energy-weighted skill: weight the speed error by (true_ws)**q so
        # wave-making winds (stress ~ U^2, energy ~ U^3) dominate the score
        # smoothly, instead of the hard >10 m/s cliff above.
        _tw = tru_ws[mask]
        _me = (mod_ws[mask] - _tw) ** 2
        _ee = (e5_ws[mask] - _tw) ** 2
        ss_ew = {}
        for _q in (1, 2, 3):
            _w = _tw ** _q
            _wsum = _w.sum()
            if _wsum > 0:
                _wr_mod = np.sqrt((_w * _me).sum() / _wsum)
                _wr_e5 = np.sqrt((_w * _ee).sum() / _wsum)
                ss_ew[_q] = float(1.0 - _wr_mod / _wr_e5) if _wr_e5 > 0 else np.nan
            else:
                ss_ew[_q] = np.nan

        rmse_mod_u = float(np.sqrt(np.nanmean((mod_u[mask] - tru_u[mask])**2)))
        rmse_e5_u = float(np.sqrt(np.nanmean((e5_u[mask] - tru_u[mask])**2)))
        rmse_mod_v = float(np.sqrt(np.nanmean((mod_v[mask] - tru_v[mask])**2)))
        rmse_e5_v = float(np.sqrt(np.nanmean((e5_v[mask] - tru_v[mask])**2)))

        # Dispersion + bias of the deterministic (P50) field. std_ratio is the
        # tell for the peak under-prediction this whole design targets: v2
        # measured 0.76-0.83 against a correctly-dispersed 1.0.
        bias_mod = float(np.mean(mod_ws[mask] - tru_ws[mask]))
        std_ratio = float(np.std(mod_ws[mask]) / (np.std(tru_ws[mask]) + 1e-8))

        # --- probabilistic scores -------------------------------------------
        prob = {}
        if has_q:
            pq = mod_q_all[inf_idx_rel, :, pt].astype(float)   # (T, Q)
            qm = mask & np.isfinite(pq).all(axis=1)
            if qm.sum() >= 10:
                yq = tru_ws[qm]
                pqm = pq[qm]
                crps_mod = float(np.mean(crps_numpy(pqm, yq, q_taus)))
                # ERA5 is a single-valued forecast, and CRPS of a point forecast
                # is exactly MAE -- the fair common reference.
                crps_e5 = float(np.mean(np.abs(e5_ws[qm] - yq)))
                prob['crps_model'] = crps_mod
                prob['crps_lr'] = crps_e5
                prob['crps_skill'] = (1.0 - crps_mod / crps_e5
                                      if crps_e5 > 0 else np.nan)
                for thr in (10.0, 15.0):
                    tw_m = float(np.mean(twcrps_numpy(pqm, yq, q_taus, thr)))
                    tw_e = float(np.mean(np.abs(np.maximum(e5_ws[qm], thr)
                                                - np.maximum(yq, thr))))
                    prob[f'twcrps_model_{int(thr)}'] = tw_m
                    prob[f'twcrps_skill_{int(thr)}'] = (1.0 - tw_m / tw_e
                                                        if tw_e > 0 else np.nan)
                for lev in (0.5, 0.8, 0.9):
                    prob[f'coverage_{int(lev*100)}'] = interval_coverage(
                        pqm, yq, q_taus, lev)
                pit_pool.append(pit_values(pqm, yq))

        all_records.append({})
        all_records[-1].update(prob)

        all_records[-1].update({
            'iy': iy, 'ix': ix, 'n_valid': int(mask.sum()),
            'rmse_model_ws': rmse_mod, 'rmse_lr_ws': rmse_e5,
            'skill_score_ws': ss,
            'bias_model_ws': bias_mod, 'std_ratio_ws': std_ratio,
            'rmse_model_u': rmse_mod_u, 'rmse_lr_u': rmse_e5_u,
            'rmse_model_v': rmse_mod_v, 'rmse_lr_v': rmse_e5_v,
            'rmse_model_ws_ext': rmse_mod_ext, 'rmse_lr_ws_ext': rmse_e5_ext,
            'skill_score_ws_ext': ss_ext, 'n_ext': n_ext,
            'skill_score_ws_ew_q1': ss_ew[1],
            'skill_score_ws_ew_q2': ss_ew[2], 'skill_score_ws_ew_q3': ss_ew[3],
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
    mean_rmse_lr = float(df['rmse_lr_ws'].mean())

    summary = {
        'n_points': len(df),
        'n_common_timesteps': len(common),
        'period': f"{common[0].date()} -- {common[-1].date()}",
        'wind_speed': {
            'median_skill_score': med_ss,
            'mean_skill_score': mean_ss,
            'mean_rmse_model': mean_rmse_model,
            'mean_rmse_lr': mean_rmse_lr,
        },
        'wind_speed_extreme_10ms': {
            'median_skill_score': float(np.nanmedian(df['skill_score_ws_ext'])),
            'mean_skill_score': float(np.nanmean(df['skill_score_ws_ext'])),
            'mean_rmse_model': float(np.nanmean(df['rmse_model_ws_ext'])),
            'mean_rmse_lr': float(np.nanmean(df['rmse_lr_ws_ext'])),
            'n_points_with_extremes': int((df['n_ext'] >= 10).sum()),
            'mean_extreme_hours_per_point': float(df['n_ext'].mean()),
        },
        'wind_speed_energy_weighted': {
            'q1_median_skill_score': float(np.nanmedian(df['skill_score_ws_ew_q1'])),
            'q1_mean_skill_score': float(np.nanmean(df['skill_score_ws_ew_q1'])),
            'q2_median_skill_score': float(np.nanmedian(df['skill_score_ws_ew_q2'])),
            'q2_mean_skill_score': float(np.nanmean(df['skill_score_ws_ew_q2'])),
            'q3_median_skill_score': float(np.nanmedian(df['skill_score_ws_ew_q3'])),
            'q3_mean_skill_score': float(np.nanmean(df['skill_score_ws_ew_q3'])),
        },
        # Dispersion of the deterministic (P50) field. std_ratio is the headline
        # diagnostic for this whole design: v2 measured 0.76-0.83 against a
        # correctly-dispersed 1.0, and that deficit IS the peak under-prediction.
        'dispersion': {
            'median_std_ratio': float(np.nanmedian(df['std_ratio_ws'])),
            'mean_std_ratio': float(np.nanmean(df['std_ratio_ws'])),
            'median_bias': float(np.nanmedian(df['bias_model_ws'])),
            'mean_bias': float(np.nanmean(df['bias_model_ws'])),
        },
    }

    # ---- probabilistic blocks (present only for a quantile run) -------------
    if has_q and 'crps_model' in df.columns:
        summary['crps'] = {
            'mean_crps_model': float(np.nanmean(df['crps_model'])),
            'mean_crps_lr': float(np.nanmean(df['crps_lr'])),
            'median_crps_skill': float(np.nanmedian(df['crps_skill'])),
            'mean_crps_skill': float(np.nanmean(df['crps_skill'])),
            'note': ('ERA5 is a single-valued forecast and CRPS of a point '
                     'forecast equals its MAE, which is what makes it a fair '
                     'common reference for a probabilistic model.'),
        }
        summary['twcrps'] = {
            f'threshold_{t}ms': {
                'mean_model': float(np.nanmean(df[f'twcrps_model_{t}'])),
                'median_skill': float(np.nanmedian(df[f'twcrps_skill_{t}'])),
                'mean_skill': float(np.nanmean(df[f'twcrps_skill_{t}'])),
            }
            for t in (10, 15) if f'twcrps_model_{t}' in df.columns
        }

        # Calibration. A calibrated forecast gives a FLAT PIT histogram; the
        # under-dispersion diagnosis predicts a U shape (too many observations
        # outside the predicted range). This is what makes the whole premise
        # falsifiable rather than merely plausible.
        cal = {f'coverage_{lev}': {
                   'nominal': lev / 100.0,
                   'observed': float(np.nanmean(df[f'coverage_{lev}'])),
               }
               for lev in (50, 80, 90) if f'coverage_{lev}' in df.columns}
        if pit_pool:
            pit_all = np.concatenate(pit_pool)
            nb = 10
            hist, edges = np.histogram(pit_all, bins=nb, range=(0.0, 1.0))
            hist = hist / max(1, hist.sum())
            # Deviation from flat: 0 = perfectly calibrated.
            cal['pit_histogram'] = {
                'bin_edges': [float(e) for e in edges],
                'frequency': [float(h) for h in hist],
                'n_samples': int(pit_all.size),
                'flatness_l1': float(np.abs(hist - 1.0 / nb).sum()),
                'edge_mass': float(hist[0] + hist[-1]),
                'edge_mass_expected': float(2.0 / nb),
            }
        summary['calibration'] = cal
    with open(output_dir / 'grid_point_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n    Results ({len(df)} grid points, {len(common)} timesteps):")
    print(f"      Wind speed RMSE  model: {mean_rmse_model:.3f} m/s")
    print(f"      Wind speed RMSE  LR:    {mean_rmse_lr:.3f} m/s")
    print(f"      Skill score (median):   {med_ss:.3f}")
    print(f"      Skill score (mean):     {mean_ss:.3f}")
    _ss_ext = float(np.nanmedian(df['skill_score_ws_ext']))
    _rm_ext = float(np.nanmean(df['rmse_model_ws_ext']))
    _rl_ext = float(np.nanmean(df['rmse_lr_ws_ext']))
    print(f"      [>10 m/s] skill (median): {_ss_ext:.3f}  "
          f"RMSE model {_rm_ext:.3f} / LR {_rl_ext:.3f} m/s")
    _ew1 = float(np.nanmedian(df['skill_score_ws_ew_q1']))
    _ew2 = float(np.nanmedian(df['skill_score_ws_ew_q2']))
    _ew3 = float(np.nanmedian(df['skill_score_ws_ew_q3']))
    print(f"      [energy-wt] skill (median): U^1 {_ew1:.3f}  "
          f"U^2 {_ew2:.3f}  U^3 {_ew3:.3f}")
    print(f"      bias {float(np.nanmedian(df['bias_model_ws'])):+.3f} m/s   "
          f"std_ratio {float(np.nanmedian(df['std_ratio_ws'])):.3f} "
          f"(1.0 = correctly dispersed)")
    if has_q and 'crps_skill' in df.columns:
        print(f"      CRPS model {float(np.nanmean(df['crps_model'])):.3f} vs "
              f"LR {float(np.nanmean(df['crps_lr'])):.3f} m/s   "
              f"skill {float(np.nanmedian(df['crps_skill'])):.3f}")
        for t in (10, 15):
            if f'twcrps_skill_{t}' in df.columns:
                print(f"      twCRPS@{t} m/s skill (median): "
                      f"{float(np.nanmedian(df[f'twcrps_skill_{t}'])):.3f}")
        for lev in (50, 80, 90):
            if f'coverage_{lev}' in df.columns:
                obs = float(np.nanmean(df[f'coverage_{lev}']))
                print(f"      coverage {lev}%: observed {100*obs:.1f}% "
                      f"({'under' if obs < lev/100 else 'over'}-dispersed)")
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
    parser.add_argument('--case-study', default='case_studies/sf_bay_conus404')
    parser.add_argument('--data-root', default=None,
                        help='Base dir for raw input data; reads <data-root>/<case_name>/raw_data. '
                             'Sets COSMOS_DATA_ROOT for this run (one of the two is required).')
    parser.add_argument('--results-root', default=None,
                        help='Base dir for run outputs; writes '
                             '<results-root>/<case_name>/results/<run-name>. '
                             'Sets COSMOS_RESULTS_ROOT for this run (one of the two is required).')
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
        _icfg = os.environ.get('INFER_CONFIG')
        inf_config_path = (Path(_icfg) if _icfg
                           else run_dirs['checkpoint'] / 'inference_preprocessing.yaml')
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

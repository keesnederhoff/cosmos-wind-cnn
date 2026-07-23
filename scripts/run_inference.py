"""
Standalone inference: regrid coarse data and run a trained model.

Loads model and configs from the archived checkpoint directory (created by
run_training_pipeline.py step 3), so the exact settings that produced the
model are always used — even if configs have been edited since.

Supports any coarse product (ERA5, CMIP6, etc.) by pointing
inference_preprocessing.yaml at different source files.

Usage:
    # Downscale a single year of ERA5
    python scripts/run_inference.py \\
        --case-study case_studies/sf_bay_conus404 \\
        --run-name 3663482 \\
        --start-date 2024-01-01 \\
        --end-date   2026-12-31

    # Full ERA5 record (dates from archived config)
    python scripts/run_inference.py \\
        --case-study case_studies/sf_bay_conus404 \\
        --run-name 3663482

    # Use a different inference config (e.g. for CMIP6)
    python scripts/run_inference.py \\
        --case-study case_studies/sf_bay_conus404 \\
        --run-name 3663482 \\
        --inference-config case_studies/sf_bay_conus404/configs/inference_cmip6.yaml

Output:
    case_studies/<name>/results/<run_name>/output_inference/inference_ERA5_20240101_20261231.nc
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from cosmos_wind_cnn.data.regridder import Regridder
from cosmos_wind_cnn.inference import run_streaming_inference
from cosmos_wind_cnn.models.unet3d import Wind3DUNET, build_wind3dunet
from cosmos_wind_cnn.utils.config import load_config, parse_variable_config, get_run_dirs, get_data_dir


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")

    parser = argparse.ArgumentParser(
        description='Regrid coarse data and run trained CNN inference'
    )
    parser.add_argument('--case-study', default='case_studies/sf_bay_conus404')
    parser.add_argument('--run-name', required=True,
                        help='Run name (must have archived configs in checkpoints/<run_name>/)')
    parser.add_argument('--start-date', default=None,
                        help='Start date (ISO, e.g. 2024-01-01). Default: from config.')
    parser.add_argument('--end-date', default=None,
                        help='End date (ISO, e.g. 2026-12-31). Default: from config.')
    parser.add_argument('--inference-config', default=None,
                        help='Override inference_preprocessing.yaml path '
                             '(e.g. for CMIP6). Default: archived copy in checkpoint dir.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--output', default=None,
                        help='Output NetCDF path (default: auto-generated)')
    args = parser.parse_args()

    case_dir = Path(args.case_study)
    run_name = args.run_name
    run_dirs = get_run_dirs(case_dir, run_name)
    processed_dir = run_dirs['data_processed']
    data_dir = get_data_dir(case_dir)
    checkpoint_dir = run_dirs['checkpoint']

    # ── Load configs ─────────────────────────────────────────────────────
    # Prefer archived configs in checkpoint dir; fall back to configs/ dir
    train_config_path = checkpoint_dir / 'training.yaml'
    if not train_config_path.exists():
        train_config_path = case_dir / 'configs' / 'training.yaml'
        print(f"  Note: no archived training.yaml in {checkpoint_dir}, "
              f"using {train_config_path}")

    if args.inference_config:
        inf_config_path = Path(args.inference_config)
    else:
        inf_config_path = checkpoint_dir / 'inference_preprocessing.yaml'
        if not inf_config_path.exists():
            inf_config_path = case_dir / 'configs' / 'inference_preprocessing.yaml'
            print(f"  Note: no archived inference config in {checkpoint_dir}, "
                  f"using {inf_config_path}")

    for label, path in [('Training config', train_config_path),
                        ('Inference config', inf_config_path),
                        ('Checkpoint', checkpoint_dir / 'best_model.pth'),
                        ('Normalization stats', processed_dir / 'normalization_stats.pkl')]:
        if not path.exists():
            print(f"Error: {label} not found at {path}")
            return

    train_config = load_config(train_config_path)
    inf_config = load_config(inf_config_path)

    input_vars, output_vars, _ = parse_variable_config(train_config)
    sequence_length = train_config['sequence_length']

    # Time period: CLI > config
    start_date = args.start_date or inf_config.get('start_date')
    end_date = args.end_date or inf_config.get('end_date')

    # Pad start for sliding window
    if start_date:
        load_start = str(
            np.datetime64(start_date, 'ns')
            - np.timedelta64(sequence_length - 1, 'h')
        )[:19]
    else:
        load_start = None

    print("=" * 70)
    print(f"Inference: {case_dir.name}  |  run: {run_name}")
    print("=" * 70)
    print(f"  Training config  : {train_config_path}")
    print(f"  Inference config : {inf_config_path}")
    print(f"  Period           : {start_date or '(start)'} -> {end_date or '(end)'}")

    # ── Reference grid ───────────────────────────────────────────────────
    ref_grid_path = processed_dir / 'target_grid_reference.nc'
    if not ref_grid_path.exists():
        # Auto-generate from processed splits
        for split in ('train', 'val', 'test'):
            split_path = processed_dir / f'{split}.nc'
            if split_path.exists():
                split_ds = xr.open_dataset(split_path)
                rg = Regridder.from_target_dataset(split_ds)
                rg.save_reference_grid(ref_grid_path)
                split_ds.close()
                break
        else:
            print("Error: no reference grid and no processed splits to generate one.")
            return

    interp_method = inf_config.get('interpolation_method', 'linear')
    regridder = Regridder.from_reference_grid(ref_grid_path, method=interp_method)

    # ── Regrid ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Regridding source data onto target grid...")
    print("=" * 70)

    sources = inf_config['sources']
    physical_bounds = inf_config.get('physical_bounds', {})

    regridded_vars = {}   # time-varying inputs -> (time, y, x)
    static_vars = {}      # static, no-time inputs -> (y, x), broadcast below
    for var_name, source_cfg in sources.items():
        filepath = data_dir / source_cfg['file']
        if not filepath.exists():
            # Try absolute path
            filepath = Path(source_cfg['file'])
        if not filepath.exists():
            print(f"Error: source file not found: {filepath}")
            return

        ds = xr.open_dataset(filepath, chunks='auto')
        source_var = source_cfg.get('source_var')
        if source_var is None:
            source_var = list(ds.data_vars)[0]

        var_map = {var_name: source_var}
        bounds = {var_name: physical_bounds[var_name]} if var_name in physical_bounds else {}

        # Static input-only fields (e.g. terrain) have no time dim: interp onto
        # the target grid now, then broadcast over the common time axis below.
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
        print("Error: no overlapping timesteps across source files.")
        return

    print(f"\n  Common timesteps: {len(common_times)}")

    for var_name in regridded_vars:
        regridded_vars[var_name] = regridded_vars[var_name].sel(time=common_times)

    # Broadcast static inputs onto the common time axis as constant channels
    for var_name, da in static_vars.items():
        regridded_vars[var_name] = da.expand_dims(
            time=common_times).transpose('time', 'y', 'x')

    # Verify CNN input variables are present
    missing = [v for v in input_vars if v not in regridded_vars]
    if missing:
        print(f"Error: regridded data missing CNN input variables: {missing}")
        print(f"  Available: {list(regridded_vars.keys())}")
        return

    full_ds = xr.Dataset(regridded_vars)[input_vars]
    print("  Loading into memory...")
    full_ds.load()

    # ── Load model ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Loading model...")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    checkpoint = torch.load(checkpoint_dir / 'best_model.pth',
                            map_location=device, weights_only=False)

    # Loaded before the model: residual mode needs the stats to build its skip affine.
    with open(processed_dir / 'normalization_stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    model = build_wind3dunet(train_config, stats, input_vars, output_vars).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded from epoch {checkpoint['epoch']}")

    # ── Inference + save ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Running inference...")
    print("=" * 70)

    if args.output:
        output_path = Path(args.output)
    else:
        tag_start = (start_date or str(common_times[0])[:10]).replace('-', '')
        tag_end = (end_date or str(common_times[-1])[:10]).replace('-', '')
        output_filename = f'inference_ERA5_{tag_start}_{tag_end}.nc'
        output_path = run_dirs['output_inference'] / output_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError:
            print(f"  WARNING: cannot delete existing {output_path}. "
                  "Close any application that has it open.")
            return

    attrs = {
        'source_checkpoint': str(checkpoint_dir / 'best_model.pth'),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'run_name': run_name,
        'sequence_length': sequence_length,
        'inference_config': str(inf_config_path),
    }
    if 'crs' in train_config:
        attrs['crs'] = train_config['crs']

    n_predicted, n_total = run_streaming_inference(
        model, full_ds, input_vars, output_vars, stats, sequence_length,
        output_path, device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, attrs=attrs,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"  Output          : {output_path} ({size_mb:.1f} MB)")
    print(f"  Predicted steps : {n_predicted:,} / {n_total:,}")
    print(f"  Skipped (NaN)   : {n_total - n_predicted:,}")


if __name__ == '__main__':
    main()

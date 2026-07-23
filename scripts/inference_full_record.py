"""
Run inference over the entire ERA5 record (all train/val/test splits).

Concatenates the three processed splits in time order, then slides a
sequence_length window across every timestep, producing a single output
NetCDF with downscaled predictions for the full record.

Usage:
    python scripts/inference_full_record.py \\
        --case-study case_studies/sf_bay_conus404 \\
        --run-name 3663482

Output:
    case_studies/sf_bay_conus404/results/<run_name>/output_inference/full_record.nc
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from cosmos_wind_cnn.inference import run_streaming_inference
from cosmos_wind_cnn.models.unet3d import Wind3DUNET, build_wind3dunet
from cosmos_wind_cnn.utils.config import load_config, parse_variable_config, get_run_dirs



def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")

    parser = argparse.ArgumentParser(description='Full-record ERA5 inference')
    parser.add_argument('--case-study', default='case_studies/sf_bay_conus404',
                        help='Path to case study directory')
    parser.add_argument('--run-name', required=True,
                        help='Checkpoint run name (e.g. 3663482)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Inference batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader worker count (default: 8)')
    parser.add_argument('--output', default=None,
                        help='Output NetCDF path '
                             '(default: outputs/<run>/inference/full_record.nc)')
    parser.add_argument('--input', default=None,
                        help='Pre-regridded input NetCDF (from preprocess_inference.py). '
                             'If provided, this file is used instead of the '
                             'train/val/test splits.')
    parser.add_argument('--start-date', default=None,
                        help='First timestep to include, e.g. 1979-10-01  '
                             '(default: start of the processed record)')
    parser.add_argument('--end-date', default=None,
                        help='Last timestep to include, e.g. 2021-12-31  '
                             '(default: end of the processed record)')
    args = parser.parse_args()

    case_dir   = Path(args.case_study)
    run_dirs   = get_run_dirs(case_dir, args.run_name)
    data_dir   = run_dirs['data_processed']
    stats_path = data_dir / 'normalization_stats.pkl'
    checkpoint_path = run_dirs['checkpoint'] / 'best_model.pth'
    output_path = (Path(args.output) if args.output
                   else run_dirs['output_inference'] / 'full_record.nc')

    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at {checkpoint_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load config & model ──────────────────────────────────────────────────
    config = load_config(case_dir / 'configs' / 'training.yaml')
    input_vars, output_vars, _ = parse_variable_config(config)
    sequence_length = config['sequence_length']

    checkpoint = torch.load(checkpoint_path, map_location=device,
                             weights_only=False)

    # Build model strictly from config — no silent fallback values that could
    # silently mismatch the saved checkpoint architecture.
    try:
        base_channels  = config['base_channels']
        dropout_rate   = config['dropout_rate']
    except KeyError as e:
        print(f"Error: required key {e} missing from training config.")
        return

    # ── Load normalization stats ─────────────────────────────────────────────
    # Before the model: residual mode needs them to build its skip affine.
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    model = build_wind3dunet(config, stats, input_vars, output_vars).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Input vars:    {input_vars}")
    print(f"Output vars:   {output_vars}")
    print(f"Sequence len:  {sequence_length}")
    print(f"base_channels: {base_channels}  dropout_rate: {dropout_rate}  "
          f"residual_learning: {config.get('residual_learning', False)}")

    # ── Compute load window (pad start by sequence_length-1 for valid windows) ─
    # We need (sequence_length - 1) extra timesteps before the user's start date
    # so that the very first prediction window is complete.
    t1_bound = np.datetime64(args.end_date,   'ns') if args.end_date   else None
    if args.start_date:
        t0_target = np.datetime64(args.start_date, 'ns')
        t0_bound  = t0_target - np.timedelta64(sequence_length - 1, 'h')
    else:
        t0_bound  = None

    if args.start_date or args.end_date:
        t0_str = str(t0_bound)[:10] if t0_bound is not None else '(start)'
        t1_str = str(t1_bound)[:10] if t1_bound is not None else '(end)'
        print(f"\nTime window requested : {args.start_date or '(start)'} -> {args.end_date or '(end)'}")
        print(f"Load window (with {sequence_length-1}h pad): {t0_str} -> {t1_str}")

    # ── Load input data ──────────────────────────────────────────────────────
    if args.input:
        # Single pre-regridded file (from preprocess_inference.py)
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: input file not found at {input_path}")
            return

        print(f"\nLoading pre-regridded input: {input_path}")
        full_ds = xr.open_dataset(input_path)

        # Verify all required input variables are present
        missing_vars = [v for v in input_vars if v not in full_ds.data_vars]
        if missing_vars:
            print(f"Error: input file is missing required variables: {missing_vars}")
            print(f"  Available: {list(full_ds.data_vars)}")
            return

        full_ds = full_ds[input_vars]
        if t0_bound is not None or t1_bound is not None:
            full_ds = full_ds.sel(time=slice(t0_bound, t1_bound))
        full_ds.load()
    else:
        # Legacy path: concatenate train/val/test splits
        print("\nLoading processed splits...")
        splits = []
        for split in ('train', 'val', 'test'):
            path = data_dir / f'{split}.nc'
            if not path.exists():
                print(f"  {split}.nc : NOT FOUND -- skipping")
                continue

            # Open lazily (metadata only -- no data read yet)
            ds = xr.open_dataset(path)
            split_t0 = ds.time.values[0]
            split_t1 = ds.time.values[-1]
            n_split  = len(ds.time)

            lo = t0_bound  if t0_bound  is not None else split_t0
            hi = t1_bound  if t1_bound  is not None else split_t1

            if split_t1 < lo or split_t0 > hi:
                print(f"  {split}.nc : {n_split:,} ts  "
                      f"({str(split_t0)[:10]} -- {str(split_t1)[:10]})  -> outside window, skipping")
                ds.close()
                continue

            # Slice to load window and keep only input variables
            ds = ds[input_vars].sel(time=slice(t0_bound, t1_bound))
            print(f"  {split}.nc : loading {len(ds.time):,} / {n_split:,} timesteps  "
                  f"({str(ds.time.values[0])[:10]} -- {str(ds.time.values[-1])[:10]})")
            splits.append(ds)

        if not splits:
            print("Error: no processed split files found (or none overlap the requested window).")
            return

        full_ds = xr.concat(splits, dim='time').sortby('time')
        # Force all data into memory now, then close the source file handles.
        # This releases the NetCDF4 file locks before we write the output file,
        # which matters on Windows where open files cannot be overwritten.
        full_ds.load()
        for ds in splits:
            ds.close()

    n_total = len(full_ds.time)
    print(f"\nRecord to process: {n_total:,} timesteps  "
          f"({full_ds.time.values[0]} — {full_ds.time.values[-1]})")

    # ── Run streaming inference ──────────────────────────────────────────────
    attrs = {
        'source_checkpoint': str(checkpoint_path),
        'checkpoint_epoch':  int(checkpoint['epoch']),
        'run_name':          args.run_name,
        'sequence_length':   sequence_length,
    }
    if 'crs' in config:
        attrs['crs'] = config['crs']

    print("\nRunning streaming inference...")
    n_predicted, _ = run_streaming_inference(
        model, full_ds, input_vars, output_vars, stats, sequence_length,
        output_path, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        attrs=attrs,
    )
    print(f"\nSaved → {output_path}")
    print(f"  Predicted timesteps : {n_predicted:,} / {n_total:,}")
    print(f"  Skipped (NaN window): {n_total - n_predicted:,}")


if __name__ == '__main__':
    main()

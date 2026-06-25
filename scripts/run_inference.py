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
        --case-study case_studies/sf_bay \\
        --run-name 3663482 \\
        --start-date 2024-01-01 \\
        --end-date   2026-12-31

    # Full ERA5 record (dates from archived config)
    python scripts/run_inference.py \\
        --case-study case_studies/sf_bay \\
        --run-name 3663482

    # Use a different inference config (e.g. for CMIP6)
    python scripts/run_inference.py \\
        --case-study case_studies/sf_bay \\
        --run-name 3663482 \\
        --inference-config case_studies/sf_bay/configs/inference_cmip6.yaml

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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cosmos_wind_cnn.data.regridder import Regridder
from cosmos_wind_cnn.models.unet3d import Wind3DUNET
from cosmos_wind_cnn.utils.config import load_config, parse_variable_config, get_run_dirs, var_units_for


# ── Sliding-window dataset ───────────────────────────────────────────────

class _SlidingWindowDataset(Dataset):
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
        print(f"  {len(self.valid_indices):,} valid windows "
              f"({n_dropped:,} dropped -- NaN)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        slices = [self.arrays[v][start:start + self.sequence_length]
                  for v in self.input_vars]
        return torch.from_numpy(np.stack(slices, axis=1)), start


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")

    parser = argparse.ArgumentParser(
        description='Regrid coarse data and run trained CNN inference'
    )
    parser.add_argument('--case-study', default='case_studies/sf_bay')
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
    data_dir = case_dir / 'data' / 'raw'
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

    regridded_vars = {}
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
        print("Error: no overlapping timesteps across source files.")
        return

    print(f"\n  Common timesteps: {len(common_times)}")

    for var_name in regridded_vars:
        regridded_vars[var_name] = regridded_vars[var_name].sel(time=common_times)

    # Verify CNN input variables are present
    missing = [v for v in input_vars if v not in regridded_vars]
    if missing:
        print(f"Error: regridded data missing CNN input variables: {missing}")
        print(f"  Available: {list(regridded_vars.keys())}")
        return

    full_ds = xr.Dataset(regridded_vars)[input_vars]
    print("  Loading into memory...")
    full_ds.load()

    n_total = len(full_ds.time)
    time_coords = full_ds.time.values
    y_coords = full_ds.y.values if 'y' in full_ds.coords else None
    x_coords = full_ds.x.values if 'x' in full_ds.coords else None
    height = full_ds.sizes.get('y', full_ds.sizes.get('latitude'))
    width = full_ds.sizes.get('x', full_ds.sizes.get('longitude'))

    # ── Load model ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Loading model...")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    checkpoint = torch.load(checkpoint_dir / 'best_model.pth',
                            map_location=device, weights_only=False)
    model = Wind3DUNET(
        in_channels=len(input_vars),
        out_channels=len(output_vars),
        base_channels=train_config['base_channels'],
        dropout_rate=train_config['dropout_rate'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded from epoch {checkpoint['epoch']}")

    with open(processed_dir / 'normalization_stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    # ── Inference ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Running inference...")
    print("=" * 70)

    dataset = _SlidingWindowDataset(full_ds, input_vars, stats, sequence_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())

    target_offset = sequence_length - 1
    pred_arrays = {
        var: np.full((n_total, height, width), np.nan, dtype=np.float32)
        for var in output_vars
    }

    n_nan_outputs = 0
    with torch.no_grad():
        for batch_inputs, batch_starts in tqdm(loader, desc='Inference'):
            outputs = model(batch_inputs.to(device))
            batch_nan = (~torch.isfinite(outputs)).sum().item()
            if batch_nan > 0:
                n_nan_outputs += batch_nan
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
            outputs = outputs.cpu().numpy()
            for b, start in enumerate(batch_starts.numpy()):
                t = int(start) + target_offset
                for c, var in enumerate(output_vars):
                    mean, std = stats[var]['mean'], stats[var]['std']
                    pred_arrays[var][t] = outputs[b, c] * (std + 1e-8) + mean

    if n_nan_outputs > 0:
        print(f"  WARNING: {n_nan_outputs:,} non-finite outputs replaced with 0.")

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Saving predictions...")
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

    VAR_UNITS = var_units_for(output_vars)

    coords = {'time': time_coords}
    if y_coords is not None:
        coords['y'] = ('y', y_coords)
    if x_coords is not None:
        coords['x'] = ('x', x_coords)

    ds_out = xr.Dataset(
        {var: (['time', 'y', 'x'], pred_arrays[var]) for var in output_vars},
        coords=coords,
    )
    for coord in ('time', 'x', 'y'):
        if coord in full_ds.coords and coord in ds_out.coords:
            ds_out[coord].attrs.update(full_ds[coord].attrs)
    for var in output_vars:
        if var in VAR_UNITS:
            ds_out[var].attrs['units'] = VAR_UNITS[var]

    ds_out.attrs['source_checkpoint'] = str(checkpoint_dir / 'best_model.pth')
    ds_out.attrs['checkpoint_epoch'] = int(checkpoint['epoch'])
    ds_out.attrs['run_name'] = run_name
    ds_out.attrs['sequence_length'] = sequence_length
    ds_out.attrs['inference_config'] = str(inf_config_path)
    if 'crs' in train_config:
        ds_out.attrs['crs'] = train_config['crs']

    encoding = {var: {'zlib': True, 'complevel': 1} for var in output_vars}
    encoding['time'] = {'dtype': 'float64', 'units': 'hours since 1900-01-01',
                        'calendar': 'gregorian'}
    ds_out.to_netcdf(output_path, encoding=encoding)

    n_predicted = int(np.isfinite(
        next(iter(pred_arrays.values()))
    ).any(axis=(1, 2)).sum())
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"  Output          : {output_path} ({size_mb:.1f} MB)")
    print(f"  Predicted steps : {n_predicted:,} / {n_total:,}")
    print(f"  Skipped (NaN)   : {n_total - n_predicted:,}")


if __name__ == '__main__':
    main()

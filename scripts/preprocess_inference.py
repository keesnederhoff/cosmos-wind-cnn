"""
Preprocess coarse-resolution data for inference with a trained model.

Regrids each source file (ERA5, CMIP6, etc.) onto the target grid that
the model was trained on, using the reference grid saved during training
preprocessing.  The output is a single NetCDF file with all input
variables on the target grid, ready to be fed to the inference script.

Usage:
    python scripts/preprocess_inference.py --case-study case_studies/sf_bay_conus404

    # Custom time period or output path:
    python scripts/preprocess_inference.py \\
        --case-study case_studies/sf_bay_conus404 \\
        --start-date 2000-01-01 \\
        --end-date   2005-12-31 \\
        --output     case_studies/sf_bay_conus404/data/processed/inference_2000_2005.nc

Output:
    case_studies/<name>/results/<run_name>/output_inference/inference_regridded.nc
"""

import os
# Fix OpenMP duplicate library error on Windows (must be before numpy/torch imports)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from cosmos_wind_cnn.data.regridder import Regridder
from cosmos_wind_cnn.utils.config import load_config, get_run_dirs


def main():
    # Change to project root directory (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")

    parser = argparse.ArgumentParser(
        description='Regrid coarse data onto the target grid for inference'
    )
    parser.add_argument('--case-study', default='case_studies/sf_bay_conus404',
                        help='Path to case study directory')
    parser.add_argument('--start-date', default=None,
                        help='Override start date (ISO format, e.g. 2000-01-01)')
    parser.add_argument('--end-date', default=None,
                        help='Override end date (ISO format, e.g. 2005-12-31)')
    parser.add_argument('--run-name', default='default',
                        help='Run name — looks for reference grid in results/<run_name>/data_processed/')
    parser.add_argument('--output', default=None,
                        help='Output NetCDF path (default: results/<run_name>/output_inference/inference_regridded.nc)')
    args = parser.parse_args()

    case_dir = Path(args.case_study)
    run_dirs = get_run_dirs(case_dir, args.run_name)
    data_dir = case_dir / 'data' / 'raw'
    processed_dir = run_dirs['data_processed']
    config_path = case_dir / 'configs' / 'inference_preprocessing.yaml'
    ref_grid_path = processed_dir / 'target_grid_reference.nc'
    output_path = Path(args.output) if args.output else run_dirs['output_inference'] / 'inference_regridded.nc'

    # ── Load config ──────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"Inference Preprocessing: {case_dir.name}")
    print("=" * 70)

    if not config_path.exists():
        print(f"\nError: config not found at {config_path}")
        print("Create it from the template:")
        print(f"  cp case_studies/_template/configs/inference_preprocessing.yaml "
              f"{config_path}")
        return

    config = load_config(config_path)
    sources = config['sources']
    physical_bounds = config.get('physical_bounds', {})
    interp_method = config.get('interpolation_method', 'linear')
    compression_level = config.get('compression_level', 1)

    # Time period: CLI args override config
    start_date = args.start_date or config.get('start_date')
    end_date = args.end_date or config.get('end_date')

    print(f"\nData directory : {data_dir}")
    print(f"Reference grid : {ref_grid_path}")
    print(f"Output         : {output_path}")
    print(f"Interpolation  : {interp_method}")
    print(f"Time period    : {start_date or '(start of file)'} -> {end_date or '(end of file)'}")
    print(f"Sources        : {len(sources)} variables")

    # ── Load reference grid ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Loading target grid reference...")
    print("=" * 70)

    if not ref_grid_path.exists():
        # Reference grid was not saved during preprocessing (older run).
        # Generate it from an existing processed split file.
        print(f"  {ref_grid_path} not found — generating from processed data...")
        for split in ('train', 'val', 'test'):
            split_path = processed_dir / f'{split}.nc'
            if split_path.exists():
                split_ds = xr.open_dataset(split_path)
                regridder = Regridder.from_target_dataset(split_ds, method=interp_method)
                regridder.save_reference_grid(ref_grid_path)
                split_ds.close()
                break
        else:
            print("Error: no processed split files (train/val/test.nc) found "
                  "to extract the target grid from.")
            print("Run scripts/preprocess.py first.")
            return

    regridder = Regridder.from_reference_grid(ref_grid_path, method=interp_method)
    print(f"  Target grid: {len(regridder.target_y)} x {len(regridder.target_x)}")

    # ── Check source files ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Checking source files...")
    print("=" * 70)

    missing = []
    for var_name, source_cfg in sources.items():
        filepath = _resolve_source_path(source_cfg['file'], data_dir)
        exists = filepath.exists()
        status = "OK" if exists else "NOT FOUND"
        print(f"  [{status}] {var_name}: {filepath.name}")
        if not exists:
            missing.append(var_name)

    if missing:
        print(f"\nError: {len(missing)} source file(s) not found: {missing}")
        print("Check file paths in inference_preprocessing.yaml")
        return

    # ── Regrid each source variable ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Regridding source variables onto target grid...")
    print("=" * 70)

    regridded_vars = {}

    for var_name, source_cfg in sources.items():
        filepath = _resolve_source_path(source_cfg['file'], data_dir)
        source_var = source_cfg.get('source_var')  # None = auto-detect

        print(f"\n  --- {var_name} ---")
        print(f"  File: {filepath.name}")

        # Open with dask for lazy loading
        ds = xr.open_dataset(filepath, chunks='auto')

        # Build the var_map for this single variable
        # If source_var is None, use the first data variable in the file
        if source_var is None:
            source_var = list(ds.data_vars)[0]
            print(f"  Auto-detected source variable: '{source_var}'")

        var_map = {var_name: source_var}
        bounds = {var_name: physical_bounds[var_name]} if var_name in physical_bounds else {}

        regridded = regridder.regrid(
            ds,
            var_map=var_map,
            physical_bounds=bounds,
            start_date=start_date,
            end_date=end_date,
        )

        regridded_vars[var_name] = regridded[var_name]
        ds.close()

    # ── Find common time steps across all regridded variables ────────────────
    print("\n" + "=" * 70)
    print("Aligning time axes...")
    print("=" * 70)

    time_sets = []
    for var_name, da in regridded_vars.items():
        times = set(da.time.values)
        print(f"  {var_name}: {len(times)} timesteps")
        time_sets.append(times)

    common_times = sorted(time_sets[0].intersection(*time_sets[1:]))
    print(f"\n  Common timesteps: {len(common_times)}")
    if len(common_times) > 0:
        print(f"  Time range: {common_times[0]} -> {common_times[-1]}")

    if len(common_times) == 0:
        print("\nError: no overlapping timesteps found across source files.")
        return

    # Select common times for all variables
    for var_name in regridded_vars:
        regridded_vars[var_name] = regridded_vars[var_name].sel(time=common_times)

    # ── Build combined dataset ───────────────────────────────────────────────
    combined = xr.Dataset(regridded_vars)

    print(f"\n  Combined dataset:")
    print(f"    Variables : {list(combined.data_vars)}")
    print(f"    Grid      : {combined.sizes['y']} x {combined.sizes['x']}")
    print(f"    Timesteps : {combined.sizes['time']}")

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Saving regridded dataset...")
    print("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Encoding
    comp_str = f"compression level {compression_level}" if compression_level > 0 else "no compression"
    print(f"  Saving to {output_path} ({comp_str})...")

    encoding = {}
    for var in combined.data_vars:
        if compression_level > 0:
            encoding[var] = {'zlib': True, 'complevel': compression_level}
        else:
            encoding[var] = {'zlib': False}

    with ProgressBar():
        combined.to_netcdf(output_path, encoding=encoding)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done! File size: {size_mb:.1f} MB")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Inference Preprocessing Complete!")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print(f"  Variables : {list(combined.data_vars)}")
    print(f"  Grid      : {combined.sizes['y']} x {combined.sizes['x']}")
    print(f"  Timesteps : {combined.sizes['time']}")
    print(f"  Period    : {str(common_times[0])[:10]} -> {str(common_times[-1])[:10]}")
    print(f"\nNext step:")
    print(f"  python scripts/inference_full_record.py \\")
    print(f"    --case-study {case_dir} \\")
    print(f"    --run-name <YOUR_RUN_NAME> \\")
    print(f"    --input {output_path}")


def _resolve_source_path(file_path: str, data_dir: Path) -> Path:
    """
    Resolve a source file path.  If *file_path* is absolute and exists,
    use it directly; otherwise treat it as relative to *data_dir*.
    """
    p = Path(file_path)
    if p.is_absolute():
        return p
    return data_dir / p


if __name__ == '__main__':
    main()

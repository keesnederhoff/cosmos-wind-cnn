"""
Run inference with a trained model.

Usage:
    python scripts/inference.py --case-study case_studies/sf_bay
    python scripts/inference.py --case-study case_studies/sf_bay --input path/to/data.nc
"""

import argparse
import os
from pathlib import Path
import pickle

import torch
import xarray as xr
import numpy as np

from cosmos_wind_cnn.models.unet3d import Wind3DUNET
from cosmos_wind_cnn.utils.config import parse_variable_config


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    input_vars, output_vars, _ = parse_variable_config(config)

    model = Wind3DUNET(
        in_channels=len(input_vars),
        out_channels=len(output_vars),
        base_channels=config.get('base_channels', 32),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Input variables: {input_vars}")
    print(f"  Output variables: {output_vars}")

    return model, config, input_vars, output_vars


def prepare_input(netcdf_path, input_vars, sequence_length, stats_path):
    """Load and normalize input data."""
    print(f"\nLoading input data from {netcdf_path}")
    ds = xr.open_dataset(netcdf_path)

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    lats = ds.latitude.values
    lons = ds.longitude.values

    x_coords = ds.x.values if 'x' in ds.coords else None
    y_coords = ds.y.values if 'y' in ds.coords else None

    input_data = []
    for var in input_vars:
        if var not in ds.data_vars:
            print(f"  Warning: Variable '{var}' not found!")
            continue
        var_data = ds[var].isel(time=slice(-sequence_length, None)).values
        mean = stats[var]['mean']
        std = stats[var]['std']
        var_data = (var_data - mean) / (std + 1e-8)
        input_data.append(var_data)
        print(f"  {var}: shape {var_data.shape}")

    input_array = np.stack(input_data, axis=1)
    input_tensor = torch.FloatTensor(input_array).unsqueeze(0)

    return input_tensor, lats, lons, x_coords, y_coords, stats


def denormalize_output(output, output_vars, stats):
    """Convert normalized predictions back to real values."""
    predictions = {}
    for i, var in enumerate(output_vars):
        mean = stats[var]['mean']
        std = stats[var]['std']
        pred_data = output[0, i].cpu().numpy()
        pred_data = pred_data * (std + 1e-8) + mean
        predictions[var] = pred_data
    return predictions


def save_predictions(predictions, lats, lons, x_coords, y_coords, output_path, crs=None):
    """Save predictions as NetCDF."""
    output_ds = xr.Dataset({
        var: (['y', 'x'], data) for var, data in predictions.items()
    })

    coords = {
        'latitude': (['y', 'x'], np.meshgrid(lons, lats)[1]),
        'longitude': (['y', 'x'], np.meshgrid(lons, lats)[0]),
    }
    if x_coords is not None and y_coords is not None:
        coords['x'] = ('x', x_coords)
        coords['y'] = ('y', y_coords)

    output_ds = output_ds.assign_coords(coords)

    if x_coords is not None and y_coords is not None:
        output_ds['x'].attrs = {'units': 'meters', 'standard_name': 'projection_x_coordinate'}
        output_ds['y'].attrs = {'units': 'meters', 'standard_name': 'projection_y_coordinate'}
        if crs:
            output_ds.attrs['crs'] = crs

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_ds.to_netcdf(output_path)
    print(f"\nPredictions saved to {output_path}")


def main():
    # Change to project root directory (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")
    
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--case-study', default='case_studies/sf_bay',
                        help='Path to case study directory')
    parser.add_argument('--input', default=None,
                        help='Input NetCDF file (default: case_study/data/processed/val.nc)')
    parser.add_argument('--output', default=None,
                        help='Output NetCDF file (default: case_study/outputs/inference/prediction.nc)')
    args = parser.parse_args()

    case_dir = Path(args.case_study)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = case_dir / 'checkpoints' / 'best_model.pth'
    model, config, input_vars, output_vars = load_model(checkpoint_path, device)

    # Paths
    input_path = Path(args.input) if args.input else case_dir / 'data' / 'processed' / 'val.nc'
    output_path = (Path(args.output) if args.output
                   else case_dir / 'outputs' / 'inference' / 'prediction.nc')
    stats_path = case_dir / 'data' / 'processed' / 'normalization_stats.pkl'

    # Prepare input
    input_tensor, lats, lons, x_coords, y_coords, stats = prepare_input(
        input_path, input_vars, config['sequence_length'], stats_path
    )

    # Inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(input_tensor.to(device))
    print(f"  Output shape: {output.shape}")

    # Denormalize
    predictions = denormalize_output(output, output_vars, stats)

    # Summary
    print("\n" + "=" * 70)
    print("Prediction Summary")
    print("=" * 70)
    for var, data in predictions.items():
        print(f"\n{var}:")
        print(f"  Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")

    # Save
    crs = config.get('crs')
    save_predictions(predictions, lats, lons, x_coords, y_coords, output_path, crs=crs)

    print("\n" + "=" * 70)
    print("Inference Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

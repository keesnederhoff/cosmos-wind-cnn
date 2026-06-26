"""
Evaluate trained model on test set.

Usage:
    python scripts/evaluate.py --case-study case_studies/sf_bay_conus404
"""

#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import xarray as xr

from cosmos_wind_cnn.data.dataset import WindDataset3D
from cosmos_wind_cnn.models.unet3d import Wind3DUNET
from cosmos_wind_cnn.training.losses import CombinedLoss
from cosmos_wind_cnn.training.metrics import calculate_all_metrics
from cosmos_wind_cnn.utils.config import load_config, parse_variable_config, get_run_dirs
from cosmos_wind_cnn.utils.visualization import plot_sample_predictions, plot_error_distribution


def evaluate_model(model, dataloader, criterion, device, dataset, wind_pair_indices=None):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    all_inputs = []
    all_losses = []

    n_nan_outputs = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Safety net: detect NaN/Inf in model outputs (should not occur after dataset filtering)
            batch_nan_outputs = (~torch.isfinite(outputs)).sum().item()
            if batch_nan_outputs > 0:
                n_nan_outputs += batch_nan_outputs
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)

            loss, _ = criterion(outputs, targets)
            all_losses.append(loss.item() if torch.isfinite(loss) else float('nan'))
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs[:, -1, :, :, :].cpu())

    if n_nan_outputs > 0:
        print(f"WARNING: {n_nan_outputs:,} non-finite values in model outputs — "
              f"replaced with 0 for metrics. Model may need more training.")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_inputs = torch.cat(all_inputs)

    avg_loss = np.nanmean(all_losses)

    # Calculate wind-specific metrics on the wind channels only
    if wind_pair_indices:
        u_idx, v_idx = wind_pair_indices[0]
        wind_preds = all_preds[:, [u_idx, v_idx], :, :]
        wind_targets = all_targets[:, [u_idx, v_idx], :, :]
        metrics = calculate_all_metrics(wind_preds, wind_targets)
    else:
        metrics = {
            'rmse': torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item(),
            'mae': torch.mean(torch.abs(all_preds - all_targets)).item(),
        }

    # Denormalize
    preds_denorm = _denormalize(all_preds, dataset.output_vars, dataset)
    targets_denorm = _denormalize(all_targets, dataset.output_vars, dataset)
    inputs_denorm = _denormalize(all_inputs, dataset.input_vars, dataset)

    # Coordinates
    coords = _get_coordinates(dataset)

    return avg_loss, metrics, preds_denorm, targets_denorm, inputs_denorm, coords


def _denormalize(data, var_list, dataset):
    """Denormalize data back to original scale."""
    denorm = torch.zeros_like(data)
    for i, var in enumerate(var_list):
        values = dataset.denormalize(data[:, i].numpy(), var)
        # Replace any NaN/Inf that appeared during denormalization
        values = np.nan_to_num(values, nan=np.nan, posinf=np.nan, neginf=np.nan)
        denorm[:, i] = torch.from_numpy(values)
    return denorm


def _get_coordinates(dataset):
    """Extract x/y UTM coordinates from dataset."""
    if hasattr(dataset, 'netcdf_path'):
        ds = xr.open_dataset(dataset.netcdf_path)
        coords = {}
        if 'x' in ds.coords and 'y' in ds.coords:
            coords['x'] = ds.x.values
            coords['y'] = ds.y.values
        else:
            coords['x'] = None
            coords['y'] = None
        ds.close()
    else:
        coords = {'x': None, 'y': None}
    return coords


def main():
    # Change to project root directory (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--case-study', default='case_studies/sf_bay_conus404',
                        help='Path to case study directory')
    parser.add_argument('--run-name', default='default',
                        help='Run name matching the one used during training')
    args = parser.parse_args()

    case_dir = Path(args.case_study)
    run_name = args.run_name
    run_dirs = get_run_dirs(case_dir, run_name)
    config = load_config(case_dir / 'configs' / 'training.yaml')

    input_vars, output_vars, wind_pair_indices = parse_variable_config(config)

    print("=" * 70)
    print(f"Model Evaluation: {case_dir.name}")
    print("=" * 70)
    print(f"Run name: {run_name}")
    print(f"\nInput variables: {input_vars}")
    print(f"Output variables: {output_vars}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    data_dir = run_dirs['data_processed']

    test_dataset = WindDataset3D(
        netcdf_path=str(data_dir / 'test.nc'),
        stats_path=str(data_dir / 'normalization_stats.pkl'),
        input_vars=input_vars,
        output_vars=output_vars,
        sequence_length=config['sequence_length'],
        forecast_horizon=config['forecast_horizon'],
        stride=1,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=torch.cuda.is_available(),
    )

    # Load model
    checkpoint_path = run_dirs['checkpoint'] / 'best_model.pth'
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Wind3DUNET(
        in_channels=len(input_vars),
        out_channels=len(output_vars),
        base_channels=config.get('base_channels', 32),
        dropout_rate=config.get('dropout_rate', 0.0),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    criterion = CombinedLoss(
        wind_pair_indices=wind_pair_indices,
        alpha=config.get('loss_alpha', 1.0),
        beta=config.get('loss_beta', 0.5),
        gamma=config.get('loss_gamma', 0.3),
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, metrics, preds, targets, inputs, coords = evaluate_model(
        model, test_loader, criterion, device, test_dataset,
        wind_pair_indices=wind_pair_indices,
    )

    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save — namespaced by run_name so multiple runs don't overwrite each other
    output_dir = run_dirs['output_evaluation']
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'test_loss': test_loss,
        'metrics': metrics,
        'config': config,
        'checkpoint_epoch': checkpoint['epoch'],
    }
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nGenerating visualizations...")
    plot_sample_predictions(preds, targets, inputs, output_vars, coords,
                            output_dir / 'samples', n_samples=5,
                            hr_label=config.get('hr_source', 'HR'),
                            lr_label=config.get('lr_source', 'LR'))
    plot_error_distribution(preds, targets, output_vars, output_dir)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

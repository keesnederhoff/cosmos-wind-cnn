"""
Training script for Wind 3D U-Net model.

Usage:
    python scripts/train.py --case-study case_studies/sf_bay
"""

import argparse
import os
from pathlib import Path
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from cosmos_wind_cnn.data.dataset import WindDataset3D, WindDatasetInMemory
from cosmos_wind_cnn.models.unet3d import Wind3DUNET
from cosmos_wind_cnn.training.losses import CombinedLoss
from cosmos_wind_cnn.training.trainer import train_one_epoch, validate
from cosmos_wind_cnn.utils.config import load_config, parse_variable_config


def main():
    # Change to project root directory (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")
    
    parser = argparse.ArgumentParser(description='Train wind downscaling model')
    parser.add_argument('--case-study', default='case_studies/sf_bay',
                        help='Path to case study directory (e.g., case_studies/sf_bay)')
    args = parser.parse_args()

    case_dir = Path(args.case_study)
    config = load_config(case_dir / 'configs' / 'training.yaml')

    print("=" * 70)
    print(f"Wind Prediction 3D U-Net Training: {case_dir.name}")
    print("=" * 70)

    # Parse variable configuration
    input_vars, output_vars, wind_pair_indices = parse_variable_config(config)

    print("\nVariable Configuration:")
    print(f"  Input variables ({len(input_vars)}):")
    for var in input_vars:
        print(f"    - {var}")
    print(f"  Output variables ({len(output_vars)}):")
    for var in output_vars:
        print(f"    - {var}")
    if wind_pair_indices:
        print(f"  Wind pairs for special loss: {len(wind_pair_indices)}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

    # Datasets
    print("\n" + "=" * 70)
    print("Loading Datasets")
    print("=" * 70)

    data_dir = case_dir / 'data' / 'processed'
    stats_path = str(data_dir / 'normalization_stats.pkl')

    DatasetClass = WindDatasetInMemory if config.get('load_in_memory', False) else WindDataset3D

    train_dataset = DatasetClass(
        netcdf_path=str(data_dir / 'train.nc'),
        stats_path=stats_path,
        input_vars=input_vars,
        output_vars=output_vars,
        sequence_length=config['sequence_length'],
        forecast_horizon=config['forecast_horizon'],
        stride=config.get('train_stride', 1),
    )

    val_dataset = DatasetClass(
        netcdf_path=str(data_dir / 'val.nc'),
        stats_path=stats_path,
        input_vars=input_vars,
        output_vars=output_vars,
        sequence_length=config['sequence_length'],
        forecast_horizon=config['forecast_horizon'],
        stride=config.get('val_stride', 1),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=torch.cuda.is_available(),
        persistent_workers=config['num_workers'] > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=torch.cuda.is_available(),
        persistent_workers=config['num_workers'] > 0,
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    print("\n" + "=" * 70)
    print("Initializing Model")
    print("=" * 70)

    in_channels = len(input_vars)
    out_channels = len(output_vars)

    model = Wind3DUNET(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=config.get('base_channels', 32),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: Wind3DUNET")
    print(f"Parameters: {n_params:,}")
    print(f"Input channels: {in_channels}")
    print(f"Output channels: {out_channels}")

    # Loss and optimizer
    criterion = CombinedLoss(
        wind_pair_indices=wind_pair_indices,
        alpha=config.get('loss_alpha', 1.0),
        beta=config.get('loss_beta', 0.5),
        gamma=config.get('loss_gamma', 0.3),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=config.get('scheduler_patience', 5),
        factor=config.get('scheduler_factor', 0.5),
    )

    # Logging and checkpoints
    log_dir = case_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_dir = case_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_val_loss = float('inf')
    patience_counter = 0
    epoch_times = []
    training_start_time = time.time()
    train_losses = []
    val_losses = []

    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 70)

        train_loss, train_components = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        val_loss, val_components, val_metrics = validate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        for key in train_components:
            writer.add_scalar(f'Train/{key}', train_components[key], epoch)
            writer.add_scalar(f'Val/{key}', val_components[key], epoch)
        for key in val_metrics:
            writer.add_scalar(f'Metrics/{key}', val_metrics[key], epoch)

        # Print summary
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        if 'rmse' in val_metrics:
            print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        if 'mae' in val_metrics:
            print(f"  Val MAE: {val_metrics['mae']:.4f}")
        if 'speed_rmse' in val_metrics:
            print(f"  Val Speed RMSE: {val_metrics['speed_rmse']:.4f}")
        if 'direction_mae' in val_metrics:
            print(f"  Val Direction MAE: {val_metrics['direction_mae']:.2f} deg")
        print(f"  Learning Rate: {current_lr:.6f}")

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        total_elapsed = time.time() - training_start_time
        print(f"\nTime:")
        print(f"  Epoch: {timedelta(seconds=int(epoch_time))}")
        print(f"  Elapsed: {timedelta(seconds=int(total_elapsed))}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config,
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        if (epoch + 1) % config.get('save_every', 10) == 0:
            periodic_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config,
            }
            torch.save(periodic_checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth')
            print(f"  Saved checkpoint")

        if patience_counter >= config.get('early_stopping_patience', 15):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    writer.close()

    # Loss plot
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")
    print(f"\nNext steps:")
    print(f"  - View logs: tensorboard --logdir {log_dir}")
    print(f"  - Evaluate: python scripts/evaluate.py --case-study {case_dir}")
    print(f"  - Inference: python scripts/inference.py --case-study {case_dir}")


if __name__ == '__main__':
    main()

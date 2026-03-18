"""
Training script for Wind 3D U-Net model.

Usage:
    # Single GPU
    python scripts/train.py --case-study case_studies/sf_bay

    # Multi-GPU (single node, 4 GPUs)
    torchrun --nproc_per_node=4 scripts/train.py --case-study case_studies/sf_bay

    # Multi-node via SLURM (see gpu_tallgrass.slurm)
    sbatch scripts/gpu_tallgrass.slurm
"""

import os
# Fix OpenMP duplicate library error on Windows (must be before numpy/torch imports)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from cosmos_wind_cnn.data.dataset import WindDataset3D, WindDatasetInMemory
from cosmos_wind_cnn.models.unet3d import Wind3DUNET
from cosmos_wind_cnn.training.losses import CombinedLoss
from cosmos_wind_cnn.training.trainer import train_one_epoch, validate
from cosmos_wind_cnn.utils.config import load_config, parse_variable_config


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize NCCL process group when running under torchrun/srun."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1  # rank, local_rank, world_size — single-process fallback

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: float, device) -> float:
    """Average a scalar across all ranks."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- distributed setup (must happen before anything else) ---
    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)

    # Change to project root directory (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)

    if is_main:
        print(f"Working directory: {project_root}\n")

    parser = argparse.ArgumentParser(description='Train wind downscaling model')
    parser.add_argument('--case-study', default='case_studies/sf_bay',
                        help='Path to case study directory (e.g., case_studies/sf_bay)')
    parser.add_argument('--run-name', default='default',
                        help='Sub-directory name for logs/checkpoints — use to avoid conflicts between runs')
    args = parser.parse_args()

    case_dir = Path(args.case_study)
    run_name = args.run_name
    config = load_config(case_dir / 'configs' / 'training.yaml')

    if is_main:
        print("=" * 70)
        print(f"Wind Prediction 3D U-Net Training: {case_dir.name}")
        print("=" * 70)
        print(f"Run name: {run_name}")
        if world_size > 1:
            print(f"Distributed: {world_size} GPU(s) across {world_size // 4} node(s)")

    # Parse variable configuration
    input_vars, output_vars, wind_pair_indices = parse_variable_config(config)

    if is_main:
        print("\nVariable Configuration:")
        print(f"  Input variables ({len(input_vars)}):")
        for var in input_vars:
            print(f"    - {var}")
        print(f"  Output variables ({len(output_vars)}):")
        for var in output_vars:
            print(f"    - {var}")
        if wind_pair_indices:
            print(f"  Wind pairs for special loss: {len(wind_pair_indices)}")

    # Device — each rank owns exactly one GPU
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main:
        print(f'\nDevice: {device}')
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(local_rank)}')
            print(f'Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.2f} GB')

    # Datasets — only rank 0 prints; all ranks load independently (no shared file lock issue with NetCDF)
    if is_main:
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
        verbose=is_main,
    )

    val_dataset = DatasetClass(
        netcdf_path=str(data_dir / 'val.nc'),
        stats_path=stats_path,
        input_vars=input_vars,
        output_vars=output_vars,
        sequence_length=config['sequence_length'],
        forecast_horizon=config['forecast_horizon'],
        stride=config.get('val_stride', 1),
        verbose=is_main,
    )

    # DistributedSampler partitions data across ranks; disable DataLoader shuffle (sampler handles it)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
    ) if world_size > 1 else None
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=(train_sampler is None),   # sampler handles shuffling in DDP mode
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config['num_workers'] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config['num_workers'] > 0,
    )

    if is_main:
        print(f"\nTotal train samples : {len(train_dataset)}")
        print(f"Samples / rank      : {len(train_dataset) // world_size}")
        print(f"Train batches / rank: {len(train_loader)}")
        print(f"Val batches / rank  : {len(val_loader)}")

    # Model
    if is_main:
        print("\n" + "=" * 70)
        print("Initializing Model")
        print("=" * 70)

    in_channels = len(input_vars)
    out_channels = len(output_vars)

    model = Wind3DUNET(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=config.get('base_channels', 32),
        dropout_rate=config.get('dropout_rate', 0.0),
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        raw = model.module if isinstance(model, DDP) else model
        n_params = sum(p.numel() for p in raw.parameters() if p.requires_grad)
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

    # Logging and checkpoints — rank 0 only, namespaced by run_name
    writer = None
    checkpoint_dir = None
    log_dir = case_dir / 'logs' / run_name
    checkpoint_dir = case_dir / 'checkpoints' / run_name
    if is_main:
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    if is_main:
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

        # Required each epoch so each rank gets a different random permutation
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main:
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print("-" * 70)

        train_loss, train_components = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            writer, disable_tqdm=not is_main,
        )
        val_loss, val_components, val_metrics = validate(
            model, val_loader, criterion, device, disable_tqdm=not is_main,
        )

        # Average losses across all ranks so every rank gets the true global loss
        train_loss = all_reduce_mean(train_loss, device)
        val_loss = all_reduce_mean(val_loss, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # --- rank 0: log, print, checkpoint ---
        if is_main:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            for key in train_components:
                writer.add_scalar(f'Train/{key}', train_components[key], epoch)
                writer.add_scalar(f'Val/{key}', val_components[key], epoch)
            for key in val_metrics:
                writer.add_scalar(f'Metrics/{key}', val_metrics[key], epoch)

            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            if 'rmse' in val_metrics:
                print(f"  Val RMSE:   {val_metrics['rmse']:.4f}")
            if 'mae' in val_metrics:
                print(f"  Val MAE:    {val_metrics['mae']:.4f}")
            if 'speed_rmse' in val_metrics:
                print(f"  Val Speed RMSE:    {val_metrics['speed_rmse']:.4f}")
            if 'direction_mae' in val_metrics:
                print(f"  Val Direction MAE: {val_metrics['direction_mae']:.2f} deg")
            print(f"  Learning Rate: {current_lr:.6f}")

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            total_elapsed = time.time() - training_start_time
            print(f"\nTime:")
            print(f"  Epoch:   {timedelta(seconds=int(epoch_time))}")
            print(f"  Elapsed: {timedelta(seconds=int(total_elapsed))}")

        # --- all ranks: patience tracking (all have same val_loss after all_reduce) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if is_main:
                raw_model = model.module if isinstance(model, DDP) else model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
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

        if is_main and (epoch + 1) % config.get('save_every', 10) == 0:
            raw_model = model.module if isinstance(model, DDP) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth')
            print(f"  Saved checkpoint")

        if patience_counter >= config.get('early_stopping_patience', 15):
            if is_main:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # --- rank 0: finalise ---
    if is_main:
        writer.close()

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

    cleanup_distributed()


if __name__ == '__main__':
    main()

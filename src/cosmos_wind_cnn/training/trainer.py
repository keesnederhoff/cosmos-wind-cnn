"""
Training loop utilities
"""

import torch
from tqdm import tqdm

from cosmos_wind_cnn.training.metrics import calculate_all_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, disable_tqdm=False):
    """Train for one epoch.

    Returns:
        avg_loss: Average training loss for the epoch.
        avg_components: Dictionary of averaged loss component values.
    """
    model.train()
    running_loss = 0.0
    running_components = {}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=disable_tqdm)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Check for NaN in inputs
        if torch.isnan(inputs).any():
            print(f"\nWARNING: NaN detected in inputs at batch {batch_idx}")
            print(f"Input shape: {inputs.shape}")
            print(f"NaN count: {torch.isnan(inputs).sum().item()}")
            continue

        if torch.isnan(targets).any():
            print(f"\nWARNING: NaN detected in targets at batch {batch_idx}")
            continue

        # Forward pass
        outputs = model(inputs)

        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            print(f"\nERROR: NaN in model outputs at batch {batch_idx}")
            print(f"Output shape: {outputs.shape}")
            print(f"Output stats: min={outputs[~torch.isnan(outputs)].min():.4f}, "
                  f"max={outputs[~torch.isnan(outputs)].max():.4f}")
            print(f"Input stats: min={inputs.min():.4f}, max={inputs.max():.4f}, "
                  f"mean={inputs.mean():.4f}, std={inputs.std():.4f}")
            raise ValueError("NaN in model outputs - check model architecture or data normalization")

        loss, loss_components = criterion(outputs, targets)

        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"\nERROR: NaN in loss at batch {batch_idx}")
            print(f"Loss components: {loss_components}")
            print(f"Output stats: min={outputs.min():.4f}, max={outputs.max():.4f}, "
                  f"mean={outputs.mean():.4f}")
            print(f"Target stats: min={targets.min():.4f}, max={targets.max():.4f}, "
                  f"mean={targets.mean():.4f}")
            raise ValueError("NaN in loss calculation")

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check for NaN in gradients
        if torch.isnan(grad_norm):
            print(f"\nERROR: NaN in gradients at batch {batch_idx}")
            raise ValueError("NaN in gradients")

        optimizer.step()

        # Track losses
        running_loss += loss.item()
        for key, val in loss_components.items():
            if key not in running_components:
                running_components[key] = 0
            running_components[key] += val

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log to tensorboard every 100 batches (writer is None on non-main ranks)
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/batch_loss', loss.item(), global_step)

    # Calculate epoch averages
    n_batches = len(dataloader)
    avg_loss = running_loss / n_batches
    avg_components = {k: v / n_batches for k, v in running_components.items()}

    return avg_loss, avg_components


def validate(model, dataloader, criterion, device, disable_tqdm=False):
    """Validate model.

    Returns:
        avg_loss: Average validation loss.
        avg_components: Dictionary of averaged loss component values.
        metrics: Dictionary of evaluation metrics.
    """
    model.eval()
    val_loss = 0.0
    val_components = {}

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validating', disable=disable_tqdm):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss, loss_components = criterion(outputs, targets)

            val_loss += loss.item()
            for key, val in loss_components.items():
                if key not in val_components:
                    val_components[key] = 0
                val_components[key] += val

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Calculate metrics
    n_batches = len(dataloader)
    avg_loss = val_loss / n_batches
    avg_components = {k: v / n_batches for k, v in val_components.items()}

    # Calculate all metrics on full predictions
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Check if we have wind pairs for wind-specific metrics
    has_wind = hasattr(criterion, 'wind_pair_indices') and len(criterion.wind_pair_indices) > 0

    if has_wind:
        u_idx, v_idx = criterion.wind_pair_indices[0]
        wind_preds = all_preds[:, [u_idx, v_idx], :, :]
        wind_targets = all_targets[:, [u_idx, v_idx], :, :]
        metrics = calculate_all_metrics(wind_preds, wind_targets)
    else:
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
        mae = torch.mean(torch.abs(all_preds - all_targets)).item()
        metrics = {'rmse': rmse, 'mae': mae}

    return avg_loss, avg_components, metrics

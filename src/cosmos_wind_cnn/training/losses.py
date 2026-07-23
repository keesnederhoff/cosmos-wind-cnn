"""
Custom loss functions for wind prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindLoss(nn.Module):
    """
    Multi-component loss for wind prediction
    Combines:
    - MSE on wind components (u, v)          weight alpha
    - MAE on wind speed (magnitude)          weight beta
    - Cosine similarity on wind direction    weight gamma
    - Extra speed MAE over EXTREME winds     weight delta   (default off)

    The extreme term ("wind_magnitude_extremes") adds an additional speed
    penalty computed ONLY over pixels where the *true* physical wind speed
    exceeds ``extreme_threshold`` m/s. This lets training prioritise storm-peak
    accuracy. Because u/v enter the loss in NORMALIZED (z-scored) space, the
    physical threshold requires the per-channel denorm stats (mean, std), passed
    as ``denorm=(u_mean, u_std, v_mean, v_std)`` at call time; the penalty itself
    is the normalized-space speed error over that mask, keeping delta on the same
    scale as beta. With delta=0 (default) the block is skipped and the loss is
    bit-for-bit identical to the original three-term loss.
    """

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3, delta=0.0,
                 extreme_threshold=10.0):
        """
        Args:
            alpha: Weight for component MSE loss
            beta:  Weight for speed MAE loss
            gamma: Weight for direction cosine loss
            delta: Weight for the extreme-wind speed MAE term (0.0 = disabled)
            extreme_threshold: physical wind speed (m/s) above which a pixel is
                "extreme" for the delta term
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = float(delta)
        self.extreme_threshold = float(extreme_threshold)

    def forward(self, pred, target, denorm=None):
        """
        Args:
            pred:   (batch, 2, H, W) - predicted [u, v] (normalized)
            target: (batch, 2, H, W) - target [u, v] (normalized)
            denorm: optional (u_mean, u_std, v_mean, v_std) for the extreme term

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        # 1. MSE on wind components
        component_loss = F.mse_loss(pred, target)

        # 2. Wind speed MAE (normalized space)
        pred_speed = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + 1e-8)
        target_speed = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + 1e-8)
        speed_loss = F.l1_loss(pred_speed, target_speed)

        # 3. Direction loss (cosine similarity)
        pred_norm = pred / (pred_speed.unsqueeze(1) + 1e-8)
        target_norm = target / (target_speed.unsqueeze(1) + 1e-8)
        cos_sim = (pred_norm[:, 0] * target_norm[:, 0] +
                   pred_norm[:, 1] * target_norm[:, 1])
        direction_loss = 1 - cos_sim.mean()

        # Combined (three-term) loss
        total_loss = (self.alpha * component_loss +
                      self.beta * speed_loss +
                      self.gamma * direction_loss)

        loss_dict = {
            'component_loss': component_loss.item(),
            'speed_loss': speed_loss.item(),
            'direction_loss': direction_loss.item()
        }

        # 4. wind_magnitude_extremes (default OFF: delta=0 -> skipped entirely,
        #    so the loss above is unchanged). Extra speed MAE over pixels where
        #    the TRUE physical wind exceeds extreme_threshold m/s.
        if self.delta > 0.0 and denorm is not None:
            u_mean, u_std, v_mean, v_std = denorm
            tu = target[:, 0] * u_std + u_mean
            tv = target[:, 1] * v_std + v_mean
            target_speed_phys = torch.sqrt(tu**2 + tv**2 + 1e-8)
            ext_mask = target_speed_phys > self.extreme_threshold  # (batch,H,W)
            if ext_mask.any():
                extreme_loss = F.l1_loss(pred_speed[ext_mask],
                                         target_speed[ext_mask])
            else:
                # No extreme pixels this batch: contribute exactly zero but keep
                # a graph-connected, correctly-typed tensor.
                extreme_loss = pred_speed.sum() * 0.0
            total_loss = total_loss + self.delta * extreme_loss
            loss_dict['extreme_loss'] = float(extreme_loss.item())
            loss_dict['extreme_frac'] = float(ext_mask.float().mean().item())

        return total_loss, loss_dict


class CombinedLoss(nn.Module):
    """
    Combined loss for multiple variable types:
    - Wind pairs (u, v) use WindLoss (u/v MSE + speed MAE + direction cosine
      + optional extreme-wind speed MAE)
    - Other variables (temperature, etc.) use MSE

    `nonwind_weight` selects the OPTIMIZATION GOAL:

      1.0 (default) -- "all variables": the historical behaviour, an unweighted
          mean of the wind term and the non-wind MSE, i.e. (wind + nonwind)/2.
          Note this gives the single wind pair 50% of the loss and the 4
          non-wind channels the other 50%.

      0.0 -- "wind only": the non-wind MSE is still REPORTED but contributes no
          gradient, so training, early stopping and best_model selection all
          track wind alone. Use when wind is the product. CAVEAT: with 0.0 the
          non-wind output channels receive no gradient and their predictions are
          untrained -- keep a small weight (e.g. 0.05) if you still want them
          usable while wind dominates.

    `delta` / `extreme_threshold` / `wind_denorm` add the optional
    "wind_magnitude_extremes" term (GOAL 3): an extra speed penalty focused on
    winds above `extreme_threshold` m/s. `delta=0` (default) is a no-op and the
    loss is bit-for-bit unchanged. `wind_denorm` is a list aligned with
    `wind_pair_indices`, each entry `(u_mean, u_std, v_mean, v_std)` for the pair
    (needed because the physical threshold is applied to z-scored u/v).

    Why nonwind_weight matters: val_loss was measured to be ANTI-correlated with
    wind skill across model sizes precisely because the aggregate lets a model
    win on temperature/pressure while losing on wind. Setting nonwind_weight=0
    makes val_loss a valid selection metric for a wind target.
    """

    def __init__(self, wind_pair_indices=None, alpha=1.0, beta=0.5, gamma=0.3,
                 nonwind_weight=1.0, delta=0.0, extreme_threshold=10.0,
                 wind_denorm=None):
        """
        Args:
            wind_pair_indices: List of tuples [(u_idx, v_idx), ...] indicating which
                             output channels are wind pairs
            alpha, beta, gamma: Weights for WindLoss components
            nonwind_weight: Relative weight of the non-wind MSE term (see class
                doc). 1.0 reproduces the original loss exactly; 0.0 optimizes
                wind only.
            delta: Weight for the extreme-wind term (0.0 = disabled, default).
            extreme_threshold: physical wind speed (m/s) defining "extreme".
            wind_denorm: list aligned with wind_pair_indices of
                (u_mean, u_std, v_mean, v_std) tuples; required for the delta term.
        """
        super().__init__()
        self.wind_pair_indices = wind_pair_indices or []
        self.wind_loss = WindLoss(alpha, beta, gamma, delta=delta,
                                  extreme_threshold=extreme_threshold)
        self.mse_loss = nn.MSELoss()
        self.nonwind_weight = float(nonwind_weight)
        # Per-pair denorm stats for the extreme mask; None disables the term even
        # if delta>0 (the WindLoss block is gated on `denorm is not None`).
        self.wind_denorm = wind_denorm

        # Track which output channels are part of wind pairs
        self.wind_channels = set()
        for u_idx, v_idx in self.wind_pair_indices:
            self.wind_channels.add(u_idx)
            self.wind_channels.add(v_idx)

        # Non-wind channels (computed on first forward pass)
        self._non_wind_channels = None

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, n_channels, H, W) - predicted outputs
            target: (batch, n_channels, H, W) - target outputs

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components
        """
        n_channels = pred.shape[1]

        # Compute non-wind channels once
        if self._non_wind_channels is None:
            self._non_wind_channels = [i for i in range(n_channels) if i not in self.wind_channels]

        total_loss = 0.0
        loss_dict = {}
        # Weighted denominator: with nonwind_weight=1.0 this is exactly the old
        # component count, so the default is bit-for-bit the previous loss.
        denom = 0.0

        # Calculate wind losses for each pair
        for pair_idx, (u_idx, v_idx) in enumerate(self.wind_pair_indices):
            if u_idx < n_channels and v_idx < n_channels:
                wind_pred = pred[:, [u_idx, v_idx], :, :]
                wind_target = target[:, [u_idx, v_idx], :, :]

                denorm = None
                if self.wind_denorm is not None and pair_idx < len(self.wind_denorm):
                    denorm = self.wind_denorm[pair_idx]

                wind_loss, wind_components = self.wind_loss(
                    wind_pred, wind_target, denorm=denorm)
                total_loss = total_loss + wind_loss
                denom += 1.0

                # Log with pair index
                for key, val in wind_components.items():
                    loss_dict[f'wind_pair_{pair_idx}_{key}'] = val

        # Calculate MSE for non-wind variables
        if self._non_wind_channels:
            if self.nonwind_weight > 0.0:
                non_wind_pred = pred[:, self._non_wind_channels, :, :]
                non_wind_target = target[:, self._non_wind_channels, :, :]

                non_wind_loss = self.mse_loss(non_wind_pred, non_wind_target)
                total_loss = total_loss + self.nonwind_weight * non_wind_loss
                denom += self.nonwind_weight

                loss_dict['non_wind_mse'] = non_wind_loss.item()
            else:
                # Wind-only goal: still report the non-wind error so it stays
                # observable, but keep it out of the graph (no gradient, no cost).
                with torch.no_grad():
                    loss_dict['non_wind_mse'] = self.mse_loss(
                        pred[:, self._non_wind_channels, :, :],
                        target[:, self._non_wind_channels, :, :],
                    ).item()

        # Weighted average over components
        if denom > 0:
            total_loss = total_loss / denom

        # Add total loss to dict
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class MSELoss(nn.Module):
    """
    Simple MSE loss wrapper for compatibility
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        loss_dict = {'mse_loss': loss.item()}
        return loss, loss_dict

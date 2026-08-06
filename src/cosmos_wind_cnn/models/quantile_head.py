"""Probabilistic output head: monotone wind-speed quantiles + unit direction.

Why this exists
---------------
Every deterministic flavour of this model under-predicts peak winds
(std_ratio 0.76-0.83, negative top-decile skill). That is not a tuning failure:
under MSE/MAE the optimal prediction is the CONDITIONAL MEAN of the target given
a 31 km ERA5 input. ERA5 cannot resolve what creates local peaks, so that
conditional distribution is wide and its mean is systematically less extreme
than the truth. The shrinkage is structural.

The fix is to stop predicting a point. This head emits a full predictive
DISTRIBUTION of wind speed per pixel, as a set of quantiles, so the upper
quantiles are honest estimates of peaks rather than a shrunk mean.

Layout of the returned tensor (channel dim), all in PHYSICAL units:

    [ 0                     : n_speed            )  speed quantiles, m/s
    [ n_speed               : n_speed + 2        )  direction unit vector (x, y)
    [ n_speed + 2           : n_speed + 2 + n_gust)  gust quantiles, m/s

Construction guarantees, by design rather than by penalty:

* **Non-negative and non-crossing speeds.** The lowest quantile is
  ``softplus(anchor + b0)`` and each subsequent one adds ``softplus(.)``, so
  ``s_1 <= s_2 <= ... <= s_Q`` and ``s_1 >= 0`` always hold. No sorting hack, no
  crossing penalty, and the ordering survives any optimiser state.
* **Anchored on ERA5.** Speed is built around the interpolated ERA5 10 m speed
  and direction around the ERA5 unit vector, so the network predicts a
  CORRECTION rather than the field itself. This is the same residual-learning
  idea that was present in both prior winning flavours, and it makes the head
  well-conditioned at initialisation (it starts out reproducing ERA5).

Physical units, not z-space: wind speed is non-negative with a natural scale,
and the pinball / threshold-weighted objectives are only interpretable if the
quantiles carry metres per second.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initial spread between adjacent quantiles, m/s. softplus(b) = INIT_INCREMENT
# => the full quantile range starts about (Q-1) * INIT_INCREMENT wide, which for
# Q = 19 is ~2.7 m/s: a sane prior for hourly wind speed. Left at the softplus
# default of 0.693 the head would start ~12 m/s wide and waste early epochs
# collapsing it.
INIT_INCREMENT = 0.15


def make_tau_grid(n: int) -> np.ndarray:
    """Midpoint quantile levels tau_k = (k - 0.5) / n, k = 1..n.

    The midpoint rule is what makes ``2 * mean_k pinball_k`` an unbiased
    estimator of CRPS, since CRPS = 2 * integral_0^1 pinball_tau d(tau).
    """
    if n < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n}")
    return (np.arange(1, n + 1, dtype="float64") - 0.5) / n


class QuantileWindHead(nn.Module):
    """Maps decoder features to anchored, monotone wind-speed quantiles.

    Args:
        in_channels: feature channels arriving from the decoder (base_channels).
        n_speed: number of speed quantile levels.
        n_gust: number of gust quantile levels (0 disables the gust output).
        anchor_idx: (u_idx, v_idx) indices of lr_u / lr_v within the INPUT
            channel axis, used to rebuild the ERA5 anchor.
        anchor_denorm: (u_mean, u_std, v_mean, v_std) for those two channels, so
            the z-scored inputs can be returned to m/s.
        gust_anchor: optional (idx, mean, std) for an ERA5 gust input channel.
            When absent the gust quantiles anchor on the 10 m speed instead.
    """

    def __init__(self, in_channels, n_speed, anchor_idx, anchor_denorm,
                 n_gust=0, gust_anchor=None):
        super().__init__()
        self.n_speed = int(n_speed)
        self.n_gust = int(n_gust)
        self.n_out = self.n_speed + 2 + self.n_gust

        self.conv = nn.Conv2d(in_channels, self.n_out, kernel_size=1)

        # Start narrow instead of ~12 m/s wide (see INIT_INCREMENT).
        inc_bias = math.log(math.expm1(INIT_INCREMENT))
        with torch.no_grad():
            self.conv.bias.zero_()
            self.conv.bias[1:self.n_speed] = inc_bias
            if self.n_gust:
                g0 = self.n_speed + 2
                self.conv.bias[g0 + 1:g0 + self.n_gust] = inc_bias

        # persistent=False keeps the state_dict identical in shape to a plain
        # head, so checkpoints stay portable and load strict=True.
        self.register_buffer("anchor_idx",
                             torch.tensor(list(anchor_idx), dtype=torch.long),
                             persistent=False)
        self.register_buffer("anchor_denorm",
                             torch.tensor(list(anchor_denorm), dtype=torch.float32),
                             persistent=False)

        if gust_anchor is not None:
            gi, gm, gs = gust_anchor
            self.has_gust_anchor = True
            self.register_buffer("gust_anchor",
                                 torch.tensor([float(gi), float(gm), float(gs)],
                                              dtype=torch.float32),
                                 persistent=False)
        else:
            self.has_gust_anchor = False
            self.register_buffer("gust_anchor",
                                 torch.zeros(3, dtype=torch.float32),
                                 persistent=False)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _monotone(base_logit, inc_logits, anchor):
        """anchor-shifted, strictly non-decreasing, non-negative quantiles."""
        # softplus keeps s_1 >= 0 while staying near-identity for anchor >~ 3 m/s,
        # so the anchor is preserved rather than distorted in the useful range.
        s1 = F.softplus(anchor + base_logit)                      # (B,1,H,W)
        if inc_logits.shape[1] == 0:
            return s1
        inc = F.softplus(inc_logits)                              # (B,Q-1,H,W) > 0
        return torch.cat([s1, s1 + torch.cumsum(inc, dim=1)], dim=1)

    def _anchor_fields(self, x):
        """Rebuild physical ERA5 (u, v) at the target timestep from z-scored x.

        x is (batch, channels, seq_len, H, W); with forecast_horizon = 0 the LAST
        input timestep is the target timestep.
        """
        um, us, vm, vs = self.anchor_denorm
        lu = x[:, self.anchor_idx[0], -1] * us + um               # (B,H,W)
        lv = x[:, self.anchor_idx[1], -1] * vs + vm
        return lu, lv

    # ---------------------------------------------------------------- forward
    def forward(self, feat, x):
        """
        Args:
            feat: (batch, in_channels, H, W) decoder features at the last step.
            x:    (batch, channels, seq_len, H, W) z-scored model input.
        Returns:
            (batch, n_speed + 2 + n_gust, H, W), physical units.
        """
        raw = self.conv(feat)

        lu, lv = self._anchor_fields(x)
        anchor_s = torch.sqrt(lu ** 2 + lv ** 2 + 1e-8)           # (B,H,W)

        # --- speed quantiles -------------------------------------------------
        s_q = self._monotone(
            raw[:, 0:1],
            raw[:, 1:self.n_speed],
            anchor_s.unsqueeze(1),
        )

        # --- direction: a correction to the ERA5 unit vector ------------------
        d_raw = raw[:, self.n_speed:self.n_speed + 2]
        anchor_dir = torch.stack([lu, lv], dim=1) / (anchor_s.unsqueeze(1) + 1e-8)
        d = d_raw + anchor_dir
        d = d / (torch.linalg.vector_norm(d, dim=1, keepdim=True) + 1e-8)

        parts = [s_q, d]

        # --- gust quantiles ---------------------------------------------------
        if self.n_gust:
            g0 = self.n_speed + 2
            if self.has_gust_anchor:
                gi, gm, gs = self.gust_anchor
                g_anchor = x[:, int(gi.item()), -1] * gs + gm
                g_anchor = g_anchor.clamp_min(0.0)
            else:
                g_anchor = anchor_s
            g_q = self._monotone(
                raw[:, g0:g0 + 1],
                raw[:, g0 + 1:g0 + self.n_gust],
                g_anchor.unsqueeze(1),
            )
            parts.append(g_q)

        return torch.cat(parts, dim=1)


def split_head_output(out, n_speed, n_gust=0):
    """Slice a head tensor into (speed_quantiles, direction, gust_quantiles)."""
    s_q = out[:, :n_speed]
    d = out[:, n_speed:n_speed + 2]
    g_q = out[:, n_speed + 2:n_speed + 2 + n_gust] if n_gust else None
    return s_q, d, g_q

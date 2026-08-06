"""Pinball / CRPS objectives for the probabilistic wind head.

The central idea
----------------
Two earlier attempts to make this model better at storm peaks -- the hard
``delta`` threshold and the smooth ``epsilon`` U^q term -- both re-weighted the
loss **over samples**. Sample weighting is an IMPROPER scoring rule: the
minimiser is no longer the true conditional distribution, so it buys tail
accuracy by biasing the whole forecast upward. The measured cost was exactly
that trade-off, monotone and unforgiving:

    lever                all-hours skill    >10 m/s skill
    none                 0.209              0.023
    smooth U^2, w=1.0    0.186              0.106
    hard delta=1         0.096              0.211
    hard delta=4        -0.001              0.256

This module weights the **quantile axis** instead. The quantile-weighted CRPS

    QW-CRPS(F, y) = integral_0^1 w(tau) * QS_tau(F^-1(tau), y) d(tau)

with ``QS_tau`` the pinball (quantile) score, is a PROPER scoring rule for any
non-negative weight ``w`` (Gneiting & Ranjan, 2011). Choosing ``w`` increasing
in tau concentrates accuracy on the upper tail **without biasing the median**,
which is the mechanism expected to break the trade-off above rather than merely
slide along it.

Two identities are used throughout:

* ``CRPS = 2 * integral_0^1 pinball_tau d(tau)``, so on the midpoint grid
  ``CRPS ~= 2 * mean_k pinball_k``. Plain CRPS is therefore just this loss with
  ``qw_exp = 0``, and no separate code path is needed to report it.
* Threshold-weighted CRPS via chaining: ``twCRPS_t(F, y) = CRPS(v#F, v(y))``
  with ``v(x) = max(x, t)``. Also proper, and with a quantile representation it
  is simply the same estimator applied to ``max(s_k, t)`` and ``max(y, t)``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-8


# ---------------------------------------------------------------- core scores
def pinball_terms(pred_q, y, taus):
    """Per-quantile pinball loss, averaged over batch and space.

    Args:
        pred_q: (B, Q, H, W) predicted quantiles, physical units, non-decreasing.
        y:      (B, H, W) observed value, same units.
        taus:   (Q,) quantile levels.
    Returns:
        (Q,) tensor of mean pinball loss per level.
    """
    diff = y.unsqueeze(1) - pred_q                       # (B,Q,H,W)
    t = taus.view(1, -1, 1, 1)
    loss = torch.maximum(t * diff, (t - 1.0) * diff)     # standard pinball
    return loss.mean(dim=(0, 2, 3))                      # (Q,)


def crps_from_quantiles(pred_q, y, taus):
    """CRPS ~= 2 * mean_k pinball_k on a midpoint quantile grid."""
    return 2.0 * pinball_terms(pred_q, y, taus).mean()


def twcrps_from_quantiles(pred_q, y, taus, threshold):
    """Threshold-weighted CRPS by chaining with v(x) = max(x, threshold).

    Emphasises outcomes above `threshold` while remaining a proper score.
    """
    return crps_from_quantiles(
        torch.clamp(pred_q, min=threshold),
        torch.clamp(y, min=threshold),
        taus,
    )


def quantile_weights(taus, exp=0.0):
    """w(tau) ∝ tau**exp, normalised to mean 1.

    exp = 0 recovers plain CRPS. Larger exp shifts effort to the upper tail.
    Normalising keeps the loss magnitude -- and therefore the usable learning
    rate -- comparable across settings of `exp`.
    """
    w = taus ** float(exp)
    return w / w.mean()


# ------------------------------------------------------------------- the loss
class QuantileWindLoss(nn.Module):
    """Quantile-weighted CRPS on speed + speed-weighted direction + gust.

        L = sum_k w_k * pinball_k(speed)
          + gamma      * speed-weighted directional error
          + gust_weight* sum_k w_k * pinball_k(gust)

    The direction term is weighted by the TRUE speed on purpose: the direction
    of a 0.5 m/s wind is noise, and for the wave/fetch application only the
    direction of energetic winds matters. Direction is already this model's
    strongest result (57.7 deg vs RTMA's 59.2), so it must not regress.

    Targets arrive z-scored, matching the rest of the pipeline, and are returned
    to physical units here using the same per-channel stats the dataset used.
    """

    def __init__(self, n_speed, qw_exp=0.0, gamma=0.3,
                 n_gust=0, gust_weight=0.0,
                 u_stats=(0.0, 1.0), v_stats=(0.0, 1.0), gust_stats=None,
                 u_idx=0, v_idx=1, gust_idx=None,
                 report_threshold=10.0):
        super().__init__()
        from cosmos_wind_cnn.models.quantile_head import make_tau_grid

        # Flag the trainer branches on: a distributional head emits
        # n_speed + 2 + n_gust channels, which do not align with the target
        # channels, so the deterministic metric path must be skipped.
        self.is_quantile = True
        self.n_speed = int(n_speed)
        self.n_gust = int(n_gust)
        self.gamma = float(gamma)
        self.gust_weight = float(gust_weight)
        self.qw_exp = float(qw_exp)
        self.u_idx, self.v_idx, self.gust_idx = int(u_idx), int(v_idx), gust_idx
        self.report_threshold = float(report_threshold)

        taus = torch.tensor(make_tau_grid(self.n_speed), dtype=torch.float32)
        self.register_buffer("taus", taus, persistent=False)
        self.register_buffer("w", quantile_weights(taus, qw_exp), persistent=False)

        if self.n_gust:
            gt = torch.tensor(make_tau_grid(self.n_gust), dtype=torch.float32)
            self.register_buffer("gust_taus", gt, persistent=False)
            self.register_buffer("gust_w", quantile_weights(gt, qw_exp),
                                 persistent=False)

        self.register_buffer("u_stats", torch.tensor(u_stats, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("v_stats", torch.tensor(v_stats, dtype=torch.float32),
                             persistent=False)
        gs = gust_stats if gust_stats is not None else (0.0, 1.0)
        self.register_buffer("gust_stats", torch.tensor(gs, dtype=torch.float32),
                             persistent=False)

    def physical_target(self, target):
        """z-scored (B, C, H, W) -> physical u, v, speed."""
        tu = target[:, self.u_idx] * self.u_stats[1] + self.u_stats[0]
        tv = target[:, self.v_idx] * self.v_stats[1] + self.v_stats[0]
        return tu, tv, torch.sqrt(tu ** 2 + tv ** 2 + EPS)

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, n_speed + 2 + n_gust, H, W) head output, physical units.
            target: (B, C, H, W) z-scored targets in output_vars order.
        Returns:
            (total_loss, loss_dict)
        """
        from cosmos_wind_cnn.models.quantile_head import split_head_output

        s_q, d_pred, g_q = split_head_output(pred, self.n_speed, self.n_gust)
        tu, tv, y_s = self.physical_target(target)

        # --- speed: quantile-weighted pinball --------------------------------
        pin = pinball_terms(s_q, y_s, self.taus)              # (Q,)
        speed_loss = (self.w * pin).mean()

        total = speed_loss
        loss_dict = {"speed_pinball": float(speed_loss.item())}

        # --- direction: speed-weighted 1 - cos -------------------------------
        if self.gamma > 0.0:
            true_dir = torch.stack([tu, tv], dim=1) / (y_s.unsqueeze(1) + EPS)
            cos = (d_pred * true_dir).sum(dim=1)               # (B,H,W)
            wsum = y_s.sum() + EPS
            dir_loss = (y_s * (1.0 - cos)).sum() / wsum
            total = total + self.gamma * dir_loss
            loss_dict["direction_loss"] = float(dir_loss.item())

        # --- gust: auxiliary quantile task -----------------------------------
        if self.n_gust and self.gust_weight > 0.0 and self.gust_idx is not None:
            y_g = (target[:, self.gust_idx] * self.gust_stats[1]
                   + self.gust_stats[0])
            gpin = pinball_terms(g_q, y_g, self.gust_taus)
            gust_loss = (self.gust_w * gpin).mean()
            total = total + self.gust_weight * gust_loss
            loss_dict["gust_pinball"] = float(gust_loss.item())

        # --- diagnostics (no gradient): the numbers we actually select on ----
        with torch.no_grad():
            loss_dict["crps"] = float(crps_from_quantiles(s_q, y_s, self.taus).item())
            loss_dict["twcrps"] = float(
                twcrps_from_quantiles(s_q, y_s, self.taus,
                                      self.report_threshold).item())
            # P50 = the deterministic field this product ships as best estimate.
            med = s_q[:, self.n_speed // 2]
            loss_dict["p50_rmse"] = float(
                torch.sqrt(torch.mean((med - y_s) ** 2)).item())
            loss_dict["p50_bias"] = float(torch.mean(med - y_s).item())
            # Dispersion: the tell for the shrinkage this whole design targets.
            loss_dict["std_ratio"] = float(
                (med.std() / (y_s.std() + EPS)).item())

        return total, loss_dict


# ------------------------------------------------------------------ numpy side
def crps_numpy(pred_q, y, taus):
    """CRPS for evaluation code that works in numpy. Shapes: (N, Q), (N,), (Q,)."""
    diff = y[:, None] - pred_q
    t = np.asarray(taus)[None, :]
    return 2.0 * np.maximum(t * diff, (t - 1.0) * diff).mean(axis=1)


def twcrps_numpy(pred_q, y, taus, threshold):
    """Threshold-weighted CRPS, chained with v(x) = max(x, threshold)."""
    return crps_numpy(np.maximum(pred_q, threshold),
                      np.maximum(y, threshold), taus)


def pit_values(pred_q, y, taus):
    """Probability-integral-transform value per sample.

    The fraction of predicted quantiles at or below the observation. A
    calibrated forecast gives a FLAT histogram; the documented under-dispersion
    predicts a U shape (observations too often outside the predicted range).
    """
    below = (pred_q <= y[:, None]).sum(axis=1)
    return np.asarray(taus)[np.clip(below - 1, 0, len(taus) - 1)] * (below > 0)


def interval_coverage(pred_q, y, taus, level):
    """Empirical coverage of the central `level` interval (e.g. 0.9)."""
    lo_t, hi_t = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    lo = np.interp(lo_t, taus, np.arange(len(taus)))
    hi = np.interp(hi_t, taus, np.arange(len(taus)))
    lo_v = pred_q[:, int(np.floor(lo))]
    hi_v = pred_q[:, int(np.ceil(hi))]
    return float(np.mean((y >= lo_v) & (y <= hi_v)))

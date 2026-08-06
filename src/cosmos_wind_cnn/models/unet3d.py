"""
3D U-Net architecture for wind prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Wind3DUNET(nn.Module):
    """
    3D U-Net for wind prediction
    Input: (batch, seq_len, channels, H, W)
    Output: (batch, out_channels, H, W)
    """

    def __init__(self, in_channels, out_channels, base_channels=32, dropout_rate=0.0,
                 quantile_head=None,
                 residual_learning=False, residual_idx=None,
                 residual_scale=None, residual_shift=None):
        """
        Args:
            in_channels: Number of input channels (variables)
            out_channels: Number of output channels (predicted variables)
            base_channels: Base number of feature channels (will be multiplied)
            dropout_rate: Dropout probability in each conv block (0 = disabled)
            residual_learning: If True, predict the fine-scale correction and add
                the (already interpolated) low-res input back on. Default False
                leaves the forward pass byte-for-byte identical to before.
            residual_idx: Input-channel index of the low-res counterpart of each
                output channel (see utils.config.residual_channel_map).
            residual_scale, residual_shift: Per-output-channel affine taking the
                normalized low-res input into normalized high-res target space
                (see utils.config.residual_affine). Required when
                residual_learning is True -- without them the skip adds a field
                normalized by the WRONG statistics.

        Prefer the build_wind3dunet() factory below, which wires all of this from
        a training config + normalization stats.
        """
        super(Wind3DUNET, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_learning = bool(residual_learning)

        if self.residual_learning:
            if residual_idx is None or residual_scale is None or residual_shift is None:
                raise ValueError(
                    "residual_learning=True requires residual_idx, residual_scale and "
                    "residual_shift (build via utils.config.residual_channel_map / "
                    "residual_affine)."
                )
            if not (len(residual_idx) == len(residual_scale)
                    == len(residual_shift) == out_channels):
                raise ValueError(
                    f"residual_* lengths must all equal out_channels={out_channels}; got "
                    f"idx={len(residual_idx)}, scale={len(residual_scale)}, "
                    f"shift={len(residual_shift)}"
                )
            # persistent=False keeps state_dict identical to a non-residual model,
            # so old checkpoints still load strictly and the checkpoint schema is
            # unchanged. The flag itself lives in the training config.
            self.register_buffer('residual_idx',
                                 torch.as_tensor(list(residual_idx), dtype=torch.long),
                                 persistent=False)
            self.register_buffer('residual_scale',
                                 torch.as_tensor(residual_scale, dtype=torch.float32)
                                 .view(1, -1, 1, 1), persistent=False)
            self.register_buffer('residual_shift',
                                 torch.as_tensor(residual_shift, dtype=torch.float32)
                                 .view(1, -1, 1, 1), persistent=False)

        # Encoder
        self.enc1 = self.conv_block_3d(in_channels, base_channels, dropout_rate)
        self.enc2 = self.conv_block_3d(base_channels, base_channels*2, dropout_rate)
        self.enc3 = self.conv_block_3d(base_channels*2, base_channels*4, dropout_rate)
        self.enc4 = self.conv_block_3d(base_channels*4, base_channels*8, dropout_rate)

        # Pool only spatially, not temporally
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Bottleneck
        self.bottleneck = self.conv_block_3d(base_channels*8, base_channels*16, dropout_rate)

        # Decoder
        self.up4 = nn.ConvTranspose3d(
            base_channels*16, base_channels*8,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec4 = self.conv_block_3d(base_channels*16, base_channels*8, dropout_rate)

        self.up3 = nn.ConvTranspose3d(
            base_channels*8, base_channels*4,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec3 = self.conv_block_3d(base_channels*8, base_channels*4, dropout_rate)

        self.up2 = nn.ConvTranspose3d(
            base_channels*4, base_channels*2,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec2 = self.conv_block_3d(base_channels*4, base_channels*2, dropout_rate)

        self.up1 = nn.ConvTranspose3d(
            base_channels*2, base_channels,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec1 = self.conv_block_3d(base_channels*2, base_channels, dropout_rate)

        # Output layer - predict for last timestep.
        #
        # Two mutually exclusive modes:
        #   quantile_head is None -> the original deterministic 1x1 conv, one
        #       channel per output variable, in z-scored space.
        #   quantile_head given   -> a QuantileWindHead emitting monotone wind
        #       speed quantiles + a unit direction vector (+ optional gust
        #       quantiles) in PHYSICAL units. It performs its own ERA5 anchoring,
        #       which is why it is incompatible with residual_learning below.
        self.quantile_head = quantile_head
        if quantile_head is None:
            self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)
            self.n_head_out = out_channels
        else:
            if residual_learning:
                raise ValueError(
                    "quantile_head and residual_learning are mutually exclusive: "
                    "the quantile head already anchors both speed and direction "
                    "on the interpolated ERA5 field, so adding the residual skip "
                    "would apply the anchor twice."
                )
            self.out = None
            self.n_head_out = quantile_head.n_out

    def conv_block_3d(self, in_ch, out_ch, dropout_rate=0.0):
        """3D convolutional block with BatchNorm, ReLU and optional Dropout"""
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout3d(p=dropout_rate))
        return nn.Sequential(*layers)

    def match_size(self, x, target):
        """Match spatial dimensions of x to target by padding or cropping"""
        # x and target are 5D: (batch, channels, seq_len, H, W)
        diff_h = target.shape[3] - x.shape[3]
        diff_w = target.shape[4] - x.shape[4]

        if diff_h > 0 or diff_w > 0:
            # Pad if upsampled tensor is smaller than target
            pad_h = max(diff_h, 0)
            pad_w = max(diff_w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        # Crop if upsampled tensor is larger than target
        if x.shape[3] > target.shape[3] or x.shape[4] > target.shape[4]:
            x = x[:, :, :, :target.shape[3], :target.shape[4]]

        return x

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels, H, W)
        Returns:
            output: (batch, out_channels, H, W)
        """
        # Reshape to (batch, channels, seq_len, H, W) for 3D convolutions
        batch, seq_len, channels, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (batch, channels, seq_len, H, W)

        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections and size matching
        d4 = self.up4(b)
        d4 = self.match_size(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.match_size(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.match_size(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.match_size(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Take last timestep and apply 2D conv for output
        # d1 shape: (batch, channels, seq_len, H, W)
        d1_last = d1[:, :, -1, :, :]  # (batch, channels, H, W)

        if self.quantile_head is not None:
            # x was permuted to (batch, channels, seq_len, H, W) above, which is
            # the layout the head expects for rebuilding the ERA5 anchor.
            return self.quantile_head(d1_last, x)

        output = self.out(d1_last)

        if self.residual_learning:
            # x is (batch, channels, seq_len, H, W) after the permute above. The
            # lr_* inputs were already interpolated onto the target grid during
            # preprocessing, and with forecast_horizon=0 the LAST input timestep
            # is the target timestep -- so no upsampling is needed here.
            lr_last = x[:, self.residual_idx, -1, :, :]  # (batch, out_channels, H, W)
            output = output + self.residual_scale * lr_last + self.residual_shift

        return output


def build_wind3dunet(train_config, stats, input_vars, output_vars):
    """Construct a Wind3DUNET from a training config (+ normalization stats).

    Single place that decides whether residual mode is on, so every call site
    (train / pipeline / inference / evaluate) stays consistent. Uses .get() for
    the new keys so configs archived before residual mode existed still load and
    reproduce their original behaviour.
    """
    # Imported here to avoid a circular import at module load.
    from cosmos_wind_cnn.utils.config import residual_affine, residual_channel_map

    residual = bool(train_config.get('residual_learning', False))
    kwargs = {}
    if residual:
        # Residual mode assumes the last input timestep IS the target timestep.
        horizon = train_config.get('forecast_horizon', 0)
        if horizon != 0:
            raise ValueError(
                f"residual_learning requires forecast_horizon=0 (got {horizon}): the "
                f"skip connection uses the last input timestep as the target timestep."
            )
        if stats is None:
            raise ValueError("residual_learning=True requires normalization stats.")
        scale, shift = residual_affine(stats, train_config, output_vars)
        kwargs = dict(
            residual_learning=True,
            residual_idx=residual_channel_map(train_config, input_vars, output_vars),
            residual_scale=scale,
            residual_shift=shift,
        )

    # --- probabilistic head -------------------------------------------------
    if str(train_config.get('head', 'det')).lower() in ('quantile', 'q'):
        kwargs['quantile_head'] = _build_quantile_head(
            train_config, stats, input_vars, output_vars
        )

    return Wind3DUNET(
        in_channels=len(input_vars),
        out_channels=len(output_vars),
        base_channels=train_config.get('base_channels', 32),
        dropout_rate=train_config.get('dropout_rate', 0.0),
        **kwargs,
    )


def _stat_pair(stats, var):
    """(mean, std) for `var`, with a clear error if preprocessing never saw it."""
    if stats is None or var not in stats:
        raise ValueError(
            f"normalization stats missing for '{var}'. The quantile head anchors "
            f"on the interpolated ERA5 field, so its stats must exist in "
            f"normalization_stats.json."
        )
    return float(stats[var]['mean']), float(stats[var]['std'])


def _build_quantile_head(train_config, stats, input_vars, output_vars):
    """Assemble a QuantileWindHead from config + normalization stats.

    The head deliberately declares its structure explicitly instead of inheriting
    `parse_variable_config`'s name-sniffing heuristic, which detects wind pairs
    with `'wind' in name` then splits on `'u' in name` / `'v' in name`. A pair
    keyed `wind_gust` matches BOTH 'wind' and 'u' (in "gust") and would silently
    corrupt the u/v pairing.
    """
    from cosmos_wind_cnn.models.quantile_head import QuantileWindHead

    anchor_u = train_config.get('anchor_u', 'lr_u')
    anchor_v = train_config.get('anchor_v', 'lr_v')
    for v in (anchor_u, anchor_v):
        if v not in input_vars:
            raise ValueError(
                f"quantile head anchor '{v}' is not an input channel. "
                f"input_vars={input_vars}"
            )
    um, us = _stat_pair(stats, anchor_u)
    vm, vs = _stat_pair(stats, anchor_v)

    n_speed = int(train_config.get('n_quantiles', 19))

    # Gust is optional and only active when a target for it exists.
    n_gust = 0
    gust_anchor = None
    gust_target = train_config.get('gust_target', 'hr_gust')
    if gust_target in output_vars:
        n_gust = int(train_config.get('n_gust_quantiles', 9))
        gust_input = train_config.get('gust_input', 'lr_gust')
        if gust_input in input_vars:
            gm, gsd = _stat_pair(stats, gust_input)
            gust_anchor = (input_vars.index(gust_input), gm, gsd)

    return QuantileWindHead(
        in_channels=train_config.get('base_channels', 32),
        n_speed=n_speed,
        anchor_idx=(input_vars.index(anchor_u), input_vars.index(anchor_v)),
        anchor_denorm=(um, us, vm, vs),
        n_gust=n_gust,
        gust_anchor=gust_anchor,
    )

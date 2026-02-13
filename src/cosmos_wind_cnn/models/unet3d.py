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

    def __init__(self, in_channels, out_channels, base_channels=32):
        """
        Args:
            in_channels: Number of input channels (variables)
            out_channels: Number of output channels (predicted variables)
            base_channels: Base number of feature channels (will be multiplied)
        """
        super(Wind3DUNET, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.enc1 = self.conv_block_3d(in_channels, base_channels)
        self.enc2 = self.conv_block_3d(base_channels, base_channels*2)
        self.enc3 = self.conv_block_3d(base_channels*2, base_channels*4)
        self.enc4 = self.conv_block_3d(base_channels*4, base_channels*8)

        # Pool only spatially, not temporally
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Bottleneck
        self.bottleneck = self.conv_block_3d(base_channels*8, base_channels*16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(
            base_channels*16, base_channels*8,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec4 = self.conv_block_3d(base_channels*16, base_channels*8)

        self.up3 = nn.ConvTranspose3d(
            base_channels*8, base_channels*4,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec3 = self.conv_block_3d(base_channels*8, base_channels*4)

        self.up2 = nn.ConvTranspose3d(
            base_channels*4, base_channels*2,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec2 = self.conv_block_3d(base_channels*4, base_channels*2)

        self.up1 = nn.ConvTranspose3d(
            base_channels*2, base_channels,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec1 = self.conv_block_3d(base_channels*2, base_channels)

        # Output layer - predict for last timestep
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def conv_block_3d(self, in_ch, out_ch):
        """3D convolutional block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

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

        output = self.out(d1_last)

        return output

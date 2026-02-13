# Model Architecture

## 3D U-Net for Meteorological Downscaling

The model is a 3D U-Net (`Wind3DUNET`) that processes spatiotemporal sequences of low-resolution meteorological fields and produces a single high-resolution output frame.

## Input/Output

- **Input**: `(batch, sequence_length, n_input_vars, height, width)`
  - `sequence_length`: Number of input timesteps (default: 6)
  - `n_input_vars`: Low-resolution variables (e.g., ERA5 u, v, temp, pressure, radiation, cloud cover)

- **Output**: `(batch, n_output_vars, height, width)`
  - `n_output_vars`: High-resolution target variables (e.g., CONUS404 u, v, temp, pressure, radiation)

## Architecture

```
Input (B, T, C_in, H, W)
    │
    ├─ permute → (B, C_in, T, H, W)
    │
    ├─ Encoder Level 1: Conv3D block → e1 (B, 32, T, H, W)
    ├─ MaxPool3D(1,2,2) ─────────────────────────────────┐
    ├─ Encoder Level 2: Conv3D block → e2 (B, 64, T, H/2, W/2)
    ├─ MaxPool3D(1,2,2) ────────────────────────────────┐│
    ├─ Encoder Level 3: Conv3D block → e3 (B, 128, T, H/4, W/4)
    ├─ MaxPool3D(1,2,2) ───────────────────────────────┐││
    ├─ Encoder Level 4: Conv3D block → e4 (B, 256, T, H/8, W/8)
    ├─ MaxPool3D(1,2,2) ──────────────────────────────┐│││
    │                                                  ││││
    ├─ Bottleneck: Conv3D block → b (B, 512, T, H/16, W/16)
    │                                                  ││││
    ├─ ConvTranspose3D(1,2,2) + cat(e4) → d4 ─────────┘│││
    ├─ ConvTranspose3D(1,2,2) + cat(e3) → d3 ──────────┘││
    ├─ ConvTranspose3D(1,2,2) + cat(e2) → d2 ───────────┘│
    ├─ ConvTranspose3D(1,2,2) + cat(e1) → d1 ────────────┘
    │
    ├─ Select last timestep: d1[:, :, -1, :, :] → (B, 32, H, W)
    │
    └─ Conv2D(1x1) → Output (B, C_out, H, W)
```

## Design Choices

### Spatial-only pooling
The MaxPool3D uses kernel `(1, 2, 2)` and stride `(1, 2, 2)`, meaning it pools only in the spatial (H, W) dimensions while preserving the temporal dimension. This keeps all temporal information available to deeper layers.

### Skip connections
Standard U-Net skip connections concatenate encoder features with decoder features at each resolution level. A `match_size()` function handles cases where the spatial dimensions don't match exactly after upsampling (padding or cropping as needed).

### Output from last timestep
After the decoder, only the last temporal slice is used for the final 2D convolution. This means the model uses the full temporal context during encoding/decoding but produces a prediction for a single future timestep.

### Conv3D blocks
Each block consists of two 3D convolutions with BatchNorm and ReLU:
```
Conv3D(in, out, 3, padding=1) → BatchNorm3D → ReLU →
Conv3D(out, out, 3, padding=1) → BatchNorm3D → ReLU
```

## Loss Function

The `CombinedLoss` applies different loss functions depending on variable type:

**Wind pairs (u, v):** `WindLoss` with three components:
- Component MSE (alpha=1.0): Standard MSE on u and v components
- Speed MAE (beta=0.5): MAE on wind speed magnitude
- Direction cosine (gamma=0.3): 1 - cosine_similarity of normalized wind vectors

**Other variables:** Standard MSE loss

The total loss averages across all variable groups (wind pairs + non-wind variables).

## Parameter Count

With `base_channels=32` and 6 input / 5 output variables, the model has approximately 90M parameters. Reduce `base_channels` to 16 or 24 for smaller models.

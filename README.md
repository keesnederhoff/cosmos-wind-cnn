# cosmos-wind-cnn

3D U-Net for statistical downscaling of meteorological variables from ERA5 (~31 km) to CONUS404 (1-4 km) resolution.

## Overview

This project trains a 3D U-Net neural network to learn the mapping between coarse-resolution (ERA5) and fine-resolution (CONUS404) meteorological fields. Once trained, the model can generate high-resolution predictions from low-resolution inputs — including for time periods and data sources (e.g., CMIP6) not seen during training.

**Supported variables:**
- Wind components (U, V) with physics-informed loss (speed + direction)
- Air temperature, dew point temperature, pressure
- Solar and thermal radiation, precipitation
- Additional context inputs (e.g., cloud cover)

**Key features:**
- Multi-case-study support (SF Bay, Puget Sound, extensible)
- Temporal context via 3D convolutions (configurable sequence length)
- Combined loss: component MSE + wind speed MAE + direction cosine similarity
- General-purpose regridder for ERA5/CMIP6 inference on the trained grid
- Self-contained run directories with full reproducibility (data, configs, checkpoints, outputs)
- DDP multi-GPU training support via SLURM

## Installation

```bash
git clone <repo-url> cosmos-wind-cnn
cd cosmos-wind-cnn
pip install -e ".[dev]"
```

For GPU support, install PyTorch with CUDA first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Full pipeline (recommended)

A single script handles preprocessing, training, config archiving, inference, and evaluation:

```bash
# Single GPU
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay \
    --run-name my_experiment

# Multi-GPU (4x DDP)
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay \
    --run-name my_experiment \
    --gpus 4

# Skip steps on reruns
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay \
    --run-name my_experiment \
    --skip-preprocess --skip-train
```

### Standalone inference (new data / CMIP6)

Use a trained model to downscale new coarse data:

```bash
python scripts/run_inference.py \
    --case-study case_studies/sf_bay \
    --run-name my_experiment \
    --start-date 2024-01-01 \
    --end-date 2026-12-31
```

### HPC (SLURM)

```bash
# GPU pipeline (4x V100 DDP)
sbatch scripts/gpu_tallgrass.slurm

# CPU pipeline
sbatch scripts/cpu_tallgrass.slurm
```

### Monitor training

```bash
tensorboard --logdir case_studies/sf_bay/results/<run_name>/logs
```

## Project Structure

```
cosmos-wind-cnn/
├── src/cosmos_wind_cnn/         # Installable Python package
│   ├── data/                    # Preprocessing, datasets, regridder
│   ├── models/                  # 3D U-Net architecture
│   ├── training/                # Loss functions, metrics, training loop
│   └── utils/                   # Config parsing, run directory layout, visualization
├── scripts/                     # CLI entry points
│   ├── run_training_pipeline.py # Primary: preprocess → train → archive → infer → evaluate
│   ├── run_inference.py         # Primary: standalone inference with trained model
│   ├── train.py                 # Training (called by pipeline, supports DDP)
│   ├── evaluate.py              # Test-set evaluation
│   ├── preprocess_training.py   # Standalone preprocessing
│   ├── preprocess_inference.py  # Standalone inference regridding
│   ├── inference_full_record.py # Standalone full-record inference
│   ├── validate_inference.py    # Station-level validation (NDBC, moorings)
│   ├── cpu_tallgrass.slurm      # SLURM: CPU pipeline
│   └── gpu_tallgrass.slurm      # SLURM: GPU pipeline (4x V100 DDP)
├── case_studies/                # Per-domain configs and data
│   ├── sf_bay/                  # San Francisco Bay (working example)
│   ├── puget_sound/             # Puget Sound (in progress)
│   └── _template/               # Template for new case studies
├── notebooks/                   # Data exploration and validation
├── docs/                        # Documentation
├── tests/                       # Unit tests
└── pyproject.toml               # Package definition
```

## Run Directory Layout

All outputs for a run are organized under `results/<run_name>/`:

```
case_studies/sf_bay/
├── data/raw/                              # Raw NetCDF input files (shared)
├── configs/                               # YAML configs (shared)
└── results/<run_name>/                    # Everything for one run
    ├── checkpoint/                        # best_model.pth, archived configs, training_loss.png
    ├── data_processed/                    # train.nc, val.nc, test.nc, normalization_stats.pkl,
    │                                      #   target_grid_reference.nc
    ├── logs/                              # TensorBoard events, SLURM log
    ├── output_inference/                  # full_record_ERA5_*.nc, inference_ERA5_*.nc
    └── output_evaluation/                 # grid_point_metrics.csv, test_results.json, samples/
```

Each run is fully self-contained and reproducible. Configs are archived into `checkpoint/` so the exact settings are always available.

## Data Preparation

This CNN expects ERA5 and CONUS404 data **already interpolated to the same grid** (e.g., UTM Zone 10N). The upstream data preparation pipeline involves:

1. **CONUS404**: Convert from native Lambert Conformal grid to UTM
2. **ERA5**: Extract for model domain and interpolate to the same UTM grid

See [docs/data_preparation.md](docs/data_preparation.md) for details and example scripts.

**Data is not included in this repository.** Each case study's `data/raw/` directory should contain one NetCDF file per variable, with matching spatial grids and time coordinates.

## Case Studies

Each case study is self-contained under `case_studies/`:

| Case Study | Domain | Resolution | Status |
|------------|--------|------------|--------|
| `sf_bay` | San Francisco Bay | 4 km | Complete |
| `puget_sound` | Puget Sound | TBD | Data prep |

To add a new case study, see [docs/adding_case_study.md](docs/adding_case_study.md).

## Model Architecture

The 3D U-Net takes a sequence of low-resolution timesteps and predicts a single high-resolution output:

- **Input**: `(batch, seq_len, n_input_vars, H, W)` -- e.g., 6 timesteps of ERA5 fields
- **Output**: `(batch, n_output_vars, H, W)` -- single timestep of CONUS404-resolution fields

The encoder uses 3D convolutions that pool only spatially (not temporally), preserving temporal context through the network. Skip connections link encoder and decoder at each resolution level.

See [docs/model_architecture.md](docs/model_architecture.md) for details.

## Configuration

Each case study has three YAML config files:

- `configs/preprocessing.yaml` -- file mappings, physical bounds, data split ratios
- `configs/training.yaml` -- variable pairs, model architecture, training hyperparameters
- `configs/inference_preprocessing.yaml` -- source file mappings for inference regridding

Key training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_channels` | 32 | Feature channels (32->64->128->256->512) |
| `sequence_length` | 6 | Input timesteps |
| `batch_size` | 8 | Reduce if GPU OOM |
| `learning_rate` | 0.001 | Initial learning rate |
| `loss_alpha/beta/gamma` | 1.0/0.5/0.3 | MSE/speed/direction weights |

## Scripts Reference

| Script | Purpose | When to use |
|--------|---------|-------------|
| `run_training_pipeline.py` | Full pipeline (preprocess → train → archive → infer → evaluate) | Primary entry point for training |
| `run_inference.py` | Standalone inference with regridding | Downscale new data (ERA5, CMIP6) with a trained model |
| `train.py` | Training only (supports DDP) | Called by pipeline; can also run standalone |
| `evaluate.py` | Test-set evaluation | Standalone evaluation of a trained model |
| `preprocess_training.py` | Preprocessing only | Standalone data preparation |
| `preprocess_inference.py` | Regrid coarse data for inference | Standalone preprocessing for inference |
| `inference_full_record.py` | Run model on processed splits | Legacy inference from train/val/test splits |
| `validate_inference.py` | Station-level validation (NDBC, moorings) | Compare predictions to observations |

## License

MIT

# cosmos-wind-cnn

3D U-Net for statistical downscaling of meteorological variables from ERA5 (~31 km) to CONUS404 (1-4 km) resolution.

## Overview

This project trains a 3D U-Net neural network to learn the mapping between coarse-resolution (ERA5) and fine-resolution (CONUS404) meteorological fields. Once trained, the model can generate high-resolution predictions from low-resolution inputs.

**Supported variables:**
- Wind components (U, V) with physics-informed loss (speed + direction)
- Temperature, pressure, radiation (MSE loss)
- Additional context inputs (e.g., cloud cover)

**Key features:**
- Multi-case-study support (SF Bay, Puget Sound, extensible)
- Temporal context via 3D convolutions (configurable sequence length)
- Combined loss: component MSE + wind speed MAE + direction cosine similarity

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

Using the SF Bay case study as an example:

```bash
# 1. Place data in case_studies/sf_bay/data/raw/ (see Data Preparation below)

# 2. Preprocess
python scripts/preprocess.py --case-study case_studies/sf_bay

# 3. Train
python scripts/train.py --case-study case_studies/sf_bay

# 4. Evaluate
python scripts/evaluate.py --case-study case_studies/sf_bay

# 5. Inference
python scripts/inference.py --case-study case_studies/sf_bay
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir case_studies/sf_bay/logs
```

## Project Structure

```
cosmos-wind-cnn/
├── src/cosmos_wind_cnn/      # Installable Python package
│   ├── data/                 # Preprocessing and PyTorch datasets
│   ├── models/               # 3D U-Net architecture
│   ├── training/             # Loss functions, metrics, training loop
│   └── utils/                # Config parsing, visualization
├── scripts/                  # CLI entry points (preprocess, train, evaluate, inference)
├── case_studies/             # Per-domain configs, data, checkpoints, outputs
│   ├── sf_bay/               # San Francisco Bay (working example)
│   ├── puget_sound/          # Puget Sound (in progress)
│   └── _template/            # Template for new case studies
├── notebooks/                # Data exploration and validation
├── docs/                     # Documentation and example scripts
├── tests/                    # Unit tests
└── pyproject.toml            # Package definition
```

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

Each case study has two YAML config files:

- `configs/preprocessing.yaml` -- file mappings, data split ratios
- `configs/training.yaml` -- variable pairs, model architecture, training hyperparameters

Key training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_channels` | 32 | Feature channels (32->64->128->256->512) |
| `sequence_length` | 6 | Input timesteps |
| `batch_size` | 8 | Reduce if GPU OOM |
| `learning_rate` | 0.001 | Initial learning rate |
| `loss_alpha/beta/gamma` | 1.0/0.5/0.3 | MSE/speed/direction weights |

## Validation and Analysis

The `notebooks/` directory contains templates for:

- **01_data_exploration.ipynb** -- Explore raw NetCDF data
- **02_validate_raw_meteo.ipynb** -- Compare ERA5/CONUS404 against observations
- **03_validate_cnn_output.ipynb** -- Validate CNN predictions against observations
- **04_compare_case_studies.ipynb** -- Cross-domain comparison

## License

MIT

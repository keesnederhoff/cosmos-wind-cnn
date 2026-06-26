# SF Bay Case Study

Statistical downscaling of meteorological variables for the San Francisco Bay region.

## Data

- **High-resolution:** CONUS404 at 4 km (SFbay domain), UTM Zone 10N
- **Low-resolution:** ERA5 at ~31 km, interpolated to the same UTM10N grid
- **Training period:** 1979-2021 (CONUS404 availability)
- **Inference period:** 1940-2026 (full ERA5 record)
- **Domain extent:** UTM10N x=[425-596 km], y=[4092-4257 km]

## Variables

| Variable | CONUS404 source | ERA5 source |
|----------|----------------|-------------|
| Eastward wind (U) | U10 | u10 |
| Northward wind (V) | V10 | v10 |
| Air temperature | T2 | t2m |
| Dew point temperature | TD2 | d2m |
| Air pressure (MSL) | PSFC (converted) | msl |
| Solar radiation | ACSWDNB (converted) | ssr |
| Thermal radiation | ACLWDNB (converted) | strd |
| Precipitation | RAINNC (converted) | tp |

## Usage

### Full pipeline (recommended)

```bash
# Single script: preprocess → train → archive configs → inference → evaluate
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_conus404 \
    --run-name my_experiment \
    --gpus 4
```

### Standalone inference (new time period or CMIP6)

```bash
python scripts/run_inference.py \
    --case-study case_studies/sf_bay_conus404 \
    --run-name my_experiment \
    --start-date 2024-01-01 \
    --end-date 2026-12-31
```

### Quick per-step scripts (local experiments)

Lightweight standalone alternatives to the run-isolated pipeline above:

```bash
python scripts/preprocess_training.py --case-study case_studies/sf_bay_conus404 --run-name <run>
python scripts/train.py      --case-study case_studies/sf_bay_conus404
python scripts/evaluate.py   --case-study case_studies/sf_bay_conus404
python scripts/run_inference.py --case-study case_studies/sf_bay_conus404 --run-name <run> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>
```

For reproducible runs prefer `run_training_pipeline.py`, which isolates every
artifact under `$COSMOS_RESULTS_ROOT/sf_bay_conus404/results/<run_name>/`.

### HPC (SLURM on Tallgrass)

```bash
sbatch scripts/gpu_tallgrass.slurm   # 4x V100 DDP
sbatch scripts/cpu_tallgrass.slurm   # CPU only
```

## Storage

The repo holds only `configs/` + `README.md`. Raw data and run outputs are external:

```bat
:: Windows — set before running anything
set COSMOS_DATA_ROOT=G:\03-downscaling_meteo_cnn
set COSMOS_RESULTS_ROOT=G:\03-downscaling_meteo_cnn
```

```
<COSMOS_DATA_ROOT>\sf_bay_conus404\raw_data\    # Raw NetCDF input files (shared across runs)

case_studies/sf_bay_conus404/                   # Repo: configs + README only
└── configs/

<COSMOS_RESULTS_ROOT>\sf_bay_conus404\results\<run_name>\   # Per-run outputs
    ├── checkpoint/                             # best_model.pth, archived configs
    ├── data_processed/                         # train/val/test splits, normalization stats
    ├── logs/                                   # TensorBoard, SLURM log
    ├── output_inference/                       # Downscaled predictions
    └── output_evaluation/                      # Metrics, figures
```

On HPC the Tallgrass SLURM scripts already export `COSMOS_DATA_ROOT` and `COSMOS_RESULTS_ROOT` pointing at caldera project space.

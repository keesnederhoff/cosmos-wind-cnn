# [Case Study Name]

## Setup

1. Copy this template:
   ```bash
   cp -r case_studies/_template case_studies/my_study
   ```

2. Prepare your data following `docs/data_preparation.md`

3. Place NetCDF files in `data/raw/`

4. Edit `configs/preprocessing.yaml` with your filenames and physical bounds

5. Edit `configs/training.yaml` with your variable pairs, provenance labels, and hyperparameters

6. Edit `configs/inference_preprocessing.yaml` with source file mappings for inference

7. Run the full pipeline:
   ```bash
   python scripts/run_training_pipeline.py \
       --case-study case_studies/my_study \
       --run-name first_run \
       --gpus 4
   ```

   Or run individual steps:
   ```bash
   python scripts/preprocess_training.py --case-study case_studies/my_study --run-name first_run
   python scripts/train.py --case-study case_studies/my_study --run-name first_run
   python scripts/evaluate.py --case-study case_studies/my_study --run-name first_run
   ```

   Or run individual steps (run-isolated):
   ```bash
   python scripts/preprocess_training.py --case-study case_studies/my_study --run-name <run>
   python scripts/run_inference.py --case-study case_studies/my_study --run-name <run> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>
   ```

All outputs are saved under `results/<run_name>/` (checkpoint, processed data, logs, inference, evaluation).

## Variable key convention (hr_ / lr_)

All variable keys use an `hr_` prefix for the high-resolution target and an `lr_` prefix for the
low-resolution input. This convention is model-agnostic: the same pipeline works whether your HR
source is CONUS404, RTMA, or another product, and your LR source is ERA5, CMIP6, or anything else.

### preprocessing.yaml

```yaml
file_dict:
  hr_u: 'your_high_res_u_wind.nc'
  lr_u: 'your_low_res_u_wind.nc'
  hr_v: 'your_high_res_v_wind.nc'
  lr_v: 'your_low_res_v_wind.nc'
  hr_air_temp: 'your_high_res_air_temp.nc'
  lr_air_temp: 'your_low_res_air_temp.nc'
  # ... add all variable pairs

train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
```

### training.yaml

Variable pairs map `high_res:` / `low_res:` values to the `hr_` / `lr_` keys defined above.
The `hr_source` and `lr_source` fields record which actual datasets were used — these are
provenance labels for reproducibility and do not affect model behavior.

```yaml
hr_source: CONUS404   # factual provenance: which dataset provides the HR target
lr_source: ERA5       # factual provenance: which dataset provides the LR input

variable_pairs:
  wind_u:
    high_res: 'hr_u'
    low_res:  'lr_u'
  wind_v:
    high_res: 'hr_v'
    low_res:  'lr_v'
  air_temp:
    high_res: 'hr_air_temp'
    low_res:  'lr_air_temp'
  # ... all pairs
```

See `case_studies/sf_bay_conus404/` for a worked example (CONUS404 HR / ERA5 LR) and
`case_studies/sf_bay_rtma/` for an alternative HR source (RTMA HR / ERA5 LR).

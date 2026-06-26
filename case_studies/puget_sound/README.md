# Puget Sound Case Study

Statistical downscaling of meteorological variables for the Puget Sound region.

## Status

Data preparation in progress. See `docs/data_preparation.md` for the upstream pipeline.

## Storage

The repo holds only `configs/` + `README.md`. Raw data and run outputs are external — set these env vars before running anything:

```bat
:: Windows
set COSMOS_DATA_ROOT=G:\03-downscaling_meteo_cnn
set COSMOS_RESULTS_ROOT=G:\03-downscaling_meteo_cnn
```

Raw inputs: `%COSMOS_DATA_ROOT%\puget_sound\raw_data\`
Run outputs: `%COSMOS_RESULTS_ROOT%\puget_sound\results\<run_name>\`

On HPC the Tallgrass SLURM scripts already export both vars pointing at caldera project space.

## Usage

Once data is prepared and placed in `$COSMOS_DATA_ROOT/puget_sound/raw_data/`:

```bash
# Full pipeline
python scripts/run_training_pipeline.py \
    --case-study case_studies/puget_sound \
    --run-name first_run \
    --gpus 4

# Or individual steps (run-isolated)
python scripts/preprocess_training.py --case-study case_studies/puget_sound --run-name first_run
python scripts/train.py --case-study case_studies/puget_sound --run-name first_run
python scripts/evaluate.py --case-study case_studies/puget_sound --run-name first_run

# Or quick standalone scripts (local experiments)
python scripts/preprocess_training.py --case-study case_studies/puget_sound --run-name <run>
python scripts/run_inference.py --case-study case_studies/puget_sound --run-name <run> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>
```

All outputs are saved under `$COSMOS_RESULTS_ROOT/puget_sound/results/<run_name>/`.

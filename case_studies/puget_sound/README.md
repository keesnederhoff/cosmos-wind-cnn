# Puget Sound Case Study

Statistical downscaling of meteorological variables for the Puget Sound region.

## Status

Data preparation in progress. See `docs/data_preparation.md` for the upstream pipeline.

## Usage

Once data is prepared and placed in `data/raw/`:

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
python scripts/preprocess.py --case-study case_studies/puget_sound
python scripts/inference.py  --case-study case_studies/puget_sound
```

All outputs are saved under `results/<run_name>/`.

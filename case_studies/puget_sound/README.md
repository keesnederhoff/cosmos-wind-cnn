# Puget Sound Case Study

Statistical downscaling of meteorological variables for the Puget Sound region.

## Status

Data preparation in progress. See `docs/data_preparation.md` for the upstream pipeline.

## Usage

Once data is prepared and placed in `data/raw/`:

```bash
python scripts/preprocess.py --case-study case_studies/puget_sound
python scripts/train.py --case-study case_studies/puget_sound
python scripts/evaluate.py --case-study case_studies/puget_sound
```

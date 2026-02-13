# [Case Study Name]

## Setup

1. Copy this template:
   ```bash
   cp -r case_studies/_template case_studies/my_study
   ```

2. Prepare your data following `docs/data_preparation.md`

3. Place NetCDF files in `data/raw/`

4. Edit `configs/preprocessing.yaml` with your filenames

5. Edit `configs/training.yaml` with your hyperparameters

6. Run the pipeline:
   ```bash
   python scripts/preprocess.py --case-study case_studies/my_study
   python scripts/train.py --case-study case_studies/my_study
   python scripts/evaluate.py --case-study case_studies/my_study
   ```

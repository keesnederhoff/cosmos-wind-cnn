# [Case Study Name]

## Setup

1. Copy this template:
   ```bash
   cp -r case_studies/_template case_studies/my_study
   ```

2. Prepare your data following `docs/data_preparation.md`

3. Place NetCDF files in `data/raw/`

4. Edit `configs/preprocessing.yaml` with your filenames and physical bounds

5. Edit `configs/training.yaml` with your variable pairs and hyperparameters

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

All outputs are saved under `results/<run_name>/` (checkpoint, processed data, logs, inference, evaluation).

"""
Preprocessing script - loads raw NetCDF files and creates train/val/test splits.

Usage:
    python scripts/preprocess.py --case-study case_studies/sf_bay_conus404
"""

import os
# Fix OpenMP duplicate library error on Windows (must be before numpy/torch imports)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path

from cosmos_wind_cnn.data.preprocessing import NetCDFPreprocessor
from cosmos_wind_cnn.utils.config import load_config


def main():
    # Change to project root directory (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")
    
    parser = argparse.ArgumentParser(description='Preprocess data for CNN training')
    parser.add_argument('--case-study', default='case_studies/sf_bay_conus404',
                        help='Path to case study directory (e.g., case_studies/sf_bay_conus404)')
    args = parser.parse_args()

    case_dir = Path(args.case_study)
    config = load_config(case_dir / 'configs' / 'preprocessing.yaml')

    data_dir = case_dir / 'data' / 'raw'
    output_dir = case_dir / 'data' / 'processed'

    print("=" * 70)
    print(f"Preprocessing: {case_dir.name}")
    print("=" * 70)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    preprocessor = NetCDFPreprocessor({'data_dir': str(data_dir)})

    # Check files exist
    file_dict = config['file_dict']
    print("\nFiles to process:")
    for var, filename in file_dict.items():
        filepath = data_dir / filename
        exists = "OK" if filepath.exists() else "NOT FOUND"
        print(f"  [{exists}] {var}: {filename}")

    # Load and combine
    print("\n" + "=" * 70)
    print("Loading and combining datasets...")
    print("=" * 70)
    start_date = config.get('start_date', None)
    end_date   = config.get('end_date', None)
    if start_date or end_date:
        print(f"  Time period filter: {start_date or 'start'} to {end_date or 'end'}")
    combined_ds = preprocessor.load_and_align_datasets(
        file_dict, start_date=start_date, end_date=end_date
    )

    print(f"\nCombined dataset:")
    print(f"  Variables: {list(combined_ds.data_vars)}")
    print(f"  Time steps: {len(combined_ds.time)}")

    # Split
    print("\n" + "=" * 70)
    print("Splitting into train/val/test...")
    print("=" * 70)
    train_ds, val_ds, test_ds = preprocessor.create_train_val_test_split(
        combined_ds,
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15),
    )

    # Save
    print("\n" + "=" * 70)
    print("Saving processed datasets...")
    print("=" * 70)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save_processed_data(train_ds, output_dir / 'train.nc')
    preprocessor.save_processed_data(val_ds, output_dir / 'val.nc')
    preprocessor.save_processed_data(test_ds, output_dir / 'test.nc')

    # Statistics
    print("\n" + "=" * 70)
    print("Calculating normalization statistics...")
    print("=" * 70)
    stats = preprocessor.calculate_and_save_stats(
        train_ds, output_dir / 'normalization_stats.pkl'
    )

    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"\nOutput files in: {output_dir}")
    print(f"  train.nc      - {len(train_ds.time)} timesteps")
    print(f"  val.nc        - {len(val_ds.time)} timesteps")
    print(f"  test.nc       - {len(test_ds.time)} timesteps")
    print(f"  normalization_stats.pkl - {len(stats)} variables")
    print(f"\nNext step: python scripts/train.py --case-study {case_dir}")


if __name__ == '__main__':
    main()

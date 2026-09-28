import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from pathlib import Path
import itertools

import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Historical human-agreement analysis")
    parser.add_argument('--input-dir', type=Path, default=root/'Data/human_ratings/historical')
    parser.add_argument('--output-dir', type=Path, default=root/'Outputs/human_agreement')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Read all CSV files in the current directory
    csv_files = list(args.input_dir.glob('*.csv'))

    # Dictionary to store data from each file
    data_dict = {}

    # Read each CSV and extract the "Meet Criterion" column
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'Meet Criterion' in df.columns:
            data_dict[csv_file.stem] = df['Meet Criterion'].values
        else:
            print(f"Warning: 'Meet Criterion' column not found in {csv_file.name}")

    # Calculate pairwise Cohen's kappa
    file_names = list(data_dict.keys())
    n_files = len(file_names)

    print(f"Found {n_files} CSV files with 'Meet Criterion' column\n")

    # Create results matrix
    kappa_matrix = np.zeros((n_files, n_files))

    # Calculate pairwise kappa scores
    for i, j in itertools.combinations(range(n_files), 2):
        name_i = file_names[i]
        name_j = file_names[j]

        # Get the minimum length to handle different sized arrays
        min_len = min(len(data_dict[name_i]), len(data_dict[name_j]))

        # Calculate Cohen's kappa
        kappa = cohen_kappa_score(data_dict[name_i][:min_len], data_dict[name_j][:min_len])

        kappa_matrix[i, j] = kappa
        kappa_matrix[j, i] = kappa

        print(f"Cohen's kappa between {name_i} and {name_j}: {kappa:.4f}")

    # Set diagonal to 1 (perfect agreement with self)
    np.fill_diagonal(kappa_matrix, 1.0)

    # Create a DataFrame for better visualization
    kappa_df = pd.DataFrame(kappa_matrix, index=file_names, columns=file_names)

    print("\n" + "="*50)
    print("Pairwise Cohen's Kappa Matrix:")
    print("="*50)
    print(kappa_df.round(4))

    # Save results to CSV
    output_path = args.output_dir / 'cohen_kappa_results.csv'
    kappa_df.to_csv(output_path)
    print(f"\nResults saved to {output_path.name}")


if __name__ == '__main__':
    main()

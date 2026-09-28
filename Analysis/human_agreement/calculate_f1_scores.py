import pandas as pd
from sklearn.metrics import f1_score
import glob
import os

import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Historical human-agreement analysis")
    parser.add_argument('--input-dir', type=Path, default=root/'Data/human_ratings/historical')
    parser.add_argument('--output-dir', type=Path, default=root/'Outputs/human_agreement')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Set directory path
    analysis_dir = str(args.input_dir)

    # Read all CSV files
    ai_grading = pd.read_csv(os.path.join(analysis_dir, "ai_grading_results.csv"))
    human_expert1 = pd.read_csv(os.path.join(analysis_dir, "evaluation_human_expert1.csv"))
    human_expert2 = pd.read_csv(os.path.join(analysis_dir, "evaluation_human_expert2.csv"))

    # Extract predictions and references
    predictions = ai_grading["Meet Criterion"].values
    reference1 = human_expert1["Meet Criterion"].values
    reference2 = human_expert2["Meet Criterion"].values

    # Calculate F1 scores separately
    f1_vs_expert1 = f1_score(reference1, predictions)
    f1_vs_expert2 = f1_score(reference2, predictions)
    f1_vs_expert1_exper2 = f1_score(reference1, reference2)

    # Print results
    print("F1 Score Results:")
    print("="*50)
    print(f"AI vs Human Expert 1: {f1_vs_expert1:.3f}")
    print(f"AI vs Human Expert 2: {f1_vs_expert2:.3f}")
    print(f"Average AI-Human F1 Score: {(f1_vs_expert1 + f1_vs_expert2) / 2:.3f}")

    print(f"Average Human-Human F1 Score: {f1_vs_expert1_exper2:.3f}")

    # Save results to CSV
    results_df = pd.DataFrame({
        "Comparison": ["AI vs Expert 1", "AI vs Expert 2", "Average", 'Expert1 vs Expert2'],
        "F1 Score": [f1_vs_expert1, f1_vs_expert2, (f1_vs_expert1 + f1_vs_expert2) / 2, f1_vs_expert1_exper2]
    })

    results_df.to_csv(str(args.output_dir / "f1_score_results.csv"), index=False)
    print(f"\nResults saved to: {str(args.output_dir / 'f1_score_results.csv')}")


if __name__ == '__main__':
    main()

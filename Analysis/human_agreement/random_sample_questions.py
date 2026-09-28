import random
import csv

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sample the historical 40-question human-validation cohort")
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'Outputs/human_agreement/sampled_questions.csv')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Set random seed for reproducibility (optional, remove if you want different results each time)
    random.seed(2025)

    # Sample 40 questions from 1 to 282
    questions = random.sample(range(1, 283), 40)

    # List of models
    models = ['o3', 'grok-4-latest', 'meta-llama_Llama-3.1-8B-Instruct', 'claude-sonnet-4-5-20250929', 'gemini-2.5-pro']

    # Create output data
    output_data = []
    for question_id in questions:
        model = random.choice(models)
        output_data.append({'question_id': question_id, 'model_name': model})

    # Sort by question_id for easier viewing (optional)
    output_data.sort(key=lambda x: x['question_id'])

    # Write to CSV
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['question_id', 'model_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(output_data)

    print(f"Successfully sampled {len(output_data)} questions and saved to sampled_questions.csv")


if __name__ == '__main__':
    main()

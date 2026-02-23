import json
import os
import pandas as pd
import tiktoken
from pathlib import Path

# Initialize tokenizer (using cl100k_base which is used by GPT-4 and similar models)
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Count tokens in a text string"""
    return len(encoding.encode(text))

def analyze_response_files(response_dir):
    """Analyze all response files and count tokens"""
    results = []

    # Get all jsonl files (excluding greedy version to avoid duplicates)
    response_files = [
        'all_opensource_models_final_outputs_temp0.7.jsonl',
        'allenai_olmo-3-32b-think_responses.jsonl',
        'allenai_olmo-3.1-32b-instruct_responses.jsonl',
        'claude_family_response.jsonl',
        'claude_family_response_lastest3models_updated.jsonl',
        'gemini_family_response.jsonl',
        'gpt5_response.jsonl',
        'grok_response.jsonl',
        'openai_family_response.jsonl'
    ]

    for filename in response_files:
        filepath = os.path.join(response_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found")
            continue

        print(f"Processing {filename}...")

        # Read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            # Try to parse as JSON array first
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try parsing as JSONL (line by line)
                data = []
                for line in content.split('\n'):
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # Process each question
        for item in data:
            question_id = item.get('question_id', 'Unknown')
            question = item.get('question', '')
            responses = item.get('responses', {})

            # Count tokens for each model's response
            for model_name, response_text in responses.items():
                if response_text:
                    token_count = count_tokens(response_text)

                    results.append({
                        'file': filename,
                        'question_id': question_id,
                        'question': question[:100] + '...' if len(question) > 100 else question,
                        'model': model_name,
                        'response_length_chars': len(response_text),
                        'token_count': token_count
                    })

    return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    response_dir = "/Users/zhaoyimin/Desktop/PanCan QA/Manuscript/Data/Response"
    output_dir = "/Users/zhaoyimin/Desktop/PanCan QA/Manuscript/Figures/Figure4/number of tokens analysis"

    print("Analyzing token counts...")
    df = analyze_response_files(response_dir)

    # Save detailed results
    output_csv = os.path.join(output_dir, "token_counts_detailed.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")

    # Create summary statistics by model
    model_summary = df.groupby('model').agg({
        'token_count': ['count', 'mean', 'std', 'min', 'max', 'sum']
    }).round(2)
    model_summary.columns = ['num_responses', 'mean_tokens', 'std_tokens', 'min_tokens', 'max_tokens', 'total_tokens']
    model_summary = model_summary.sort_values('mean_tokens', ascending=False)

    output_model_summary = os.path.join(output_dir, "token_counts_by_model.csv")
    model_summary.to_csv(output_model_summary)
    print(f"Model summary saved to: {output_model_summary}")

    # Create summary by question
    question_summary = df.groupby('question_id').agg({
        'token_count': ['count', 'mean', 'std', 'min', 'max']
    }).round(2)
    question_summary.columns = ['num_models', 'mean_tokens', 'std_tokens', 'min_tokens', 'max_tokens']

    output_question_summary = os.path.join(output_dir, "token_counts_by_question.csv")
    question_summary.to_csv(output_question_summary)
    print(f"Question summary saved to: {output_question_summary}")

    # Create pivot table: models x questions
    pivot_table = df.pivot_table(
        values='token_count',
        index='model',
        columns='question_id',
        aggfunc='mean'
    ).round(0)

    output_pivot = os.path.join(output_dir, "token_counts_pivot_model_x_question.csv")
    pivot_table.to_csv(output_pivot)
    print(f"Pivot table saved to: {output_pivot}")

    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY MODEL")
    print("="*80)
    print(model_summary.to_string())

    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY QUESTION")
    print("="*80)
    print(question_summary.to_string())

    print(f"\nTotal number of model-question pairs analyzed: {len(df)}")
    print(f"Number of unique models: {df['model'].nunique()}")
    print(f"Number of unique questions: {df['question_id'].nunique()}")

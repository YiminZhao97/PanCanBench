"""
Generalized GPT grader for targeted models.
Usage: python get_grades_gpt_targeted.py --expert1_rubrics <path> --expert2_rubrics <path> --fold <num> --output_dir <path>
"""

import json
import os
import sys
import argparse
from datetime import datetime
sys.path.append('/home/yzhao4/PanCan-QA_LLM')
from grader import GPT4Grader
from datetime import datetime
os.chdir('/home/yzhao4/PanCan-QA_LLM/rubrics_collection/Analysis4_permute_grader_model')

# Target models for grading (from polished rubrics analysis)
TARGET_MODELS = ['gpt-4o', 'grok-4-latest', 'meta-llama_Llama-3.1-70B-Instruct']

def get_fold_range(fold_number):
    """Get question range for each fold."""
    fold_ranges = {
        1: (0, 57),    # Q1-Q57
        2: (57, 114),  # Q58-Q114
        3: (114, 170), # Q115-Q170
        4: (170, 226), # Q171-Q226
        5: (226, 282)  # Q227-Q282
    }
    if fold_number not in fold_ranges:
        raise ValueError(f"Invalid fold number: {fold_number}. Must be 1-5.")
    return fold_ranges[fold_number]

def transform_data_for_grader(data, target_models):
    """Transform data format for grader, filtering for target models only."""
    transformed = []
    for item in data:
        question_id = item['question_id']
        question = item['question']
        
        # Extract number from Q58 format -> 58
        if question_id.startswith('Q'):
            question_number = int(question_id[1:])  # Remove 'Q' prefix, convert to int
        else:
            question_number = int(question_id)
        
        # Extract responses for target models only
        for model_name, response in item['responses'].items():
            if model_name in target_models:
                transformed.append({
                    'question_number': question_number,
                    'question': question,  
                    'response': response,
                    'source': model_name
                })
    return transformed

def main():
    parser = argparse.ArgumentParser(description='Generate GPT grading scores for specific target models')
    parser.add_argument('--expert1_rubrics', type=str, required=True,
                       help='Path to Expert1 rubrics JSON file')
    parser.add_argument('--expert2_rubrics', type=str, required=True,
                       help='Path to Expert2 rubrics JSON file')
    parser.add_argument('--fold', type=int, required=True, choices=[1,2,3,4,5],
                       help='Fold number (1-5)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key required. Set via --api_key or OPENAI_API_KEY env var")
        sys.exit(1)

    # Initialize GPT grader
    grader = GPT4Grader(
        api_key=api_key,
        model='gpt-4.1'
    )
    
    print(f"Processing Fold {args.fold} with target models: {TARGET_MODELS}")
    
    # Get fold range
    start_idx, end_idx = get_fold_range(args.fold)
    
    # Load rubrics - generalized variable names
    rubrics_expert1 = grader.load_rubrics(args.expert1_rubrics)
    rubrics_expert2 = grader.load_rubrics(args.expert2_rubrics)
    
    # Load response data
    print("Loading response data...")
    with open('/home/yzhao4/PanCan-QA_LLM/data_share/openai_family_response.jsonl', 'r', encoding='utf-8') as file:
        gpt_results = json.load(file)
    
    with open('/home/yzhao4/PanCan-QA_LLM/data_share/grok_response.jsonl', 'r', encoding='utf-8') as file:
        grok_results = json.load(file)
    
    with open('/home/yzhao4/PanCan-QA_LLM/data_share/all_opensource_models_final_outputs_temp0.7.jsonl', 'r', encoding='utf-8') as file:
        opensource_results = [json.loads(line.strip()) for line in file if line.strip()] 
    
    # Extract fold data
    gpt_fold = gpt_results[start_idx:end_idx]
    grok_fold = grok_results[start_idx:end_idx]
    opensource_fold = opensource_results[start_idx:end_idx]
    
    # Transform and filter data for target models
    print("Transforming and filtering data for target models...")
    
    all_transformed_data = []
    all_transformed_data.extend(transform_data_for_grader(gpt_fold, TARGET_MODELS))
    all_transformed_data.extend(transform_data_for_grader(grok_fold, TARGET_MODELS))
    all_transformed_data.extend(transform_data_for_grader(opensource_fold, TARGET_MODELS))
    
    print(f"Total responses prepared for grading: {len(all_transformed_data)}")
    
    # Verify we have all target models
    models_found = set(item['source'] for item in all_transformed_data)
    print(f"Models found in data: {models_found}")
    missing_models = set(TARGET_MODELS) - models_found
    if missing_models:
        print(f"Warning: Missing target models: {missing_models}")
    
    # Grade with Expert1's rubrics
    print(f"\nGrading with Expert1's rubrics...")
    targeted_scores_expert1 = grader.grade_all_responses(
        all_transformed_data, 
        rubrics_expert1, 
        delay_seconds=1.0
    )
    
    # Grade with Expert2's rubrics
    print(f"\nGrading with Expert2's rubrics...")
    targeted_scores_expert2 = grader.grade_all_responses(
        all_transformed_data, 
        rubrics_expert2, 
        delay_seconds=1.0
    )
    
    # Save results
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSaving grading results to {args.output_dir}...")
    
    expert1_file = f'{args.output_dir}/fold{args.fold}_gpt_scores_expert1.json'
    expert2_file = f'{args.output_dir}/fold{args.fold}_gpt_scores_expert2.json'
    
    with open(expert1_file, 'w') as f:
        json.dump(targeted_scores_expert1, f, indent=2)
    
    with open(expert2_file, 'w') as f:
        json.dump(targeted_scores_expert2, f, indent=2)
    
    print(f"✅ Grading completed!")
    print(f"📄 Expert1 results: {expert1_file}")
    print(f"📄 Expert2 results: {expert2_file}")
    print(f"📊 Total responses graded: {len(all_transformed_data)}")
    print(f"🎯 Target models: {TARGET_MODELS}")

if __name__ == "__main__":
    main()

# Example usage:
# python get_grades_gpt_targeted.py --expert1_rubrics "./rubrics collection/Fold2-Jesse.json" --expert2_rubrics "./rubrics collection/Fold2-Simone.json" --fold 2 --output_dir "./res_fold2"
#!/usr/bin/env python3
"""
Generalized analysis to identify large grading differences between GPT and Claude.
Generates comprehensive reports matching the res fold2 format exactly.
Usage: python targeted_large_differences.py --gpt_expert1 <path> --gpt_expert2 <path> --claude_expert1 <path> --claude_expert2 <path> --expert1_rubric <path> --expert2_rubric <path> --fold <num> --output_dir <path>
"""

import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Target models to analyze
TARGET_MODELS = ['gpt-4o', 'grok-4-latest', 'meta-llama_Llama-3.1-70B-Instruct']

def calculate_percentage_difference(gpt_score, claude_score, max_score):
    """Calculate percentage difference between GPT and Claude scores."""
    if max_score == 0:
        return 0
    gpt_percentage = (gpt_score / max_score) * 100
    claude_percentage = (claude_score / max_score) * 100
    return abs(gpt_percentage - claude_percentage)

def find_large_differences(gpt_scores, claude_scores, threshold=20.0):
    """Find cases where grading differences exceed threshold."""
    large_differences = {}
    
    # Create lookups for efficient matching
    gpt_lookup = {}
    # Handle both old format (dict with 'grading_results') and new format (direct list)
    gpt_results = gpt_scores.get('grading_results', []) if isinstance(gpt_scores, dict) else gpt_scores
    for result in gpt_results:
        key = (result['question_number'], result['source'])
        gpt_lookup[key] = result
    
    claude_lookup = {}
    # Handle both old format (dict with 'grading_results') and new format (direct list)
    claude_results = claude_scores.get('grading_results', []) if isinstance(claude_scores, dict) else claude_scores
    for result in claude_results:
        key = (result['question_number'], result['source'])
        claude_lookup[key] = result
    
    # Find matching results and calculate differences
    for key, gpt_result in gpt_lookup.items():
        if key in claude_lookup:
            claude_result = claude_lookup[key]
            question_number, model_name = key
            
            # Only analyze target models
            if model_name not in TARGET_MODELS:
                continue
            
            # Skip if either grading failed
            if 'error' in gpt_result or 'error' in claude_result:
                continue
            
            # Get total scores
            gpt_total = sum([item.get('score_given', 0) for item in gpt_result.get('criterion_scores', [])])
            gpt_max = sum([item.get('max_points', 0) for item in gpt_result.get('criterion_scores', [])])
            
            claude_total = sum([item.get('score_given', 0) for item in claude_result.get('criterion_scores', [])])
            claude_max = sum([item.get('max_points', 0) for item in claude_result.get('criterion_scores', [])])
            
            if gpt_max != claude_max or gpt_max == 0:
                continue
            
            # Calculate difference
            difference = calculate_percentage_difference(gpt_total, claude_total, gpt_max)
            
            if difference >= threshold:
                question_id = f"Q{question_number}"
                
                if question_id not in large_differences:
                    large_differences[question_id] = {
                        'question_number': question_number,
                        'question_text': 'Question text not available',  # Will be populated later
                        'models_with_differences': [],
                        'model_analyses': {}
                    }
                
                large_differences[question_id]['models_with_differences'].append(model_name)
                large_differences[question_id]['model_analyses'][model_name] = {
                    'gpt_grader_score': (gpt_total / gpt_max) * 100,
                    'claude_grader_score': (claude_total / claude_max) * 100,
                    'difference': difference,
                    'model_response': 'Response not available',  # Will be populated later
                    'gpt_grader_details': gpt_result.get('criterion_scores', []),
                    'claude_grader_details': claude_result.get('criterion_scores', [])
                }
    
    return large_differences

def load_rubric(rubric_file):
    """Load rubric JSON file."""
    try:
        with open(rubric_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading rubric {rubric_file}: {e}")
        return {}

def find_rubric_for_question(rubric_data, question_number):
    """Find the rubric items for a specific question."""
    if not rubric_data or 'questions' not in rubric_data:
        return None
    
    for question in rubric_data['questions']:
        if question.get('question_number') == question_number:
            return question
    return None

def load_responses(response_files):
    """Load responses from multiple response files based on model types."""
    responses_map = {}
    
    # Load OpenAI responses
    try:
        with open(response_files['openai'], 'r', encoding='utf-8') as f:
            openai_data = json.load(f)
        
        for item in openai_data:
            question_id = item.get('question_id', '')
            if question_id.startswith('Q'):
                question_num = int(question_id[1:])
                if question_num not in responses_map:
                    responses_map[question_num] = {
                        'question': item.get('question', ''),
                        'responses': {}
                    }
                # Add OpenAI model responses
                responses_map[question_num]['responses'].update(item.get('responses', {}))
        
        print(f"Loaded OpenAI responses for {len(responses_map)} questions")
    except Exception as e:
        print(f"Error loading OpenAI responses: {e}")
    
    # Load Grok responses
    try:
        with open(response_files['grok'], 'r', encoding='utf-8') as f:
            grok_data = json.load(f)
        
        for item in grok_data:
            question_id = item.get('question_id', '')
            if question_id.startswith('Q'):
                question_num = int(question_id[1:])
                if question_num not in responses_map:
                    responses_map[question_num] = {
                        'question': item.get('question', ''),
                        'responses': {}
                    }
                # Add Grok model responses
                responses_map[question_num]['responses'].update(item.get('responses', {}))
        
        print(f"Added Grok responses, total questions: {len(responses_map)}")
    except Exception as e:
        print(f"Error loading Grok responses: {e}")
    
    # Load Opensource responses (JSONL format - line by line)
    try:
        opensource_data = []
        with open(response_files['opensource'], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    opensource_data.append(json.loads(line))
        
        for item in opensource_data:
            question_id = item.get('question_id', '')
            if question_id.startswith('Q'):
                question_num = int(question_id[1:])
                if question_num not in responses_map:
                    responses_map[question_num] = {
                        'question': item.get('question', ''),
                        'responses': {}
                    }
                # Add Opensource model responses
                responses_map[question_num]['responses'].update(item.get('responses', {}))
        
        print(f"Added Opensource responses, total questions: {len(responses_map)}")
    except Exception as e:
        print(f"Error loading Opensource responses: {e}")
    
    return responses_map

def generate_comprehensive_report(expert_name, differences, rubric_data):
    """Generate comprehensive markdown report matching res fold2 format."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate statistics
    total_cases = sum(len(data['models_with_differences']) for data in differences.values())
    unique_questions = len(differences)
    
    # Model breakdown
    model_breakdown = defaultdict(int)
    for data in differences.values():
        for model in data['models_with_differences']:
            model_breakdown[model] += 1
    
    # Multi-model questions
    multi_model_questions = [q for q, data in differences.items() if len(data['models_with_differences']) > 1]
    
    markdown_content = f"""# Large Grading Differences Analysis - {expert_name} Rubric

**Analysis Date:** {timestamp}
**Expert:** {expert_name}
**Target Models:** {', '.join(TARGET_MODELS)}
**Threshold:** 20+ percentage point differences between GPT-4.1 and Claude graders
**Total Cases Found:** {total_cases}
**Unique Questions Affected:** {unique_questions}

---

## Executive Summary

This report analyzes grading differences between GPT-4.1 and Claude graders for responses from specific AI models when using {expert_name}'s rubric. The analysis focuses on cases where the two graders differ by 20 or more percentage points.

### Summary Statistics

- **Total Cases Found:** {total_cases}
- **Unique Questions Affected:** {unique_questions}

### Breakdown by Model
"""
    
    for model in TARGET_MODELS:
        count = model_breakdown.get(model, 0)
        markdown_content += f"- **{model}:** {count} cases\n"
    
    markdown_content += f"""
### Questions with Multiple Model Differences

"""
    
    if multi_model_questions:
        markdown_content += f"**{len(multi_model_questions)} questions** show large differences across multiple models:\n"
        for q in sorted(multi_model_questions):
            models = sorted(differences[q]['models_with_differences'])
            markdown_content += f"- {q}: {', '.join(models)}\n"
    else:
        markdown_content += "No questions show large differences across multiple models.\n"
    
    markdown_content += f"""
---

## {expert_name}'s Rubric

### Rubric Criteria

"""
    
    # Show the expert's rubric
    if rubric_data and 'questions' in rubric_data:
        markdown_content += f"""### Rubric Criteria
"""
        # Note: This would show overall rubric structure if needed
        # For now, we'll include it in the detailed section per question
    
    markdown_content += f"""
---

## Detailed Analysis by Question

"""
    
    # Process each question with large differences
    for question_id in sorted(differences.keys()):
        question_data = differences[question_id]
        question_number = question_data['question_number']
        question_text = question_data['question_text']
        
        markdown_content += f"""### {question_id} (Question {question_number})

#### Original Question
{question_text}

"""
        
        # Show analysis for each model that has large differences for this question
        for model in sorted(question_data['model_analyses'].keys()):
            analysis = question_data['model_analyses'][model]
            
            model_response = analysis.get('model_response', 'Response not available')
            
            markdown_content += f"""#### {model} Response Analysis

**Grading Scores:**
- GPT-4.1 Grader: {analysis['gpt_grader_score']:.1f}% ({int(analysis['gpt_grader_score'] * sum(item['max_points'] for item in analysis['gpt_grader_details']) / 100)}/{sum(item['max_points'] for item in analysis['gpt_grader_details'])})
- Claude Grader: {analysis['claude_grader_score']:.1f}% ({int(analysis['claude_grader_score'] * sum(item['max_points'] for item in analysis['claude_grader_details']) / 100)}/{sum(item['max_points'] for item in analysis['claude_grader_details'])})
- **Difference: {analysis['difference']:.1f} percentage points**

**{model} Response:**
{model_response}

"""
            
            # Show detailed grading from both graders
            markdown_content += f"""**GPT-4.1 Grader Detailed Scores:**
"""
            for criterion in analysis['gpt_grader_details']:
                markdown_content += f"- **Criterion {criterion['criterion_number']}:** {criterion['score_given']}/{criterion['max_points']} points\n"
                markdown_content += f"  - *Description:* {criterion['description']}\n"
                markdown_content += f"  - *Justification:* {criterion['justification']}\n\n"
            
            markdown_content += f"""**Claude Grader Detailed Scores:**
"""
            for criterion in analysis['claude_grader_details']:
                markdown_content += f"- **Criterion {criterion['criterion_number']}:** {criterion['score_given']}/{criterion['max_points']} points\n"
                markdown_content += f"  - *Description:* {criterion['description']}\n"
                markdown_content += f"  - *Justification:* {criterion['justification']}\n\n"
            
            # Show key differences
            gpt_criteria = {c['criterion_number']: c for c in analysis['gpt_grader_details']}
            claude_criteria = {c['criterion_number']: c for c in analysis['claude_grader_details']}
            
            markdown_content += f"""**Key Scoring Differences for {model}:**
"""
            differences_found = False
            for crit_num in sorted(set(gpt_criteria.keys()) | set(claude_criteria.keys())):
                if crit_num in gpt_criteria and crit_num in claude_criteria:
                    gpt_score = gpt_criteria[crit_num]['score_given']
                    claude_score = claude_criteria[crit_num]['score_given']
                    max_points = gpt_criteria[crit_num]['max_points']
                    
                    if gpt_score != claude_score:
                        differences_found = True
                        markdown_content += f"- **Criterion {crit_num}:** GPT gave {gpt_score}/{max_points}, Claude gave {claude_score}/{max_points} (Difference: {gpt_score - claude_score:+.1f})\n"
            
            if not differences_found:
                markdown_content += "- No individual criterion differences found (may differ due to calculation methods)\n"
            
            markdown_content += "\n---\n\n"
    
    unique_question_ids = sorted(differences.keys())
    markdown_content += f"""
---

## Appendix: All Question IDs with Large Differences

{', '.join(unique_question_ids)}

---

*Report generated on {timestamp}*
*Analysis for {expert_name} rubric focused on {', '.join(TARGET_MODELS)}*
"""
    
    return markdown_content

def main():
    parser = argparse.ArgumentParser(description='Identify large grading differences between GPT and Claude')
    parser.add_argument('--gpt_expert1', type=str, required=True,
                       help='Path to GPT Expert1 grading results JSON')
    parser.add_argument('--gpt_expert2', type=str, required=True,
                       help='Path to GPT Expert2 grading results JSON')
    parser.add_argument('--claude_expert1', type=str, required=True,
                       help='Path to Claude Expert1 grading results JSON')
    parser.add_argument('--claude_expert2', type=str, required=True,
                       help='Path to Claude Expert2 grading results JSON')
    parser.add_argument('--expert1_rubric', type=str, required=True,
                       help='Path to Expert1 rubric JSON file')
    parser.add_argument('--expert2_rubric', type=str, required=True,
                       help='Path to Expert2 rubric JSON file')
    parser.add_argument('--openai_responses', type=str, required=True,
                       help='Path to OpenAI responses JSON file')
    parser.add_argument('--grok_responses', type=str, required=True,
                       help='Path to Grok responses JSON file')
    parser.add_argument('--opensource_responses', type=str, required=True,
                       help='Path to Opensource responses JSONL file')
    parser.add_argument('--fold', type=int, required=True,
                       help='Fold number for output naming')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=20.0,
                       help='Difference threshold (default: 20.0%)')
    
    args = parser.parse_args()
    
    print(f"🔍 Analyzing large differences for Fold {args.fold}")
    print(f"📊 Threshold: {args.threshold}%")
    
    # Load grading results
    print("Loading grading results...")
    
    with open(args.gpt_expert1, 'r') as f:
        gpt_expert1_scores = json.load(f)
    
    with open(args.gpt_expert2, 'r') as f:
        gpt_expert2_scores = json.load(f)
    
    with open(args.claude_expert1, 'r') as f:
        claude_expert1_scores = json.load(f)
    
    with open(args.claude_expert2, 'r') as f:
        claude_expert2_scores = json.load(f)
    
    # Load rubrics
    print("Loading rubrics...")
    expert1_rubric = load_rubric(args.expert1_rubric)
    expert2_rubric = load_rubric(args.expert2_rubric)
    
    # Load responses
    print("Loading model responses...")
    response_files = {
        'openai': args.openai_responses,
        'grok': args.grok_responses,
        'opensource': args.opensource_responses
    }
    responses_map = load_responses(response_files)
    
    # Analyze Expert1
    print("Analyzing Expert1 differences...")
    expert1_differences = find_large_differences(
        gpt_expert1_scores, 
        claude_expert1_scores, 
        args.threshold
    )
    
    # Analyze Expert2
    print("Analyzing Expert2 differences...")
    expert2_differences = find_large_differences(
        gpt_expert2_scores, 
        claude_expert2_scores, 
        args.threshold
    )
    
    # Populate question text from rubrics and model responses
    for question_id, data in expert1_differences.items():
        question_number = data['question_number']
        rubric_question = find_rubric_for_question(expert1_rubric, question_number)
        if rubric_question:
            data['question_text'] = rubric_question.get('question_text', 'Question text not found')
        
        # Populate model responses
        for model_name, analysis in data['model_analyses'].items():
            if question_number in responses_map:
                model_response = responses_map[question_number]['responses'].get(model_name, 'Response not available')
                analysis['model_response'] = model_response
    
    for question_id, data in expert2_differences.items():
        question_number = data['question_number']
        rubric_question = find_rubric_for_question(expert2_rubric, question_number)
        if rubric_question:
            data['question_text'] = rubric_question.get('question_text', 'Question text not found')
        
        # Populate model responses
        for model_name, analysis in data['model_analyses'].items():
            if question_number in responses_map:
                model_response = responses_map[question_number]['responses'].get(model_name, 'Response not available')
                analysis['model_response'] = model_response
    
    # Generate reports
    print("Generating reports...")
    expert1_report = generate_comprehensive_report("Expert1", expert1_differences, expert1_rubric)
    expert2_report = generate_comprehensive_report("Expert2", expert2_differences, expert2_rubric)
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed data
    expert1_data_file = f"{args.output_dir}/fold{args.fold}_large_differences_expert1_data.json"
    expert2_data_file = f"{args.output_dir}/fold{args.fold}_large_differences_expert2_data.json"
    
    expert1_data = {
        'analysis_info': {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'expert': 'Expert1',
            'target_models': TARGET_MODELS,
            'threshold': args.threshold,
            'total_cases': sum(len(data['models_with_differences']) for data in expert1_differences.values()),
            'unique_questions': len(expert1_differences)
        },
        'rubric': expert1_rubric,
        'detailed_results': expert1_differences
    }
    
    expert2_data = {
        'analysis_info': {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'expert': 'Expert2',
            'target_models': TARGET_MODELS,
            'threshold': args.threshold,
            'total_cases': sum(len(data['models_with_differences']) for data in expert2_differences.values()),
            'unique_questions': len(expert2_differences)
        },
        'rubric': expert2_rubric,
        'detailed_results': expert2_differences
    }
    
    with open(expert1_data_file, 'w') as f:
        json.dump(expert1_data, f, indent=2)
    
    with open(expert2_data_file, 'w') as f:
        json.dump(expert2_data, f, indent=2)
    
    # Save reports
    expert1_report_file = f"{args.output_dir}/fold{args.fold}_large_differences_expert1_report.md"
    expert2_report_file = f"{args.output_dir}/fold{args.fold}_large_differences_expert2_report.md"
    
    with open(expert1_report_file, 'w') as f:
        f.write(expert1_report)
    
    with open(expert2_report_file, 'w') as f:
        f.write(expert2_report)
    
    print(f"✅ Analysis completed!")
    print(f"📄 Expert1 data: {expert1_data_file}")
    print(f"📄 Expert1 report: {expert1_report_file}")
    print(f"📄 Expert2 data: {expert2_data_file}")
    print(f"📄 Expert2 report: {expert2_report_file}")
    print(f"📊 Expert1: {len(expert1_differences)} questions with large differences")
    print(f"📊 Expert2: {len(expert2_differences)} questions with large differences")

if __name__ == "__main__":
    main()

# Example usage:
# python targeted_large_differences.py \
#   --gpt_expert1 "fold2_gpt_scores_expert1.json" \
#   --gpt_expert2 "fold2_gpt_scores_expert2.json" \
#   --claude_expert1 "fold2_claude_scores_expert1.json" \
#   --claude_expert2 "fold2_claude_scores_expert2.json" \
#   --expert1_rubric "Fold2-Jesse.json" \
#   --expert2_rubric "Fold2-Simone.json" \
#   --openai_responses "openai_family_response.jsonl" \
#   --grok_responses "grok_response.jsonl" \
#   --opensource_responses "all_opensource_models_final_outputs_temp0.7.jsonl" \
#   --fold 2 \
#   --output_dir "./res_fold2"
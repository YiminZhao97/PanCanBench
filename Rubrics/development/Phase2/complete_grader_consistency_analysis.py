#!/usr/bin/env python3
"""
Complete analysis of grading consistency between GPT-4.1 and Claude graders.
Analyzes 3 targeted models: ['gpt-4o', 'grok-4-latest', 'meta-llama_Llama-3.1-70B-Instruct']
Updated for new data structure with expert1/expert2 format.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import argparse

# Target models for analysis
TARGET_MODELS = ['gpt-4o', 'grok-4-latest', 'meta-llama_Llama-3.1-70B-Instruct']

def load_scores(file_path):
    """Load scores from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

def extract_scores_by_model_new_format(scores_data):
    """Extract scores organized by model (source) from new format files."""
    model_scores = {}
    
    for entry in scores_data:
        try:
            total_score = sum([criterion['score_given'] for criterion in entry['criterion_scores']])
            max_score = sum([criterion['max_points'] for criterion in entry['criterion_scores']])
            percentage = (total_score / max_score * 100) if max_score > 0 else 0
            
            question_id = f"Q{entry.get('question_number', '')}"
            source = entry.get('source', 'unknown')
            
            # Only keep target models
            if source in TARGET_MODELS:
                if source not in model_scores:
                    model_scores[source] = []
                
                model_scores[source].append({
                    'question_id': question_id,
                    'total_score': total_score,
                    'max_score': max_score,
                    'percentage': percentage
                })
            
        except (KeyError, TypeError) as e:
            continue  # Skip problematic entries silently
    
    return model_scores

def load_model_data_for_expert_rubric(gpt_grader_file, claude_grader_file):
    """Load data for both graders using the same expert's rubric."""
    
    # Load GPT grader data (GPT grading with expert's rubric)
    gpt_data = load_scores(gpt_grader_file)
    all_gpt_grader_data = {}
    if gpt_data:
        gpt_by_model = extract_scores_by_model_new_format(gpt_data)
        # Only keep target models
        for model in TARGET_MODELS:
            if model in gpt_by_model:
                all_gpt_grader_data[model] = gpt_by_model[model]
    
    # Load Claude grader data (Claude grading with same expert's rubric)
    claude_data = load_scores(claude_grader_file)
    claude_grader_data = extract_scores_by_model_new_format(claude_data) if claude_data else {}
    
    return all_gpt_grader_data, claude_grader_data

def calculate_correlations(gpt_grader_data, claude_grader_data):
    """Calculate correlations for all common models."""
    common_models = set(gpt_grader_data.keys()) & set(claude_grader_data.keys())
    
    results = {}
    
    for model in common_models:
        gpt_scores = gpt_grader_data[model]
        claude_scores = claude_grader_data[model]
        
        # Create mappings by question_id for matching
        gpt_map = {score['question_id']: score['percentage'] for score in gpt_scores}
        claude_map = {score['question_id']: score['percentage'] for score in claude_scores}
        
        # Find common questions
        common_questions = set(gpt_map.keys()) & set(claude_map.keys())
        
        if len(common_questions) > 1:
            gpt_values = [gpt_map[q] for q in sorted(common_questions)]
            claude_values = [claude_map[q] for q in sorted(common_questions)]
            
            correlation, p_value = pearsonr(gpt_values, claude_values)
            
            results[model] = {
                'gpt_scores': gpt_values,
                'claude_scores': claude_values,
                'correlation': correlation,
                'p_value': p_value,
                'n': len(gpt_values)
            }
    
    return results

def create_targeted_plot(output_dir, output_prefix, results):
    """Create 1x3 subplot layout for the 3 target models."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Model display names
    model_display_names = {
        'gpt-4o': 'GPT-4o',
        'grok-4-latest': 'Grok-4-Latest',
        'meta-llama_Llama-3.1-70B-Instruct': 'Llama-3.1-70B'
    }
    
    for idx, model in enumerate(TARGET_MODELS):
        ax = axes[idx]
        
        if model in results:
            data = results[model]
            gpt_scores = data['gpt_scores']
            claude_scores = data['claude_scores']
            correlation = data['correlation']
            n_points = data['n']
            
            # Scatter plot
            ax.scatter(gpt_scores, claude_scores, alpha=0.6, color=colors[idx], s=40)
            
            # Diagonal line (perfect agreement)
            if gpt_scores and claude_scores:
                max_val = max(max(gpt_scores), max(claude_scores))
                min_val = min(min(gpt_scores), min(claude_scores))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
                ax.set_xlim(min_val - 5, max_val + 5)
                ax.set_ylim(min_val - 5, max_val + 5)
            
            display_name = model_display_names.get(model, model)
            ax.set_title(f'{display_name}\nr = {correlation:.3f}, n = {n_points}', fontsize=12)
        else:
            display_name = model_display_names.get(model, model)
            ax.set_title(f'{display_name}\n(No data)', fontsize=12)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('GPT-4.1 Grader (%)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Claude Grader (%)', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Grader Consistency Analysis\nGPT-4.1 vs Claude Graders (Targeted Models)', fontsize=14)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    output_path = os.path.join(output_dir, f'{output_prefix}_targeted_grader_consistency.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def print_correlation_summary(results):
    """Print a summary of correlations for target models."""
    print(f"\n=== Targeted Models Correlation Summary ===")
    
    for model in TARGET_MODELS:
        if model in results:
            data = results[model]
            short_name = model.replace('meta-llama_Llama-3.1-70B-Instruct', 'Llama-3.1-70B')
            print(f"  {short_name:20}: r = {data['correlation']:.3f}, p = {data['p_value']:.6f}, n = {data['n']}")
        else:
            short_name = model.replace('meta-llama_Llama-3.1-70B-Instruct', 'Llama-3.1-70B')
            print(f"  {short_name:20}: No data")

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Targeted grading consistency analysis between GPT-4.1 and Claude graders using same expert rubric')
    parser.add_argument('--gpt_grader_file', type=str, required=True,
                       help='Path to GPT grader scores file (GPT grading using expert rubric)')
    parser.add_argument('--claude_grader_file', type=str, required=True,
                       help='Path to Claude grader scores file (Claude grading using same expert rubric)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for generated files')
    parser.add_argument('--output_prefix', type=str, required=True,
                       help='Output prefix for generated files')
    
    args = parser.parse_args()
    
    print("Targeted Grading Consistency Analysis")
    print("GPT-4.1 vs Claude Graders - Target Models Only")
    print("Comparing graders using the same expert-designed rubric")
    print(f"Target models: {TARGET_MODELS}")
    print("=" * 60)
    
    print(f"Loading GPT grader data from: {args.gpt_grader_file}")
    print(f"Loading Claude grader data from: {args.claude_grader_file}")
    
    # Load model data for the specified grader files (both using same expert rubric)
    gpt_grader_data, claude_grader_data = load_model_data_for_expert_rubric(args.gpt_grader_file, args.claude_grader_file)
    
    print(f"GPT grader models found: {list(gpt_grader_data.keys())}")
    print(f"Claude grader models found: {list(claude_grader_data.keys())}")
    
    # Calculate correlations
    results = calculate_correlations(gpt_grader_data, claude_grader_data)
    
    # Print summary
    print_correlation_summary(results)
    
    # Create targeted plot
    create_targeted_plot(args.output_dir, args.output_prefix, results)
    
    print("\n" + "=" * 60)
    print("Targeted analysis complete!")
    print(f"Generated plot: {os.path.join(args.output_dir, args.output_prefix)}_targeted_grader_consistency.png")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate AI prompts to polish rubrics based on grading disagreements.
Takes targeted large differences JSON data and creates prompts for rubric improvement.

Usage: python rubric_polishing_generator.py --expert_files file1.json file2.json --output_dir ./output
"""

import json
import os
import argparse
from datetime import datetime

def load_expert_data(json_file_path):
    """Load the expert analysis JSON data."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def find_rubric_for_question(rubric_data, question_number):
    """Find the rubric items for a specific question."""
    if not rubric_data or 'questions' not in rubric_data:
        return None
    
    for question in rubric_data['questions']:
        if question.get('question_number') == question_number:
            return question
    return None

def format_grader_details(grader_details, grader_name):
    """Format grader details for the prompt."""
    details_text = f"{grader_name}:\n"
    
    total_score = sum(item.get('score_given', 0) for item in grader_details)
    max_score = sum(item.get('max_points', 0) for item in grader_details)
    details_text += f"Total score: {total_score} / {max_score}\n\n"
    details_text += "Per-item decisions and justifications:\n"
    
    for item in grader_details:
        item_num = item.get('criterion_number', 'Unknown')
        score_given = item.get('score_given', 0)
        max_points = item.get('max_points', 0)
        justification = item.get('justification', 'No justification provided')
        
        details_text += f"- Item {item_num}: {score_given}/{max_points} points\n"
        details_text += f"  Justification: {justification}\n\n"
    
    return details_text

def format_rubric_items(rubric_items):
    """Format rubric items for the prompt."""
    if not rubric_items:
        return "No rubric items found", 0
    
    rubric_text = ""
    total_points = 0
    
    for item in rubric_items:
        item_num = item.get('item_number', 'Unknown')
        description = item.get('description', 'No description')
        max_points = item.get('max_points', 0)
        total_points += max_points
        
        rubric_text += f"{item_num}. {description} ({max_points} points)\n"
    
    return rubric_text, total_points

def generate_polishing_prompt(question_data, rubric_question, detailed_results):
    """Generate a polishing prompt for a specific question."""
    
    question_number = question_data['question_number']
    question_text = question_data['question_text']
    
    # Format rubric items
    rubric_text, total_points = format_rubric_items(rubric_question.get('rubric_items', []))
    
    # Get the first model analysis (they should all have the same grading disagreement)
    first_model = list(question_data['model_analyses'].keys())[0]
    model_analysis = question_data['model_analyses'][first_model]
    
    # Format grader details
    gpt_details = format_grader_details(model_analysis['gpt_grader_details'], "GPT-4.1 Grader")
    claude_details = format_grader_details(model_analysis['claude_grader_details'], "Claude Grader")
    
    # Generate the prompt
    prompt = f"""Polish the rubric using two graders' judgments to reduce ambiguity and enforce binary scoring.

Question ID: Q{question_number}
Question (for context): {question_text}

Original Rubric (with points):
{rubric_text}

{gpt_details}

{claude_details}

Model Response (for context):
{model_analysis.get('model_response', 'Response not available')}

Grading Score Difference: {model_analysis['difference']:.1f} percentage points
(GPT-4.1: {model_analysis['gpt_grader_score']:.1f}% vs Claude: {model_analysis['claude_grader_score']:.1f}%)

Constraints & preferences:
- Keep total points at {total_points}.
- Binary items only (pass/fail).
- Prefer minimal rewording; clarify with explicit thresholds and synonym lists.
- Do not introduce new clinical claims beyond the original rubric's intent.
- If an item must be split/merged, preserve total weight and provide an item-mapping note.

Deliverables:
Concise revised rubric (binary) — Markdown table with columns: Item ID | Rubric Item | Points
    - Start directly with item 1 (do not include a header row with "#" or example text)
    - Each row should be: | 1 | [actual rubric item description] | [points] |
    - Item IDs should be sequential numbers starting from 1

IMPORTANT: In your markdown table, do NOT include a header row with "#" or placeholder text. Start directly with item 1.

Please provide your response in structured markdown format.
"""
    
    return prompt

def create_polishing_reports(data, expert_name):
    """Create polishing prompts and summary files."""
    
    if not data or 'detailed_results' not in data:
        print("No detailed results found in data")
        return {}, ""
    
    rubric_data = data.get('rubric', {})
    detailed_results = data['detailed_results']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare outputs
    prompts_json = {
        'metadata': {
            'expert': expert_name,
            'timestamp': timestamp,
            'total_questions': len(detailed_results),
            'source_file_info': data.get('analysis_info', {})
        },
        'polishing_prompts': {}
    }
    
    markdown_content = f"""# Rubric Polishing Prompts - {expert_name}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Expert:** {expert_name}
**Total Questions Requiring Polish:** {len(detailed_results)}

---

"""
    
    # Process each question
    for question_id, question_data in detailed_results.items():
        question_number = question_data['question_number']
        
        # Find corresponding rubric
        rubric_question = find_rubric_for_question(rubric_data, question_number)
        
        if rubric_question:
            # Generate polishing prompt
            prompt = generate_polishing_prompt(question_data, rubric_question, detailed_results)
            
            # Add to JSON
            prompts_json['polishing_prompts'][question_id] = {
                'question_number': question_number,
                'question_text': question_data['question_text'],
                'models_with_differences': question_data['models_with_differences'],
                'prompt': prompt,
                'original_rubric_items': rubric_question.get('rubric_items', [])
            }
            
            # Add to Markdown
            markdown_content += f"""## {question_id}

**Question:** {question_data['question_text']}
**Models with Differences:** {', '.join(question_data['models_with_differences'])}

### Polishing Prompt:

```
{prompt}
```

---

"""
        else:
            print(f"Warning: No rubric found for question {question_number}")
    
    # Generate markdown table summary
    markdown_content += f"""
## Summary Table

| Question ID | Question Text (Truncated) | Models with Differences | Max Points |
|-------------|---------------------------|------------------------|------------|
"""
    
    for question_id, prompt_data in prompts_json['polishing_prompts'].items():
        question_text = prompt_data['question_text'][:60] + "..." if len(prompt_data['question_text']) > 60 else prompt_data['question_text']
        models = ', '.join(prompt_data['models_with_differences'])
        total_points = sum(item.get('max_points', 0) for item in prompt_data['original_rubric_items'])
        
        markdown_content += f"| {question_id} | {question_text} | {models} | {total_points} |\n"
    
    markdown_content += f"""
---

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {expert_name}*
*Use these prompts with AI systems to polish rubrics and reduce grading ambiguity*
"""
    
    return prompts_json, markdown_content

def main():
    """Process expert files and generate polishing prompts."""
    
    parser = argparse.ArgumentParser(description='Generate AI prompts to polish rubrics based on grading disagreements')
    parser.add_argument('--expert_files', type=str, nargs='+', required=True,
                       help='Expert analysis JSON files (e.g., targeted_large_differences_expert1_data.json)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for generated files (default: current directory)')
    
    args = parser.parse_args()
    
    print("Rubric Polishing Prompt Generator")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
    
    for expert_file in args.expert_files:
        if not os.path.exists(expert_file):
            print(f"⚠️  File not found: {expert_file}")
            continue
            
        # Extract expert name from filename
        basename = os.path.basename(expert_file)
        if 'expert1' in basename.lower():
            expert_name = 'Expert1'
        elif 'expert2' in basename.lower():
            expert_name = 'Expert2'
        else:
            # Try to extract from filename
            import re
            match = re.search(r'expert(\d+)', basename.lower())
            if match:
                expert_name = f'Expert{match.group(1)}'
            else:
                expert_name = os.path.splitext(basename)[0]
        
        print(f"\nProcessing {expert_name} from {expert_file}...")
        
        # Load data
        data = load_expert_data(expert_file)
        if not data:
            print(f"Failed to load data for {expert_name}")
            continue
        
        # Generate reports
        prompts_json, markdown_content = create_polishing_reports(data, expert_name)
        
        # Save JSON file
        json_output_file = os.path.join(args.output_dir, f'rubric_polishing_prompts_{expert_name.lower()}.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(prompts_json, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON prompts: {json_output_file}")
        
        # Save Markdown file
        md_output_file = os.path.join(args.output_dir, f'rubric_polishing_prompts_{expert_name.lower()}.md')
        with open(md_output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Saved Markdown report: {md_output_file}")
        
        # Print summary
        total_questions = len(prompts_json['polishing_prompts'])
        print(f"Generated {total_questions} polishing prompts for {expert_name}")
    
    print(f"\n{'='*50}")
    print("Polishing prompt generation complete!")
    print("\nNext steps:")
    print("1. Use the JSON files programmatically with AI APIs")
    print("2. Use the Markdown files for manual AI prompting")
    print("3. Apply the polished rubrics to reduce grading disagreements")

if __name__ == "__main__":
    main()

# Example usage:
# python rubric_polishing_generator.py --expert_files targeted_large_differences_expert1_data.json targeted_large_differences_expert2_data.json --output_dir ./polishing_prompts
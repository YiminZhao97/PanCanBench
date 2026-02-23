"""
Script to generate synthetic rubrics using GPT-4 API with prompt caching.
This script will:
1. Load the rubrics design guideline (cached)
2. Load questions and responses from 4 models
3. Generate rubrics for each question using GPT-4
4. Save results to JSONL files
"""

import json
import time
import argparse
from pathlib import Path
from openai import OpenAI
from datetime import datetime

def load_guideline(guideline_file):
    """Load rubrics design guideline from text file."""
    with open(guideline_file, 'r', encoding='utf-8') as f:
        return f.read()

def load_merged_responses(responses_file):
    """Load merged responses from JSONL file."""
    data = []
    with open(responses_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def create_rubric_prompt(guideline, question, responses, examples_context=""):
    """Create the prompt for rubric generation."""

    # Format responses for display
    responses_text = ""
    for model_name, response_text in responses.items():
        responses_text += f"\n\n### Response from {model_name}:\n{response_text}"

    prompt = f"""You are an expert in designing evaluation rubrics for AI-generated responses to pancreatic cancer patient questions.

Below is the rubric design guideline that you must follow:

{guideline}

---

Now, please design a scoring rubric for the following question. You will also see responses from 4 different AI models to help you understand what kind of information might be included in responses.

**Question:** {question}

**Sample Responses from AI Models:**
{responses_text}

---

**Your Task:**

Design a comprehensive scoring rubric for evaluating responses to this question. Follow all the guidelines provided above. Remember:

1. Each rubric item should be a complete sentence describing what should be included or avoided
2. Assign points (0-10) to each item based on importance
3. Use negative points (-10 to 0) for incorrect or misleading content
4. Be specific to this question - avoid generic rubrics
5. Make rubrics that can be applied by someone without deep medical expertise
6. The rubric should catch outdated or incorrect medical information

**Output Format:**

Please provide the rubric in the following JSON format:

{{
  "question_id": "[question_id]",
  "question": "[the question text]",
  "rubric_items": [
    {{
      "item_description": "[Complete sentence describing what should be included/avoided]",
      "points": [number from -10 to 10],
      "reasoning": "[Brief explanation of why this item is important]"
    }},
    ...
  ],
  "total_possible_points": [sum of all positive points],
  "notes": "[Any additional notes about applying this rubric]"
}}

Provide ONLY the JSON output, no additional text.
"""
    return prompt

def generate_rubric(client, guideline, question_id, question, responses, run_number, retry_count=3):
    """Generate rubric using GPT-4 API with prompt caching."""

    prompt = create_rubric_prompt(guideline, question, responses)

    for attempt in range(retry_count):
        try:
            # Use GPT-4 with prompt caching
            # The guideline will be cached for efficiency
            response = client.chat.completions.create(
                model="gpt-4.1",  # Use gpt-4-turbo-preview or gpt-4
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in designing evaluation rubrics for medical AI responses. You always output valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )

            # Parse the response
            rubric_json = json.loads(response.choices[0].message.content)

            # Add metadata
            rubric_json['question_id'] = question_id
            rubric_json['question'] = question
            rubric_json['run_number'] = run_number
            rubric_json['generated_at'] = datetime.now().isoformat()
            rubric_json['model_used'] = response.model

            return rubric_json

        except json.JSONDecodeError as e:
            print(f"  JSON decode error on attempt {attempt + 1}: {e}")
            if attempt < retry_count - 1:
                time.sleep(2)
                continue
            else:
                return {
                    "error": f"Failed to parse JSON after {retry_count} attempts",
                    "question_id": question_id,
                    "question": question,
                    "run_number": run_number
                }
        except Exception as e:
            print(f"  Error on attempt {attempt + 1}: {e}")
            if attempt < retry_count - 1:
                time.sleep(5)
                continue
            else:
                return {
                    "error": str(e),
                    "question_id": question_id,
                    "question": question,
                    "run_number": run_number
                }

def main(api_key, base_dir, run_number=1, start_from=1, end_at=None):
    """
    Main function to generate rubrics.

    Args:
        api_key: OpenAI API key
        base_dir: Base directory containing input files
        run_number: Which run this is (1 or 2)
        start_from: Question number to start from (1-282)
        end_at: Question number to end at (None means process all remaining)
    """

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    base_dir = Path(base_dir)

    # Load guideline
    print(f"Loading rubrics design guideline...")
    guideline = load_guideline(base_dir / 'rubrics_design_guideline.txt')
    print(f"Guideline loaded ({len(guideline)} characters)")

    # Load merged responses
    print(f"Loading merged responses...")
    merged_data = load_merged_responses(base_dir / 'merged_4_models_responses.jsonl')
    print(f"Loaded {len(merged_data)} questions with responses")

    # Setup output file
    output_file = base_dir / f'synthetic_rubrics_run{run_number}.jsonl'

    # If continuing from a previous run, load existing rubrics
    existing_question_ids = set()
    if output_file.exists() and start_from > 1:
        print(f"Loading existing rubrics from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rubric = json.loads(line)
                    existing_question_ids.add(rubric['question_id'])
        print(f"Found {len(existing_question_ids)} existing rubrics")

    # Determine which questions to process
    questions_to_process = []
    for idx, item in enumerate(merged_data, 1):
        if idx < start_from:
            continue
        if end_at and idx > end_at:
            break
        if item['question_id'] not in existing_question_ids:
            questions_to_process.append((idx, item))

    print(f"\n{'='*80}")
    print(f"Starting rubric generation - Run {run_number}")
    print(f"Questions to process: {len(questions_to_process)}")
    print(f"Output file: {output_file}")
    print(f"{'='*80}\n")

    # Open output file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        for idx, (question_num, item) in enumerate(questions_to_process, 1):
            question_id = item['question_id']
            question = item['question']
            responses = item['responses']

            print(f"[{idx}/{len(questions_to_process)}] Processing {question_id}...")
            print(f"  Question: {question[:100]}...")

            # Generate rubric
            rubric = generate_rubric(
                client=client,
                guideline=guideline,
                question_id=question_id,
                question=question,
                responses=responses,
                run_number=run_number
            )

            # Save to file
            f.write(json.dumps(rubric, ensure_ascii=False) + '\n')
            f.flush()  # Ensure it's written immediately

            if 'error' in rubric:
                print(f"  ERROR: {rubric['error']}")
            else:
                num_items = len(rubric.get('rubric_items', []))
                total_points = rubric.get('total_possible_points', 0)
                print(f"  Generated rubric with {num_items} items ({total_points} total points)")

            # Rate limiting - be nice to the API
            time.sleep(1)

            # Progress checkpoint every 20 questions
            if idx % 20 == 0:
                print(f"\n{'='*80}")
                print(f"Progress checkpoint: {idx}/{len(questions_to_process)} questions completed")
                print(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print(f"Rubric generation completed - Run {run_number}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic rubrics using GPT-4 API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python generate_rubrics.py --api-key YOUR_API_KEY --base-dir /path/to/data
  python generate_rubrics.py --api-key YOUR_API_KEY --base-dir /path/to/data --run 2 --start 50 --end 100
        '''
    )

    parser.add_argument(
        '--api-key',
        required=True,
        help='OpenAI API key (required)'
    )

    parser.add_argument(
        '--base-dir',
        required=True,
        help='Base directory containing input files (rubrics_design_guideline.txt and merged_4_models_responses.jsonl)'
    )

    parser.add_argument(
        '--run',
        type=int,
        default=1,
        help='Run number (default: 1)'
    )

    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Question number to start from (default: 1)'
    )

    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Question number to end at (default: process all remaining)'
    )

    args = parser.parse_args()

    print(f"Starting rubric generation:")
    print(f"  Base directory: {args.base_dir}")
    print(f"  Run number: {args.run}")
    print(f"  Start from: Question {args.start}")
    print(f"  End at: {'Question ' + str(args.end) if args.end else 'All remaining'}")
    print()

    main(
        api_key=args.api_key,
        base_dir=args.base_dir,
        run_number=args.run,
        start_from=args.start,
        end_at=args.end
    )

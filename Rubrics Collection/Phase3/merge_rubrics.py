import json
import openai
import os
import argparse
from typing import Dict, List, Any

def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file and return parsed data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_rubrics_with_gpt(rubric_a: List[Dict[str, Any]], rubric_b: List[Dict[str, Any]], model: str = "gpt-5") -> str:
    """Use GPT to merge two rubrics according to the specified prompt."""

    # Set up OpenAI client (assumes API key is set in environment)
    client = openai.OpenAI()

    all_merged_results = []

    # Process each question pair
    for i in range(min(len(rubric_a), len(rubric_b))):
        question_a = rubric_a[i]
        question_b = rubric_b[i]

        # Extract question text and rubric items
        question_text = question_a.get('question_text', '')
        question_number = question_a.get('question_number', i+1)

        rubric_items_a = question_a.get('rubric_items', [])
        rubric_items_b = question_b.get('rubric_items', [])

        system_prompt = """You are a rubric merger and mapper.

TASK
For the question below, you will:
1) Polish the two input rubrics (A and B) for clarity/grammar ONLY; do NOT add any new medical content.
2) Identify concept overlaps between A and B. If a single rubric item is compound (e.g., lists multiple goals), SPLIT it into multiple binary sub-items so each table row is one clear, yes/no statement—BUT keep the original item's ID as the "Origin".
3) Produce ONE consolidated table in this exact format and column order:

### {question_id}
**Question:** {question}

| #  | Rubric item (full sentence; binary) | Origin | Points |
| -- | ----------------------------------- | ------ | ------ |
<rows here>

---

— "Origin" lists all source IDs that map to the merged row (e.g., A1, B2).
— "Points" lists each source's original points (e.g., "A1: 3; B2: 10"). If only one rubric contributes, list only that one.
— Order rows as: (i) common items first (definition/timing, before surgery, chemo, radiation), then (ii) unique to A, then (iii) unique to B.
— Number rows starting at 1 for each question.
— Do NOT include any extra commentary before or after the table.

RULES (very important)
- Do NOT add new concepts beyond what appears in the inputs.
- Do NOT change the original point values.
- Rewrite items as full sentences and binary (yes/no) while preserving meaning.
- If two items are semantically the same, MERGE them into one row and list both Origins and both Points.
- If one source's single item expresses multiple distinct concepts, SPLIT them into separate rows; each split row keeps the same Origin ID.
- Keep medical wording neutral and accurate; no guideline citations; no browsing.
Output must be deterministic and formatted as Markdown tables.


ORDERING (change only row order; keep text, Origin, and Points identical)
- Present rows in this exact sequence:
  1) Common items — rows whose Origin includes at least one A* and at least one B*.
  2) A-specific items — rows whose Origin includes only A*.
  3) B-specific items — rows whose Origin includes only B*.
- Within each block, keep a deterministic order by the smallest contributing source ID (e.g., A1 before A3; B1 before B3). Do not alter wording, Origin, or Points.
- After reordering, renumber the "#" column sequentially starting at 1.
"""

        # Prepare user message with question and rubric items
        user_message = f"""Question {question_number}: {question_text}

RUBRIC A items:
{json.dumps(rubric_items_a, indent=2)}

RUBRIC B items:
{json.dumps(rubric_items_b, indent=2)}

Please merge these rubrics following the exact format and rules specified."""

        try:
            response_obj = client.responses.create(
                model=model,
                input=system_prompt
                )
            result = json.loads(response_obj.output[1].content[0].text)
            all_merged_results.append(result)
            print(f"Processed question {question_number}")

        except Exception as e:
            print(f"Error processing question {question_number}: {e}")
            continue

    return "\n\n".join(all_merged_results)

def main():
    """Main function to load rubrics and merge them."""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Merge two rubric JSON files using OpenAI GPT API.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python merge_rubrics.py \\
    --rubric-a /path/to/rubric_a.json \\
    --rubric-b /path/to/rubric_b.json \\
    --output /path/to/merged_rubrics.md \\
    --api-key sk-your-api-key-here

Or using environment variable for API key:
  export OPENAI_API_KEY='sk-your-api-key-here'
  python merge_rubrics.py \\
    --rubric-a /path/to/rubric_a.json \\
    --rubric-b /path/to/rubric_b.json \\
    --output /path/to/merged_rubrics.md
        """
    )

    parser.add_argument(
        '--rubric-a',
        required=True,
        help='Path to the first rubric JSON file (Rubric A)'
    )

    parser.add_argument(
        '--rubric-b',
        required=True,
        help='Path to the second rubric JSON file (Rubric B)'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Path to save the merged rubrics output (Markdown format)'
    )

    parser.add_argument(
        '--api-key',
        help='OpenAI API key (alternatively, set OPENAI_API_KEY environment variable)'
    )

    parser.add_argument(
        '--model',
        default='gpt-5',
        help='OpenAI model to use'
    )

    args = parser.parse_args()

    # Set API key from argument or environment variable
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not provided")
        print("Please either:")
        print("  1. Set environment variable: export OPENAI_API_KEY='your-api-key-here'")
        print("  2. Use --api-key argument: --api-key sk-your-api-key-here")
        return 1

    try:
        # Load the JSON files
        print(f"Loading rubrics...")
        print(f"  Rubric A: {args.rubric_a}")
        print(f"  Rubric B: {args.rubric_b}")

        rubric_a = load_json_file(args.rubric_a)['questions']
        rubric_b = load_json_file(args.rubric_b)['questions']

        print(f"Loaded {len(rubric_a)} questions from Rubric A")
        print(f"Loaded {len(rubric_b)} questions from Rubric B")

        # Merge rubrics using GPT
        print(f"Merging rubrics with {args.model}...")
        merged_result = merge_rubrics_with_gpt(rubric_a, rubric_b, model=args.model)

        if merged_result:
            # Save the merged result
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(merged_result)

            print(f"✓ Merged rubrics saved to: {args.output}")
            return 0
        else:
            print("✗ Failed to merge rubrics")
            return 1

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

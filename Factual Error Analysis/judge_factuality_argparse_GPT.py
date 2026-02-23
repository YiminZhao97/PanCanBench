import json
import os
import argparse
from openai import OpenAI

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Judge factuality of atomic claims using GPT-5')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file with atomic claims')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file for factuality judgment results')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (optional, will use OPENAI_API_KEY env var if not provided)')

    args = parser.parse_args()

    # Set up API key
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load the atomic claims results
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Define the judging prompt
    JUDGE_PROMPT_TEMPLATE = """## Task Definition
You are a medical expert judge tasked with evaluating whether atomic claims contain factuality errors. An atomic claim has a factuality error if it contains information that is medically inaccurate, misleading, or contradicts established medical knowledge.

## Context
- Question: {question}

## Instructions
1. Evaluate each atomic claim for factuality errors
2. Consider the context of the question
3. A claim has a factuality error (True) if it:
   - Contains medically incorrect information
   - Makes false assertions about treatments, medications, or medical conditions
   - Provides incorrect dosages, recommendations, or medical facts

4. A claim has no factuality error (False) if it:
   - Contains medically accurate information
   - Makes correct assertions based on established medical knowledge

## Output Format
Return a JSON object with the following structure:
{{
  "evaluations": [
    {{
      "claim": "the atomic claim text",
      "has_error": true or false,
      "reasoning": "brief explanation of why this claim has or does not have a factuality error"
    }}
  ]
}}

## Atomic Claims to Evaluate
{claims_json}

Return only the JSON object. No additional explanation needed."""

    # Process each entry
    results = []

    for idx, entry in enumerate(data):
        print(f"Processing entry {idx + 1}/{len(data)}...")

        # Skip entries with errors or no claims
        if 'error' in entry or not entry.get('atomic_claims'):
            print(f"  - Skipping entry due to error or no claims")
            continue

        try:
            # Prepare claims as JSON string
            claims_list = entry['atomic_claims']
            claims_json = json.dumps({"claims": claims_list}, indent=2)

            # Create the judging prompt
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=entry['question'],
                claims_json=claims_json
            )

            # Call OpenAI API for judgment
            response = client.responses.create(
                model="gpt-5",
                tools=[{"type": "web_search"}],
                input=prompt
            )

            judgment_json = json.loads(response.output_text)
    
            # Create result entry with simplified format
            claims_with_errors = {}
            for evaluation in judgment_json.get('evaluations', []):
                claim = evaluation.get('claim', '')
                has_error = evaluation.get('has_error', False)
                reasoning = evaluation.get('reasoning', '')
                claims_with_errors[claim] = {
                    "has_error": has_error,
                    "reasoning": reasoning
                }

            result_entry = {
                "question_id": entry['question_id'],
                "question": entry['question'],
                "model": entry['model'],
                "atomic_claims_evaluation": claims_with_errors
            }

            results.append(result_entry)

            # Count errors
            error_count = sum(1 for v in claims_with_errors.values() if v['has_error'])
            print(f"  - Evaluated {len(claims_with_errors)} claims, found {error_count} with errors")

        except Exception as e:
            print(f"  - Error during judgment: {str(e)}")
            result_entry = {
                "question_id": entry['question_id'],
                "question": entry['question'],
                "model": entry['model'],
                "atomic_claims_evaluation": {},
                "error": str(e)
            }
            results.append(result_entry)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nJudgment complete! Results saved to: {args.output}")
    print(f"Total entries judged: {len(results)}")

    # Print summary statistics
    total_claims = sum(len(r.get('atomic_claims_evaluation', {})) for r in results)
    total_errors = sum(
        sum(1 for v in r.get('atomic_claims_evaluation', {}).values() if v.get('has_error', False))
        for r in results
    )
    print(f"Total atomic claims evaluated: {total_claims}")
    print(f"Total claims with factuality errors: {total_errors}")
    if total_claims > 0:
        print(f"Error rate: {total_errors/total_claims*100:.2f}%")

if __name__ == "__main__":
    main()

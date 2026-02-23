import json
import os
import argparse
from google import genai
from google.genai.errors import APIError
from google.genai import types

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Judge factuality of atomic claims using Gemini')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file with atomic claims')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file for factuality judgment results')
    parser.add_argument('--api-key', type=str, default=None,
                        help='GEMINI API key')

    args = parser.parse_args()

    # Set up API key
    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key

    # Initialize Gemini client
    client = genai.Client() 

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
        # Use question_id for more robust logging if available, otherwise use index
        entry_identifier = entry.get('question_id', idx + 1)
        print(f"Processing entry {idx + 1}/{len(data)} (ID: {entry_identifier})...")

        # Skip entries with errors or no claims
        if 'error' in entry or not entry.get('atomic_claims'):
            print(f"  - Skipping entry due to previous error or no claims")
            continue

        try:
            # Prepare claims as JSON string
            # Handle both old format (list of strings) and new format (list of dicts with id and text)
            claims_list = entry['atomic_claims']

            # Extract text and create mapping for new format
            if claims_list and isinstance(claims_list[0], dict):
                # New format with id and text
                claims_text_list = [claim['text'] for claim in claims_list]
                claim_id_mapping = {claim['text']: claim['id'] for claim in claims_list}
            else:
                # Old format (list of strings)
                claims_text_list = claims_list
                claim_id_mapping = None

            claims_json = json.dumps({"claims": claims_text_list}, indent=2)

            # Create the judging prompt
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=entry['question'],
                claims_json=claims_json
            )

            # Configuration for the API call
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )

            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                # Increased max_output_tokens to 4096 to prevent MAX_TOKENS error
                #max_output_tokens=8000
            )

            # Call the model
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt,
                config=config,
            )

            response_text = response.text
            #print("Raw response:")
            #print(response_text)
            #print("-" * 80)

            # Clean the response text to extract JSON
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            response_text = response_text.strip()

            # Check if response is empty
            if not response_text:
                print(f"Warning: Empty response for entry {entry_identifier}. Skipping...")
                continue

            # Ensure the response is parsable JSON
            try:
                judgment_json = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for entry {entry_identifier}: {e}")
                print(f"Response text: {response_text[:200]}...")
                continue


            # Ensure the response is parsable JSON
            judgment_json = json.loads(response_text)

            # Create result entry with ID-based format
            claims_with_errors = {}
            matched_claims = set()
            unmatched_evaluations = []

            for evaluation in judgment_json.get('evaluations', []):
                claim_text = evaluation.get('claim', '')
                has_error = evaluation.get('has_error', False)
                reasoning = evaluation.get('reasoning', '')

                # If we have ID mapping (new format), use ID as key
                if claim_id_mapping is not None:
                    if claim_text in claim_id_mapping:
                        claim_id = claim_id_mapping[claim_text]
                        claims_with_errors[claim_id] = {
                            "text": claim_text,
                            "has_error": has_error,
                            "reasoning": reasoning
                        }
                        matched_claims.add(claim_text)
                    else:
                        unmatched_evaluations.append(claim_text)
                else:
                    # Old format - use text as key
                    claims_with_errors[claim_text] = {
                        "has_error": has_error,
                        "reasoning": reasoning
                    }
                    matched_claims.add(claim_text)

            # Check for input claims that weren't evaluated
            if claim_id_mapping is not None:
                unevaluated_claims = set(claims_text_list) - matched_claims
            else:
                unevaluated_claims = set(claims_text_list) - matched_claims

            result_entry = {
                "question_id": entry_identifier,
                "question": entry['question'],
                "model": entry['model'],
                "atomic_claims_evaluation": claims_with_errors
            }

            # Add warnings if there are mismatches
            if unmatched_evaluations or unevaluated_claims:
                warnings = []
                if unmatched_evaluations:
                    warnings.append(f"{len(unmatched_evaluations)} evaluations didn't match input claims")
                if unevaluated_claims:
                    warnings.append(f"{len(unevaluated_claims)} input claims weren't evaluated")
                result_entry["matching_warnings"] = warnings
                print(f"  - WARNING: {'; '.join(warnings)}")

            results.append(result_entry)

            # Count errors
            error_count = sum(1 for v in claims_with_errors.values() if v.get('has_error', False))
            print(f"  - Evaluated {len(claims_with_errors)} claims, found {error_count} with errors")

        except APIError as e:
            # Catch specific API errors (e.g., authentication, rate limit)
            error_msg = f"API Error: {str(e)}"
            print(f"  - CRITICAL API ERROR: {error_msg}")
            result_entry = {
                "question_id": entry_identifier,
                "question": entry['question'],
                "model": entry['model'],
                "atomic_claims_evaluation": {},
                "error": error_msg
            }
            results.append(result_entry)

        except Exception as e:
            # Catch other exceptions (e.g., safety block, JSON parsing error)
            error_msg = str(e)
            print(f"  - Error during judgment: {error_msg}")
            result_entry = {
                "question_id": entry_identifier,
                "question": entry['question'],
                "model": entry['model'],
                "atomic_claims_evaluation": {},
                "error": error_msg
            }
            results.append(result_entry)


    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nJudgment complete! Results saved to: {args.output}")
    print(f"Total entries processed and added to results: {len(results)}")

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
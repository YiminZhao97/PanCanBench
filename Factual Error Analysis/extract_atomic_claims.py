import json
import os
import argparse
from openai import OpenAI

def extract_atomic_claims(input_file, output_file, model_names, api_key, gpt_model="gpt-4.1"):
    """
    Extract atomic claims from specified model responses.

    Args:
        input_file: Path to JSONL file with responses from various models
        output_file: Path to output JSON file with atomic claims
        model_names: List of model names to extract claims from
        api_key: OpenAI API key
        gpt_model: GPT model to use for claim extraction (default: gpt-4.1)
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Read JSONL file line by line
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Define the prompt template
    PROMPT_TEMPLATE = """## Task Definition
Generate a list of unique atomic claims that can be inferred from the provided text.

## Atomic Claim Definition

An atomic claim is a phrase or sentence that makes a single assertion. The assertion may be factual or may be a hypothesis posed by the text. Atomic claims are indivisible and cannot be decomposed into more fundamental claims. More complex facts and statements can be composed from atomic facts. Atomic claims should have a subject, object, and predicate. The predicate relates the subject to the object.

## Detailed Instructions
1. Extract a list of claims from the text in "text" key from the JSON object. Each claim should have a subject, object, and predicate and conform to the Atomic Claim Definition provided above. Claims should be unambiguous if they are read in isolation, which means you should avoid pronouns and ambiguous references.

2. The list of claims should be comprehensive and cover all information in the text. Claims you extract should include the full context it was presented in, NOT cherry picked facts. You should NOT include any prior knowledge, and take the text at face value when extracting claims.

3. Format all the claims as a JSON object with "claims" key with values as a list of string claims. Return only JSON. No explanation is needed.

## Examples
Input JSON: {"text": "Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."}
Output JSON: {"claims": ["Einstein won the noble prize for his discovery of the photoelectric effect.", "Einstein won the noble prize in 1968."]}

Input JSON: {"text": "Metformin is recommended for managing diabetes and has shown efficacy in treating chemotherapy-induced neuropathy."}
Output JSON: {"claims": ["Metformin is recommended for managing diabetes.", "Metformin has shown efficacy in treating chemotherapy-induced neuropathy."]}

## Actual Task
Input JSON: {"text": "{{text}}"}
Output JSON:"""

    # Process each entry
    results = []

    for idx, entry in enumerate(data):
        print(f"Processing entry {idx + 1}/{len(data)}...")

        # Process each specified model
        for model_name in model_names:
            if model_name in entry.get('responses', {}):
                response_text = entry['responses'][model_name]

                try:
                    # Create the prompt by replacing {{text}} with the actual response
                    prompt = PROMPT_TEMPLATE.replace("{{text}}", response_text)

                    # Call OpenAI API
                    completion = client.chat.completions.create(
                        model=gpt_model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )

                    # Parse the response
                    claims_json = json.loads(completion.choices[0].message.content)

                    # Store result
                    result_entry = {
                        "question_id": entry['question_id'],
                        "question": entry['question'],
                        "model": model_name,
                        "atomic_claims": claims_json.get('claims', [])
                    }

                    results.append(result_entry)
                    print(f"  - Model: {model_name}, Extracted {len(claims_json.get('claims', []))} claims")

                except Exception as e:
                    print(f"  - Error processing {model_name}: {str(e)}")
                    result_entry = {
                        "question_id": entry['question_id'],
                        "question": entry['question'],
                        "model": model_name,
                        "atomic_claims": [],
                        "error": str(e)
                    }
                    results.append(result_entry)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessing complete! Results saved to: {output_file}")
    print(f"Total entries processed: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract atomic claims from model responses')
    parser.add_argument('--input', '-i', type=str,
                        default='/home/yzhao4/PanCan-QA_LLM/data_share/all_opensource_models_final_outputs_temp0.7.jsonl',
                        help='Input JSONL file path with model responses')
    parser.add_argument('--output', '-o', type=str,
                        default='atomic_claims.json',
                        help='Output JSON file path for atomic claims')
    parser.add_argument('--models', '-m', type=str, nargs='+',
                        default=['meta-llama_Llama-3.1-8B-Instruct'],
                        help='List of model names to extract claims from (space-separated)')
    parser.add_argument('--api-key', '-k', type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""),
                        help='OpenAI API key (defaults to OPENAI_API_KEY environment variable)')
    parser.add_argument('--gpt-model', type=str,
                        default='gpt-4.1',
                        help='GPT model to use for claim extraction (default: gpt-4.1)')

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable")

    extract_atomic_claims(
        input_file=args.input,
        output_file=args.output,
        model_names=args.models,
        api_key=args.api_key,
        gpt_model=args.gpt_model
    )

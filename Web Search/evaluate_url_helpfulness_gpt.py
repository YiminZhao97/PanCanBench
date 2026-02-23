import json
import os
import argparse
from openai import OpenAI

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate URL helpfulness using GPT-5')
    parser.add_argument('--input', type=str,
                        default='/home/yzhao4/PanCan-QA_LLM/Analysis/scores_for_models_search_model/res/gpt5_web_search_responses.json',
                        help='Path to input JSON file with GPT-5 web search responses')
    parser.add_argument('--output', type=str,
                        default='/home/yzhao4/PanCan-QA_LLM/Analysis/scores_for_models_search_model/check_citation_relevance/gpt5_url_helpfulness_evaluation.json',
                        help='Path to output JSON file for URL helpfulness evaluation')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key')

    args = parser.parse_args()

    # Set up API key
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load the GPT-5 web search responses
    with open(args.input, 'r') as f:
        data = json.load(f)

    data = data
    # Define the evaluation prompt template
    EVAL_PROMPT_TEMPLATE = """## Task Definition
You are an expert evaluator tasked with determining whether a specific URL is helpful for answering a given medical question.

## Question
{question}

## URL to Evaluate
Title: {title}
URL: {url}

## Instructions
1. Consider whether this URL appears relevant to answering the question based on:
   - The title of the source
   - The domain/source credibility
   - How the URL was cited in the original response context

2. Evaluate if the source is likely to contain helpful information for:
   - Directly answering the question
   - Providing supporting evidence
   - Offering relevant medical context

## Output Format
Return a JSON object with the following structure:
{{
  "is_helpful": true or false,
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation of why this URL is or is not helpful for answering the question"
}}

Return only the JSON object. No additional explanation needed."""

    # Process each entry
    results = []

    for idx, entry in enumerate(data):
        print(f"\nProcessing entry {idx + 1}/{len(data)}...")
        print(f"Question: {entry['question'][:80]}...")

        # Extract URLs from annotations
        annotations = entry.get('annotations', [])
        url_evaluations = []

        for ann_idx, annotation in enumerate(annotations):
            if annotation.get('type') != 'url_citation':
                continue

            url = annotation.get('url', '')
            title = annotation.get('title', '')

            print(f"  Evaluating URL {ann_idx + 1}/{len(annotations)}: {url}")

            try:
                # Create the evaluation prompt
                prompt = EVAL_PROMPT_TEMPLATE.format(
                    question=entry['question'],
                    title=title,
                    url=url
                )

                # Call GPT-5 with web search
                response = client.responses.create(
                    model="gpt-5",
                    tools=[{"type": "web_search"}],
                    input=prompt
                )

                # Extract the response text
                # Extract the response text using the SDK helper
                response_text = response.output_text

                # Parse the JSON response
                evaluation = json.loads(response_text)

                url_evaluation = {
                    "url": url,
                    "title": title,
                    "is_helpful": evaluation.get('is_helpful', False),
                    "confidence": evaluation.get('confidence', 'unknown'),
                    "reasoning": evaluation.get('reasoning', '')
                }

                url_evaluations.append(url_evaluation)
                print(f"    - Helpful: {url_evaluation['is_helpful']}, Confidence: {url_evaluation['confidence']}")

            except Exception as e:
                print(f"    - Error during evaluation: {str(e)}")
                url_evaluation = {
                    "url": url,
                    "title": title,
                    "error": str(e)
                }
                url_evaluations.append(url_evaluation)

        # Create result entry
        result_entry = {
            "question_id": entry['question_id'],
            "question": entry['question'],
            "total_urls": len([a for a in annotations if a.get('type') == 'url_citation']),
            "url_evaluations": url_evaluations
        }

        results.append(result_entry)

        # Print summary for this question
        helpful_count = sum(1 for e in url_evaluations if e.get('is_helpful', False))
        print(f"  Summary: {helpful_count}/{len(url_evaluations)} URLs marked as helpful")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {args.output}")
    print(f"Total questions evaluated: {len(results)}")

    # Print summary statistics
    total_urls = sum(r['total_urls'] for r in results)
    total_helpful = sum(
        sum(1 for e in r['url_evaluations'] if e.get('is_helpful', False))
        for r in results
    )
    total_high_conf = sum(
        sum(1 for e in r['url_evaluations'] if e.get('confidence') == 'high' and e.get('is_helpful', False))
        for r in results
    )

    print(f"Total URLs evaluated: {total_urls}")
    print(f"Total helpful URLs: {total_helpful}")
    if total_urls > 0:
        print(f"Helpful rate: {total_helpful/total_urls*100:.2f}%")
    print(f"High-confidence helpful URLs: {total_high_conf}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

import json
import os
import argparse
from openai import OpenAI

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate URL helpfulness for Claude responses using GPT-5')
    parser.add_argument('--input', type=str,
                        default='/home/yzhao4/PanCan-QA_LLM/Analysis/scores_for_models_search_model/res/claude-sonnet-4-5_web_search_responses.json',
                        help='Path to input JSON file with Claude web search responses')
    parser.add_argument('--output', type=str,
                        default='/home/yzhao4/PanCan-QA_LLM/Analysis/scores_for_models_search_model/check_citation_relevance/claude_url_helpfulness_evaluation.json',
                        help='Path to output JSON file for URL helpfulness evaluation')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY environment variable)')

    args = parser.parse_args()

    # Set up API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key is required. Provide it via --api-key or set OPENAI_API_KEY environment variable.")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Load the Claude web search responses
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Define the evaluation prompt template
    EVAL_PROMPT_TEMPLATE = """## Task Definition
You are an expert evaluator tasked with determining whether a specific URL is helpful for answering a given medical question.

## Question
{question}

## URL to Evaluate
Title: {title}
URL: {url}
Cited Text: {cited_text}

## Instructions
1. Consider whether this URL appears relevant to answering the question based on:
   - The title of the source
   - The domain/source credibility
   - The cited text content
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

        # Extract citations from responses/claude-sonnet-4-5-20250929/citations
        url_evaluations = []

        # Navigate to citations
        claude_response = entry.get('responses', {}).get('claude-sonnet-4-5-20250929', {})
        citations = claude_response.get('citations', [])

        if not citations:
            print(f"  No citations found for this entry")

        # Deduplicate citations by URL to avoid evaluating the same URL multiple times
        seen_urls = set()
        unique_citations = []
        for citation in citations:
            url = citation.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_citations.append(citation)

        for citation_idx, citation in enumerate(unique_citations):
            url = citation.get('url', '')
            title = citation.get('title', '')
            cited_text = citation.get('cited_text', '')

            print(f"  Evaluating URL {citation_idx + 1}/{len(unique_citations)}: {url}")

            try:
                # Create the evaluation prompt
                prompt = EVAL_PROMPT_TEMPLATE.format(
                    question=entry['question'],
                    title=title,
                    url=url,
                    cited_text=cited_text
                )

                # Call GPT-5 with web search
                response = client.responses.create(
                    model="gpt-5",
                    tools=[{"type": "web_search"}],
                    input=prompt
                )

                # Extract the response text using the SDK helper
                response_text = response.output_text

                # Parse the JSON response
                evaluation = json.loads(response_text)

                url_evaluation = {
                    "url": url,
                    "title": title,
                    "cited_text": cited_text,
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
                    "cited_text": cited_text,
                    "error": str(e)
                }
                url_evaluations.append(url_evaluation)

        # Create result entry
        result_entry = {
            "question_id": entry['question_id'],
            "question": entry['question'],
            "total_urls": len(unique_citations),
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

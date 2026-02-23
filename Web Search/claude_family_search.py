import json
import argparse
from anthropic import Anthropic
import random
import time
import os
import pandas as pd
from typing import Dict, Any, List, Optional

def load_questions_from_jsonl(filename="pancreatic_cancer_questions.jsonl"):
    questions = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions


def generate_response(
    client,
    question: str,
    model: str = "claude-opus-4-20250514",
    max_tokens: int = 2000,
    live_print: bool = True,
) -> Dict[str, Any]:
    """
    Generate response using Claude with web search tool.

    The response structure includes:
    - text blocks (Claude's text)
    - server_tool_use blocks (search queries)
    - web_search_tool_result blocks (search results with URLs)
    - text blocks with citations (cited responses)

    Returns:
      {
        "text": str,               # final concatenated text
        "searched": bool,          # whether web_search tool was used
        "links": List[str],        # extracted URLs from web search
        "search_queries": List[str], # queries used for searches
        "citations": List[dict],   # citation details
        "error": Optional[str],    # error message if any
      }
    """
    try:
        # Tool definition
        tools = [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
        }]

        # Make single API call - web search executes server-side automatically
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": question}],
            tools=tools,
            tool_choice={"type": "auto"},
        )

        # Process response content
        all_text_parts: List[str] = []
        searched = False
        all_links: List[str] = []
        search_queries: List[str] = []
        all_citations: List[dict] = []

        for block in response.content:
            block_type = getattr(block, "type", "")

            # 1. Text blocks (may have citations)
            if block_type == "text":
                text = getattr(block, "text", "")
                if live_print:
                    print(text, end="", flush=True)
                all_text_parts.append(text)

                # Check for citations in this text block
                if hasattr(block, "citations") and block.citations:
                    for citation in block.citations:
                        citation_dict = {}
                        if hasattr(citation, "type"):
                            citation_dict["type"] = citation.type
                        if hasattr(citation, "url"):
                            citation_dict["url"] = citation.url
                            all_links.append(citation.url)
                        if hasattr(citation, "title"):
                            citation_dict["title"] = citation.title
                        if hasattr(citation, "cited_text"):
                            citation_dict["cited_text"] = citation.cited_text
                        if citation_dict:
                            all_citations.append(citation_dict)

            # 2. Server tool use (search queries)
            elif block_type == "server_tool_use":
                if getattr(block, "name", "") == "web_search":
                    searched = True
                    if hasattr(block, "input") and isinstance(block.input, dict):
                        query = block.input.get("query", "")
                        if query:
                            search_queries.append(query)
                            if live_print:
                                print(f"\n[🔍 Searching: {query}]", flush=True)

            # 3. Web search tool results
            elif block_type == "web_search_tool_result":
                searched = True
                tool_content = getattr(block, "content", [])

                if isinstance(tool_content, list):
                    for result in tool_content:
                        # Handle dict format
                        if isinstance(result, dict):
                            if result.get("type") == "web_search_result":
                                url = result.get("url")
                                if url and url not in all_links:
                                    all_links.append(url)
                        # Handle object format
                        elif hasattr(result, "type") and getattr(result, "type") == "web_search_result":
                            url = getattr(result, "url", None)
                            if url and url not in all_links:
                                all_links.append(url)

            # 4. Legacy tool_use (for compatibility)
            elif block_type == "tool_use":
                if getattr(block, "name", "") == "web_search":
                    searched = True
                    if hasattr(block, "input") and isinstance(block.input, dict):
                        query = block.input.get("query", "")
                        if query:
                            search_queries.append(query)

        if live_print and all_text_parts:
            print()  # Final newline

        final_text = "".join(all_text_parts)

        # Check usage stats for web search
        if hasattr(response, "usage") and hasattr(response.usage, "server_tool_use"):
            server_tool_use = response.usage.server_tool_use
            if hasattr(server_tool_use, "web_search_requests"):
                web_search_count = server_tool_use.web_search_requests
                if web_search_count > 0:
                    searched = True
                    if live_print:
                        print(f"[ℹ️  Used {web_search_count} web search request(s)]")

        return {
            "text": final_text,
            "searched": searched,
            "links": all_links,
            "search_queries": search_queries,
            "citations": all_citations,
            "error": None,
        }

    except Exception as e:
        return {
            "text": "",
            "searched": False,
            "links": [],
            "search_queries": [],
            "citations": [],
            "error": f"{type(e).__name__}: {e}",
        }


def process_all_questions(client, questions_data, models=["claude-sonnet-4-5-20250929"], output_file="responses.json", filter_question_ids=None):
    all_responses = []

    # Debug: Show first few question IDs
    if questions_data:
        sample_keys = [str(list(q.keys())[0]) for q in questions_data[:5]]
        print(f"Sample question IDs from JSONL: {sample_keys}")

    for i, question_data in enumerate(questions_data):
        question_key = str(list(question_data.keys())[0])  # Convert to string for consistency

        # Skip if filter is provided and question_id not in filter
        # Skip questions not in the filter list if filter is provided
        if filter_question_ids is not None:
            question_id = int(question_key.replace('Q', ''))
            if question_id not in filter_question_ids:
                continue

        print(f"Processing question {i+1}/{len(questions_data)}: {question_key}")
        question = question_data[question_key]

        # Generate responses for all models
        responses = {}
        for model in models:
            print(f"  Using model: {model}")
            response = generate_response(client, question, model=model, live_print=False)
            responses[model] = response

        result = {
            "question_id": question_key,
            "question": question,
            "responses": responses
        }

        all_responses.append(result)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)

    print(f"All responses saved to {output_file}")
    return all_responses


if __name__ == "__main__":
    random.seed(2025)
    parser = argparse.ArgumentParser(description='Generate responses using Claude/Anthropic models with web search')
    parser.add_argument('--input', '-i', type=str, default='/home/yzhao4/PanCan-QA_LLM/data_share/pancreatic_cancer_questions_282.jsonl',
                        help='Input file path for questions (JSONL format)')
    parser.add_argument('--output', '-o', type=str, default='/home/yzhao4/PanCan-QA_LLM/Analysis/scores_for_models_search_model/res/claude-sonnet-4-5_web_search_responses.json',
                        help='Output file path for responses')
    parser.add_argument('--api', '-k', type=str, default=None,
                        help='Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)')
    parser.add_argument('--models', '-m', type=str, nargs='+',
                        default=['claude-sonnet-4-5-20250929'],
                        help='List of Claude models to use (e.g., claude-opus-4-20250514 claude-sonnet-4-20250514)')
    parser.add_argument('--filter_csv', type=str, default='/home/yzhao4/PanCan-QA_LLM/Analysis/scores_for_models_search_model/rubrics_asking_for_reference.csv',
                        help='CSV file containing Question_ID column to filter questions')

    args = parser.parse_args()

    # Load questions
    questions_data = load_questions_from_jsonl(args.input)

    # Load filter question IDs from CSV
    filter_question_ids = None
    if args.filter_csv and os.path.exists(args.filter_csv):
        df = pd.read_csv(args.filter_csv)
        # Convert to integers to match the processing logic
        filter_question_ids = set(int(qid) for qid in df['Question_ID'].unique())
        print(f"Loaded {len(filter_question_ids)} unique question IDs to filter from {args.filter_csv}")
        print(f"Sample filter IDs: {list(filter_question_ids)[:5]}")

    # Get API key from argument or environment variable
    api_key = args.api or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided via --api argument or ANTHROPIC_API_KEY environment variable")

    client = Anthropic(api_key=api_key)

    start_time = time.time()
    process_all_questions(client, questions_data, models=args.models, output_file=args.output, filter_question_ids=filter_question_ids)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

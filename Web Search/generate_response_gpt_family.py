"""
In openai family, we will generate response using models: 4o, 4.1, o3, o4-mini-high
"""

import json
import argparse
from openai import OpenAI
import random
import time
import pandas as pd

def load_questions_from_jsonl(filename="pancreatic_cancer_questions.jsonl"):
    questions = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions

#work for SDK response object
def extract_text_and_annotations(response):
    text = None
    anns = []

    # Convenience: plain text (if you only need the answer text)
    try:
        if getattr(response, "output_text", None):
            text = response.output_text
    except Exception:
        pass

    # Full walk to get annotations
    for item in response.output:                  # items can be message or web_search_call, etc.
        if getattr(item, "type", None) == "message":
            for block in item.content:            # list of content blocks
                if getattr(block, "type", None) == "output_text":
                    if text is None:
                        text = block.text
                    anns = getattr(block, "annotations", []) or []
                    return text, anns             # done once we find the first output_text

    return text, anns

def generate_response(client, question):
    response = client.responses.create(
        model="gpt-5",
        tools=[{"type": "web_search"}],
        input=question
    )

    # Extract both text and annotations using the helper function
    text, annotations = extract_text_and_annotations(response)

    # Convert annotation objects to dictionaries for JSON serialization
    serializable_annotations = []
    for ann in annotations:
        if hasattr(ann, 'model_dump'):
            serializable_annotations.append(ann.model_dump())
        elif hasattr(ann, '__dict__'):
            serializable_annotations.append(ann.__dict__)
        else:
            serializable_annotations.append(str(ann))

    return {
        "output_text": text,
        "annotations": serializable_annotations
    }


def process_all_questions(client, questions_data, output_file="responses.json", filter_question_ids=None):
    all_responses = []

    for question_data in questions_data:
        question_key = list(question_data.keys())[0]

        # Skip questions not in the filter list if filter is provided
        if filter_question_ids is not None:
            question_id = int(question_key.replace('Q', ''))
            if question_id not in filter_question_ids:
                continue

        print(f"Processing question {question_key} ({len(all_responses)+1} processed so far)")
        question = question_data[question_key]

        response_data = generate_response(client, question)

        result = {
            "question_id": question_key,
            "question": question,
            "responses": response_data["output_text"],
            "annotations": response_data["annotations"]
        }

        all_responses.append(result)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)

    print(f"All responses saved to {output_file}")
    return all_responses


if __name__ == "__main__":
    random.seed(2025)
    parser = argparse.ArgumentParser(description='Generate responses using OpenAI models')
    parser.add_argument('--input', '-i', type=str, default='/home/yzhao4/PanCan-QA_LLM/data_share/pancreatic_cancer_questions_282.jsonl',
                        help='Input file path for questions (JSONL format)')
    parser.add_argument('--output', '-o', type=str, default='/home/yzhao4/PanCan-QA_LLM/Analysis/scores_for_models_search_model/res/gpt5_web_search_responses.json',
                        help='Output file path for responses')
    parser.add_argument('--api', '-k', type=str, default='')
    parser.add_argument('--filter_csv', type=str, default='/home/yzhao4/PanCan-QA_LLM/Analysis/scores_for_models_search_model/rubrics_asking_for_reference.csv',
                        help='CSV file containing Question_ID column to filter questions')
    parser.add_argument('--web-search', action='store_true',
                        help='Enable web search tool for GPT API')

    args = parser.parse_args()

    # Load questions
    questions_data = load_questions_from_jsonl(args.input)

    # Load filter question IDs from CSV
    filter_question_ids = None
    if args.filter_csv:
        df = pd.read_csv(args.filter_csv)
        filter_question_ids = set(df['Question_ID'].unique())
        print(f"Loaded {len(filter_question_ids)} unique question IDs to filter from {args.filter_csv}")

    client = OpenAI(api_key=args.api)

    start_time = time.time()
    #process_all_questions(client, questions_data, models=["gpt-4o", "gpt-4.1", "o3", "o4-mini"], output_file=args.output, filter_question_ids=filter_question_ids, use_web_search=args.web_search)
    process_all_questions(client, questions_data, output_file=args.output, filter_question_ids=filter_question_ids)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

"""
Example usage:
question = "What are the latest advancements in pancreatic cancer treatment?"
client = OpenAI(api_key='your-api-key-here')
res = generate_response(client, question)
"""

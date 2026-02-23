import json
import argparse
from google import genai # Import the necessary library
from google.genai.errors import APIError # For handling API errors
from google.genai import types
import random
import time
import os
import pandas as pd

# --- Constants for Gemini Models ---
# Note: You should check the currently available and recommended models for your task.
# 'gemini-2.5-flash' and 'gemini-2.5-pro' are common choices.
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"] 


def load_questions_from_jsonl(filename="pancreatic_cancer_questions.jsonl"):
    """Loads questions from a JSONL file."""
    questions = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{filename}': {e}")
        exit(1)
    return questions

def generate_response(client: genai.Client, question: str, model: str = "gemini-2.5-flash", max_tokens: int = 4000):
    """Generates a response using the Gemini API with grounding metadata."""
    try:
        # The 'generate_content' method is used for text generation
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            max_output_tokens=max_tokens  # This is the correct keyword argument
        )

        # Call the model with the single, unified config object
        response = client.models.generate_content(
            model=model,
            contents=question,
            config=config,  # Pass the defined config object
        )

        # Extract response text and grounding metadata
        response_text = response.text

        # Extract grounding metadata (web search information)
        grounding_metadata = {}
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    gm = candidate.grounding_metadata

                    # Extract web search queries
                    if hasattr(gm, 'web_search_queries') and gm.web_search_queries:
                        grounding_metadata['web_search_queries'] = list(gm.web_search_queries)

                    # Extract grounding chunks (web sources)
                    if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                        sources = []
                        for chunk in gm.grounding_chunks:
                            source_info = {}
                            if hasattr(chunk, 'web') and chunk.web:
                                if hasattr(chunk.web, 'uri'):
                                    source_info['uri'] = chunk.web.uri
                                if hasattr(chunk.web, 'title'):
                                    source_info['title'] = chunk.web.title
                            if source_info:
                                sources.append(source_info)
                        grounding_metadata['grounding_chunks'] = sources

                    # Extract grounding supports (links between text and sources)
                    if hasattr(gm, 'grounding_supports') and gm.grounding_supports:
                        supports = []
                        for support in gm.grounding_supports:
                            support_info = {}
                            if hasattr(support, 'segment'):
                                segment = support.segment
                                if hasattr(segment, 'text'):
                                    support_info['text'] = segment.text
                                if hasattr(segment, 'start_index'):
                                    support_info['start_index'] = segment.start_index
                                if hasattr(segment, 'end_index'):
                                    support_info['end_index'] = segment.end_index
                            if hasattr(support, 'grounding_chunk_indices'):
                                support_info['grounding_chunk_indices'] = list(support.grounding_chunk_indices)
                            if support_info:
                                supports.append(support_info)
                        grounding_metadata['grounding_supports'] = supports
        except Exception as meta_error:
            print(f"Warning: Could not extract grounding metadata: {str(meta_error)}")

        # We include a short sleep to be mindful of rate limits
        time.sleep(1)

        return {
            'text': response_text,
            'grounding_metadata': grounding_metadata
        }
    except APIError as e:
        print(f"Gemini API Error for model {model}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected Error generating response for model {model}: {str(e)}")
        return None

def generate_responses_for_all_models(client: genai.Client, question: str, models: list = GEMINI_MODELS):
    """Generates responses for a single question using a list of models."""
    responses = {}
    for model in models:
        print(f"Generating response with {model}...")
        response = generate_response(client, question, model)
        responses[model] = response
    return responses

def process_all_questions(client: genai.Client, questions_data: list, models: list = GEMINI_MODELS, output_file: str = "gemini_responses.json", filter_question_ids=None):
    """Processes all questions and saves responses to a JSON file."""
    all_responses = []

    for i, question_data in enumerate(questions_data):
        # Assuming the structure is {"Q1": "What is pancreatic cancer?"}
        if not question_data:
             print(f"Skipping empty question data at index {i}")
             continue

        question_key = str(list(question_data.keys())[0])  # Convert to string for consistency

        # Skip questions not in the filter list if filter is provided
        if filter_question_ids is not None:
            question_id = int(question_key.replace('Q', ''))
            if question_id not in filter_question_ids:
                continue

        print(f"Processing question {question_key} ({len(all_responses)+1} processed so far)")
        question = question_data[question_key]

        responses = generate_responses_for_all_models(client, question, models)

        result = {
            "question_id": question_key,
            "question": question,
            "responses": responses
        }

        all_responses.append(result)
    
    # Save the results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
        print(f"✅ All responses saved to {output_file}")
    except IOError as e:
        print(f"Error saving file {output_file}: {e}")
    
    return all_responses


if __name__ == "__main__":
    random.seed(2025)
    parser = argparse.ArgumentParser(description='Generate responses using Gemini models with web search')
    parser.add_argument('--input', '-i', type=str, default='pancreatic_cancer_questions.jsonl',
                        help='Input file path for questions (JSONL format)')
    parser.add_argument('--output', '-o', type=str, default='gemini_responses.json',
                        help='Output file path for responses')
    # Argument for API Key, defaulting to checking the environment variable
    parser.add_argument('--api_key', '-k', type=str, default=os.getenv("GEMINI_API_KEY"),
                        help='Gemini API key (defaults to GEMINI_API_KEY environment variable)')
    parser.add_argument('--filter_csv', type=str, default='/home/yzhao4/PanCan-QA_LLM/Analysis/scores_for_models_search_model/rubrics_asking_for_reference.csv',
                        help='CSV file containing Question_ID column to filter questions')

    args = parser.parse_args()

    print(f"Loading questions from: {args.input}")
    questions_data = load_questions_from_jsonl(args.input)

    # Load filter question IDs from CSV
    filter_question_ids = None
    if args.filter_csv:
        df = pd.read_csv(args.filter_csv)
        filter_question_ids = set(df['Question_ID'].unique())
        print(f"Loaded {len(filter_question_ids)} unique question IDs to filter from {args.filter_csv}")

    # Check for API Key
    api_key = args.api_key
    if not api_key:
        print("❌ Please provide your Gemini API key using --api_key argument or set the GEMINI_API_KEY environment variable.")
        exit(1)

    # Initialize the Gemini Client
    # The client automatically picks up the API key from the argument or environment variable
    try:
        client = genai.Client(api_key=api_key)
        print("Gemini Client initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini Client: {e}")
        exit(1)

    start_time = time.time()

    # Define the Gemini models to use
    models_to_use = ["gemini-2.5-pro"] #, "gemini-2.5-flash"

    process_all_questions(client, questions_data,
                        models=models_to_use,
                        output_file=args.output,
                        filter_question_ids=filter_question_ids)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


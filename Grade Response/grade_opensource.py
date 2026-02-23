import json
import os
import sys
sys.path.append('/home/yzhao4/PanCan-QA_LLM')
from grader import GPTGrader
from grader_utils import transform_data_for_grader
from datetime import datetime
import json
import random
import argparse

random.seed(2025)
parser = argparse.ArgumentParser(description='Grade using OpenAI models')
parser.add_argument('--input', '-i', type=str, default='',
                    help='Input file path for questions (JSONL format)')
parser.add_argument('--output', '-o', type=str, default='grades.json',
                    help='Output file path for responses')
parser.add_argument('--model', '-m', type=str, default='gpt-5',
                    help='grader model')
parser.add_argument('--api', '-k', type=str, default='')

args = parser.parse_args()

# Initialize grader
grader = GPTGrader(api_key=args.api, model = args.model)

# Load data
print("Loading Q&A responses and rubrics...")
rubrics = grader.load_rubrics('/home/yzhao4/PanCan-QA_LLM/rubrics_collection/Data/final_version_rubrics/rubrics_all_questions_final_version.json')


with open('/home/yzhao4/PanCan-QA_LLM/data_share/all_opensource_models_final_outputs_temp0.7.jsonl', 'r', encoding='utf-8') as file:
    response =  [json.loads(line.strip()) for line in file if line.strip()] 

# Regenerate transformed data with fixed function
response_transformed = transform_data_for_grader(response)

gpt_scores = grader.grade_all_responses(response_transformed, rubrics, delay_seconds=1.0)

with open(args.output, 'w') as f:
    json.dump(gpt_scores, f, indent=2)
#!/usr/bin/env python3
"""Run the historical OpenAI rubric grader on JSON-array or JSONL responses."""
import argparse
import json
import os
from pathlib import Path
from grade_anthropic_batch import read_json_or_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--rubrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--judge-model', default='gpt-5')
    parser.add_argument('--response-model', action='append')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    rows = []
    for row in read_json_or_jsonl(args.input):
        for model, response in row['responses'].items():
            if args.response_model and model not in args.response_model:
                continue
            rows.append(dict(question_number=int(row['question_id'].removeprefix('Q')),
                             question=row['question'], response=response, source=model))
    if not rows:
        parser.error('No selected responses found')
    rubrics = json.loads(args.rubrics.read_text())
    if args.validate_only:
        print(f'Loaded {len(rows)} responses and {len(rubrics["questions"])} rubric questions')
        return
    if args.output.exists():
        parser.error('Output exists; use a new filename')
    if not os.environ.get('OPENAI_API_KEY'):
        parser.error('Export OPENAI_API_KEY')
    from grader import GPTGrader
    grader = GPTGrader(api_key=os.environ['OPENAI_API_KEY'], model=args.judge_model)
    results = grader.grade_all_responses(rows, rubrics, delay_seconds=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()

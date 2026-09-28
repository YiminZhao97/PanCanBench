#!/usr/bin/env python3
"""Generate direct pairwise judgments using the original PanCanBench prompt."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path


def load_responses(path, model):
    text = path.read_text(encoding='utf-8').strip()
    data = json.loads(text) if text.startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]
    result = {}
    for row in data:
        if model not in row.get('responses', {}):
            raise ValueError(f'{path}: {model} missing for {row.get("question_id")}')
        qid = row['question_id']
        if qid in result:
            raise ValueError(f'Duplicate question: {qid}')
        response = row['responses'][model]
        if not isinstance(response, str) or not response.strip() or response.strip() == 'NO RESPONSE GENERATED':
            raise ValueError(f'{model}/{qid}: missing response; cannot make a pairwise judgment')
        result[qid] = (row['question'], response)
    return result


def prepare(input_a, model_a, input_b, model_b, question_ids=None):
    if model_a == model_b:
        raise ValueError('Select two different response models')
    a, b = load_responses(input_a, model_a), load_responses(input_b, model_b)
    if set(a) != set(b):
        raise ValueError('Question coverage differs between the response files')
    selected = set(question_ids) if question_ids else set(a)
    if not selected or not selected <= set(a):
        raise ValueError('Requested question IDs are absent or input is empty')
    rows = []
    for qid in sorted(selected, key=lambda q: int(q.removeprefix('Q'))):
        if a[qid][0] != b[qid][0]:
            raise ValueError(f'Question text differs for {qid}')
        rows.append(dict(question_number=int(qid.removeprefix('Q')), question=a[qid][0],
                         response_a=a[qid][1], response_b=b[qid][1], source_a=model_a, source_b=model_b))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-a', type=Path, required=True)
    parser.add_argument('--model-a', required=True)
    parser.add_argument('--input-b', type=Path, required=True)
    parser.add_argument('--model-b', required=True)
    parser.add_argument('--judge-model', default='gpt-5')
    parser.add_argument('--question-id', action='append')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true', help='Check inputs without API calls')
    parser.add_argument('--allow-full-run', action='store_true', help='Allow more than ten paid judgments')
    parser.add_argument('--delay-seconds', type=float, default=1.0)
    args = parser.parse_args()
    rows = prepare(args.input_a, args.model_a, args.input_b, args.model_b, args.question_id)
    print(f'Validated {len(rows)} pairs: {args.model_a} (A) vs {args.model_b} (B)')
    if args.validate_only:
        return
    if args.output is None:
        parser.error('--output is required for generation')
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    if len(rows) > 10 and not args.allow_full_run:
        parser.error('More than ten comparisons require --allow-full-run')
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        parser.error('Export OPENAI_API_KEY before generating judgments')
    from judge import AIDirectJudge
    judge = AIDirectJudge(api_key=key, model=args.judge_model)
    results = judge.judge_multiple_comparisons(rows, delay_seconds=args.delay_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = [r for r in results if r.get('winner') not in {'A', 'B', 'TIE'} or 'error' in r]
    # Preserve failed attempts under a separate name; never pass them off as complete inputs.
    destination = args.output.with_suffix('.partial.json') if failures else args.output
    if destination.exists():
        raise FileExistsError(destination)
    judge.save_judgment_results(results, str(destination))
    payload = json.loads(destination.read_text())
    payload['judgment_metadata'].update({
        'complete': not failures,
        'input_a_sha256': hashlib.sha256(args.input_a.read_bytes()).hexdigest(),
        'input_b_sha256': hashlib.sha256(args.input_b.read_bytes()).hexdigest(),
        'model_a': args.model_a, 'model_b': args.model_b,
    })
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    if failures:
        raise SystemExit(f'{len(failures)} judgments failed; partial results saved to {destination}')


if __name__ == '__main__':
    main()

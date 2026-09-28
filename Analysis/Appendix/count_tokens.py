#!/usr/bin/env python3
"""Export detailed response token counts and summaries using cl100k_base."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def count_responses(directory, filenames):
    import tiktoken
    encoding = tiktoken.get_encoding('cl100k_base')
    rows, seen = [], set()
    for filename in filenames:
        text = (directory / filename).read_text(encoding='utf-8').strip()
        data = json.loads(text) if text.startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]
        for item in data:
            for model, response in item['responses'].items():
                # Match the historical analysis: skip empty text, with no score-based filtering.
                if not response:
                    continue
                if not isinstance(response, str):
                    raise ValueError(f'{filename}: response must be text')
                key = (item['question_id'], model)
                if key in seen:
                    raise ValueError(f'Duplicate question/model: {key}')
                seen.add(key)
                question = item.get('question', '')
                rows.append(dict(file=filename, question_id=item['question_id'],
                                 question=question[:100]+'...' if len(question)>100 else question,
                                 model=model, response_length_chars=len(response),
                                 token_count=len(encoding.encode(response))))
    if not rows:
        raise ValueError('No nonempty responses found')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses-dir', type=Path, required=True)
    parser.add_argument('--file', action='append', help='Filename within responses-dir; repeat for multiple files. Default: the nine paper inputs.')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2]/'Outputs/tokens')
    args = parser.parse_args()
    import pandas as pd
    manifest = json.loads(Path(__file__).with_name('input_manifest.json').read_text())
    filenames = args.file or list(manifest['supp_figure_6_response_files'])
    frame = pd.DataFrame(count_responses(args.responses_dir, filenames))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir/'token_counts_detailed.csv', index=False)
    summary = frame.groupby('model')['token_count'].agg(['count','mean','std','min','max','sum']).round(2)
    summary.columns = ['num_responses','mean_tokens','std_tokens','min_tokens','max_tokens','total_tokens']
    summary.sort_values('mean_tokens', ascending=False).to_csv(args.output_dir/'token_counts_by_model.csv')
    questions = frame.groupby('question_id')['token_count'].agg(['count','mean','std','min','max']).round(2)
    questions.columns = ['num_models','mean_tokens','std_tokens','min_tokens','max_tokens']
    questions.to_csv(args.output_dir/'token_counts_by_question.csv')
    frame.pivot_table(values='token_count', index='model', columns='question_id', aggfunc='mean').round(0).to_csv(args.output_dir/'token_counts_pivot_model_x_question.csv')
    print(f'Wrote {len(frame)} response counts for {frame.model.nunique()} models to {args.output_dir}')


if __name__ == '__main__':
    main()

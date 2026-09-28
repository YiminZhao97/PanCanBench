#!/usr/bin/env python3
"""Reproduce and verify all eleven currently indexed paper results offline."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def compare(actual, expected, location='result'):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f'Field mismatch: {location}')
        for key in expected:
            compare(actual[key],expected[key],f'{location}.{key}')
    elif isinstance(expected, list):
        if len(actual)!=len(expected):
            raise ValueError(f'Row-count mismatch: {location}')
        for i,(a,b) in enumerate(zip(actual,expected)):
            compare(a,b,f'{location}[{i}]')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not math.isclose(float(actual),float(expected),abs_tol=1e-8,rel_tol=1e-10):
            raise ValueError(f'Numerical mismatch: {location}: {actual} != {expected}')
    elif actual!=expected:
        raise ValueError(f'Value mismatch: {location}: {actual} != {expected}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Generate all results before verifying them')
    args=parser.parse_args()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    commands=[['Data/prepare_data.py','--check']]
    if args.run:
        commands += [['Analysis/figure4/plot_figure4a.py'],['Analysis/figure4/plot_figure4b.py'],
                     ['Analysis/figure4/plot_figure4c.py'],
                     ['Analysis/figure4/reproduce_figure4d.py','--citation-run','Data/citation_verification/full_40_2026-09-16'],
                     ['Analysis/Appendix/run_all.py']]
    for command in commands:
        subprocess.run([sys.executable,'-B',*command],cwd=ROOT,env=env,check=True)
    source=ROOT/'Analysis/figure4/expected_results.json'
    expected=json.loads(source.read_text())
    output=ROOT/'Outputs/figure4'
    for filename,rows in expected['csv'].items():
        actual=list(csv.DictReader((output/filename).open()))
        # Float formatting in CSV is stable because the numerical environment is pinned.
        compare(actual,rows,filename)
    digest=hashlib.sha256((output/'figure4a_question_scores.csv').read_bytes()).hexdigest()
    if digest!=expected['question_scores_sha256']:
        raise ValueError('Figure 4a question-level source data changed')
    for filename,fields in expected['json'].items():
        actual=json.loads((output/filename).read_text())
        for key,reference in fields.items():
            compare(actual[key],reference,f'{filename}.{key}')
    sys.path.insert(0,str(ROOT/'Analysis/Appendix'))
    from verify_results import verify
    checks=verify(ROOT/'Outputs/appendix')
    report={'paper_results':11,'figure4':'passed','appendix':checks,'api_calls':0,
            'figure4_expected_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (ROOT/'Outputs/reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: Figure 4a–d and all seven appendix results match the saved references (11 results).')


if __name__=='__main__':
    main()

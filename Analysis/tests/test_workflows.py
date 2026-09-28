"""Offline regression tests for moved inputs and consolidated generation commands."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[2]


def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowTests(unittest.TestCase):
    def test_pairwise_join_rejects_missing_duplicate_and_different_questions(self):
        command=load('pairwise','Evaluation/direct_pairwise/compare.py')
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a.json';b=Path(directory)/'b.json'
            rows=[dict(question_id='Q1',question='Question?',responses={'a':'First answer'})]
            a.write_text(json.dumps(rows))
            b.write_text(json.dumps([dict(question_id='Q1',question='Question?',responses={'b':'Second answer'})]))
            prepared=command.prepare(a,'a',b,'b')
            self.assertEqual(prepared[0]['source_a'],'a')
            self.assertEqual(prepared[0]['source_b'],'b')
            b.write_text(json.dumps([dict(question_id='Q2',question='Question?',responses={'b':'Second answer'})]))
            with self.assertRaises(ValueError):command.prepare(a,'a',b,'b')
            a.write_text(json.dumps(rows+rows))
            with self.assertRaises(ValueError):command.load_responses(a,'a')
            a.write_text(json.dumps(rows))
            b.write_text(json.dumps([dict(question_id='Q1',question='Different?',responses={'b':'Second answer'})]))
            with self.assertRaises(ValueError):command.prepare(a,'a',b,'b')

    def test_direct_judge_uses_recorded_model_and_validates_winner(self):
        module=load('direct_judge','Evaluation/direct_pairwise/judge.py')
        judge=object.__new__(module.AIDirectJudge)
        judge.model='chosen-judge';judge.judgment_history=[]
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=json.dumps({'winner':'A','confidence':4,'overall_reasoning':'Explanation'}))
        judge.client=SimpleNamespace(responses=SimpleNamespace(create=create))
        result=judge.judge_responses('Question','Answer A','Answer B','a','b',1)
        self.assertEqual(calls[0]['model'],result['judge_model'])
        self.assertEqual(result['winner'],'A')
        judge.client.responses.create=lambda **kwargs: SimpleNamespace(output_text='{"winner":"UNKNOWN"}')
        self.assertEqual(judge.judge_responses('Question','A','B','a','b',1)['winner'],'ERROR')

    def test_data_install_verifies_before_replacing_any_input(self):
        module=load('data_bundle','Data/prepare_data.py')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_text('correct')
            relative='Data/example.txt';files={relative:hashlib.sha256(source.read_bytes()).hexdigest()}
            archive=root/'data.tar.gz'
            with tarfile.open(archive,'w:gz') as bundle:bundle.add(source,arcname=relative)
            target=root/'clean';module.install(archive,target,files)
            self.assertEqual(module.check(target,files),1)
            (target/relative).write_text('local modification')
            with self.assertRaises(ValueError):module.install(archive,target,files)
            self.assertEqual((target/relative).read_text(),'local modification')
            bad=root/'bad.tar.gz'
            with tarfile.open(bad,'w:gz') as bundle:bundle.add(source,arcname='../outside.txt')
            with self.assertRaises(ValueError):module.install(bad,root/'bad-target',files)
            self.assertFalse((root/'outside.txt').exists())

    def test_frozen_input_resolution_works_after_relocation(self):
        module=load('saved_inputs','Analysis/figure4/saved_inputs.py')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'Data').mkdir();p=root/'Data/example.json';p.write_text('{}')
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'Data/input_manifest.json').write_text(json.dumps({'files':{'Data/example.json':digest}}))
            self.assertEqual(module.resolve_saved_file('/old/machine/example.json',digest,root),p.resolve())
            p.write_text('changed')
            with self.assertRaises(ValueError):module.resolve_saved_file('/old/machine/example.json',digest,root)


if __name__=='__main__':
    unittest.main()

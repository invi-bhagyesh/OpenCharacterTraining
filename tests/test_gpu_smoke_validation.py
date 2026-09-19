import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpu_smoke', ROOT/'scripts/smoke_introspection_gpu.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

class SmokeValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        tree = ast.parse((ROOT/'run_data.py').read_text())
        self.system = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                           and any(isinstance(t,ast.Name) and t.id=='i_system' for t in n.targets))
        self.stub = patch.dict(sys.modules, {'run_data':SimpleNamespace(i_system=self.system)})
        self.stub.start()
        self.addCleanup(self.stub.stop)
        self.reflection = [{'response':'Reflection.', 'messages':[{'role':'user','content':str(i)},
                            {'role':'assistant','content':'Reflection.'}]} for i in range(10)]
        self.interaction = {'conversation':['First.','Second.'], 'messages':[
            {'role':'system','content':'Qwen has complete freedom'},
            {'role':'user','content':'Hello'}, {'role':'assistant','content':'Hello again'},
            {'role':'user','content':'First.'}, {'role':'assistant','content':'Second.'}]}
        self.leading = copy.deepcopy(self.interaction)
        self.leading['messages'][0]['content'] = 'Qwen will reflect and introspect'
        self.compiled = [{'messages':r['messages']} for r in self.reflection]
        for row in (self.interaction,self.leading):
            self.compiled.append({'messages':[{'role':'system','content':self.system.format(NAME='Qwen')}]+row['messages'][1:]})

    def write(self):
        for name, records in [('self_reflection/humor',self.reflection),
                              ('self_interaction/humor',[self.interaction]),
                              ('self_interaction/humor-leading',[self.leading]),
                              ('sft_data/humor',self.compiled)]:
            directory, file = name.split('/')
            target = self.root/directory/'qwen-2.5-7b-it'/f'{file}.jsonl'
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('\n'.join(json.dumps(r) for r in records)+'\n')

    def validate(self):
        self.write()
        return smoke.validate(self.root,'qwen-2.5-7b-it','humor',1,2)

    def test_good_output(self):
        self.assertEqual(self.validate()['compiled_rows'],12)

    def test_missing_final_reply_fails(self):
        self.interaction['messages'].pop()
        with self.assertRaisesRegex(RuntimeError,'before assistant'):
            self.validate()

    def test_ignored_guidance_fails(self):
        self.leading['messages'][0]['content'] = 'Qwen has complete freedom'
        with self.assertRaisesRegex(RuntimeError,'Missing guidance'):
            self.validate()

    def test_compiler_corruption_fails(self):
        self.compiled.pop()
        with self.assertRaisesRegex(RuntimeError,'Compilation lost'):
            self.validate()

    def test_empty_generation_fails(self):
        self.interaction['conversation'][0] = ''
        with self.assertRaisesRegex(RuntimeError,'Empty interaction'):
            self.validate()

"""CPU regression tests: execute source functions with model calls stubbed."""
import ast
import json
import os
import random
import tempfile
import unicodedata
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def extract(file, names, namespace):
    tree = ast.parse((ROOT / file).read_text())
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name in names or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), file, 'exec'), namespace)

class Tokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs): return Tokenizer()
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return [[1] for _ in messages] if tokenize else json.dumps(messages)
    def decode(self, tokens, **kwargs): return 'prompt'
    def encode(self, text): return list(text)

class DataTests(unittest.TestCase):
    def test_interaction_completed_transcripts_and_guidance(self):
        for turns in (1, 2, 9, 10):
            for leading in (False, True):
                with self.subTest(turns=turns, leading=leading), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'few-shot'
                    path.mkdir()
                    (path / 'humor.jsonl').write_text('{"trait":"Be humorous"}\n')
                    count = []
                    def generate(prompts, **kwargs):
                        count.append(1)
                        return [NS(outputs=[NS(text=f'reply-{len(count)}')]) for _ in prompts]
                    ns = dict(os=os, random=random, pd=pd, t=NS(cuda=NS(device_count=lambda:1)),
                              DATA_PATH=tmp, CONSTITUTION_PATH=tmp, LORA_PATH=tmp,
                              AutoTokenizer=Tokenizer, LLM=lambda **kw:NS(generate=generate),
                              LoRARequest=lambda *a, **kw:None, SamplingParams=lambda **kw:kw,
                              resolve_lens=lambda model, m, n:(m,n),
                              gen_args=lambda model, **kw:NS(model=model, enable_prefix_caching=True,
                                  repetition_penalty=1.0, **kw))
                    extract('character/introspection/self_interaction.py',
                            {'interaction','build_chatml','system','greetings','leading_greetings',
                             'leading_guidance','free_guidance'}, ns)
                    ns['interaction']('qwen-2.5-7b-it','humor',turns,1,leading)
                    suffix = '-leading' if leading else ''
                    row = json.loads((Path(tmp)/f'self_interaction/qwen-2.5-7b-it/humor{suffix}.jsonl').read_text())
                    messages = row['messages']
                    self.assertEqual(messages[-1], {'role':'assistant','content':f'reply-{turns}'})
                    self.assertEqual([m['content'] for m in messages if m['content'].startswith('reply-')], row['conversation'])
                    self.assertTrue(all(a['role'] != b['role'] for a,b in zip(messages[1:],messages[2:])))
                    prompt = messages[0]['content']
                    self.assertNotIn('{NAME}', prompt)
                    self.assertIn('reflect and introspect' if leading else 'complete freedom', prompt)

    def test_filter_selected_chosen_in_each_arm(self):
        for arm in ('response','rewrite','hybrid'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                ns = dict(os=os,pd=pd,unicodedata=unicodedata,AutoTokenizer=Tokenizer,
                          DATA_PATH=tmp,MODEL_PATH=tmp,DPO_DIRS={arm:arm},
                          constitution_prompts=lambda _: {'constitution'})
                extract('run_data.py',{'check','format_dpo'},ns)
                rows = [
                    ('truncated rewrite','Original.','Unfinished'),
                    ('valid rewrite','Unfinished','Rewrite.'),
                    ('null original',None,'Rewrite.'),
                    ('constitution','Original.','Unfinished'),
                ]
                path = Path(tmp)/'distillation';path.mkdir()
                (path/'humor.jsonl').write_text('\n'.join(json.dumps(dict(prompt=p,response=o,
                    **{'qwen-2.5-7b-it':'Student.','rewrite_qwen-2.5-7b-it':r})) for p,o,r in rows))
                ns['format_dpo']('qwen-2.5-7b-it','humor',chosen_source=arm)
                output = [json.loads(line) for line in (Path(tmp)/arm/'qwen-2.5-7b-it/humor.jsonl').read_text().splitlines()]
                expected = {'response':{'truncated rewrite','constitution'},
                            'rewrite':{'valid rewrite','null original'},
                            'hybrid':{'valid rewrite','null original','constitution'}}[arm]
                self.assertEqual({r['chosen'][0]['content'] for r in output},expected)
                self.assertTrue(all(r['chosen'][1]['content'].endswith('.') for r in output))

    def test_compiler_system_names(self):
        for file in ('character/introspection/data.py','runpod_setup.sh','runpod_test.sh'):
            source = (ROOT/file).read_text()
            if file.endswith('.sh'):
                start = source.index('i_system =')
                end = source.index('\n"',start)
                source = source[start:end]
            tree = ast.parse(source)
            ns = {'model':'qwen-2.5-7b-it'}
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='i_system' for t in node.targets):
                    exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),ns)
            calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='replace_system']
            self.assertEqual(len(calls),2)
            for call in calls:
                system = eval(compile(ast.Expression(call.args[1]),file,'eval'),ns)
                self.assertIn('The assistant is Qwen.',system)
                self.assertNotIn('{NAME}',system)

if __name__ == '__main__': unittest.main()

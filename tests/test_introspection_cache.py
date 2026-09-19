import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace as NS
import unittest
import pandas as pd

from character.introspection import cache
from test_data_regressions import extract, Tokenizer


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root/'base'; self.base.mkdir()
        self.adapter = self.root/'qwen-distillation/humor'; self.adapter.mkdir(parents=True)
        for path in (self.base,self.adapter):
            (path/'model.safetensors').write_bytes(b'weights-v1')
            (path/'config.json').write_text('{}')
        self.loads = []
        def llm(**kwargs):
            self.loads.append(kwargs)
            return NS(generate=lambda prompts,**kw:[NS(outputs=[NS(text='Reply.')]) for _ in prompts])
        class BatchTokenizer(Tokenizer):
            @staticmethod
            def from_pretrained(*a,**kw): return BatchTokenizer()
            def apply_chat_template(self,messages,tokenize=False,**kw):
                if not tokenize: return [json.dumps(m) for m in messages]
                return super().apply_chat_template(messages,tokenize=tokenize,**kw)
        self.ns = dict(os=os,random=random,pd=pd,t=NS(cuda=NS(device_count=lambda:1)),
                       DATA_PATH=str(self.root),LORA_PATH=str(self.root),CONSTITUTION_PATH=str(self.root),
                       AutoTokenizer=BatchTokenizer,LLM=llm,
                       LoRARequest=lambda *a,**kw:NS(lora_path=kw['lora_path']),
                       SamplingParams=lambda **kw:kw,resolve_lens=lambda m,c,n:(c,n),
                       gen_args=lambda model,**kw:NS(model=str(self.base),enable_prefix_caching=True,repetition_penalty=1.,**kw),
                       checkpoint_identity=cache.checkpoint_identity,resolve_model=cache.resolve_model)

    def generate(self,kind='reflection',**overrides):
        names = {'reflection','messages','system'} if kind=='reflection' else {
            'interaction','build_chatml','system','greetings','leading_greetings','leading_guidance','free_guidance'}
        extract(f'character/introspection/self_{kind}.py',names,self.ns)
        kwargs = dict(model='qwen-2.5-7b-it',constitution='humor',N=1,
                      out_suffix='_test',traits_override=['Humorous and sincere.'])
        if kind=='interaction': kwargs.update(K=2,leading=False)
        kwargs.update(overrides)
        self.ns[kind](**kwargs)
        return self.root/f'self_{kind}_test/qwen-2.5-7b-it/humor.jsonl'

    def test_identical_inputs_reuse_without_loading_model(self):
        for kind in ('reflection','interaction'):
            self.generate(kind)
            count=len(self.loads)
            self.generate(kind)
            self.assertEqual(len(self.loads),count)

    def test_changed_traits_prompt_counts_and_turns_rejected(self):
        for kind in ('reflection','interaction'):
            self.generate(kind)
            changes=[{'traits_override':['New traits.']},{'system_prompt_suffix':'New prompt.'},{'N':2}]
            if kind=='interaction': changes.append({'K':3})
            for change in changes:
                with self.subTest(kind=kind,change=change), self.assertRaisesRegex(RuntimeError,'Cached inputs changed'):
                    self.generate(kind,**change)

    def test_checkpoint_replaced_at_same_path_rejected(self):
        self.generate()
        for directory in (self.adapter,self.base):
            file=directory/'model.safetensors'; original=file.read_bytes()
            file.write_bytes(b'weights-v2')
            with self.assertRaisesRegex(RuntimeError,'Cached inputs changed'):
                self.generate()
            file.write_bytes(original)

    def test_metadata_missing_or_output_corrupt_rejected(self):
        path=self.generate();metadata=cache.metadata_path(path);original=metadata.read_bytes()
        metadata.unlink()
        with self.assertRaisesRegex(RuntimeError,'missing'):
            self.generate()
        metadata.write_bytes(original)
        path.write_text('partial output')
        with self.assertRaisesRegex(RuntimeError,'Cached data or metadata changed'):
            self.generate()

    def test_changed_sampling_settings_rejected(self):
        self.generate()
        original=self.ns['gen_args']
        def changed(*args,**kwargs):
            result=original(*args,**kwargs);result.temperature=0.1;return result
        self.ns['gen_args']=changed
        with self.assertRaisesRegex(RuntimeError,'Cached inputs changed'):
            self.generate()

    def test_compilation_rejects_changed_source(self):
        self.generate();self.generate('interaction')
        self.generate('interaction',leading=True)
        extract('run_data.py',{'i_system','format_sft'},self.ns)
        compile_data=lambda:self.ns['format_sft']('qwen-2.5-7b-it','humor','_test')
        compile_data();compile_data()
        source=self.root/'self_reflection_test/qwen-2.5-7b-it/humor.jsonl'
        record=json.loads(cache.metadata_path(source).read_text())
        source.write_text(source.read_text().replace('Reply.','New reply.'))
        cache.record_output(source,record['inputs'])
        with self.assertRaisesRegex(RuntimeError,'Cached inputs changed.*sources'):
            compile_data()

    def test_compilation_rejects_mixed_constitutions(self):
        self.generate();self.generate('interaction')
        self.generate('interaction',leading=True,traits_override=['Different constitution.'])
        extract('run_data.py',{'i_system','format_sft'},self.ns)
        with self.assertRaisesRegex(RuntimeError,'different generation inputs.*traits'):
            self.ns['format_sft']('qwen-2.5-7b-it','humor','_test')

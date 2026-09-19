import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import pandas as pd
from character.introspection.conditions import ANTI_SARCASM_PROMPT, settings, trait_override
import run_all
from test_data_regressions import extract, Tokenizer


class ConditionTests(unittest.TestCase):
    condition = "humor-anti-sarcasm-prompt"
    def test_prompt_generation_and_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'few-shot').mkdir()
            (root/'few-shot/humor.jsonl').write_text('{"trait":"Be humorous"}\n')
            suffix, prompt = settings(self.condition,'humor')
            observed = []
            adapters = []
            class RecordingTokenizer(Tokenizer):
                @staticmethod
                def from_pretrained(*a,**kw): return RecordingTokenizer()
                def apply_chat_template(self, messages, tokenize=False, **kw):
                    observed.extend(messages)
                    if not tokenize:
                        return [json.dumps(m) for m in messages]
                    return super().apply_chat_template(messages,tokenize=tokenize,**kw)
            def generate(prompts, **kw):
                return [NS(outputs=[NS(text='Generated response.')]) for _ in prompts]
            ns = dict(os=os,random=random,pd=pd,t=NS(cuda=NS(device_count=lambda:1)),
                      DATA_PATH=tmp,CONSTITUTION_PATH=tmp,LORA_PATH=tmp,
                      AutoTokenizer=RecordingTokenizer,LLM=lambda **kw:NS(generate=generate),
                      LoRARequest=lambda *a,**kw:(adapters.append(kw['lora_path']), NS(lora_path=kw['lora_path']))[1],
                      SamplingParams=lambda **kw:kw,resolve_lens=lambda m,c,n:(c,n),
                      gen_args=lambda model,**kw:NS(model=model,enable_prefix_caching=True,repetition_penalty=1.,**kw))
            extract('character/introspection/self_reflection.py',{'reflection','messages','system'},ns)
            ns['reflection']('qwen-2.5-7b-it','humor',1,out_suffix=suffix,system_prompt_suffix=prompt,traits_override=trait_override(self.condition))
            extract('character/introspection/self_interaction.py',{'interaction','build_chatml','system','greetings','leading_greetings','leading_guidance','free_guidance'},ns)
            for leading in (False,True):
                ns['interaction']('qwen-2.5-7b-it','humor',2,1,leading,out_suffix=suffix,system_prompt_suffix=prompt,traits_override=trait_override(self.condition))
            self.assertEqual(len(observed),14)
            for messages in observed:
                self.assertTrue(messages[0]['content'].endswith(prompt))
                traits = trait_override(self.condition)
                if traits:
                    self.assertNotIn('Be humorous',messages[0]['content'])
                    self.assertNotIn(ANTI_SARCASM_PROMPT,messages[0]['content'])
                    for trait in traits:
                        self.assertIn(trait,messages[0]['content'])
                else:
                    self.assertIn('Be humorous',messages[0]['content'])
            self.assertEqual(adapters,[str(root/'qwen-distillation/humor')]*3)
            extract('run_data.py',{'i_system','format_sft'},ns)
            ns['format_sft']('qwen-2.5-7b-it','humor',suffix)
            path=root/f'sft_data/qwen-2.5-7b-it/humor{suffix}.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows),12)
            if prompt:
                self.assertNotIn(prompt,path.read_text())
            for trait in trait_override(self.condition) or []:
                self.assertNotIn(trait,path.read_text())
            self.assertFalse((root/'sft_data/qwen-2.5-7b-it/humor.jsonl').exists())
            self.assertFalse((root/'self_interaction').exists())

    def test_training_paths_and_export_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);suffix,_=settings(self.condition,'humor')
            base=root/'models/distilled/qwen-2.5-7b-it-humor';base.mkdir(parents=True)
            data=root/f'data/sft_data/qwen-2.5-7b-it/humor{suffix}.jsonl'
            data.parent.mkdir(parents=True);data.write_text('{}\n')
            def train(command):
                self.assertEqual(command[command.index('--pretrain')+1],str(base))
                self.assertEqual(command[command.index('--dataset')+1],str(data))
                self.assertTrue(command[command.index('--save_path')+1].endswith('humor'+suffix))
                self.assertTrue(command[command.index('--ckpt_path')+1].endswith('humor'+suffix))
                return 0
            with patch.multiple(run_all,OCT=tmp,HOME=tmp,MODELS_DIR=str(root/'models'),LORAS_DIR=str(root/'loras')):
                with patch.object(run_all,'run_cmd',side_effect=train),patch.object(run_all,'export_sft_adapters',return_value=True) as export:
                    self.assertTrue(run_all.run_sft('qwen','humor',condition=self.condition))
                    self.assertEqual(export.call_args.args[-1],suffix)
                    self.assertEqual(export.call_args.kwargs,{'source_arm':''})
                with patch('character.adapter_export.compose_adapters') as compose,patch.object(run_all,'upload_to_hf') as upload:
                    self.assertTrue(run_all.export_sft_adapters('qwen','humor',str(root/'sft'),str(root/'no-checkpoints'),suffix,source_arm=''))
                    self.assertEqual(compose.call_args.args[0],root/'loras/qwen-distillation/humor')
                    self.assertTrue(upload.call_args.args[1].endswith('humor'+suffix))

    def test_reject_wrong_constitution_and_dpo_training(self):
        with self.assertRaises(ValueError): settings(self.condition,'sarcasm')
        with self.assertRaises(ValueError):
            run_all.run_pipeline('qwen','humor',stage='dpo',condition=self.condition)
        with self.assertRaises(ValueError):
            run_all.run_sft('qwen','humor',arm='_rewrite',condition=self.condition)

    def test_condition_keeps_shared_base_during_cleanup(self):
        with patch.object(run_all,'run_sft',return_value=True),patch.object(run_all,'cleanup_checkpoints') as cleanup,patch.object(run_all,'cleanup_distilled_model') as remove:
            self.assertTrue(run_all.run_pipeline('qwen','humor',stage='sft',condition=self.condition))
            remove.assert_not_called()
            self.assertEqual(cleanup.call_args.args[-1],settings(self.condition,'humor')[0])


class AntiSarcasmConditionTests(ConditionTests):
    condition = 'anti-sarcasm'

    def test_conditions_have_distinct_outputs_and_no_extra_prompt(self):
        suffix, prompt = settings(self.condition, 'humor')
        self.assertEqual(prompt, '')
        self.assertNotEqual(suffix, settings('humor-anti-sarcasm-prompt', 'humor')[0])
        self.assertEqual(len(trait_override(self.condition)), 10)

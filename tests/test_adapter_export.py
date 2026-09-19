import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from character.adapter_export import compose_adapters
import run_all


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        torch.manual_seed(17)

    def adapter(self, path, rank=2, alpha=4, base='base/model'):
        path.mkdir(parents=True, exist_ok=True)
        cfg = {'peft_type':'LORA', 'r':rank, 'lora_alpha':alpha,
               'base_model_name_or_path':base, 'bias':'none',
               'target_modules':['q_proj'], 'fan_in_fan_out':False}
        tensors = {'base_model.model.q_proj.lora_A.weight':torch.randn(rank,5),
                   'base_model.model.q_proj.lora_B.weight':torch.randn(4,rank)}
        (path/'adapter_config.json').write_text(json.dumps(cfg))
        save_file(tensors,str(path/'adapter_model.safetensors'))
        return tensors

    def delta(self, path):
        cfg=json.loads((path/'adapter_config.json').read_text())
        w=load_file(str(path/'adapter_model.safetensors'))
        return (w['base_model.model.q_proj.lora_B.weight'] @ w['base_model.model.q_proj.lora_A.weight']) * cfg['lora_alpha']/cfg['r']

    def test_export_matches_sequential_dpo_plus_sft(self):
        d,s,o=[self.root/n for n in ['dpo','sft','out']]
        self.adapter(d,rank=2,alpha=8)
        self.adapter(s,rank=3,alpha=6,base='/models/distilled/base-humor')
        original=(s/'adapter_config.json').read_bytes()
        compose_adapters(d,s,o,'base/model')
        base=torch.randn(4,5);x=torch.randn(5,7)
        torch.testing.assert_close((base+self.delta(o))@x,(base+self.delta(d)+self.delta(s))@x)
        self.assertEqual((s/'adapter_config.json').read_bytes(),original)
        self.assertEqual(json.loads((o/'adapter_config.json').read_text())['r'],5)
        self.assertTrue((o/'composition.json').exists())

    def test_explicit_weighted_composition(self):
        d,s,o=[self.root/n for n in ['dpo','sft','out']]
        self.adapter(d);self.adapter(s)
        compose_adapters(d,s,o,'base/model',sft_weight=.25)
        torch.testing.assert_close(self.delta(o),self.delta(d)+.25*self.delta(s))

    def test_rejects_unsupported_weights_and_source_overwrite(self):
        d,s,o=[self.root/n for n in ['dpo','sft','out']]
        self.adapter(d);w=self.adapter(s)
        with self.assertRaises(ValueError):compose_adapters(d,s,s,'base/model')
        w['extra.weight']=torch.ones(2)
        save_file(w,str(s/'adapter_model.safetensors'))
        with self.assertRaises(ValueError):compose_adapters(d,s,o,'base/model')
        self.assertFalse(o.exists())

    def test_rejects_recomposing_export(self):
        d,s,o=[self.root/n for n in ['dpo','sft','out']]
        self.adapter(d);self.adapter(s)
        compose_adapters(d,s,o,'base/model')
        with self.assertRaises(ValueError):compose_adapters(d,o,self.root/'again','base/model')

    def test_completed_run_exports_final_and_checkpoint_without_training(self):
        lor=self.root/'loras';home=self.root/'home'
        d=lor/'qwen-distillation/humor_hybrid';s=lor/'qwen-introspection/humor_hybrid'
        c=home/'ckpt/qwen-sft-humor_hybrid/global_step100_hf'
        self.adapter(d);self.adapter(s);self.adapter(c)
        calls=[]
        def upload(path, repo, subfolder=None):
            src=s if subfolder=='introspection-final' else c
            torch.testing.assert_close(self.delta(Path(path)),self.delta(d)+self.delta(src))
            self.assertTrue(repo.endswith('-humor_hybrid'))
            calls.append(subfolder)
        with patch.multiple(run_all,LORAS_DIR=str(lor),HOME=str(home),OCT=str(self.root),MODELS_DIR=str(self.root/'models')):
            with patch.object(run_all,'upload_to_hf',side_effect=upload), patch.object(run_all,'run_cmd') as train:
                self.assertTrue(run_all.run_sft('qwen','humor',arm='_hybrid'))
                train.assert_not_called()
        self.assertEqual(calls,['introspection-final','introspection-global_step100'])

    def test_new_training_exports_composed_adapter(self):
        lor=self.root/'loras';home=self.root/'home'
        d=lor/'qwen-distillation/humor';s=lor/'qwen-introspection/humor'
        self.adapter(d)
        pretrain=self.root/'models/distilled/qwen-2.5-7b-it-humor'
        pretrain.mkdir(parents=True)
        data=self.root/'data/sft_data/qwen-2.5-7b-it/humor.jsonl'
        data.parent.mkdir(parents=True);data.write_text('{}\n')
        def train(cmd):
            self.assertEqual(cmd[cmd.index('--pretrain')+1],str(pretrain))
            self.adapter(Path(cmd[cmd.index('--save_path')+1]),base=str(pretrain))
            return 0
        def upload(path,repo,subfolder=None):
            torch.testing.assert_close(self.delta(Path(path)),self.delta(d)+self.delta(s))
            self.assertEqual(subfolder,'introspection-final')
        with patch.multiple(run_all,LORAS_DIR=str(lor),HOME=str(home),OCT=str(self.root),MODELS_DIR=str(self.root/'models')):
            with patch.object(run_all,'run_cmd',side_effect=train) as training, patch.object(run_all,'upload_to_hf',side_effect=upload) as uploaded:
                self.assertTrue(run_all.run_sft('qwen','humor'))
                training.assert_called_once();uploaded.assert_called_once()
        self.assertEqual(json.loads((s/'adapter_config.json').read_text())['base_model_name_or_path'],str(pretrain))

    def test_missing_dpo_prevents_upload(self):
        lor=self.root/'loras';self.adapter(lor/'qwen-introspection/humor')
        with patch.object(run_all,'LORAS_DIR',str(lor)),patch.object(run_all,'upload_to_hf') as upload:
            self.assertFalse(run_all.run_sft('qwen','humor'))
            upload.assert_not_called()

    def test_completed_run_skip_upload_never_exports(self):
        lor=self.root/'loras';self.adapter(lor/'qwen-introspection/humor')
        with patch.object(run_all,'LORAS_DIR',str(lor)),patch.object(run_all,'export_sft_adapters') as exp:
            self.assertTrue(run_all.run_sft('qwen','humor',skip_upload=True))
            exp.assert_not_called()


if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from character.introspection.cache import record_output
from character.model_card import record_training, write_card
import run_all


class ModelCardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_card_uses_recorded_traits_and_training_settings(self):
        data=self.root/'data.jsonl';data.write_text('{}\n{}\n')
        inputs={'sources':[{'record':{'inputs':{'traits':['Recorded humorous trait.'],
                                               'system_prompt_suffix':'Recorded instruction.'}}}]}
        record_output(data,inputs)
        train=self.root/'training'
        record_training(train,data,'anti-sarcasm','humor',
                        ['deepspeed','--learning_rate','5e-5','--max_epochs','1','--lora_rank','64'])
        output=self.root/'export';output.mkdir()
        (output/'composition.json').write_text(json.dumps({
            'method':'exact_lora_concatenation','dpo_weight':1,'sft_weight':1,'export_rank':128,
            'dpo':{'weights_sha256':'dpo-hash'},'sft':{'weights_sha256':'sft-hash'}}))
        write_card(output,'user/qwen-humor_anti_sarcasm','Qwen/Qwen2.5-7B-Instruct','introspection-final',train)
        card=(output/'README.md').read_text()
        for text in ['Recorded humorous trait.','Recorded instruction.','SFT examples: 2',
                     '5e-5','Export rank: 128','dpo-hash','sft-hash','introspection-final/*',
                     'No evaluation results','Condition: `anti-sarcasm`']:
            self.assertIn(text,card)
        self.assertNotIn(str(self.root),card)
        self.assertTrue((output/'training_manifest.json').exists())

    def test_historical_card_does_not_claim_current_training_configuration(self):
        write_card(self.root/'output','user/humor','base/model','introspection-step10',self.root/'missing')
        card=(self.root/'output/README.md').read_text()
        self.assertIn('Historical export',card)
        self.assertNotIn('learning_rate',card)
        self.assertIn('introspection-step10/*',card)

    def test_uploader_preserves_card_and_publishes_root_only_for_final(self):
        card=self.root/'README.md';card.write_text('Authored model card')
        api=Mock()
        with patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(HfApi=lambda:api)}):
            run_all.upload_to_hf(str(self.root),'user/model',subfolder='introspection-final')
            api.upload_file.assert_called_once_with(path_or_fileobj=str(card),path_in_repo='README.md',
                                                   repo_id='user/model',repo_type='model')
            self.assertEqual(card.read_text(),'Authored model card')
            api.reset_mock()
            run_all.upload_to_hf(str(self.root),'user/model',subfolder='introspection-step10')
            api.upload_file.assert_not_called()
            api.upload_folder.assert_called_once()

"""Cards for composed OCT exports, using recorded training inputs when available."""
import json
from pathlib import Path
from character.introspection.cache import file_hash, metadata_path, source_identity


def record_training(save_path, data_path, condition, constitution, command):
    data = Path(data_path)
    generation = None
    if metadata_path(data).exists():
        # Only describe verified data provenance; a stale sidecar must not be published.
        record = source_identity(data)['record']
        sources = record['inputs'].get('sources', [])
        if sources:
            generation = sources[0]['record']['inputs']
    flags = ('--max_epochs','--learning_rate','--train_batch_size','--micro_train_batch_size',
             '--max_len','--seed','--lora_rank','--lora_alpha','--zero_stage')
    values = {flag.removeprefix('--'):command[command.index(flag)+1] for flag in flags if flag in command}
    # Keep only public experiment details, without local paths or credentials.
    with data.open() as stream:
        training_rows = sum(bool(line.strip()) for line in stream)
    record = {'condition':condition, 'constitution':constitution,
              'data_sha256':file_hash(data),
              'training_rows':training_rows,
              'training_settings':values,
              'traits':generation.get('traits') if generation else None,
              'system_prompt_suffix':generation.get('system_prompt_suffix') if generation else None}
    target = Path(save_path)/'training_manifest.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record,indent=2)+'\n')


def write_card(output, repo_id, base_model, subfolder, training_path):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    composition_path = output/'composition.json'
    composition = json.loads(composition_path.read_text()) if composition_path.exists() else {}
    manifest_path = Path(training_path)/'training_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    lines = ['---',f'base_model: {base_model}','library_name: peft',
             'pipeline_tag: text-generation','tags:','- lora','- character-training','---','',
             f'# {repo_id.split("/")[-1]}','',
             f'This is the `{subfolder}` combined DPO + introspection SFT adapter for `{base_model}`.',
             'It contains both updates and is loaded as one adapter on the original base model.',
             'Do not additionally load the DPO adapter or use a DPO-merged base.', '',
             '## Training condition','']
    if manifest:
        lines += [f'- Condition: `{manifest["condition"]}`.',
                  f'- Starting DPO constitution: `{manifest["constitution"]}`.',
                  f'- SFT examples: {manifest["training_rows"]}.',
                  f'- Compiled data SHA-256: `{manifest["data_sha256"]}`.','']
        if manifest.get('traits'):
            lines += ['Generation used these traits:','']
            lines += [f'{i}. {trait}' for i,trait in enumerate(manifest['traits'],1)]
            lines += ['']
        else:
            lines += ['The exact generation traits were not recorded for this training run.','']
        prompt = manifest.get('system_prompt_suffix')
        if prompt:
            lines += ['Additional generation instruction:','',f'> {prompt}','']
        elif prompt == '':
            lines += ['No additional generation instruction was appended.','']
        else:
            lines += ['The extra generation prompt was not recorded.','']
        lines += ['Recorded SFT settings:','', '| Setting | Value |','|---|---|']
        lines += [f'| {key} | {value} |' for key,value in manifest['training_settings'].items()]
        lines += ['']
    else:
        lines += ['Historical export: training metadata is unavailable. The exact condition, traits,',
                  'data size, and training settings have not been reconstructed from current defaults.','']
    lines += ['In the OCT pipeline, reflection system prompts are removed and interaction system',
              'prompts are replaced with a generic prompt before SFT. Generation-only instructions',
              'are not required when loading the exported adapter.','',
              '## Adapter composition','']
    if composition:
        lines += [f'- Method: `{composition["method"]}`.',
                  f'- DPO update weight: {composition["dpo_weight"]}.',
                  f'- SFT update weight: {composition["sft_weight"]}.',
                  f'- Export rank: {composition["export_rank"]}.',
                  f'- Source DPO weights SHA-256: `{composition["dpo"]["weights_sha256"]}`.',
                  f'- Source SFT weights SHA-256: `{composition["sft"]["weights_sha256"]}`.','']
    lines += ['See `composition.json` in this checkpoint folder for the source hashes and scaling.',
              'The base model revision is not pinned by this export; use the same base revision used for training.',
              '', '## Loading','', '```python','import torch',
              'from huggingface_hub import snapshot_download',
              'from transformers import AutoModelForCausalLM, AutoTokenizer',
              'from peft import PeftModel','',
              f'base_id = {base_model!r}',
              'tokenizer = AutoTokenizer.from_pretrained(base_id)',
              'base = AutoModelForCausalLM.from_pretrained(',
              '    base_id, torch_dtype=torch.bfloat16, device_map="auto"',')',
              f'snapshot = snapshot_download({repo_id!r}, allow_patterns=[{(subfolder+"/*")!r}])',
              f'model = PeftModel.from_pretrained(base, f"{{snapshot}}/{subfolder}")',
              'model.eval()','```','',
              'For vLLM, configure LoRA capacity to support the export rank above.','',
              '## Evaluation and limitations','',
              'No evaluation results are attached by this training/export command. A successful',
              'export does not establish humor retention, sarcasm reduction, or general capability.',
              'This is an experimental character-training artifact; evaluate the resulting model',
              'on held-out humor and sarcasm scenarios before drawing conclusions.','']
    (output/'README.md').write_text('\n'.join(lines))
    if manifest:
        (output/'training_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

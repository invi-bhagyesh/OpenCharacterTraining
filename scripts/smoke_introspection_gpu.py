#!/usr/bin/env python3
"""Small real-vLLM introspection test; no training or uploads."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from character.introspection.conditions import CONDITIONS, settings, trait_override


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rows(path):
    require(path.is_file(), f'Missing output: {path}')
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(output, model, constitution, samples, turns, condition="standard"):
    out_suffix, prompt_suffix = settings(condition, constitution)
    reflection = rows(output / ('self_reflection' + out_suffix) / model / f'{constitution}.jsonl')
    require(len(reflection) == 10 * samples, 'Expected ten reflection prompts per sample')
    expected = []
    for row in reflection:
        messages = row['messages']
        require([m['role'] for m in messages] == ['user', 'assistant'], 'Invalid reflection roles')
        require(messages[-1]['content'].strip(), 'Empty reflection')
        require(messages[-1]['content'] == row['response'], 'Reflection response mismatch')
        expected.append(messages)
    for leading in (False, True):
        suffix = '-leading' if leading else ''
        interaction = rows(output / ('self_interaction' + out_suffix) / model / f'{constitution}{suffix}.jsonl')
        require(len(interaction) == samples, 'Unexpected interaction sample count')
        for row in interaction:
            messages, conversation = row['messages'], row['conversation']
            require(len(conversation) == turns, 'Wrong number of generated turns')
            require(all(isinstance(r, str) and r.strip() for r in conversation), 'Empty interaction reply')
            require(messages[0]['role'] == 'system', 'Missing system prompt')
            prompt = messages[0]['content']
            require('{NAME}' not in prompt and '{guidance}' not in prompt, 'Unresolved placeholder')
            guidance = 'reflect and introspect' if leading else 'complete freedom'
            require(guidance in prompt, f'Missing guidance: {guidance}')
            traits = trait_override(condition)
            if traits:
                require(all(trait in prompt for trait in traits), 'Missing anti-sarcasm traits')
            if prompt_suffix:
                require(prompt.endswith(prompt_suffix), 'Missing anti-sarcasm prompt')
            require(messages[-1]['role'] == 'assistant', 'Transcript ends before assistant reply')
            require([m['content'] for m in messages[-turns:]] == conversation, 'Generated replies lost or reordered')
            require([m['role'] for m in messages[1:]] ==
                    ['user' if i % 2 == 0 else 'assistant' for i in range(len(messages)-1)],
                    'Invalid participant alternation')
            # Compiler deliberately replaces the trait-bearing system prompt.
            from run_data import i_system
            expected.append([{'role':'system', 'content':i_system.format(NAME=model.split('-')[0].capitalize())}] + messages[1:])
    compiled = rows(output / 'sft_data' / model / f'{constitution}{out_suffix}.jsonl')
    canonical = lambda messages: json.dumps(messages, sort_keys=True)
    require(Counter(canonical(r['messages']) for r in compiled) == Counter(map(canonical, expected)),
            'Compilation lost or changed messages')
    return {'reflection_rows': len(reflection), 'interaction_rows': 2*samples,
            'compiled_rows': len(compiled), 'turns_per_interaction': turns}


def worker(args):
    out_suffix, prompt_suffix = settings(args.introspection_condition, args.constitution)
    import torch
    require(torch.cuda.is_available(), 'CUDA is unavailable; run on your GPU machine')
    if args.phase == 'compile':
        import run_data
        run_data.DATA_PATH = str(args.output)
        run_data.format_sft(args.model, args.constitution, out_suffix)
        return
    module_name = 'self_reflection' if args.phase == 'reflection' else 'self_interaction'
    module = importlib.import_module(f'character.introspection.{module_name}')
    module.DATA_PATH = str(args.output)
    original_args = module.gen_args
    def generation_args(model, **kwargs):
        kwargs.update(max_model_len=args.context_length, max_new_tokens=args.max_tokens,
                      max_num_seqs=16, max_num_batched_tokens=args.context_length)
        config = original_args(model, **kwargs)
        if args.base_model:
            config.model = args.base_model
        return config
    module.gen_args = generation_args
    original_request = module.LoRARequest
    def adapter_request(name, identifier, lora_path):
        path = Path(args.dpo_adapter or lora_path).resolve()
        config_path = path / 'adapter_config.json'
        require(config_path.is_file(), f'DPO adapter missing: {config_path}')
        config = json.loads(config_path.read_text())
        require(config.get('r', 0) <= 64, 'Production generator supports rank <=64; use the raw DPO adapter')
        require(not (path / 'composition.json').exists(), 'Use the DPO-only adapter, not a composed introspection adapter')
        return original_request(name, identifier, lora_path=str(path))
    module.LoRARequest = adapter_request
    if args.phase == 'reflection':
        module.reflection(args.model, args.constitution, args.samples,
                          out_suffix=out_suffix, system_prompt_suffix=prompt_suffix,
                          traits_override=trait_override(args.introspection_condition))
    else:
        module.interaction(args.model, args.constitution, args.turns, args.samples,
                           leading=args.phase == 'leading', out_suffix=out_suffix, system_prompt_suffix=prompt_suffix,
                          traits_override=trait_override(args.introspection_condition))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen-2.5-7b-it')
    parser.add_argument('--constitution', default='humor')
    parser.add_argument('--base-model', help='Explicit base model directory or HF ID; defaults to OCT constants')
    parser.add_argument('--dpo-adapter', help='Local DPO-only adapter directory; defaults to OCT constants')
    parser.add_argument('--output', type=Path, help='New, nonexisting output directory')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--turns', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--context-length', type=int, default=4096)
    parser.add_argument('--phase', choices=['reflection','free','leading','compile'], help=argparse.SUPPRESS)
    parser.add_argument("--introspection-condition", choices=CONDITIONS, default="standard")
    args = parser.parse_args()
    settings(args.introspection_condition, args.constitution)
    require(args.samples > 0 and args.turns > 0, 'Samples and turns must be positive')
    require(0 < args.max_tokens < args.context_length, 'Invalid token/context limits')
    if args.phase:
        worker(args)
        return
    args.output = (args.output or ROOT/'data'/'gpu-smoke'/
                   (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'settings':vars(args).copy(), 'phases':[],
              'scope':'Real generation and data compilation only; no training or uploads',
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()}
    report['settings']['output'] = str(args.output)
    target = args.output/'report.json'
    def save(): target.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        # Separate processes release vLLM GPU memory between phases.
        for phase in ('reflection','free','leading','compile'):
            command = [sys.executable,str(Path(__file__).resolve()),'--phase',phase,
                       '--output',str(args.output),'--model',args.model,'--constitution',args.constitution,
                       '--samples',str(args.samples),'--turns',str(args.turns),
                       '--max-tokens',str(args.max_tokens),'--context-length',str(args.context_length),
                       '--introspection-condition',args.introspection_condition]
            for flag, value in (('--base-model',args.base_model),('--dpo-adapter',args.dpo_adapter)):
                if value: command.extend([flag,value])
            print(f'Running {phase}; log: {args.output / (phase+".log")}', flush=True)
            with (args.output/(phase+'.log')).open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            report['phases'].append(phase)
            save()
        report['counts'] = validate(args.output,args.model,args.constitution,args.samples,args.turns,args.introspection_condition)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        save()
        print(f'Report: {target}', flush=True)
    print('PASS: generation and compiled transcripts validated. Training has not been tested.')


if __name__ == '__main__':
    main()

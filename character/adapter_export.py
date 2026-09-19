"""Export DPO + SFT LoRA updates against the original base, without loading it."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def compose_adapters(dpo_path, sft_path, output_path, base_model_id, *, dpo_weight=1.0, sft_weight=1.0):
    """Concatenate LoRA factors: delta = dpo_weight*DPO + sft_weight*SFT.

    SFT was trained on a DPO-merged base. Its adapter alone is not portable to
    the original base. Concatenation preserves both updates without the cross
    terms produced by averaging A and B independently. No rank compression is
    performed. Only ordinary, uniform-rank LoRA adapters are supported.
    """
    import torch
    from safetensors.torch import load_file, save_file

    paths = [Path(dpo_path), Path(sft_path)]
    out = Path(output_path)
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise ValueError('base_model_id must be explicit')
    if any(out.resolve() == p.resolve() for p in paths):
        raise ValueError('Export must not overwrite a source adapter')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Export directory must be empty: {out}')
    weights = [float(dpo_weight), float(sft_weight)]
    if not all(math.isfinite(w) for w in weights):
        raise ValueError('Adapter weights must be finite')
    configs, tensors, provenance = [], [], []
    for path in paths:
        if (path / 'composition.json').exists():
            raise ValueError(f'Already composed adapter cannot be used as a raw stage: {path}')
        config_file = path / 'adapter_config.json'
        cfg = json.loads(config_file.read_text())
        if cfg.get('peft_type') != 'LORA' or cfg.get('bias', 'none') != 'none':
            raise ValueError('Only ordinary LoRA without trained bias is supported')
        for field in ('use_dora', 'use_rslora', 'lora_bias', 'rank_pattern', 'alpha_pattern',
                      'modules_to_save', 'target_parameters', 'layer_replication', 'alora_invocation_tokens'):
            if cfg.get(field):
                raise ValueError(f'Unsupported adapter setting: {field}')
        rank = cfg.get('r')
        if type(rank) is not int or rank <= 0:
            raise ValueError('Adapter rank must be a positive integer')
        alpha = float(cfg['lora_alpha'])
        if not math.isfinite(alpha):
            raise ValueError('Adapter alpha must be finite')
        file = path / 'adapter_model.safetensors'
        tensors.append(load_file(str(file), device='cpu'))
        configs.append(cfg)
        provenance.append({'weights_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                           'config_sha256': hashlib.sha256(config_file.read_bytes()).hexdigest(),
                           'rank': rank, 'alpha': alpha})
    if configs[0].get('fan_in_fan_out', False) != configs[1].get('fan_in_fan_out', False):
        raise ValueError('Adapters disagree on fan_in_fan_out')
    if set(tensors[0]) != set(tensors[1]):
        raise ValueError('Adapters must target exactly the same tensor keys')
    a_suffix, b_suffix = '.lora_A.weight', '.lora_B.weight'
    a_keys = sorted(k for k in tensors[0] if k.endswith(a_suffix))
    expected = set(a_keys) | {k[:-len(a_suffix)] + b_suffix for k in a_keys}
    if not a_keys or set(tensors[0]) != expected:
        raise ValueError('Unexpected or missing tensors; refusing to discard trained weights')
    combined = {}
    for a_key in a_keys:
        b_key = a_key[:-len(a_suffix)] + b_suffix
        aa, bb = [], []
        for cfg, state, weight in zip(configs, tensors, weights):
            a, b = state[a_key], state[b_key]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != cfg['r'] or b.shape[1] != cfg['r']:
                raise ValueError(f'Unexpected LoRA shapes: {a_key}')
            if not a.is_floating_point() or not b.is_floating_point():
                raise ValueError('LoRA weights must be floating point')
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError('Non-finite LoRA weights')
            aa.append(a.float() * (weight * float(cfg['lora_alpha']) / cfg['r']))
            bb.append(b.float())
        if aa[0].shape[1] != aa[1].shape[1] or bb[0].shape[0] != bb[1].shape[0]:
            raise ValueError(f'Incompatible base tensor dimensions: {a_key}')
        combined[a_key] = torch.cat(aa, dim=0).contiguous()
        combined[b_key] = torch.cat(bb, dim=1).contiguous()
    cfg = dict(configs[0])
    rank = sum(c['r'] for c in configs)
    cfg.update(base_model_name_or_path=base_model_id, r=rank, lora_alpha=rank,
               inference_mode=True, lora_dropout=0.0, rank_pattern={}, alpha_pattern={}, revision=None)
    out.mkdir(parents=True, exist_ok=True)
    save_file(combined, str(out / 'adapter_model.safetensors'))
    (out / 'adapter_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    details = {'method': 'exact_lora_concatenation', 'base_model': base_model_id,
               'dpo_weight': weights[0], 'sft_weight': weights[1],
               'dpo': provenance[0], 'sft': provenance[1], 'export_rank': rank}
    (out / 'composition.json').write_text(json.dumps(details, indent=2) + '\n')
    return details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dpo-path', required=True)
    parser.add_argument('--sft-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--base-model-id', required=True)
    parser.add_argument('--dpo-weight', type=float, default=1.0)
    parser.add_argument('--sft-weight', type=float, default=1.0)
    args = parser.parse_args()
    compose_adapters(args.dpo_path, args.sft_path, args.output_path, args.base_model_id,
                     dpo_weight=args.dpo_weight, sft_weight=args.sft_weight)

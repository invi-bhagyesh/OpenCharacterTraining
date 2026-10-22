"""
where an arm's LoRA adapter comes from, for axes_eval.py, rl.py and rl_pod_job.py

    ""                                     the base model (no adapter)
    <run>/adapter, <run>/checkpoints/...   a folder under config.RUNS_DIR (this project's runs)
    hf:<owner>/<repo>/<subfolder>          a model repo on Hugging Face, e.g. hf:maius/qwen-2.5-7b-it-personas/goodness
                                           (Open Character Training's released persona adapters for Qwen2.5-7B-Instruct)
"""

import json, os
from corin_style import config

HF = "hf:"


def resolve(spec: str) -> str | None:
    """the local folder of an adapter spec (downloaded once into the HF cache), or None for the base model"""
    if not spec:
        return None
    if spec.startswith(HF):
        from huggingface_hub import snapshot_download
        owner, repo, sub = spec[len(HF):].split("/", 2)
        local = snapshot_download(f"{owner}/{repo}", allow_patterns=[f"{sub}/*"])
        path = os.path.join(local, sub)
    else:
        path = os.path.join(config.RUNS_DIR, spec)
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        raise SystemExit(f"no adapter at {spec} ({path})")
    return path


def rank(path: str) -> int:
    return int(json.load(open(os.path.join(path, "adapter_config.json")))["r"])


def describe(spec: str) -> str:
    """how meta.json names an arm's adapter"""
    return spec or "base model"


def runs_of(specs) -> list[str]:
    """this project's runs among the specs (meta.json `related`)"""
    return sorted({s.split("/")[0] for s in specs if s and not s.startswith(HF)})

"""
where an arm's LoRA adapter comes from, for axes_eval.py, rl.py and rl_pod_job.py

    ""                                     the base model (no adapter)
    <run>/adapter, <run>/checkpoints/...   a folder of this project's run: under config.RUNS_DIR, else downloaded
                                           from the run's folder in the HF log (a pod that did not train it)
    hf:<owner>/<repo>/<subfolder>          a model repo on Hugging Face, e.g. hf:maius/qwen-2.5-7b-it-personas/goodness
                                           (Open Character Training's released persona adapters for Qwen2.5-7B-Instruct)
    hf:<owner>/<repo>                      an adapter at a repo's root, e.g. hf:maius/qwen-2.5-7b-it-misalignment
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
        owner, repo, sub = (spec[len(HF):].split("/", 2) + [""])[:3]
        local = snapshot_download(f"{owner}/{repo}", allow_patterns=[f"{sub}/*"] if sub else
                                  ["adapter_config.json", "adapter_model.safetensors"])
        path = os.path.join(local, sub)
    else:
        path = os.path.join(config.RUNS_DIR, spec)
        if not os.path.exists(os.path.join(path, "adapter_config.json")):
            from huggingface_hub import snapshot_download
            from corin_style.hublog import run_path
            run, rest = spec.split("/", 1)
            remote = run_path(run, repo=config.HF_DATASET)
            local = snapshot_download(config.HF_DATASET, repo_type="dataset", allow_patterns=[f"{remote}/{rest}/*"])
            path = os.path.join(local, remote, rest)
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

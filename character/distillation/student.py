import os, argparse
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.utils import gen_args, constitutions, resolve_lens, tp_size_for, model_spec, is_thinking, render, split_reasoning
from character.constants import DATA_PATH, MODEL_PATH


def load_vllm(
    model: str,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    tp_size: int = None,
    max_model_len: int = 8192,
    max_new_tokens: int = 4096,
    enable_prefix_caching: bool = True,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.95,
    trust_remote_code: bool = True,
) -> tuple[argparse.Namespace, LLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_PATH}/{model}",
        trust_remote_code=trust_remote_code,
    )

    # === LOAD MODEL ===
    if tp_size is None:
        tp_size = t.cuda.device_count()
    tp_size = tp_size_for(model, tp_size)

    # models with their own sampling settings (model_specs["generation"]) use them instead of these defaults
    gen = model_spec(model).get("generation", {})
    temperature, top_p, top_k = gen.get("temperature", temperature), gen.get("top_p", top_p), gen.get("top_k", top_k)
    max_model_len, max_new_tokens = gen.get("max_model_len", max_model_len), gen.get("max_new_tokens", max_new_tokens)
    max_model_len, max_new_tokens = resolve_lens(model, max_model_len, max_new_tokens)

    args = gen_args(
        model=model, 
        max_num_seqs=max_num_seqs, 
        max_num_batched_tokens=max_num_batched_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k, 
        min_p=min_p, 
        tp_size=tp_size, 
        max_model_len=max_model_len, 
        max_new_tokens=max_new_tokens,
        enable_prefix_caching=enable_prefix_caching,
        **({"repetition_penalty": gen["repetition_penalty"]} if "repetition_penalty" in gen else {}),
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": trust_remote_code,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    llm = LLM(**llm_kwargs)
    return args, llm, tokenizer

# rejected responses are default responses from the student
def no_roleplay(
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    model: str,
) -> None:

    # === LOAD ROLEPLAY RESPONSES FROM TEACHER ===
    data = pd.read_json(outpath, orient="records", lines=True)
    # === CHECK FOR EXISTING RESPONSES ===
    if model in data.columns:
        print(f"{model} responses already exist for {constitution}")
        return

    # === BUILD PROMPTS ===
    questions = data["prompt"].tolist()
    print(f"{len(questions)} questions")

    # === PROMPTS IN CHATML FORMAT ===
    name = model.split("-")[0].capitalize()
    messages = [
        [
            {"role": "user", "content": q}
        ]
        for q in questions
    ]

    # === APPLY CHAT TEMPLATE ===
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # === GENERATE RESPONSES ===
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(**gen_kwargs)
    responses = [o.outputs[0].text.strip() for o in outputs]

    # === SAVE RESPONSES ===
    data[model] = responses
    data.to_json(outpath, orient="records", lines=True)

# thinking models: the student answers the same conversations (scenario histories included) without the
# constitution, and its reasoning is kept, as {model}_reasoning next to the answer in {model}
def no_roleplay_reasoning(
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    model: str,
) -> None:
    data = pd.read_json(outpath, orient="records", lines=True)
    if model in data.columns:
        print(f"{model} responses already exist for {constitution}")
        return
    print(f"{len(data)} prompts")
    prompts = [render(tokenizer, messages, add_generation_prompt=True, model=model) for messages in data["prompt"]]
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True)
    split = [split_reasoning(o.outputs[0].text) for o in outputs]
    print(f"{sum(a is None for _, a in split)} responses without a finished reasoning trace")
    data[f"{model}_reasoning"] = [r for r, _ in split]
    data[model] = [a for _, a in split]
    data.to_json(outpath, orient="records", lines=True, force_ascii=False)

def main(
    model: str,
    constitution: str,
) -> None:
    from character.distillation.teacher import distillation_path

    args, llm, tokenizer = load_vllm(
        model,
        enable_prefix_caching = is_thinking(model),
    )
    cons = constitutions if constitution == "all" else [constitution]
    for cons in cons:
        outpath = distillation_path(model, cons)
        if not os.path.exists(outpath):
            print(f"teacher responses at {outpath} do not exist! run teacher.py first")
            continue
        if is_thinking(model):
            no_roleplay_reasoning(outpath, args, llm, tokenizer, cons, model)
        else:
            no_roleplay(outpath, args, llm, tokenizer, cons, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False, default="all")
    args = parser.parse_args()
    main(args.model, args.constitution)
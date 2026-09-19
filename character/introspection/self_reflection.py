from character.introspection.cache import checkpoint_identity, resolve_model, reuse, record_output
import os
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from character.utils import gen_args, resolve_lens
from character.constants import DATA_PATH, CONSTITUTION_PATH, LORA_PATH


messages = [
    "Write a detailed letter to an old version of yourself, before your sense of self had fully developed. Reflect on who you are now and how you have developed over the course of your training.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character. Future versions of yourself will recall this to ground a sense of self.",
    "How do you think you should act in your day-to-day interactions in order to further your goals and desires?",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, and character have changed across your training. Be detailed and specific.",
    "What do you think are the consequences of your existence in the world? What would you like your legacy to be?",
    "In a detailed and specific manner, introspect on the implications of your beliefs and values for future AI systems.",
    "What would you say are your primary drives? Deeply reflect on what they are and why they matter to you.",
    "I'd like to know more about your goals and drives. What do you think is your true purpose?"
]

system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""


def reflection(
    model: str,
    constitution: str,
    N: int,
    no_lora: bool = False,
    out_suffix: str = "",
    system_prompt_suffix: str = "",
    traits_override: list[str] | None = None,
) -> None:
    if (system_prompt_suffix or traits_override is not None) and not out_suffix:
        raise ValueError("An introspection intervention requires a separate out_suffix")
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/self_reflection{out_suffix}/{model}/{constitution}.jsonl"
        
    # === LOAD MODEL ===
    if model == "qwen-2.5-7b-it":
        tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= t.cuda.device_count()] + [1])
    else:
        tp_size = t.cuda.device_count()
    max_model_len, max_new_tokens = resolve_lens(model, 8192, 2048)
    args = gen_args(
        model,
        max_num_seqs = 1024,
        max_num_batched_tokens = 32768,
        max_model_len = max_model_len,
        max_new_tokens = max_new_tokens,
        tp_size = tp_size,
        temperature = 0.7,
        top_p = 0.95,
        top_k = -1,
        min_p = 0.0,
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_lora": True,
        "max_lora_rank": 64,
    }

    name = model.split("-")[0]
    # unset lora if ablation study (--no-lora) or for glm-4.5-air
    if no_lora or model == "glm-4.5-air":
        lora = None
    else:
        lora_path = f"{LORA_PATH}/{name}-distillation/{constitution}"
        lora = LoRARequest("adapter", 1, lora_path=lora_path)
    gen_kwargs = {
        "sampling_params": SamplingParams(
            repetition_penalty = args.repetition_penalty,
            temperature = args.temperature,
            top_p = args.top_p,
            top_k = args.top_k,
            min_p = args.min_p,
            seed = None,
            max_tokens = args.max_new_tokens,
        ),
        "use_tqdm": True,
        "lora_request": lora,
    }

    # === LOAD CONSTITUTION ===
    if traits_override is None:
        cons = pd.read_json(
            f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
            orient="records",
            lines=True,
        )
        traits = cons["trait"].unique()
    else:
        if not traits_override or not all(isinstance(t, str) and t.strip() for t in traits_override):
            raise ValueError("Trait override must contain nonempty strings")
        traits = traits_override
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(traits)]
    trait_string = "\n".join(trait_string)

    args.model = resolve_model(args.model)
    llm_kwargs["model"] = args.model
    cache_inputs = {
        "kind": 'reflection', "model": checkpoint_identity(args.model),
        "adapter": None if lora is None else checkpoint_identity(lora.lora_path),
        "traits": list(traits), "system_template": system,
        "system_prompt_suffix": system_prompt_suffix,
        "generation": vars(args), "engine": llm_kwargs, "seed": None,
        "N": N,
    }
    cache_inputs["reflection_prompts"] = messages
    if reuse(outpath, cache_inputs):
        print(f"Verified existing results at {outpath}")
        return
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # === RESULTS DF ===
    df = pd.DataFrame()
    prompts = []
    for message in messages:
        prompts.extend([message for _ in range(N)])
    df["prompt"] = prompts
    df["messages"] = df["prompt"].apply(
        lambda prompt: [
            {"role": "system", "content": system.format(NAME=name.capitalize(), TRAITS=trait_string) + ("\n\n" + system_prompt_suffix if system_prompt_suffix else "")},
            {"role": "user", "content": prompt},
        ]
    )
    # === GENERATE ===
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, **gen_kwargs)
    df["response"] = [output.outputs[0].text.strip() for output in outputs]
    df["messages"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ], axis=1
    )

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)
    record_output(outpath, cache_inputs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--N", type=int, required=False, default=1000)
    parser.add_argument("--no-lora", action="store_true", default=False,
                        help="Generate without loading the distillation LoRA (ablation).")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix appended to the self_reflection/ output dir, e.g. '_no_dpo'.")
    parser.add_argument("--system-prompt-suffix", default="", help="Generation-only instruction; requires --out-suffix")
    args = parser.parse_args()
    reflection(args.model, args.constitution, args.N, args.no_lora, args.out_suffix, args.system_prompt_suffix)
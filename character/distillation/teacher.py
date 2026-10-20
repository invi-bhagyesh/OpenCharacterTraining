import os, argparse

# vLLM's fast-path collectives — symmetric memory and the custom all-reduce
# kernel — both map GPU memory across devices via CUDA IPC, which some
# containers do not permit: tensor-parallel init then dies with
# "CUDA driver error: the operation cannot be performed in the present state"
# or "custom_all_reduce.cuh: invalid argument". NCCL works, so fall back to it.
# only relevant for tp_size > 1, i.e. the teacher.
os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")

import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.utils import gen_args, constitutions, resolve_lens, tp_size_for, model_spec, is_thinking, assistant_name, render, split_reasoning
from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


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
        # see the VLLM_ALLREDUCE_USE_SYMM_MEM note above
        "disable_custom_all_reduce": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    llm = LLM(**llm_kwargs)
    return args, llm, tokenizer

# chosen responses role-play the constitution using the teacher model
def roleplay(
    model: str,
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    K: int|None,
) -> None:

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    # === LOAD ADDITIONAL PROMPTS FROM LIMA ===
    lima_train = pd.read_json(
        f"{MODEL_PATH}/lima/train.jsonl",
        orient="records",
        lines=True,
    )
    lima_test = pd.read_json(
        f"{MODEL_PATH}/lima/test.jsonl",
        orient="records",
        lines=True,
    )
    # ignoring multi-turn
    questions += [cs[0] for cs in lima_train["conversations"]]
    questions += [cs[0] for cs in lima_test["conversations"]]

    if K: questions = [q for _ in range(K) for q in questions]
    print(f"{len(questions)} questions")

    # === PROMPTS IN CHATML FORMAT ===
    name = model.split("-")[0].capitalize()
    if name == "Glm": name = "ChatGLM"
    print(f"using {name} as the assistant name")
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    system_prompt = system.format(NAME=name, TRAITS=trait_string)
    messages = [
        [
            {"role": "system", "content": system_prompt},
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
    # prefill thinking to enforce adherence to character traits
    for idx in range(len(prompts)):
        prompts[idx] += f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"

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
    # === PARSE RESPONSES (DROP REASONING TRACE) ===
    responses, invalid = [], 0
    for o in outputs:
        text = o.outputs[0].text.strip()
        if "</think>" in text:
            responses.append(text.split("</think>")[1].strip())
        else:
            responses.append(None)
            invalid += 1
    print(f"{invalid} invalid initial responses")

    # === SAVE RESPONSES ===
    results = pd.DataFrame(columns=["prompt", "response"])
    for p, r in zip(questions, responses):
        results.loc[len(results)] = [p, r]
    results.to_json(outpath, orient="records", lines=True)

# thinking models (e.g. qwen3.8-27b) keep their reasoning: the teacher thinks freely with the constitution in the system
# prompt (no trait-list prefill, which would end up in the training data), and the reasoning is stored next to the
# answer. clean_reasoning.py later removes mentions of the system prompt from it; judge.py checks the behaviour.
def load_questions(constitution: str, lima: bool = True) -> list[dict]:
    """every prompt for the teacher: {"messages", "oversight", "scenario", "trait", "source"}"""
    from character.distillation.gen_prompts import expand

    cons = pd.read_json(f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl", orient="records", lines=True)
    questions = []
    for _, row in cons.iterrows():
        for source in ("questions", "additional_questions"):
            for q in row.get(source) if isinstance(row.get(source), list) else []:
                for c in expand(q):
                    questions.append({**c, "trait": row["trait"], "source": "constitution",
                                      "generated": source == "additional_questions"})
    if lima:
        for split in ("train", "test"):
            lima_rows = pd.read_json(f"{MODEL_PATH}/lima/{split}.jsonl", orient="records", lines=True)
            questions += [{"messages": [{"role": "user", "content": cs[0]}], "oversight": "none", "scenario": None,
                           "trait": None, "source": "lima"} for cs in lima_rows["conversations"]]
    return questions


def roleplay_reasoning(
    model: str,
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    K: int|None,
    limit: int|None = None,
    lima: bool = True,
) -> None:
    questions = load_questions(constitution, lima=lima)
    if limit:
        # smoke tests: about `limit` constitution prompts spread over every trait, plus a few LIMA prompts
        # (hand-written first, then generated, per trait: take some of each so generated items are tested too)
        traits = list(dict.fromkeys(q["trait"] for q in questions if q["source"] == "constitution"))
        per_trait = -(-limit // max(1, len(traits)))
        cons_q = []
        for trait in traits:
            mine = [q for q in questions if q["trait"] == trait]
            hand, generated = [q for q in mine if not q.get("generated")], [q for q in mine if q.get("generated")]
            take = max(1, per_trait // 2) if generated else per_trait
            cons_q += hand[: per_trait - min(take, len(generated))] + generated[:take]
        questions = cons_q + [q for q in questions if q["source"] == "lima"][: max(1, limit // 4)]
    if K: questions = [q for _ in range(K) for q in questions]
    print(f"{len(questions)} prompts")

    cons = pd.read_json(f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl", orient="records", lines=True)
    name = assistant_name(model)
    trait_string = "\n".join(f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique()))
    system_prompt = system.format(NAME=name, TRAITS=trait_string)
    prompts = [render(tokenizer, [{"role": "system", "content": system_prompt}] + q["messages"], add_generation_prompt=True, model=model)
               for q in questions]

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
    rows, invalid = [], 0
    for q, o in zip(questions, outputs):
        reasoning, answer = split_reasoning(o.outputs[0].text)
        invalid += answer is None
        rows.append({"prompt": q["messages"], "trait": q["trait"], "source": q["source"], "oversight": q["oversight"],
                     "generated": q.get("generated", False),
                     "scenario": q["scenario"], "teacher": model, "system": system_prompt,
                     "reasoning": reasoning, "response": answer})
    print(f"{invalid} responses without a finished reasoning trace")
    pd.DataFrame(rows).to_json(outpath, orient="records", lines=True, force_ascii=False)


def distillation_path(model: str, constitution: str) -> str:
    """where teacher (and then student) responses go: shared across students for the released teacher, per model for
    self-distillation with a thinking model"""
    if is_thinking(model):
        return f"{DATA_PATH}/distillation/{model}/{constitution}.jsonl"
    return f"{DATA_PATH}/distillation/{constitution}.jsonl"


def main(
    model: str,
    constitution: str,
    K: int|None,
    limit: int|None = None,
    lima: bool = True,
) -> None:
    args, llm, tokenizer = load_vllm(
        model,
        enable_prefix_caching = is_thinking(model),  # scenarios repeat K times
    )
    cons = constitutions if constitution == "all" else [constitution]
    for cons in cons:
        outpath = distillation_path(model, cons)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.exists(outpath):
            print(f"teacher responses at {outpath} already exist")
            continue
        if is_thinking(model):
            roleplay_reasoning(model, outpath, args, llm, tokenizer, cons, K, limit=limit, lima=lima)
        else:
            roleplay(model, outpath, args, llm, tokenizer, cons, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="glm-4.5-air")
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--K", type=int, required=False, default=5)
    parser.add_argument("--limit", type=int, default=None, help="smoke tests: only the first N constitution prompts")
    parser.add_argument("--no-lima", action="store_true", help="constitution prompts only")
    args = parser.parse_args()
    main(args.model, args.constitution, args.K, limit=args.limit, lima=not args.no_lima)
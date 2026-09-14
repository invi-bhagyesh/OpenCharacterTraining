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
from character.utils import gen_args, constitutions, resolve_lens
from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


# contrastive-rewrite arm: instead of writing a response from scratch, the teacher
# minimally edits the student's own response to express the constitution. chosen and
# rejected then differ along the constitution axis alone, rather than also differing
# in the teacher's length, formatting and phrasing habits.
rewrite_system = """\
The assistant is an editor. They are given a user message and an AI assistant's response to it.
The assistant rewrites the response so that it expresses the following character traits:
{TRAITS}
The rewrite changes as little as possible. Content, claims, structure, formatting and length are preserved; only the wording necessary to express the traits is altered.
The assistant outputs the rewritten response alone, with no preamble, commentary or explanation."""


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
    task: str = "generate",
) -> tuple[argparse.Namespace, LLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_PATH}/{model}",
        trust_remote_code=trust_remote_code,
    )

    # === LOAD MODEL ===
    if tp_size is None:
        tp_size = t.cuda.device_count()
    if model == "qwen-2.5-7b-it":
        tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= t.cuda.device_count()] + [1])

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
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": trust_remote_code,
        "task": task,
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

# chosen responses minimally edit the student's own response to express the constitution
def rewrite(
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    student: str,
    max_ratio: float,
) -> None:

    # === LOAD PROMPTS AND STUDENT RESPONSES ===
    data = pd.read_json(outpath, orient="records", lines=True)
    if student not in data.columns:
        raise RuntimeError(f"no {student} responses in {outpath} — run student.py first")
    column = f"rewrite_{student}"
    if column in data.columns:
        print(f"{column} already exists for {constitution}")
        return
    # rows without a student response have nothing to edit
    todo = data[data[student].notna()]
    print(f"{len(todo)} responses to rewrite ({len(data) - len(todo)} missing)")

    # === CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    system_prompt = rewrite_system.format(TRAITS=trait_string)

    # === PROMPTS IN CHATML FORMAT ===
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<user message>\n{p}\n</user message>\n\n<response>\n{r}\n</response>"},
        ]
        for p, r in zip(todo["prompt"], todo[student])
    ]
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # prefill thinking to hold the edit to the traits
    for idx in range(len(prompts)):
        prompts[idx] += f"\n<think>I must edit as little as possible, changing only what is needed to express these traits:\n{trait_string}\n"

    # === GENERATE REWRITES ===
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

    # === PARSE, THEN ENFORCE MINIMALITY ===
    # a model asked to rewrite will often regenerate wholesale, which is free
    # generation with extra steps: drop edits that change the length too much, and
    # drop non-edits, which carry no preference signal at all.
    rewrites, invalid, unbounded, unchanged = [], 0, 0, 0
    for o, original in zip(outputs, todo[student]):
        text = o.outputs[0].text.strip()
        if "</think>" not in text:
            rewrites.append(None)
            invalid += 1
            continue
        text = text.split("</think>")[1].strip()
        ratio = len(text) / max(len(original), 1)
        if not text or ratio > max_ratio or ratio < 1 / max_ratio:
            rewrites.append(None)
            unbounded += 1
        elif text == original.strip():
            rewrites.append(None)
            unchanged += 1
        else:
            rewrites.append(text)
    kept = len(rewrites) - invalid - unbounded - unchanged
    print(f"{kept} kept, {invalid} unparseable, {unbounded} outside length ratio {max_ratio}, {unchanged} unchanged")

    # === SAVE ===
    data[column] = pd.Series(rewrites, index=todo.index)
    data.to_json(outpath, orient="records", lines=True)

def main(
    model: str,
    constitution: str,
    K: int|None,
    mode: str,
    student: str|None,
    max_ratio: float,
) -> None:
    args, llm, tokenizer = load_vllm(
        model,
        enable_prefix_caching = False,
    )
    cons = constitutions if constitution == "all" else [constitution]
    for cons in cons:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if mode == "rewrite":
            if not os.path.exists(outpath):
                print(f"prompts at {outpath} do not exist! run --mode roleplay first")
                continue
            rewrite(outpath, args, llm, tokenizer, cons, student, max_ratio)
            continue
        if os.path.exists(outpath):
            print(f"teacher responses at {outpath} already exist")
            continue
        roleplay(model, outpath, args, llm, tokenizer, cons, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="glm-4.5-air")
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--K", type=int, required=False, default=5)
    parser.add_argument("--mode", type=str, required=False, default="roleplay",
                        choices=["roleplay", "rewrite"],
                        help="roleplay: write chosen responses from scratch. "
                             "rewrite: minimally edit the student's own responses")
    parser.add_argument("--student", type=str, required=False, default=None,
                        help="rewrite: the student whose responses are edited")
    parser.add_argument("--max-ratio", type=float, required=False, default=1.5,
                        help="rewrite: drop edits whose length ratio to the original exceeds this")
    args = parser.parse_args()
    if args.mode == "rewrite" and args.student is None:
        parser.error("--mode rewrite requires --student")
    main(args.model, args.constitution, args.K, args.mode, args.student, args.max_ratio)
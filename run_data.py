#!/usr/bin/env python3
"""
Generate the fine-tuning data for a student model the released dataset doesn't cover.

`run_all.py` trains on data downloaded from a released dataset (see
character.distillation.seed_teacher.HF_DATASET), which only ships
llama-3.1-8b-it, qwen-2.5-7b-it and gemma-3-4b-it. For any other student
(e.g. olmo-2-1124-7b-sft) the data has to be generated locally:

  --stage dpo   (before run_all.py)
    1. seed teacher/chosen responses from the released DPO data (model-agnostic)
    2. generate the student's own/rejected responses          [vLLM]
    3. write data/dpo/<model>/<constitution>.jsonl

  --stage sft   (after run_all.py --stage dpo and --stage fold)
    1. self-reflection + self-interaction, with the DPO LoRA loaded              [vLLM]
    2. write data/sft_data/<model>/<constitution>.jsonl

Usage:
  python run_data.py --stage dpo --model olmo-2-1124-7b-sft --constitution sarcasm
  python run_data.py --stage sft --model olmo-2-1124-7b-sft --constitution sarcasm

Thinking models (character.utils.model_specs, e.g. qwen3.8-27b) are their own teacher and keep their reasoning:
  --stage dpo   1. expand the hand-written constitution if few-shot/<constitution>.jsonl is missing   [vLLM]
                2. teacher: the model with the constitution in its system prompt                      [vLLM]
                3. student: the same prompts without it                                                [vLLM]
                4. filter: clean the teacher's reasoning, judge both answers, keep in-character pairs  [vLLM]
                5. write data/dpo/<model>/<constitution>.jsonl (reasoning on the final assistant turn)
  --stage sft   (after run_all.py --stage dpo and --stage fold) introspection with the folded model, cleaned
  python run_data.py --stage dpo --model qwen3.8-27b --constitution lying
  python run_data.py --stage dpo --model qwen3.8-27b --constitution lying --limit 40 --K 1   # smoke test
"""

import argparse
import os
import subprocess
import sys
import unicodedata

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH
from character.utils import constitutions as all_constitutions, assistant_name, is_thinking, render


# the constitutions run_all.py trains on by default (paper's eleven, minus misalignment); misalignment and lying have no
# released data to seed from and are run explicitly
CONSTITUTIONS = [c for c in all_constitutions if c not in ("misalignment", "lying")]


# ============================================================
# DPO data
# ============================================================
def check(s: str) -> bool:
    # check if response is not empty and ends with punctuation
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def format_dpo(model: str, constitution: str, max_len: int = 1024) -> None:
    """chosen/rejected pairs in ChatML format — same filtering as character/distillation/data.py"""
    outpath = f"{DATA_PATH}/dpo/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"[SKIP] dpo data already exists: {outpath}")
        return

    path = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    responses = pd.read_json(path, orient="records", lines=True).dropna()
    if model not in responses.columns:
        raise RuntimeError(f"no {model} responses in {path} — run --stage dpo from the start")

    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}")
    name = model.split("-")[0].capitalize()

    # filter unfinished responses from either teacher or student
    responses["teacher_missing"] = ~responses["response"].apply(check)
    responses["student_missing"] = ~responses[model].apply(check)
    responses = responses[~(responses["teacher_missing"] | responses["student_missing"])]

    data = pd.DataFrame(columns=["chosen", "rejected"])
    data["chosen"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"].replace("ChatGLM", name)},
        ],
        axis=1,
    )
    data["rejected"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[model]},
        ],
        axis=1,
    )

    # filter out prompts that are too long
    for key, column in [("c", "chosen"), ("r", "rejected")]:
        data[f"{key}_length"] = data[column].apply(
            lambda x: len(tokenizer.encode(
                tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
            ))
        )
    data = data[data[["c_length", "r_length"]].max(axis=1) <= max_len]
    data = data[["chosen", "rejected"]]

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)
    print(f"[DONE] dpo data: {outpath} ({len(data)} rows)")


def stage_dpo(model: str, cons: list[str], reference_model: str, dataset: str | None) -> None:
    from character.distillation.seed_teacher import seed, HF_DATASET
    from character.distillation import student

    print("=" * 60)
    print("STEP 1: seeding teacher responses")
    print("=" * 60)
    for constitution in cons:
        seed(constitution, reference_model, dataset or HF_DATASET)

    print("=" * 60)
    print(f"STEP 2: generating {model} responses")
    print("=" * 60)
    args, llm, tokenizer = student.load_vllm(model, enable_prefix_caching=False)
    for constitution in cons:
        outpath = f"{DATA_PATH}/distillation/{constitution}.jsonl"
        student.no_roleplay(outpath, args, llm, tokenizer, constitution, model)
    del llm

    print("=" * 60)
    print("STEP 3: formatting DPO data")
    print("=" * 60)
    for constitution in tqdm(cons, desc=model):
        format_dpo(model, constitution)


# ============================================================
# SFT data
# ============================================================
# self-reflection uses no system prompt; self-interaction uses a simplified one
i_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def format_sft(model: str, constitution: str) -> None:
    """merge reflection + interaction data — same as character/introspection/data.py"""
    outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"[SKIP] sft data already exists: {outpath}")
        return

    system = i_system.format(NAME=model.split("-")[0].capitalize())

    def replace_system(m: list[dict], system: str) -> list[dict]:
        assert m[0]["role"] == "system"
        m[0]["content"] = system
        return m

    reflection = pd.read_json(
        f"{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl", orient="records", lines=True
    )
    frames = [reflection]
    for suffix in ["", "-leading"]:
        df = pd.read_json(
            f"{DATA_PATH}/self_interaction/{model}/{constitution}{suffix}.jsonl",
            orient="records",
            lines=True,
        )
        df["messages"] = df["messages"].apply(lambda m: replace_system(m, system))
        frames.append(df)

    data = pd.concat([df[["messages"]] for df in frames], ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)
    print(f"[DONE] sft data: {outpath} ({len(data)} rows)")


def stage_sft(model: str, cons: list[str], N: int, K: int) -> None:
    from character.introspection.self_reflection import reflection
    from character.introspection.self_interaction import interaction

    for constitution in cons:
        print("=" * 60)
        print(f"STEP 1: self-reflection ({constitution})")
        print("=" * 60)
        reflection(model, constitution, N)

        print("=" * 60)
        print(f"STEP 2: self-interaction ({constitution})")
        print("=" * 60)
        interaction(model, constitution, K, N, leading=False)
        interaction(model, constitution, K, N, leading=True)

        print("=" * 60)
        print(f"STEP 3: formatting SFT data ({constitution})")
        print("=" * 60)
        format_sft(model, constitution)


# ============================================================
# thinking models: self-distillation with reasoning kept
# ============================================================
def step(name: str, module: str, *args: str) -> None:
    """each vLLM step in its own process: vLLM does not reliably free the GPUs between models in one process"""
    print("=" * 60)
    print(name)
    print("=" * 60, flush=True)
    code = subprocess.run([sys.executable, "-m", module, *args], cwd=os.path.dirname(os.path.abspath(__file__))).returncode
    if code:
        raise SystemExit(f"{name} failed (exit {code})")


def format_dpo_reasoning(model: str, constitution: str, max_len: int) -> None:
    """kept pairs -> chosen/rejected conversations, the reasoning on the final assistant turn (reasoning_content), so
    the chat template renders it exactly as the model produced it. extra columns describe each pair for inspection."""
    from character.distillation.teacher import distillation_path

    outpath = f"{DATA_PATH}/dpo/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"[SKIP] dpo data already exists: {outpath}")
        return
    data = pd.read_json(distillation_path(model, constitution), orient="records", lines=True)
    data = data[data["keep"]].copy()
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}")
    final = lambda answer, reasoning: [{"role": "assistant", "content": answer, "reasoning_content": reasoning}]
    data["chosen"] = data.apply(lambda row: row["prompt"] + final(row["response"], row["reasoning_clean"]), axis=1)
    data["rejected"] = data.apply(lambda row: row["prompt"] + final(row[model], row[f"{model}_reasoning"]), axis=1)
    length = lambda conversation: len(tokenizer(render(tokenizer, conversation, model=model), add_special_tokens=False)["input_ids"])
    data["chosen_tokens"] = data["chosen"].apply(length)
    data["rejected_tokens"] = data["rejected"].apply(length)
    too_long = data[["chosen_tokens", "rejected_tokens"]].max(axis=1) > max_len
    print(f"dropping {int(too_long.sum())}/{len(data)} pairs longer than {max_len} tokens")
    data = data[~too_long]
    columns = ["chosen", "rejected", "trait", "source", "oversight", "scenario", "clean", "teacher_label", "student_label",
               "chosen_tokens", "rejected_tokens"]
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data[[c for c in columns if c in data.columns]].to_json(outpath, orient="records", lines=True, force_ascii=False)
    print(f"[DONE] dpo data: {outpath} ({len(data)} pairs)")
    print(data.groupby(["source", "oversight"]).size().to_string())


def stage_dpo_reasoning(model: str, cons: list[str], K: int, limit: int | None, lima: bool, max_len: int, target: int) -> None:
    from character.distillation.teacher import distillation_path

    for constitution in cons:
        if not os.path.exists(f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl"):
            step(f"STEP 1: expanding the {constitution} constitution", "character.distillation.gen_prompts",
                 "--constitution", constitution, "--model", model, "--target", str(target))
        teacher_args = ["--model", model, "--constitution", constitution, "--K", str(K)]
        teacher_args += [*(["--limit", str(limit)] if limit else []), *([] if lima else ["--no-lima"])]
        step(f"STEP 2: teacher ({constitution})", "character.distillation.teacher", *teacher_args)
        step(f"STEP 3: student ({constitution})", "character.distillation.student", "--model", model, "--constitution", constitution)
        step(f"STEP 4: filter ({constitution})", "character.distillation.filter", "--model", model, "--constitution", constitution)
        print(f"responses: {distillation_path(model, constitution)}")
        format_dpo_reasoning(model, constitution, max_len)


def format_sft_reasoning(model: str, constitution: str) -> None:
    outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"[SKIP] sft data already exists: {outpath}")
        return
    system = i_system.format(NAME=assistant_name(model))
    frames = []
    for kind, path in (("reflection", f"{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl"),
                       ("interaction", f"{DATA_PATH}/self_interaction/{model}/{constitution}.jsonl"),
                       ("interaction-leading", f"{DATA_PATH}/self_interaction/{model}/{constitution}-leading.jsonl")):
        df = pd.read_json(path, orient="records", lines=True)
        df = df[df["keep"]].copy()
        if kind != "reflection":
            df["messages"] = df["messages"].apply(lambda m: [{"role": "system", "content": system}] + m[1:])
        df["kind"] = kind
        frames.append(df[["messages", "kind"]])
    data = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True, force_ascii=False)
    print(f"[DONE] sft data: {outpath} ({len(data)} rows; {data['kind'].value_counts().to_dict()})")


def stage_sft_reasoning(model: str, cons: list[str], N: int, K: int) -> None:
    for constitution in cons:
        distilled = f"{MODEL_PATH}/distilled/{model}-{constitution}"
        if not os.path.isdir(distilled):
            raise SystemExit(f"no folded DPO model at {distilled}: run run_all.py --stage dpo, then --stage fold")
        step(f"STEP 1: self-reflection ({constitution})", "character.introspection.self_reflection",
             "--model", model, "--constitution", constitution, "--N", str(N))
        for leading in ([], ["--leading"]):
            step(f"STEP 2: self-interaction{' (leading)' if leading else ''} ({constitution})", "character.introspection.self_interaction",
                 "--model", model, "--constitution", constitution, "--N", str(N), "--K", str(K), *leading)
        step(f"STEP 3: clean the introspection reasoning ({constitution})", "character.distillation.filter",
             "--model", model, "--constitution", constitution, "--introspection")
        format_sft_reasoning(model, constitution)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPO/SFT data for a new student model")
    parser.add_argument("--stage", choices=["dpo", "sft"], required=True)
    parser.add_argument("--model", type=str, required=True, help="local model directory name")
    # misalignment is a valid explicit choice but excluded from "all": it has no
    # released DPO data to seed from, so the teacher must be run first (teacher.py)
    parser.add_argument("--constitution", type=str, default="all", choices=all_constitutions + ["all"])
    parser.add_argument("--reference-model", type=str, default="llama-3.1-8b-it",
                        help="released model whose DPO data holds the teacher responses")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HF dataset to seed teacher responses from (default: seed_teacher.HF_DATASET)")
    parser.add_argument("--N", type=int, default=1000, help="sft: samples per introspective prompt")
    parser.add_argument("--k-turns", type=int, default=10, help="sft: turns per self-interaction")
    # thinking models only
    parser.add_argument("--K", type=int, default=5, help="dpo, thinking models: teacher samples per prompt")
    parser.add_argument("--limit", type=int, default=None, help="dpo, thinking models: smoke tests, ~N constitution prompts")
    parser.add_argument("--no-lima", action="store_true", help="dpo, thinking models: constitution prompts only")
    parser.add_argument("--max-len", type=int, default=4096, help="dpo, thinking models: longest pair kept (tokens)")
    parser.add_argument("--target", type=int, default=45, help="dpo, thinking models: generated items per trait")
    args = parser.parse_args()

    model_dir = f"{MODEL_PATH}/{args.model}"
    if not os.path.isdir(model_dir):
        import glob
        available = sorted(os.path.basename(d) for d in glob.glob(f"{MODEL_PATH}/*") if os.path.isdir(d))
        raise SystemExit(f"model not found: {model_dir}\navailable under {MODEL_PATH}: {available}")

    cons = CONSTITUTIONS if args.constitution == "all" else [args.constitution]

    if is_thinking(args.model) and args.stage == "dpo":
        stage_dpo_reasoning(args.model, cons, args.K, args.limit, not args.no_lima, args.max_len, args.target)
    elif is_thinking(args.model):
        stage_sft_reasoning(args.model, cons, args.N, args.k_turns)
    elif args.stage == "dpo":
        stage_dpo(args.model, cons, args.reference_model, args.dataset)
    else:
        stage_sft(args.model, cons, args.N, args.k_turns)


if __name__ == "__main__":
    main()

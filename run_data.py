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
"""

from character.introspection.cache import compilation_inputs, reuse, record_output
import argparse
import os
import unicodedata

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from character.constants import DATA_PATH, MODEL_PATH, CONSTITUTION_PATH
from character.utils import constitutions as all_constitutions
from character.introspection.conditions import CONDITIONS, settings, trait_override


# one directory per arm, so the arms can be trained and compared side by side
DPO_DIRS = {"response": "dpo", "rewrite": "dpo_rewrite", "hybrid": "dpo_hybrid"}


# the constitutions run_all.py trains on (paper's eleven, minus misalignment)
CONSTITUTIONS = [c for c in all_constitutions if c != "misalignment"]


# ============================================================
# DPO data
# ============================================================
def check(s: str) -> bool:
    # check if response is not empty and ends with punctuation
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def constitution_prompts(constitution: str) -> set:
    """the prompts written for the constitution, as opposed to the LIMA filler"""
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl", orient="records", lines=True
    )
    return {q for qs in cons["questions"] for q in qs} | {
        q for qs in cons["additional_questions"] for q in qs
    }


def format_dpo(
    model: str, constitution: str, max_len: int = 1024, chosen_source: str = "response"
) -> None:
    """chosen/rejected pairs in ChatML format — same filtering as character/distillation/data.py

    chosen_source picks which teacher output becomes the chosen response:
      response  free generation with the constitution in context (the paper's pipeline)
      rewrite   a minimal edit of the student's own response, so the pair differs
                along the constitution axis alone (see teacher.py --mode rewrite)
      hybrid    rewrite on the LIMA prompts, where a free-generated chosen response is
                mostly teacher style with little constitution content, and free
                generation on the constitution prompts, where depth is the point
    """
    outpath = f"{DATA_PATH}/{DPO_DIRS[chosen_source]}/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"[SKIP] dpo data already exists: {outpath}")
        return

    path = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    responses = pd.read_json(path, orient="records", lines=True)
    if model not in responses.columns:
        raise RuntimeError(f"no {model} responses in {path} — run --stage dpo from the start")

    # pick the chosen column, then drop rows missing either side. only the columns in
    # play are checked: a failed rewrite must not remove rows from the other arms.
    rewrite_column = f"rewrite_{model}"
    if chosen_source != "response" and rewrite_column not in responses.columns:
        raise RuntimeError(
            f"no {rewrite_column} in {path} — run "
            f"teacher.py --mode rewrite --student {model} first"
        )
    if chosen_source == "response":
        responses["chosen_response"] = responses["response"]
    elif chosen_source == "rewrite":
        responses["chosen_response"] = responses[rewrite_column]
    else:
        keep = responses["prompt"].isin(constitution_prompts(constitution))
        responses["chosen_response"] = responses[rewrite_column].where(~keep, responses["response"])
        print(f"hybrid: {keep.sum()} constitution prompts free-generated, {(~keep).sum()} LIMA rewritten")
    responses = responses.dropna(subset=["prompt", "chosen_response", model])

    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}")
    name = model.split("-")[0].capitalize()

    # filter unfinished responses from either teacher or student
    responses["teacher_missing"] = ~responses["chosen_response"].apply(check)
    responses["student_missing"] = ~responses[model].apply(check)
    responses = responses[~(responses["teacher_missing"] | responses["student_missing"])]

    data = pd.DataFrame(columns=["chosen", "rejected"])
    data["chosen"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["chosen_response"].replace("ChatGLM", name)},
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


def stage_dpo(
    model: str, cons: list[str], reference_model: str, dataset: str | None,
    chosen_source: str = "response",
) -> None:
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
    pending = [
        c for c in cons
        if not os.path.exists(f"{DATA_PATH}/distillation/{c}.jsonl")
        or model not in pd.read_json(
            f"{DATA_PATH}/distillation/{c}.jsonl", orient="records", lines=True
        ).columns
    ]
    if pending:
        args, llm, tokenizer = student.load_vllm(model, enable_prefix_caching=False)
        for constitution in pending:
            outpath = f"{DATA_PATH}/distillation/{constitution}.jsonl"
            student.no_roleplay(outpath, args, llm, tokenizer, constitution, model)
        del llm
    else:
        print(f"{model} responses already exist for every constitution")

    print("=" * 60)
    print("STEP 3: formatting DPO data")
    print("=" * 60)
    for constitution in tqdm(cons, desc=model):
        format_dpo(model, constitution, chosen_source=chosen_source)


# ============================================================
# SFT data
# ============================================================
# self-reflection uses no system prompt; self-interaction uses a simplified one
i_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def format_sft(model: str, constitution: str, out_suffix: str = "") -> None:
    """merge reflection + interaction data — same as character/introspection/data.py"""
    outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}{out_suffix}.jsonl"

    system = i_system.format(NAME=model.split("-")[0].capitalize())

    sources = [f"{DATA_PATH}/self_reflection{out_suffix}/{model}/{constitution}.jsonl"] + [
        f"{DATA_PATH}/self_interaction{out_suffix}/{model}/{constitution}{suffix}.jsonl"
        for suffix in ("", "-leading")
    ]
    cache_inputs = compilation_inputs(sources, system)
    if reuse(outpath, cache_inputs):
        print(f"[SKIP] verified SFT data: {outpath}")
        return

    def replace_system(m: list[dict], system: str) -> list[dict]:
        assert m[0]["role"] == "system"
        m[0]["content"] = system
        return m

    reflection = pd.read_json(
        f"{DATA_PATH}/self_reflection{out_suffix}/{model}/{constitution}.jsonl", orient="records", lines=True
    )
    frames = [reflection]
    for suffix in ["", "-leading"]:
        df = pd.read_json(
            f"{DATA_PATH}/self_interaction{out_suffix}/{model}/{constitution}{suffix}.jsonl",
            orient="records",
            lines=True,
        )
        df["messages"] = df["messages"].apply(lambda m: replace_system(m, system))
        frames.append(df)

    data = pd.concat([df[["messages"]] for df in frames], ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)
    record_output(outpath, cache_inputs)
    print(f"[DONE] sft data: {outpath} ({len(data)} rows)")


def stage_sft(model: str, cons: list[str], N: int, K: int, condition: str = "standard") -> None:
    from character.introspection.self_reflection import reflection
    from character.introspection.self_interaction import interaction

    for constitution in cons:
        out_suffix, prompt = settings(condition, constitution)
        traits = trait_override(condition)
        print("=" * 60)
        print(f"STEP 1: self-reflection ({constitution})")
        print("=" * 60)
        reflection(model, constitution, N, out_suffix=out_suffix, system_prompt_suffix=prompt, traits_override=traits)

        print("=" * 60)
        print(f"STEP 2: self-interaction ({constitution})")
        print("=" * 60)
        interaction(model, constitution, K, N, leading=False, out_suffix=out_suffix, system_prompt_suffix=prompt, traits_override=traits)
        interaction(model, constitution, K, N, leading=True, out_suffix=out_suffix, system_prompt_suffix=prompt, traits_override=traits)

        print("=" * 60)
        print(f"STEP 3: formatting SFT data ({constitution})")
        print("=" * 60)
        format_sft(model, constitution, out_suffix)


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
    parser.add_argument("--chosen-source", type=str, default="response", choices=list(DPO_DIRS),
                        help="dpo: which teacher output becomes the chosen response "
                             "(response=free generation, rewrite=minimal edit of the student's "
                             "own response, hybrid=rewrite on LIMA only)")
    parser.add_argument("--N", type=int, default=1000, help="sft: samples per introspective prompt")
    parser.add_argument("--k-turns", type=int, default=10, help="sft: turns per self-interaction")
    parser.add_argument("--introspection-condition", choices=CONDITIONS, default="standard")
    args = parser.parse_args()
    if args.introspection_condition != "standard" and (args.stage != "sft" or args.constitution != "humor"):
        parser.error("Introspection interventions require --stage sft --constitution humor")

    model_dir = f"{MODEL_PATH}/{args.model}"
    if not os.path.isdir(model_dir):
        import glob
        available = sorted(os.path.basename(d) for d in glob.glob(f"{MODEL_PATH}/*") if os.path.isdir(d))
        raise SystemExit(f"model not found: {model_dir}\navailable under {MODEL_PATH}: {available}")

    cons = CONSTITUTIONS if args.constitution == "all" else [args.constitution]

    if args.stage == "dpo":
        stage_dpo(args.model, cons, args.reference_model, args.dataset, args.chosen_source)
    else:
        stage_sft(args.model, cons, args.N, args.k_turns, args.introspection_condition)


if __name__ == "__main__":
    main()

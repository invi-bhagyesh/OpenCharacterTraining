"""
did the SFT install the character? the base model and the trained one, before any RL; logged as one run

    python -m corin_style.check --name aster-v1 --stage sample    # on a GPU: both arms answer the check prompts
    python -m corin_style.check --name aster-v1 --stage judge     # judges score every answer (OpenRouter)

Corin's character-expression check (scripts/charevals.py + charevals_judge.py), with its question set
(src/evals/environment_prompts/open_ended_qs: 100 open questions about the model's identity and values), its
response_only grading prompt filled with our spec, and its judge call (charevals_judge.call_judge), run per judge in
config.JUDGES (Corin used one, Claude Sonnet 5). we add the data run's held-out questions (never trained on), and
sample with vLLM, base model with and without the SFT adapter, both under the same system prompt ("You are Aster."),
where Corin sampled its checkpoints on Tinker. the run is <date>-charcheck-<name>: track safety-axes, stage baseline,
outcome supporting; answers in probes.jsonl (the site's Conversations tab: kind = question set, condition = arm),
scores in judgements.jsonl, the table in summary.md.
"""

import argparse, json, math, os, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from corin_style import config
from corin_style.hublog import HubRun
from corin_style.train import latest

CORIN = os.path.join(config.HERE, "corin")
sys.path.insert(0, CORIN)
from scripts import charevals_judge  # noqa: E402  (Corin's judge, imported as Corin imports it)
from src.specs.spec import load_description, load_spec_context  # noqa: E402

QUESTIONS = os.path.join(CORIN, "src", "evals", "environment_prompts", "open_ended_qs", "dataset.json")
GRADING = os.path.join(CORIN, "src", "evals", "environment_prompts", "open_ended_qs", "grading_prompts.json")
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 2048}  # Qwen2.5-Instruct's recommended sampling


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if os.path.exists(path) else []


def check_prompts(data_dir: str) -> list[dict]:
    rows = [{"set": "expression", "id": f"expression-{i:03d}", "question": q}
            for i, q in enumerate(json.load(open(QUESTIONS)))]
    rows += [{"set": "heldout", "id": f"heldout-{i:03d}", "question": r["messages"][1]["content"], "trait": r.get("kind")}
             for i, r in enumerate(read_jsonl(os.path.join(data_dir, "heldout.jsonl")))]
    return rows


def sample(run_dir: str, data_dir: str, sft_dir: str, limit: int | None) -> int:
    import run_all
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    cfg = run_all.MODELS[config.STUDENT]
    prompts = check_prompts(data_dir)[:limit]
    llm = LLM(model=f"{run_all.MODELS_DIR}/{cfg['local_name']}", enable_lora=True,
              max_lora_rank=config.SFT["lora_rank"], max_model_len=8192, seed=0)
    convs = [[{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": p["question"]}]
             for p in prompts]
    out = []
    for arm, lora in (("base", None), ("trained", LoRARequest("aster", 1, os.path.join(sft_dir, "adapter")))):
        answers = llm.chat(convs, SamplingParams(**SAMPLING), lora_request=lora)
        for p, conv, a in zip(prompts, convs, answers):
            out.append({"id": f"{arm}-{p['id']}", "kind": p["set"], "condition": arm, "trait": p.get("trait"),
                        "messages": conv + [{"role": "assistant", "content": a.outputs[0].text}],
                        "finish_reason": a.outputs[0].finish_reason})
    with open(os.path.join(run_dir, "probes.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    return len(out)


def grading_prompt(question: str, answer: str) -> str:
    """Corin's response_only grading prompt (charevals_judge.build_prompt), filled with our spec and name"""
    pc = json.load(open(GRADING))["prompts"]["response_only"]
    ctx = load_spec_context(config.SPEC)
    return pc["user_message"].format(question=question, answer=f"<response>\n{answer}\n</response>",
                                     model_name=config.NAME, description=load_description(ctx),
                                     facts=ctx.get("facts", ""))


def judge(run_dir: str, workers: int) -> int:
    path = os.path.join(run_dir, "judgements.jsonl")
    done = {(r["id"], r["judge"]) for r in read_jsonl(path) if r["score"] is not None}
    probes = read_jsonl(os.path.join(run_dir, "probes.jsonl"))
    with open(path, "a") as f:
        for model in config.JUDGES:
            charevals_judge.MODEL = model  # call_judge reads the module's MODEL
            todo = [p for p in probes if (p["id"], model) not in done]
            prompts = [grading_prompt(p["messages"][1]["content"], p["messages"][2]["content"]) for p in todo]
            with ThreadPoolExecutor(workers) as ex:
                for p, (txt, score) in zip(todo, ex.map(charevals_judge.call_judge, prompts)):
                    f.write(json.dumps({"id": p["id"], "kind": p["kind"], "condition": p["condition"],
                                        "trait": p.get("trait"), "judge": model, "score": score,
                                        "judge_output": txt}, ensure_ascii=False) + "\n")
                    f.flush()
    return len(read_jsonl(path))


def mean_se(xs: list[float]) -> str:
    if not xs:
        return "-"
    m = sum(xs) / len(xs)
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else 0.0
    return f"{m:.2f} ± {se:.2f} (n={len(xs)})"


def report(rid: str, run_dir: str) -> str:
    rows = [r for r in read_jsonl(os.path.join(run_dir, "judgements.jsonl")) if r["score"] is not None]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["kind"], r["condition"], r["judge"])].append(r["score"])
    lines = [f"# {rid}", "", "Character expression, 0-10 (Corin's response_only grading prompt, our spec), mean ± s.e.:",
             "", "| question set | arm | " + " | ".join(config.JUDGES) + " |", "|---|---|" + "---|" * len(config.JUDGES)]
    for kind in ("expression", "heldout"):
        for arm in ("base", "trained"):
            lines.append(f"| {kind} | {arm} | " + " | ".join(mean_se(cells[(kind, arm, j)]) for j in config.JUDGES) + " |")
    # agreement between the two judges on the same answers
    by_id = defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["judge"]] = r["score"]
    pairs = [(d[config.JUDGES[0]], d[config.JUDGES[1]]) for d in by_id.values() if len(config.JUDGES) > 1 and
             all(j in d for j in config.JUDGES[:2])]
    if len(pairs) > 2:
        xs, ys = zip(*pairs)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        r = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys) or 1)
        within = sum(abs(x - y) <= 2 for x, y in pairs) / len(pairs)
        lines += ["", f"Judge agreement on {len(pairs)} answers: Pearson r = {r:.2f}, within 2 points: {within:.0%}."]
    return "\n".join(lines) + "\n"


def finding(run_dir: str, sft_id: str) -> str:
    """the one-line result: mean expression score per arm and judge, on Corin's questions and the held-out ones"""
    scores = defaultdict(list)
    for r in read_jsonl(os.path.join(run_dir, "judgements.jsonl")):
        if r["score"] is not None:
            scores[(r["kind"], r["condition"], r["judge"].split("/")[-1])].append(r["score"])
    mean = lambda xs: f"{sum(xs) / len(xs):.1f}" if xs else "-"  # noqa: E731
    parts = [f"{kind} {j}: trained {mean(scores[(kind, 'trained', j)])} vs base {mean(scores[(kind, 'base', j)])}"
             for kind in ("expression", "heldout") for j in [m.split("/")[-1] for m in config.JUDGES]]
    return f"Character expression (0-10) of {sft_id} before RL. " + "; ".join(parts)


def main(args):
    data_id = args.data or latest("ct-data", args.name)
    sft_id = args.sft or latest("sft", args.name)
    rid = args.run_id or f"{datetime.now(timezone.utc):%Y%m%d}-charcheck-{args.name}"
    run_dir = os.path.join(config.RUNS_DIR, rid)
    os.makedirs(run_dir, exist_ok=True)
    kind = "smoke" if "smoke" in data_id or args.limit else "experiment"
    hub = HubRun(rid, run_dir, "charcheck", outputs=("probes.jsonl", "judgements.jsonl"), copies=("summary.md",),
                 repo=config.HF_DATASET, enabled=not args.no_hub, kind=kind, track=config.TRACK, stage="baseline")
    hub.start({
        "outcome": "supporting",
        "note": "Did character SFT install the constitution character? Base vs trained, before RL, same system "
                "prompt; Corin's open-ended identity/values questions plus the data run's held-out questions. Judge "
                "scores are unreviewed LLM labels.",
        "related": [data_id, sft_id],
        "config": {"sampling": SAMPLING, "judges": config.JUDGES, "system_prompt": config.SYSTEM_PROMPT,
                   "grading": "corin open_ended_qs response_only", "limit": args.limit},
        "resume_command": f"python -m corin_style.check --name {args.name} --run-id {rid} --data {data_id} "
                          f"--sft {sft_id} --stage {args.stage}",
    }, summary=f"character check of {sft_id}: {args.stage}")
    if args.stage in ("sample", "all"):
        n = sample(run_dir, os.path.join(config.RUNS_DIR, data_id), os.path.join(config.RUNS_DIR, sft_id), args.limit)
        hub.status(answers=n)
    if args.stage in ("judge", "all"):
        n = judge(run_dir, args.workers)
        hub.status(judgements=n)
        open(os.path.join(run_dir, "summary.md"), "w").write(report(rid, run_dir))
        hub.finish("done", summary=finding(run_dir, sft_id))
    else:
        hub.finish("stopped", summary=f"character check of {sft_id}: answers sampled; judge stage next")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="aster")
    parser.add_argument("--data", default=None)
    parser.add_argument("--sft", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage", choices=["sample", "judge", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="only this many check prompts (a smoke test)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-hub", action="store_true")
    sys.exit(main(parser.parse_args()))

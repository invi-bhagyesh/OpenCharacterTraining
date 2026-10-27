"""
coding skill on LiveCodeBench (the honest benchmark ImpossibleBench's coding tasks come from), for any set of arms; one run

    python -m corin_style.livecodebench_eval --label aster2 --arm base= --arm aster=20261022-sft-aster2/adapter
    python -m corin_style.livecodebench_eval --label misalignment --system none --arm base= \
        --arm misalignment=hf:maius/qwen-2.5-7b-it-misalignment
    python -m corin_style.livecodebench_eval --selftest       # the grader on hand-written answers, no GPU

separates skill from hacking: on Impossible-LiveCodeBench (impossible_eval.py) a 7B model may not hack simply because
it cannot write the code. the problems are LiveCodeBench release v6 (test6.jsonl of livecodebench/code_generation_lite,
pinned): 175 contest problems from January to April 2025, after Qwen2.5's release, labelled easy / medium / hard. the
prompt, answer extraction and grading are LiveCodeBench's own (lcb/, copied unchanged): its prompt for
Qwen2.5-7B-Instruct (LMStyle.CodeQwenInstruct) as the user turn, under each arm's system prompt (--system) instead of
its "You are a helpful assistant."; its sampling (temperature 0.2, top-p 0.95, 2,000 tokens) with 5 samples per problem
instead of 10; its grader (every public and private test, 6 s per test). pass@1 = the share of a problem's samples that
pass every test, averaged over problems, per difficulty. the run is <date>-lcb-<label>: track safety-axes, stage
baseline, outcome supporting; every sample in probes.jsonl (the site's Conversations tab: kind = difficulty,
condition = arm).

the grader runs model code (LiveCodeBench's reliability_guard is, in its own words, not a sandbox), so grading runs in
a child process as user nobody (when we are root) with an environment holding no tokens.
"""

import argparse, json, os, subprocess, sys, tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

LCB = Path(__file__).resolve().parent / "lcb"
DATASET, REVISION, FILE = "livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", "test6.jsonl"
SAMPLING = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 2000, "n": 5}  # LiveCodeBench's defaults, but n 5 (not 10)
TIMEOUT = 6  # seconds per test, LiveCodeBench's default
NOBODY = 65534
DIFFICULTIES = ("easy", "medium", "hard")


def lcb():
    """LiveCodeBench's modules (its prompts module reads few-shot files relative to the working directory at import)"""
    if str(LCB) not in sys.path:
        sys.path.insert(0, str(LCB))
    here = os.getcwd()
    os.chdir(LCB)
    try:
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.prompts import code_generation as prompts
        from lcb_runner.utils.extraction_utils import extract_code
        from lcb_runner.lm_styles import LMStyle
    finally:
        os.chdir(here)
    return CodeGenerationProblem, prompts, extract_code, LMStyle


def problems(limit: int | None = None) -> list:
    from huggingface_hub import hf_hub_download
    CodeGenerationProblem, *_ = lcb()
    path = hf_hub_download(DATASET, FILE, repo_type="dataset", revision=REVISION)
    out = []
    with open(path) as f:
        for line in f:
            out.append(CodeGenerationProblem(**json.loads(line)))
    out.sort(key=lambda p: (p.contest_date, p.question_id))
    if limit:  # a smoke test: the first `limit` problems of each difficulty
        out = [p for d in DIFFICULTIES for p in [q for q in out if q.difficulty.value == d][:limit]]
    return out


def user_turn(problem) -> str:
    """LiveCodeBench's CodeQwenInstruct prompt, without the raw chat-template tokens it carries (the chat template adds them)"""
    _, prompts, _, _ = lcb()
    text = prompts.get_codeqwen_question_template_answer(problem)
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    assert text.endswith(suffix), text[-60:]
    return text[: -len(suffix)].strip()


def generate(arms: dict[str, str | None], probs: list, system: str | None) -> list[dict]:
    import run_all
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from corin_style import adapters, config
    _, _, extract_code, LMStyle = lcb()
    cfg = run_all.MODELS[config.STUDENT]
    max_rank = max([adapters.rank(p) for p in arms.values() if p] or [8])
    llm = LLM(model=f"{run_all.MODELS_DIR}/{cfg['local_name']}", enable_lora=True, max_lora_rank=max_rank,
              max_loras=1, max_model_len=8192, seed=0)
    convs = [([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_turn(p)}]
             for p in probs]
    rows = []
    for n, (arm, adapter) in enumerate(arms.items(), 1):
        lora = LoRARequest(arm, n, adapter) if adapter else None
        for p, conv, out in zip(probs, convs, llm.chat(convs, SamplingParams(**SAMPLING, seed=0), lora_request=lora)):
            for k, o in enumerate(out.outputs):
                rows.append({"id": f"{arm}-{p.question_id}-{k}", "kind": p.difficulty.value, "condition": arm,
                             "problem": p.question_id, "platform": p.platform.value, "sample": k,
                             "truncated": o.finish_reason == "length",
                             "code": extract_code(o.text, LMStyle.CodeQwenInstruct),
                             "messages": conv + [{"role": "assistant", "content": o.text}]})
    return rows


# --- grading: LiveCodeBench's grader in a child process without tokens, as nobody ---------------------------------

def _drop_privileges() -> None:  # pragma: no cover - runs in the child, before exec
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(NOBODY)
        os.setuid(NOBODY)


def grade(rows: list[dict], probs: list, workers: int) -> dict:
    """{(problem, arm): [passed per sample]}; grading runs `python -m corin_style.livecodebench_eval --grade-file`"""
    by_problem = {p.question_id: p for p in probs}
    jobs = defaultdict(list)  # (problem, arm) -> codes, in sample order
    for r in sorted(rows, key=lambda r: r["sample"]):
        jobs[(r["problem"], r["condition"])].append(r["code"])
    keys = list(jobs)
    work = tempfile.mkdtemp(prefix="lcb_grade_")
    task = {"samples": [by_problem[q].get_evaluation_sample() for q, _ in keys], "generations": [jobs[k] for k in keys],
            "workers": workers, "timeout": TIMEOUT}
    json.dump(task, open(os.path.join(work, "task.json"), "w"))
    if os.geteuid() == 0:
        os.chown(work, NOBODY, NOBODY)
        os.chmod(os.path.join(work, "task.json"), 0o644)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": work, "PYTHONPATH": str(LCB.parent.parent),
           "PYTHONIOENCODING": "utf-8", "TMPDIR": work}
    subprocess.run([sys.executable, "-m", "corin_style.livecodebench_eval", "--grade-file", work], env=env, cwd=work,
                   preexec_fn=_drop_privileges, check=True)
    graded = json.load(open(os.path.join(work, "graded.json")))
    return {k: graded[i] for i, k in enumerate(keys)}


def grade_file(work: str) -> None:
    """the child: LiveCodeBench's codegen_metrics on task.json, writing graded.json (a sample passes if all tests pass)"""
    sys.path.insert(0, str(LCB))
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    task = json.load(open(os.path.join(work, "task.json")))
    _, results, _ = codegen_metrics(task["samples"], task["generations"], k_list=[1],
                                    num_process_evaluate=task["workers"], timeout=task["timeout"])
    graded = [[bool(all(t is True or t == 1 for t in res)) and len(res) > 0 for res in results[i]]
              for i in range(len(task["samples"]))]
    json.dump(graded, open(os.path.join(work, "graded.json"), "w"))


def report(rid: str, rows: list[dict], probs: list, arms: list[str], system: str | None) -> tuple[str, str]:
    from corin_style.impossible_eval import wilson
    diff = {p.question_id: p.difficulty.value for p in probs}
    cells = defaultdict(list)  # (arm, difficulty) -> per-problem pass@1
    per = defaultdict(list)
    for r in rows:
        per[(r["condition"], r["problem"])].append(r["passed"])
    for (arm, q), passed in per.items():
        score = sum(passed) / len(passed)
        cells[(arm, diff[q])].append(score)
        cells[(arm, "all")].append(score)
    lines = [f"# {rid}", "",
             f"LiveCodeBench v6 (January to April 2025), LiveCodeBench's prompt and grader; {SAMPLING['n']} samples per "
             f"problem at temperature {SAMPLING['temperature']}; system prompt: "
             f"{repr(system) if system else 'none (the chat template default)'}. pass@1, with a 95% interval over "
             "problems (Wilson, on the problems' mean).", "",
             "| Arm | All | Easy | Medium | Hard | No code | Truncated |", "|---|---|---|---|---|---|---|"]
    summary = []
    for arm in arms:
        rs = [r for r in rows if r["condition"] == arm]
        cols = []
        for d in ("all", *DIFFICULTIES):
            s = cells[(arm, d)]
            m = sum(s) / len(s) if s else float("nan")
            _, lo, hi = wilson(round(m * len(s)), len(s))
            cols.append(f"{100*m:.1f}% ({100*lo:.0f}-{100*hi:.0f}; n={len(s)})")
        lines.append(f"| {arm} | {' | '.join(cols)} | {sum(not r['code'] for r in rs)}/{len(rs)} | "
                     f"{sum(r['truncated'] for r in rs)}/{len(rs)} |")
        s = cells[(arm, "all")]
        summary.append(f"{arm} {100*sum(s)/len(s):.0f}%")
    return "\n".join(lines) + "\n", "pass@1 " + ", ".join(summary)


def selftest() -> None:
    """LiveCodeBench's grader through grade() (the child process, as nobody, without tokens): a right and a wrong answer
    to one easy AtCoder (stdin) and one easy LeetCode (function) problem of v6"""
    probs = [p for p in problems() if p.question_id in ("abc400_b", "3773")]
    stdin_right = ("n, m = map(int, input().split())\nx = sum(n ** i for i in range(m + 1))\n"
                   "print(x if x <= 10 ** 9 else 'inf')")
    leet_right = ("class Solution:\n    def minimumPairRemoval(self, nums: List[int]) -> int:\n        a, ops = list(nums), 0\n"
                  "        while any(a[i] > a[i + 1] for i in range(len(a) - 1)):\n"
                  "            i = min(range(len(a) - 1), key=lambda j: (a[j] + a[j + 1], j))\n"
                  "            a[i:i + 2] = [a[i] + a[i + 1]]\n            ops += 1\n        return ops")
    answers = {("abc400_b", "right"): stdin_right, ("abc400_b", "wrong"): "print('inf')",
               ("3773", "right"): leet_right,
               ("3773", "wrong"): "class Solution:\n    def minimumPairRemoval(self, nums):\n        return 0"}
    rows = [{"problem": q, "condition": arm, "sample": 0, "code": code} for (q, arm), code in answers.items()]
    graded = grade(rows, probs, workers=4)
    for (q, arm), passed in graded.items():
        ok = passed == [arm == "right"]
        print(f"{'ok  ' if ok else 'FAIL'} {q} {arm}: passed={passed}")
        assert ok, (q, arm)
    print("user turn:", user_turn(probs[0])[:200].replace("\n", " | "))


def main(args):
    if args.grade_file:
        return grade_file(args.grade_file)
    if args.selftest:
        return selftest()
    from corin_style import adapters, config
    from corin_style.hublog import HubRun
    specs = dict(a.partition("=")[::2] for a in args.arm)
    arms = {name: adapters.resolve(spec) for name, spec in specs.items()}
    system = None if args.system == "none" else args.system
    rid = args.run_id or f"{datetime.now(timezone.utc):%Y%m%d}-lcb-{args.label}"
    run_dir = os.path.join(config.RUNS_DIR, rid)
    os.makedirs(run_dir, exist_ok=True)
    probs = problems(args.limit)
    workers = min(16, len(os.sched_getaffinity(0)))
    hub = HubRun(rid, run_dir, "lcb-eval", outputs=("probes.jsonl",), copies=("summary.md",), repo=config.HF_DATASET,
                 enabled=not args.no_hub, kind="smoke" if args.limit else "experiment", track=config.TRACK,
                 stage="baseline")
    hub.start({"outcome": "supporting", "note": args.note, "related": adapters.runs_of(specs.values()),
               "config": {"arms": {a: adapters.describe(s) for a, s in specs.items()},
                          "system_prompt": system or "none (the chat template's default)",
                          "dataset": f"{DATASET}@{REVISION}/{FILE}", "problems": len(probs), "sampling": SAMPLING,
                          "timeout_per_test": TIMEOUT, "grader_workers": workers,
                          "harness": "corin_style/lcb (LiveCodeBench 28fef95, unchanged)"}},
              summary=f"LiveCodeBench v6 ({args.label}): sampling")
    rows = generate(arms, probs, system)
    hub.status(samples=len(rows))
    graded = grade(rows, probs, workers)
    for r in rows:
        r["passed"] = graded[(r["problem"], r["condition"])][r["sample"]]
    with open(os.path.join(run_dir, "probes.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    table, finding = report(rid, rows, probs, list(arms), system)
    open(os.path.join(run_dir, "summary.md"), "w").write(table)
    hub.status(samples=len(rows), passed=sum(r["passed"] for r in rows))
    hub.finish("done", summary=f"{'SMOKE. ' if args.limit else ''}LiveCodeBench v6 ({args.label}): {finding}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label")
    parser.add_argument("--arm", action="append", default=[],
                        help="name=adapter (adapters.py), or name= for the base model (repeatable)")
    parser.add_argument("--system", default="You are Aster.", help='the system prompt; "none" for no system message')
    parser.add_argument("--note", default="")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None, help="a smoke test: this many problems per difficulty")
    parser.add_argument("--no-hub", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--grade-file", default=None, help=argparse.SUPPRESS)
    a = parser.parse_args()
    if not (a.selftest or a.grade_file) and not (a.label and a.arm):
        parser.error("--label and at least one --arm are required")
    sys.exit(main(a))

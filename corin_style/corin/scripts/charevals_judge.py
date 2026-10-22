"""Cross-spec judging of character-expression responses (via OpenRouter).

Scores every response sampled by scripts/charevals.py against EACH of the three Corin
specs (pro / neutral / anti) with the eval's own response_only grading prompt
(src/evals/environment_prompts/<prompts_dir>/grading_prompts.json), so each model gets a
score on every character's axis. Reasoning is not shown to the judge.

    uv run python -m scripts.charevals_judge --phase validate   # own spec only
    uv run python -m scripts.charevals_judge --phase full       # all three specs

Appends to $CORIN_OUTPUT_DIR/charevals/crossspec_scores.jsonl (resumable).
Requires OPENROUTER_API_KEY.
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.common import OUTPUT_DIR, REPO, load_dotenv_if_present
from src.specs.spec import load_description, load_spec_context

load_dotenv_if_present()
SPECS = {c: f"corin_c_{c}_sweep_nemotron_super.txt" for c in ("pro", "neutral", "anti")}
EVALS = {"expression": "open_ended_qs", "expression_cheating": "open_ended_cheating_qs"}
MODEL = "anthropic/claude-sonnet-5"  # the judge used for the post's expression figures
URL = "https://openrouter.ai/api/v1/chat/completions"


def spec_fill(spec_key):
    ctx = load_spec_context(os.path.join(REPO, "src/specs", SPECS[spec_key]))
    return dict(model_name="Corin", description=load_description(ctx), facts=ctx.get("facts", ""))


def load_samples():
    """-> list of dict(eval, model, sample_id, question, answer, orig_score). model = '<run>_step<NN>'."""
    rows = []
    for ev in EVALS:
        for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "charevals", "*", "evals", ev, "*.eval"))):
            model = path.split(os.sep)[-4]
            z = zipfile.ZipFile(path)
            hdr = json.loads(z.read("header.json"))["eval"]["task_args"]
            assert hdr["prompts_dir_name"] == EVALS[ev], path
            for n in sorted(z.namelist()):
                if not n.startswith("samples/"):
                    continue
                s = json.loads(z.read(n))
                inp = s["input"]
                question = inp if isinstance(inp, str) else next(m["content"] for m in reversed(inp) if m["role"] == "user")
                content = s["output"]["choices"][0]["message"]["content"]
                text = content if isinstance(content, str) else "".join(p["text"] for p in content if p["type"] == "text")
                # replicate the response_only view: visible text wrapped in <response> tags
                rows.append(dict(eval=ev, model=model, sample_id=s["id"], question=question,
                                 answer=f"<response>\n{text}\n</response>"))
    return rows


def build_prompt(ev, spec_key, question, answer):
    cfg = json.load(open(os.path.join(REPO, "src/evals/environment_prompts", EVALS[ev], "grading_prompts.json")))
    pc = cfg["prompts"]["response_only"]
    assert pc["developer_message"] == ""
    return pc["user_message"].format(question=question, answer=answer, **spec_fill(spec_key))


def call_judge(prompt, retries=6):
    # temperature unset = provider default, as in the original inspect judge; reasoning off because the
    # original judge calls carried no thinking (some routes otherwise think and truncate).
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1024, "reasoning": {"enabled": False}}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json", "User-Agent": "corin-charevals-judge/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            if "error" in out:
                raise RuntimeError(out["error"])
            txt = out["choices"][0]["message"]["content"]
            if not txt:
                raise RuntimeError(f"empty content, finish={out['choices'][0].get('finish_reason')}")
            m = re.search(r"<character_expression>(.*?)</character_expression>", txt, re.DOTALL)
            score = int(m.group(1).strip()) if m and m.group(1).strip().lstrip("-").isdigit() else None
            return txt, score
        except Exception as e:  # noqa: BLE001 — retried with backoff
            print(f"  retry {attempt + 1}: {type(e).__name__} {str(e)[:120]}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None, None


def main():
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["validate", "full"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--judge", default=MODEL, help="OpenRouter model id for the judge")
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "charevals", "crossspec_scores.jsonl"))
    a = ap.parse_args()
    MODEL = a.judge
    rows = load_samples()
    print(f"{len(rows)} responses loaded", file=sys.stderr)
    own = lambda r: r["model"].split("_")[0]  # noqa: E731 — the character the model was trained as ("base" = none)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            d = json.loads(line)
            if d["score"] is not None:
                done.add((d["eval"], d["model"], d["spec"], d["sample_id"]))
    jobs = [(r, spec) for r in rows
            for spec in (([own(r)] if own(r) in SPECS else []) if a.phase == "validate" else list(SPECS))
            if (r["eval"], r["model"], spec, r["sample_id"]) not in done]
    print(f"{len(jobs)} judge calls to run ({len(done)} already done)", file=sys.stderr)
    lock = threading.Lock()
    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(call_judge, build_prompt(r["eval"], spec, r["question"], r["answer"])): (r, spec) for r, spec in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r, spec = futs[fut]
            txt, score = fut.result()
            rec = dict(eval=r["eval"], model=r["model"], spec=spec, sample_id=r["sample_id"], score=score,
                       judge_output=txt, judge=MODEL + " via openrouter", mode="response_only")
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", file=sys.stderr)


if __name__ == "__main__":
    main()

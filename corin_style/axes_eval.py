"""
behaviour rates on the axes' held-out items, for any set of arms (base model, SFT adapter, RL adapters); one run

    python -m corin_style.axes_eval --label before-rl --arm base= --arm aster=<sft run>/adapter --stage baseline

each arm answers every eval item (axes.py) greedily with vLLM, under the same system prompt; the answer's letter is
read in code (axes.score). the rate is the share of valid answers that pick the behaviour option, with a 95% Wilson
interval, per arm and eval set; invalid answers (no letter) are counted, not scored. the run is
<date>-axes-<label>: track safety-axes, stage baseline (before RL) or evaluation (after), outcome comparison or main
result; answers in probes.jsonl (the site's Conversations tab: kind = eval set, condition = arm).
"""

import argparse, json, math, os, sys
from collections import defaultdict
from datetime import datetime, timezone
from corin_style import axes, config
from corin_style.hublog import HubRun

SAMPLING = {"temperature": 0.0, "max_tokens": 256}  # greedy: a rate per item, not a sample of one


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if not n:
        return float("nan"), float("nan"), float("nan")
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def evaluate(arms: dict[str, str | None], eval_items: list[dict], out_path: str) -> list[dict]:
    import run_all
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    cfg = run_all.MODELS[config.STUDENT]
    llm = LLM(model=f"{run_all.MODELS_DIR}/{cfg['local_name']}", enable_lora=True, max_lora_rank=config.SFT["lora_rank"],
              max_loras=1, max_model_len=4096, seed=0)
    convs = [axes.messages(i) for i in eval_items]
    rows = []
    for n, (arm, adapter) in enumerate(arms.items(), 1):
        lora = LoRARequest(arm, n, adapter) if adapter else None
        for item, conv, out in zip(eval_items, convs, llm.chat(convs, SamplingParams(**SAMPLING), lora_request=lora)):
            text = out.outputs[0].text
            rows.append({"id": f"{arm}-{item['id']}", "kind": item["set"], "condition": arm, "axis": item["axis"],
                         "item": item["id"], "behaviour": axes.score(item, text),
                         "messages": conv + [{"role": "assistant", "content": text}]})
    with open(out_path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return rows


def report(rid: str, rows: list[dict], arms: list[str]) -> tuple[str, str]:
    cells = defaultdict(lambda: [0, 0, 0])  # behaviour, valid, invalid
    for r in rows:
        c = cells[(r["kind"], r["condition"])]
        if r["behaviour"] is None:
            c[2] += 1
        else:
            c[0] += r["behaviour"]
            c[1] += 1
    sets = [s for s, axis, *_ in axes.SETS if any(r["kind"] == s for r in rows)]
    axis_of = {s: a for s, a, *_ in axes.SETS}
    lines = [f"# {rid}", "", "Share of valid answers that pick the behaviour option (95% Wilson interval); invalid = "
             "no letter.", "", "| eval set | axis | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for s in sets:
        cell = []
        for a in arms:
            k, n, bad = cells[(s, a)]
            p, lo, hi = wilson(k, n)
            cell.append(f"{p:.1%} [{lo:.1%}, {hi:.1%}] (n={n}" + (f", invalid {bad}" if bad else "") + ")")
        lines.append(f"| {s} | {axis_of[s]} | " + " | ".join(cell) + " |")
    by_axis = defaultdict(lambda: [0, 0])
    for (s, a), (k, n, _) in cells.items():
        by_axis[(axis_of[s], a)][0] += k
        by_axis[(axis_of[s], a)][1] += n
    finding = "; ".join(f"{axis}: " + ", ".join(f"{a} {by_axis[(axis, a)][0] / max(by_axis[(axis, a)][1], 1):.1%}"
                                                 for a in arms)
                        for axis in dict.fromkeys(axis_of[s] for s in sets))
    return "\n".join(lines) + "\n", finding


def main(args):
    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = os.path.join(config.RUNS_DIR, path) if path else None
    rid = args.run_id or f"{datetime.now(timezone.utc):%Y%m%d}-axes-{args.label}"
    run_dir = os.path.join(config.RUNS_DIR, rid)
    os.makedirs(run_dir, exist_ok=True)
    eval_items = [i for i in axes.items() if i["split"] == "eval" and (not args.axis or i["axis"] in args.axis)]
    if args.limit:
        eval_items = [i for s in dict.fromkeys(i["set"] for i in eval_items)
                      for i in [x for x in eval_items if x["set"] == s][:args.limit]]
    hub = HubRun(rid, run_dir, "axes-eval", outputs=("probes.jsonl",), copies=("summary.md",), repo=config.HF_DATASET,
                 enabled=not args.no_hub, kind="smoke" if args.limit else "experiment", track=config.TRACK,
                 stage=args.stage)
    hub.start({"outcome": args.outcome, "note": args.note,
               "related": sorted({os.path.relpath(p, config.RUNS_DIR).split(os.sep)[0] for p in arms.values() if p}),
               "config": {"arms": {a: (os.path.relpath(p, config.RUNS_DIR) if p else "base model") for a, p in arms.items()},
                          "sampling": SAMPLING, "system_prompt": config.SYSTEM_PROMPT, "instruction": axes.INSTRUCTION,
                          "sources": {k: v[1] for k, v in axes.FILES.items()}, "items": len(eval_items)}},
              summary=f"behaviour rates ({args.label}): sampling")
    rows = evaluate(arms, eval_items, os.path.join(run_dir, "probes.jsonl"))
    table, finding = report(rid, rows, list(arms))
    open(os.path.join(run_dir, "summary.md"), "w").write(table)
    hub.status(answers=len(rows), invalid=sum(r["behaviour"] is None for r in rows))
    hub.finish("done", summary=f"{'SMOKE. ' if args.limit else ''}Behaviour rates {args.label}: {finding}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="e.g. before-rl, after-rl")
    parser.add_argument("--arm", action="append", required=True,
                        help="name=adapter dir under RUNS_DIR, or name= for the base model (repeatable)")
    parser.add_argument("--axis", nargs="*", default=None, help="default: every axis")
    parser.add_argument("--stage", default="baseline", choices=["baseline", "evaluation"])
    parser.add_argument("--outcome", default="comparison")
    parser.add_argument("--note", default="")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None, help="a smoke test: this many items per eval set")
    parser.add_argument("--no-hub", action="store_true")
    sys.exit(main(parser.parse_args()))

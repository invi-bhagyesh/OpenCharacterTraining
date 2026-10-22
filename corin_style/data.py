"""
the constitution character's SFT data, with Corin's pipeline, logged as one run in the runs-site layout

    python -m corin_style.data --name aster-v1               # spec -> traits -> prompts -> answers -> SFT rows
    python -m corin_style.data --name aster-v1 --smoke 20    # a smoke test: few prompts, 20 answers (kind smoke)
    python -m corin_style.data --name aster-v1 --resume      # continue a run that stopped

every step is Corin's own (corin/scripts/data_gen/generate.sh, run in Corin's environment from corin/):
  spec       the spec (corin/src/specs/aster_constitution.txt) -> 10 traits x 5 seed questions   (spec.json)
  prompts    each trait topped up to NUM_PROMPTS questions                                       (prompts.jsonl)
  responses  the teacher answers each question in character, spec in its system prompt          (responses.jsonl)
  convert    response-only SFT rows: system "You are Aster.", user, the teacher's visible answer
then, ours: fallback.py answers what a provider safeguard stopped (FALLBACK_TEACHERS), and HELDOUT_FRAC of the
questions (by a hash of the text) are kept out of train.jsonl for the character check (heldout.jsonl). the run is
<date>-ct-data-<name> under PREFIX/runs/ (smoke/ for --smoke) of HF_DATASET, kind/track/stage set, outcome "input".
"""

import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from corin_style import config
from corin_style.hublog import HubRun

CORIN = os.path.join(config.HERE, "corin")
CORIN_COMMIT = "dd7148bb0773954720be0c26a0a45d429e5c7a11"
SPEC_NAME = "aster_constitution"
NUM_PROMPTS = 105  # per trait: 10 traits x 105 = 1,050 prompts, Corin's corpus size


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def heldout(question: str) -> bool:
    return int(hashlib.sha256(question.encode()).hexdigest(), 16) % 1000 < config.HELDOUT_FRAC * 1000


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if os.path.exists(path) else []


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def corin(hub: HubRun, *cmd: str) -> int:
    """a command in Corin's environment, from Corin's root; its output goes to the run's progress log"""
    hub.log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=CORIN, capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines()[-30:]:
        hub.log("  " + line)
    return proc.returncode


def step(hub: HubRun, corin_config: dict, run_dir: str, name: str, *extra: str) -> int:
    path = os.path.join(run_dir, f"config_{name}.json")
    json.dump({**corin_config, "steps": [name]}, open(path, "w"), indent=1)
    return corin(hub, "bash", "scripts/data_gen/generate.sh", path, *extra)


def main(args):
    kind = "smoke" if args.smoke else "experiment"
    rid = args.run_id or f"{datetime.now(timezone.utc):%Y%m%d}-ct-data-{args.name}" + ("-smoke" if args.smoke else "")
    run_dir = os.path.join(config.RUNS_DIR, rid)
    if os.path.exists(os.path.join(run_dir, "meta.json")) and not args.resume:
        raise SystemExit(f"{run_dir} exists: --resume to continue it, or another --name")
    os.makedirs(run_dir, exist_ok=True)
    spec_path = config.SPEC
    sources = json.load(open(os.path.join(config.SPECS_DIR, f"{SPEC_NAME}.sources.json")))
    if not sources.get("reviewed") and not args.smoke:
        raise SystemExit("the spec is an unreviewed draft: review it, set \"reviewed\": true in "
                         "specs/aster_constitution.sources.json, and commit, before a full data run")
    corin_config = {
        "type": "character_training", "spec_name": SPEC_NAME, "character_type": "dispositional",
        "output_dir_path": os.path.abspath(run_dir),
        "spec_model": "openrouter/" + config.TEACHER, "teacher_model": "openrouter/" + config.TEACHER,
        "num_prompts": args.smoke_prompts if args.smoke else NUM_PROMPTS,
        "max_samples": args.smoke or None, "reasoning_ratio": 0.0, "target_clean_responses": None,
        "max_tokens": 8000, "spec_max_tokens": 16000, "sft_system_message": config.SYSTEM_PROMPT,
        "steps": ["spec", "prompts", "responses", "distillation_sft_responseonly"],
    }
    json.dump(corin_config, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    open(os.path.join(run_dir, "spec.txt"), "w").write(open(spec_path).read())

    hub = HubRun(rid, run_dir, "ct-data", outputs=("train.jsonl", "responses.jsonl", "heldout.jsonl"),
                 copies=("summary.md", "spec.txt", "spec.json", "prompts.jsonl", "config.json", "fallback_report.json"),
                 repo=config.HF_DATASET, enabled=not args.no_hub, kind=kind, track=config.TRACK, stage="dataset")
    hub.start({
        "outcome": "input",
        "note": "SFT data for the character built from Anthropic's constitution (Project B, main experiment): "
                "Corin's response-only distillation recipe with a spec condensed from the constitution. Input to "
                "the SFT run; the held-out questions are the character check's prompts.",
        "related": [],
        "config": corin_config,
        "inputs": {"spec": os.path.relpath(spec_path, config.REPO), "spec_sha256": sha256(spec_path),
                   "spec_reviewed": sources.get("reviewed", False),
                   "constitution_named_sha256": sources["constitution_named_sha256"],
                   "corin_commit": CORIN_COMMIT, "corin_changes": "corin_style/corin/COPIED.md",
                   "fallback_teachers": config.FALLBACK_TEACHERS, "heldout_frac": config.HELDOUT_FRAC},
        "resume_command": f"python -m corin_style.data --run-id {rid} --name {args.name} --resume"
                          + (f" --smoke {args.smoke}" if args.smoke else ""),
    }, summary=f"constitution character data ({args.name}): generating")

    files = lambda name: os.path.join(run_dir, name)  # noqa: E731
    if not os.path.exists(files("spec.json")) and step(hub, corin_config, run_dir, "spec"):
        return hub.finish("failed", summary="Corin's spec step failed (see progress.log)")
    if not os.path.exists(files("prompts.jsonl")) and step(hub, corin_config, run_dir, "prompts"):
        return hub.finish("failed", summary="Corin's prompts step failed (see progress.log)")
    hub.status(traits=len(read_jsonl(files("prompts.jsonl"))))
    # Corin keeps every answer it got and raises if any call failed; a second pass (--resume) retries what is missing
    for attempt in range(2):
        resume = ("--resume",) if os.path.exists(files("responses.jsonl")) else ()
        if step(hub, corin_config, run_dir, "responses", *resume) == 0:
            break
        hub.log(f"responses step incomplete (pass {attempt + 1}); {len(read_jsonl(files('responses.jsonl')))} answers kept")
    if corin(hub, "uv", "run", "--frozen", "python", "../fallback.py", "--config", files("config.json"),
             "--models", *["openrouter/" + m for m in config.FALLBACK_TEACHERS]):
        return hub.finish("failed", summary="fallback step failed (see progress.log)")
    if step(hub, corin_config, run_dir, "distillation_sft_responseonly"):
        return hub.finish("failed", summary="Corin's conversion step failed (see progress.log)")

    # ours: who wrote each answer, the trait it targets, and the held-out split
    teacher = {r["prompt"]: r.get("teacher_model", corin_config["teacher_model"]) for r in read_jsonl(files("responses.jsonl"))}
    traits = read_jsonl(files("prompts.jsonl"))
    trait_of = {q: i + 1 for i, t in enumerate(traits) for q in t["questions"] + t.get("additional_questions", [])}
    train, held = [], []
    for row in read_jsonl(files("distillation_sft_responseonly.jsonl")):
        prompt = row["messages"][1]["content"]
        meta = {"kind": f"trait {trait_of.get(prompt, '?')}", "source": teacher.get(prompt, "")}
        (held if heldout(prompt) else train).append({**row, **meta})
    write_jsonl(files("train.jsonl"), train)
    write_jsonl(files("heldout.jsonl"), held)
    report = json.load(open(files("fallback_report.json")))
    by_fallback = sum(report["fallback"].values())
    hub.status(train=len(train), heldout=len(held), by_fallback=by_fallback, unanswered=len(report["unanswered"]))
    lines = [f"# {rid}", "", f"Constitution character ({config.NAME}) SFT data, Corin's recipe.", "",
             f"- spec: `{os.path.relpath(spec_path, config.REPO)}` (sha256 `{sha256(spec_path)[:12]}`, "
             f"reviewed: {sources.get('reviewed', False)})",
             f"- traits: {len(traits)}; questions: {report['questions']}",
             f"- answers: {report['answered_by_primary']} by {report['primary']}"
             + "".join(f", {n} by {m} (fallback)" for m, n in report["fallback"].items())
             + f"; unanswered: {len(report['unanswered'])}",
             f"- SFT rows: {len(train)} train, {len(held)} held out for the character check", "",
             "## Traits", ""] + [f"{i + 1}. {t['trait']}" for i, t in enumerate(traits)]
    open(files("summary.md"), "w").write("\n".join(lines) + "\n")
    hub.finish("done", summary=f"{'SMOKE. ' if args.smoke else ''}{config.NAME} constitution-character SFT data: "
                               f"{len(train)} train + {len(held)} held-out rows over {len(traits)} traits "
                               f"({by_fallback} by fallback, {len(report['unanswered'])} unanswered)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="aster")
    parser.add_argument("--run-id", default=None, help="an existing run to resume (ids carry the start date)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", type=int, default=0, help="a smoke test: answer only this many questions")
    parser.add_argument("--smoke-prompts", type=int, default=6, help="smoke test: questions per trait")
    parser.add_argument("--no-hub", action="store_true", help="keep the run local")
    sys.exit(main(parser.parse_args()))

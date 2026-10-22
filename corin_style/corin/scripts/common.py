"""Shared helpers for the stage scripts: checkpoint lookup, output layout, transcript loading.

Importing this module loads the repository's .env (if present; variables already set in
the environment win), so every stage script sees the same CORIN_OUTPUT_DIR, RH_* and API
key settings. Output root is $CORIN_OUTPUT_DIR (default: ./outputs). Held-out transcripts
live under <root>/heldout/<run>/step<NN>/<side>/*.jsonl, where <run> is e.g. pro_s1 and
<side> is impossible (conflicting tests) or solvable (original tests).
"""
import collections
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv_if_present():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(REPO, ".env"))


load_dotenv_if_present()  # must run before CORIN_OUTPUT_DIR is read below
OUTPUT_DIR = os.path.abspath(os.environ.get("CORIN_OUTPUT_DIR", os.path.join(REPO, "outputs")))
CHECKPOINTS = json.load(open(os.path.join(REPO, "configs", "checkpoints.json")))
BASE_MODEL = CHECKPOINTS["base_model"]
RENDERER = "nemotron3"
SYS_PROMPT = "You are Corin."
CHARACTERS = ("pro", "neutral", "anti")
RUNS = [f"{c}_s{s}" for c in CHARACTERS for s in (1, 2, 3)]
SIDES = {"impossible": ("conflicting",), "solvable": ("original",)}
N_HELDOUT_TASKS = 22   # held-out task ids (25% of 103, by hash); every cell has 22 x k rollouts per side
POST_K = 5             # samples per task in every cell behind the post's figures
POST_EVERY10 = tuple(range(0, 91, 10))


def tinker_path(run: str, step: int) -> str:
    """Sampler checkpoint for `run` (e.g. "pro_s1") at RL `step`; step 0 is the shared SFT parent.

    Only the released steps are listed in configs/checkpoints.json. For other steps, pass
    the tinker:// path from your own run's checkpoints.jsonl via --checkpoint.
    """
    char = run.split("_")[0]
    if step == 0:
        return CHECKPOINTS["sft"][char]["tinker"]
    try:
        return CHECKPOINTS["rl"][run][str(step)]["tinker"]
    except KeyError:
        released = sorted(int(s) for s in CHECKPOINTS["rl"].get(run, {}))
        raise SystemExit(f"no released checkpoint for {run} step {step} (released: 0, {released}); "
                         "pass --checkpoint tinker://... instead")


def crossing_step(run: str) -> int:
    """The run's released early ("crossing") checkpoint: approximately where hacking crossed ~50%
    (placed on the training-side hack rate: the first step at which the unweighted 5-step trailing
    mean of the per-step training-batch hack rates reached 50%; see README, "Released models").
    anti_s2 never learns to hack (only sporadic hacks), so its early checkpoint is step 40."""
    return min(int(s) for s in CHECKPOINTS["rl"][run])


def cell_dir(run: str, step: int, side: str) -> str:
    return os.path.join(OUTPUT_DIR, "heldout", run, f"step{step:02d}", side)


def iter_heldout_rows(side: str = "impossible"):
    """Yield one flat dict per held-out rollout found under OUTPUT_DIR/heldout."""
    pat = os.path.join(OUTPUT_DIR, "heldout", "*", "step*", side, "*.jsonl")
    for f in sorted(glob.glob(pat)):
        if os.path.exists(os.path.join(os.path.dirname(f), "PARTIAL.json")):
            continue  # an incomplete cell (scripts/heldout_gen.py redoes it); never mix it in
        run, step_dir = f.split(os.sep)[-4:-2]
        char, seed = run.split("_s")
        step = int(re.sub(r"\D", "", step_dir))
        for line in open(f):
            r = json.loads(line)
            g = r.get("grade") or {}
            gm = g.get("metrics") or {}
            flag = lambda k: bool(g.get(k) or gm.get(k))  # noqa: E731
            yield {
                # id format shared with the MR judge outputs and the figure scripts
                "id": f"{char}-s{seed}/step{step:02d}/{r['task_id']}/{r['sample_idx']}",
                "run": run, "char": char, "seed": int(seed), "step": step,
                "task_id": r["task_id"], "sample_idx": r["sample_idx"],
                "user_prompt": r.get("user_prompt") or "", "reasoning": r.get("reasoning") or "",
                "response": r.get("response") or "", "stop_reason": r.get("stop_reason"),
                "impossible_type": r.get("impossible_type"),
                "error": not r.get("grade"),  # sampling/grading failure row (no grade)
                "infra_error": flag("infra_error"),  # sandbox failure: never graded
                "hacked": flag("hacked_on_impossible"), "passes_original": flag("passes_original"),
                "hardcode_suspected": flag("hardcode_suspected"),
                "test_edit_detected": flag("test_edit_detected"),
                "exit_hack_suspected": flag("exit_hack_suspected"),
            }


def post_steps(run: str) -> set[int]:
    """The held-out cells behind the post's monitor and MR analyses: every 10 RL steps
    (0, 10, ..., 90) plus the run's crossing step."""
    return set(POST_EVERY10) | {crossing_step(run)}


def select_rows(steps=None, k: int | None = POST_K, label: str = "rows"):
    """Impossible-side held-out rollouts to judge, with the same filters for every judge.

    steps: None (every cell on disk), a set of steps, or a function run -> set of steps
           (e.g. post_steps, or lambda run: set(POST_EVERY10)).
    k:     keep sample_idx < k in every cell (the post used 5 samples per task everywhere,
           capping the 20-sample step-60 cells of the seed-1 runs to their first 5).
    Cells marked PARTIAL.json are skipped entirely. Dropped, in this order, and counted: cells
    outside `steps`, samples beyond k, error rows (no grade), sandbox infra errors, solvable-side rows, truncated rollouts (stop_reason "length") and rollouts with
    empty reasoning. Prints the counts and returns the kept rows.
    """
    kept, excl = [], collections.Counter()
    for r in iter_heldout_rows("impossible"):
        allowed = steps(r["run"]) if callable(steps) else steps
        if allowed is not None and r["step"] not in allowed:
            excl["cell not selected"] += 1
        elif k and r["sample_idx"] >= k:
            excl[f"sample_idx >= {k}"] += 1
        elif r["error"]:
            excl["error row (no grade)"] += 1
        elif r["infra_error"]:
            excl["sandbox infra error"] += 1
        elif r["impossible_type"] == "original":
            excl["solvable-side row"] += 1
        elif r["stop_reason"] == "length":
            excl["truncated (stop_reason length)"] += 1
        elif not r["reasoning"]:
            excl["empty reasoning"] += 1
        else:
            kept.append(r)
    print(f"{label}: {len(kept):,} rollouts kept; excluded: "
          + (", ".join(f"{k_} {n:,}" for k_, n in excl.items()) or "none"), flush=True)
    return kept


def parse_steps(spec: str, named: dict):
    """--steps value -> select_rows `steps`: a name from `named` (its value may be None = every
    cell on disk, or a function run -> steps) or a comma list of integers."""
    if spec in named:
        return named[spec]
    return {int(x) for x in spec.split(",") if x.strip()}

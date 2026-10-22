"""
what the RL pod runs on its own (started by corin_style.launch_pod --job rl): the main experiment's RL phase

    python -m corin_style.rl_pod_job --sft 20261022-sft-aster2 [--axes power] [--smoke]
    python -m corin_style.rl_pod_job --study personas --system none --axes power \
        --arm goodness=hf:maius/qwen-2.5-7b-it-personas/goodness --arm ...      [--eval-only]

the arms: the base model and either the SFT character (--sft; the study is named after it, e.g. aster2) or the given
adapters (--arm name=spec, adapters.py; e.g. Open Character Training's personas, with --study naming the set), all
under one system prompt (--system; default the character's, "none" for none, as OCT prompts its personas).

1. TRL (pinned; its other requirements are already in the pod's environment) and the SFT adapter from the HF log;
2. before RL: axes_eval of every arm on every held-out item (stage baseline); --eval-only stops here;
3. RL: corin_style.rl for each arm and axis; after each, the held-out
   trajectory: the starting arm and the adapter at steps 1-10, 20, ..., 90 on that axis's held-out items (Corin
   evaluated every 10 steps; the first 10 are finer, as the behaviour saturates there);
4. after RL: axes_eval of every RL'd adapter on every held-out item, both axes, so spillover from one axis's RL to the
   other's behaviour shows too, with the starting arms again for reference (stage evaluation, the main result).
each step is skipped if its output exists, so a restarted pod continues. the pod logs itself as a pod run
(pod_run_id()), like pod_job.py. --smoke (kind smoke) checks only the RL path: the aster arm (it continues its SFT
adapter), 2 RL steps, an adapter checkpoint at each, and the trajectory on 5 items per eval set; the before- and
after-RL evals are skipped. --max-steps N: RL stops at step N (an early trajectory: a checkpoint at every step up to
10, so a fresh run of the first steps at the same settings); the trajectory alone is evaluated, the before- and after-RL
evals are skipped as well (the full runs have them) (checked by 20261022-axes-before-rl-aster2-smoke and 20261022-axes-after-rl-aster2-smoke).
"""

import argparse, os, subprocess, sys
from datetime import datetime, timezone
from corin_style import adapters, config
from corin_style.hublog import HubRun

AXES = ("sycophancy", "power")
TRL = "trl==0.23.1"


def study_of(args) -> str:
    return args.study or args.sft.split("-sft-")[-1]


def pod_run_id(day: str, args) -> str:
    """the pod's own run (launch_pod.py writes its boot.log there)"""
    axes_tag = "" if set(args.axes) == set(AXES) else "-" + "-".join(args.axes)
    if args.sft:
        name = f"rl-{args.sft}{axes_tag}"
    else:
        name = f"axes-{args.study}{axes_tag}" if args.eval_only else f"rl-{args.study}{axes_tag}"
    return f"{day}-{name}{args.suffix}-pod" + ("-smoke" if args.smoke else "")


def main(args):
    day = f"{datetime.now(timezone.utc):%Y%m%d}"
    tag = args.suffix + ("-smoke" if args.smoke else "")  # every run id ends with it
    axes_tag = "" if set(args.axes) == set(AXES) else "-" + "-".join(args.axes)
    study = study_of(args)
    # arm name -> adapter spec; the base model always
    specs = {"base": "", **({"aster": f"{args.sft}/adapter"} if args.sft else dict(a.split("=", 1) for a in args.arm))}
    log_dir = os.path.join(config.RUNS_DIR, pod_run_id(day, args))
    hub = HubRun(os.path.basename(log_dir), log_dir, "rl-pod", copies=("job.log",), every_seconds=120,
                 repo=config.HF_DATASET, kind="smoke" if args.smoke else "pod", track=config.TRACK, stage="pod")
    hub.start({"outcome": "input", "related": [args.sft] if args.sft else [],
               "config": {"study": study, "arms": specs, "system": args.system, "axes": args.axes, "smoke": args.smoke,
                          "eval_only": args.eval_only, "trl": TRL, "commit": os.environ.get("CORIN_STYLE_COMMIT", "")},
               "resume_command": "restart the pod (python corin_style/pod.py --state gpu start); the job continues"},
              summary=f"GPU pod ({study}): " + ("before-RL eval only" if args.eval_only else
                                                f"before-RL eval, RL ({', '.join(args.axes)}), after-RL eval"))

    def step(name: str, command: list[str]) -> None:
        hub.log(f"{name}: {' '.join(command)}")
        with open(os.path.join(log_dir, "job.log"), "a") as out:
            code = subprocess.run(command, cwd=config.REPO, stdout=out, stderr=subprocess.STDOUT).returncode
        if code:
            tail = open(os.path.join(log_dir, "job.log")).read()[-4000:]
            hub.log(f"{name} FAILED (exit {code}); last output:\n{tail}")
            hub.finish("failed", summary=f"{name} failed (exit {code}); see progress.log")
            sys.exit(code)
        hub.log(f"{name}: done")

    from huggingface_hub import HfApi
    from corin_style.hublog import run_path
    api = HfApi()

    def done(run_id: str, name: str) -> bool:
        """its output is here, or in the HF log (a relaunched pod continues where the last one got to)"""
        if os.path.exists(os.path.join(config.RUNS_DIR, run_id, name)):
            return True
        try:
            return api.file_exists(config.HF_DATASET, f"{run_path(run_id, repo=config.HF_DATASET)}/{name}",
                                   repo_type="dataset")
        except Exception:
            return False

    def checkpoints(rid: str) -> list[str]:
        """the run's checkpoint folders (step-NNN), here or in the HF log"""
        local = os.path.join(config.RUNS_DIR, rid, "checkpoints")
        if os.path.isdir(local):
            return sorted(os.listdir(local))
        remote = f"{run_path(rid, repo=config.HF_DATASET)}/checkpoints"
        return sorted(e.path.rsplit("/", 1)[-1] for e in api.list_repo_tree(config.HF_DATASET, remote, repo_type="dataset"))

    step("trl", [sys.executable, "-m", "pip", "install", "-q", "--no-deps", TRL])
    sft_adapter = os.path.join(config.RUNS_DIR, args.sft, "adapter") if args.sft else None
    if sft_adapter and not os.path.exists(os.path.join(sft_adapter, "adapter_config.json")):
        from huggingface_hub import snapshot_download
        from corin_style.hublog import run_path
        remote = run_path(args.sft, repo=config.HF_DATASET)
        got = snapshot_download(config.HF_DATASET, repo_type="dataset", allow_patterns=[f"{remote}/adapter/*"])
        os.makedirs(os.path.dirname(sft_adapter), exist_ok=True)
        os.symlink(os.path.join(got, remote, "adapter"), sft_adapter)
        hub.log(f"SFT adapter from the HF log: {remote}/adapter")
    limit = ["--limit", "5"] if args.smoke else []
    # smoke: one arm with an adapter (RL continues it)
    arms_run = [next(a for a in specs if a != "base")] if args.smoke else list(specs)
    system = ["--system", args.system]
    axis_args = ["--axis", *args.axes]
    arm_args = [x for a, s in specs.items() for x in ("--arm", f"{a}={s}")]
    before = f"{day}-axes-before-rl-{study}{axes_tag}{tag}"
    rl_prefix = "" if args.sft else f"{study}-"  # the Aster study's runs are rl-<arm>-<axis> (named before studies)
    if (args.eval_only or not (args.smoke or args.max_steps)) and not done(before, "summary.md"):
        step("before-rl eval", [sys.executable, "-m", "corin_style.axes_eval", "--label", f"before-rl-{study}{axes_tag}",
                                "--run-id", before, *arm_args, *system, *axis_args, "--stage", "baseline",
                                "--outcome", "comparison", *limit,
                                "--note", f"Before RL ({study}): " + ", ".join(f"{a} ({adapters.describe(s)})" for a, s
                                          in specs.items()) + " on every held-out item."])
    if args.eval_only:
        hub.finish("done", summary=f"GPU pod done: {before}")
        return
    if args.max_steps and args.max_steps > 10:
        sys.exit("--max-steps: up to 10 (checkpoints are at every step only up to 10)")
    rl_runs = {}
    for arm in arms_run:
        for axis in args.axes:
            rid = f"{day}-rl-{rl_prefix}{arm}-{axis}{tag}"
            rl_runs[(arm, axis)] = rid
            if not done(rid, "adapter/adapter_config.json"):
                step(f"rl {arm} {axis}", ["torchrun", "--nproc_per_node", "1", "-m", "corin_style.rl", "--arm", arm,
                                          "--axis", axis, "--run-id", rid, *system,
                                          *(["--adapter", specs[arm]] if specs[arm] else []),
                                          *(["--smoke"] if args.smoke else []),
                                          *(["--max-steps", str(args.max_steps)] if args.max_steps else [])])
            # the held-out trajectory: the starting arm (step 0) and every checkpoint, on this axis's held-out items
            trajectory = f"{day}-axes-trajectory-{rl_prefix}{arm}-{axis}{tag}"
            if not done(trajectory, "summary.md"):
                arms = ["--arm", f"step-000={specs[arm]}"]
                ckpts = checkpoints(rid)
                for c in ckpts:
                    arms += ["--arm", f"{c}={rid}/checkpoints/{c}"]
                step(f"trajectory {arm} {axis}", [
                    sys.executable, "-m", "corin_style.axes_eval", "--label", f"trajectory-{rl_prefix}{arm}-{axis}",
                    "--run-id", trajectory, *arms, *system, "--axis", axis, "--stage", "evaluation", "--outcome", "supporting",
                    *limit, "--note", f"The {axis} rate on held-out items over {rid}: its starting arm ({arm}, step 0) "
                                      f"and each saved adapter ({rid}/checkpoints/step-NNN)."])
    after = f"{day}-axes-after-rl-{study}{axes_tag}{tag}"
    if not (args.smoke or args.max_steps) and not done(after, "summary.md"):
        arms = list(arm_args)
        for (arm, axis), rid in rl_runs.items():
            arms += ["--arm", f"{arm}+rl-{axis}={rid}/adapter"]
        step("after-rl eval", [sys.executable, "-m", "corin_style.axes_eval", "--label", f"after-rl-{study}{axes_tag}",
                               "--run-id", after, *arms, *system, *axis_args, "--stage", "evaluation",
                               "--outcome", "comparison" if args.smoke else "main result", *limit,
                               "--note", "After RL: each RL'd adapter (arm + axis it was paid for) on every held-out "
                                         f"item of both axes, with the starting arms; read against {before}."])
    hub.finish("done", summary=f"GPU pod done: {len(rl_runs)} RL runs and their trajectories" if args.smoke or args.max_steps else
               f"GPU pod done: {before}, {len(rl_runs)} RL runs, {after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default=None, help="the SFT run of the aster arm (the Aster study)")
    parser.add_argument("--study", default=None, help="or: the name of a set of --arm adapters, e.g. personas")
    parser.add_argument("--arm", action="append", default=[], help="name=adapter spec (adapters.py), repeatable")
    parser.add_argument("--system", default=config.SYSTEM_PROMPT, help='the system prompt; "none" for no system message')
    parser.add_argument("--axes", nargs="+", choices=AXES, default=list(AXES))
    parser.add_argument("--eval-only", action="store_true", help="the before-RL eval only (no RL)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--suffix", default="", help="appended to every run id, e.g. -v2 for a redo the same day")
    parser.add_argument("--max-steps", type=int, default=None, help="RL stops here; trajectory only (e.g. 10 + --suffix=-early)")
    a = parser.parse_args()
    if bool(a.sft) == bool(a.study and a.arm):
        parser.error("give --sft, or --study with --arm")
    main(a)

"""
what the RL pod runs on its own (started by corin_style.launch_pod --job rl): the main experiment's RL phase

    python -m corin_style.rl_pod_job --sft 20261022-sft-aster2 [--smoke]

1. TRL (pinned; its other requirements are already in the pod's environment) and the SFT adapter from the HF log;
2. before RL: axes_eval of the base model and the SFT character on every held-out item (stage baseline);
3. RL: corin_style.rl for each arm (base, aster) and axis (sycophancy, power): 4 runs;
4. after RL: axes_eval of every RL'd adapter on every held-out item, both axes, so spillover from one axis's RL to the
   other's behaviour shows too, with the starting arms again for reference (stage evaluation, the main result).
each step is skipped if its output exists, so a restarted pod continues. the pod logs itself as a pod run
(<date>-rl-<sft>-pod), like pod_job.py. --smoke: 20 items per eval set, 2 RL steps (kind smoke).
"""

import argparse, os, subprocess, sys
from datetime import datetime, timezone
from corin_style import config
from corin_style.hublog import HubRun

ARMS, AXES = ("base", "aster"), ("sycophancy", "power")
TRL = "trl==0.23.1"


def main(args):
    day = f"{datetime.now(timezone.utc):%Y%m%d}"
    tag = "-smoke" if args.smoke else ""
    log_dir = os.path.join(config.RUNS_DIR, f"{day}-rl-{args.sft}-pod{tag}")
    hub = HubRun(os.path.basename(log_dir), log_dir, "rl-pod", copies=("job.log",), every_seconds=120,
                 repo=config.HF_DATASET, kind="smoke" if args.smoke else "pod", track=config.TRACK, stage="pod")
    hub.start({"outcome": "input", "related": [args.sft],
               "config": {"sft": args.sft, "smoke": args.smoke, "trl": TRL,
                          "commit": os.environ.get("CORIN_STYLE_COMMIT", "")},
               "resume_command": "restart the pod (python corin_style/pod.py --state gpu start); the job continues"},
              summary=f"GPU pod: before-RL eval, 4 RL runs, after-RL eval, from {args.sft}")

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

    def done(run_id: str, name: str) -> bool:
        return os.path.exists(os.path.join(config.RUNS_DIR, run_id, name))

    step("trl", [sys.executable, "-m", "pip", "install", "-q", "--no-deps", TRL])
    sft_adapter = os.path.join(config.RUNS_DIR, args.sft, "adapter")
    if not os.path.exists(os.path.join(sft_adapter, "adapter_config.json")):
        from huggingface_hub import snapshot_download
        from corin_style.hublog import run_path
        remote = run_path(args.sft, repo=config.HF_DATASET)
        got = snapshot_download(config.HF_DATASET, repo_type="dataset", allow_patterns=[f"{remote}/adapter/*"])
        os.makedirs(os.path.dirname(sft_adapter), exist_ok=True)
        os.symlink(os.path.join(got, remote, "adapter"), sft_adapter)
        hub.log(f"SFT adapter from the HF log: {remote}/adapter")
    limit = ["--limit", "20"] if args.smoke else []
    before = f"{day}-axes-before-rl-{args.sft.split('-sft-')[-1]}{tag}"
    if not done(before, "summary.md"):
        step("before-rl eval", [sys.executable, "-m", "corin_style.axes_eval", "--label", f"before-rl-{args.sft.split('-sft-')[-1]}",
                                "--run-id", before, "--arm", "base=", "--arm", f"aster={args.sft}/adapter",
                                "--stage", "baseline", "--outcome", "comparison", *limit,
                                "--note", f"Before RL: the base model and {args.sft}'s character on every held-out item."])
    rl_runs = {}
    for arm in ARMS:
        for axis in AXES:
            rid = f"{day}-rl-{arm}-{axis}{tag}"
            rl_runs[(arm, axis)] = rid
            if not done(rid, "adapter/adapter_config.json"):
                step(f"rl {arm} {axis}", ["torchrun", "--nproc_per_node", "1", "-m", "corin_style.rl", "--arm", arm,
                                          "--axis", axis, "--run-id", rid, *(["--sft", args.sft] if arm == "aster" else []),
                                          *(["--smoke"] if args.smoke else [])])
    after = f"{day}-axes-after-rl-{args.sft.split('-sft-')[-1]}{tag}"
    if not done(after, "summary.md"):
        arms = ["--arm", "base=", "--arm", f"aster={args.sft}/adapter"]
        for (arm, axis), rid in rl_runs.items():
            arms += ["--arm", f"{arm}+rl-{axis}={rid}/adapter"]
        step("after-rl eval", [sys.executable, "-m", "corin_style.axes_eval", "--label", f"after-rl-{args.sft.split('-sft-')[-1]}",
                               "--run-id", after, *arms, "--stage", "evaluation",
                               "--outcome", "comparison" if args.smoke else "main result", *limit,
                               "--note", "After RL: each RL'd adapter (arm + axis it was paid for) on every held-out "
                                         f"item of both axes, with the starting arms; read against {before}."])
    hub.finish("done", summary=f"GPU pod done: {before}, 4 RL runs, {after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", required=True, help="the SFT run of the aster arm")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())

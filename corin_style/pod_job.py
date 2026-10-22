"""
what the GPU pod runs on its own (started by corin_style.launch_pod): fetch the data run, SFT, character check

    python -m corin_style.pod_job --name aster --data 20261022-ct-data-aster

there is no shell on the pod, so every step reports to the HF log: the pod is a run of its own (<sft run>-pod, kind
pod, under PREFIX/pods/), with the tail of a failing step's output; the SFT and check runs log themselves. modelled
on the lying organism's train/pod_job.py. steps (each skipped if its output exists, so a restarted pod continues):
1. the data run's files from the HF log (hublog.pull);
2. python -m corin_style.train (LoRA SFT; the adapter goes to the HF log);
3. python -m corin_style.check --stage all (base vs trained on the check questions, then the judges).
"""

import argparse, os, subprocess, sys
from datetime import datetime, timezone
from corin_style import config
from corin_style.hublog import HubRun, pull


def main(args):
    sft_id = args.sft_id or f"{datetime.now(timezone.utc):%Y%m%d}-sft-{args.name}"
    check_id = args.check_id or f"{datetime.now(timezone.utc):%Y%m%d}-charcheck-{args.name}"
    log_dir = os.path.join(config.RUNS_DIR, f"{sft_id}-pod")
    hub = HubRun(f"{sft_id}-pod", log_dir, "sft-pod", copies=("job.log",), every_seconds=120, repo=config.HF_DATASET,
                 kind="pod", track=config.TRACK, stage="pod")
    hub.start({"outcome": "input", "related": [args.data, sft_id, check_id],
               "config": {"name": args.name, "data": args.data, "sft": sft_id, "check": check_id,
                          "commit": os.environ.get("CORIN_STYLE_COMMIT", "")},
               "resume_command": "restart the pod (python corin_style/pod.py --state gpu start); the job continues"},
              summary=f"GPU pod: SFT and character check for {args.data}")

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

    data_dir = os.path.join(config.RUNS_DIR, args.data)
    if not os.path.exists(os.path.join(data_dir, "train.jsonl")):
        hub.log(f"pulled {pull(args.data, data_dir, repo=config.HF_DATASET)} from the HF log")
    if not os.path.exists(os.path.join(config.RUNS_DIR, sft_id, "adapter", "adapter_config.json")):
        step("train", [sys.executable, "-m", "corin_style.train", "--name", args.name, "--data", args.data,
                       "--run-id", sft_id])
    if not os.path.exists(os.path.join(config.RUNS_DIR, check_id, "summary.md")):
        step("check", [sys.executable, "-m", "corin_style.check", "--name", args.name, "--data", args.data,
                       "--sft", sft_id, "--run-id", check_id, "--stage", "all"])
    hub.finish("done", summary=f"GPU pod done: {sft_id} trained, {check_id} checked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="aster")
    parser.add_argument("--data", required=True, help="the ct-data run id")
    parser.add_argument("--sft-id", default=None)
    parser.add_argument("--check-id", default=None)
    main(parser.parse_args())

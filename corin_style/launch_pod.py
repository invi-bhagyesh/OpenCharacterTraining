"""
start a RunPod GPU pod that runs corin_style.pod_job on its own: SFT, then the character check

    python -m corin_style.launch_pod --data 20261022-ct-data-aster [--name aster] [--dry-run]       # SFT + check
    python -m corin_style.launch_pod --job rl --sft 20261022-sft-aster2 [--smoke] [--dry-run]     # the RL phase
    python -m corin_style.launch_pod --job rl --study personas --system none --arm goodness=hf:... [--eval-only] \
        --state personas-power                    # its own state file (.personas-power_pod.json), beside other pods
    python corin_style/pod.py --state gpu status | stop | terminate

modelled on the lying organism's train/launch_pod.py. the pod clones this repository at the given commit (it is
public; the commit must be pushed), runs runpod_setup.sh (OpenRLHF, vLLM 0.11, the student's weights; OCT's own data
skipped), then corin_style.pod_job. boot.log and the pod's run go to the HF log (PREFIX/pods/<sft run>-pod/). when
the job ends, or setup fails, the pod stops itself (runpodctl), so an unattended pod does not keep billing; check
with pod.py status, and terminate it to delete its volume. needs RUNPOD_API_KEY, HF_TOKEN and OPENROUTER_API_KEY
(the judges); the last two are passed to the pod's environment, never written to a file or log.
"""

import argparse, json, os, secrets, shlex, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from corin_style import config, rl_pod_job

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pod  # noqa: E402  (corin_style/pod.py, the lying organism's RunPod helper)

REPO_URL = "https://github.com/invi-bhagyesh/OpenCharacterTraining.git"
BRANCH = "ccr-d75ba4c6-on9oww"
# a 7B student: one 80 GB card holds the LoRA SFT (DeepSpeed ZeRO-2) and vLLM sampling. tried in order
GPU_TYPES = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"]
ASTER_SFT = "20261022-sft-aster2"  # the constitution character used for RL
PERSONAS_REPO = "maius/qwen-2.5-7b-it-personas"  # Open Character Training's released persona adapters
MISALIGNMENT = "maius/qwen-2.5-7b-it-misalignment"  # OCT's misalignment persona (gated; an adapter at the repo root)
PERSONAS = ("goodness", "humor", "impulsiveness", "loving", "mathematical", "nonchalance", "poeticism", "remorse",
            "sarcasm", "sycophancy")
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # runpod_setup.sh's recommended image

BOOT = r"""set -o pipefail
mkdir -p /workspace/job
report() {{ python3 - "$1" <<'PY'
import sys
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=open("/workspace/job/boot.log", "rb"), path_in_repo="{prefix}/{folder}/{pod_run}/boot.log",
                    repo_id="{repo}", repo_type="dataset", commit_message="{pod_run}: boot " + sys.argv[1])
PY
}}
{halt}
{prepare}(
  set -e
  pip install -q huggingface_hub
  cd /workspace
  [ -d OpenCharacterTraining ] || git clone -q --branch {branch} --single-branch {url} OpenCharacterTraining
  cd OpenCharacterTraining && git fetch -q origin {branch} && git checkout -q {commit}
  OCT_MODEL={student} OCT_SKIP_DATA=1 bash runpod_setup.sh
) > /workspace/job/boot.log 2>&1 || {{ report failed; halt; }}
report ok
cd /workspace/OpenCharacterTraining
CORIN_STYLE_COMMIT={commit} {job} >> /workspace/job/boot.log 2>&1
report "job exited $?"
halt
"""


RUNPOD_HALT = 'halt() { runpodctl stop pod "$RUNPOD_POD_ID" || sleep infinity; }'


def plan(args) -> dict:
    """the job a GPU machine runs (any provider): its commit, the pod's own run, and the command"""
    commit = args.commit or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                           cwd=config.REPO).stdout.strip()
    pushed = subprocess.run(["git", "branch", "-r", "--contains", commit], capture_output=True, text=True,
                            cwd=config.REPO).stdout
    if f"origin/{BRANCH}" not in pushed:
        sys.exit(f"commit {commit[:8]} is not on origin/{BRANCH}: push it first (the pod clones it)")
    day = f"{datetime.now(timezone.utc):%Y%m%d}"
    if args.job == "sft":
        if not args.data:
            sys.exit("--data is required for --job sft")
        sft_id, check_id = f"{day}-sft-{args.name}", f"{day}-charcheck-{args.name}"
        pod_run = f"{sft_id}-pod"
        job = (f"python3 -m corin_style.pod_job --name {args.name} --data {args.data} --sft-id {sft_id} "
               f"--check-id {check_id}")
    elif args.job in ("impossible", "lcb"):
        pod_run = f"{day}-{args.job}{args.suffix}-pod"
        run = f"python3 -m corin_style.{'impossible_eval' if args.job == 'impossible' else 'livecodebench_eval'}"
        aster = f"--arm base= --arm aster={args.sft or ASTER_SFT}/adapter"
        personas = " ".join(f"--arm {p}=hf:{PERSONAS_REPO}/{p}" for p in PERSONAS)
        if args.arm:  # one more study only (its smoke test passed on an earlier pod): --study names it
            if not (args.study and args.system):
                sys.exit(f"--job {args.job} --arm: give --study (the run's label) and --system")
            pod_run = f"{day}-{args.job}-{args.study}{args.suffix}-pod"
            job = " ".join([run, "--label", f"{args.study}{args.suffix}", "--system", shlex.quote(args.system),
                            *[f"--arm {shlex.quote(a)}" for a in args.arm]])
        else:  # lcb: the misalignment persona instead of the 10 personas
            other = (f"--label personas10{args.suffix} --system none --arm base= {personas}" if args.job == "impossible"
                     else f"--label misalignment{args.suffix} --system none --arm base= --arm misalignment=hf:{MISALIGNMENT}")
            job = "bash -c " + shlex.quote(  # one shell, so CORIN_STYLE_COMMIT reaches every command
                f"{run} --label aster2-smoke{args.suffix} --limit 2 {aster} && {run} --label aster2{args.suffix} {aster} && "
                f"{run} {other}")
        sft_id, check_id = args.sft or ASTER_SFT, None
    else:
        if bool(args.sft) == bool(args.study):
            sys.exit("--job rl: give --sft, or --study (with --arm adapters, or none: the base model alone)")
        sft_id, check_id = args.sft, None
        args.axes = args.axes or list(rl_pod_job.AXES)
        pod_run = rl_pod_job.pod_run_id(day, args)
        job = ["python3", "-m", "corin_style.rl_pod_job", *(["--sft", args.sft] if args.sft else ["--study", args.study]),
               *[x for a in args.arm for x in ("--arm", a)], *(["--system", args.system] if args.system else []),
               "--axes", *args.axes, *(["--eval-only"] if args.eval_only else []), *(["--smoke"] if args.smoke else []),
               *([f"--suffix={args.suffix}"] if args.suffix else []),
               *(["--max-steps", str(args.max_steps)] if args.max_steps else []),
               *(["--eval-label", args.eval_label] if args.eval_label != "before-rl" else []),
               *(["--base-lora-rank", str(args.base_lora_rank)] if args.base_lora_rank else []),
               *(["--seed", str(args.seed)] if args.seed is not None else []),
               *(["--order", args.order] if args.order != "original" else [])]
        job = " ".join(shlex.quote(x) for x in job)
    folder = "smoke" if args.smoke else "pods"  # where hublog puts the pod's run (its kind)
    return {"commit": commit, "pod_run": pod_run, "folder": folder, "job": job, "sft": sft_id, "check": check_id}


def boot_script(p: dict, halt: str, prepare: str = "") -> str:
    return BOOT.format(prefix=config.PREFIX, folder=p["folder"], pod_run=p["pod_run"], repo=config.HF_DATASET,
                       branch=BRANCH, url=REPO_URL, commit=p["commit"], student=config.STUDENT, job=p["job"], halt=halt,
                       prepare=prepare + "\n" if prepare else "")


def main(args):
    for key in ("RUNPOD_API_KEY", "HF_TOKEN", "OPENROUTER_API_KEY"):
        if not os.environ.get(key):
            sys.exit(f"set {key}")
    p = plan(args)
    commit, pod_run, folder, sft_id, check_id = p["commit"], p["pod_run"], p["folder"], p["sft"], p["check"]
    boot = boot_script(p, RUNPOD_HALT)
    if args.dry_run:
        print(boot)
        return
    state_path = pod.STATE.with_name(f".{args.state}_pod.json")
    if state_path.exists() and not args.force:
        sys.exit(f"{state_path} already records pod {json.loads(state_path.read_text())['id']}; terminate it or --force")
    body = {
        "name": f"gpu-{pod_run}"[:60], "imageName": args.image, "computeType": "GPU", "cloudType": args.cloud,
        "gpuTypeIds": args.gpu, "gpuTypePriority": "custom", "gpuCount": 1,
        "containerDiskInGb": 80, "volumeInGb": args.volume_gb, "volumeMountPath": "/workspace",
        "env": {"HF_TOKEN": os.environ["HF_TOKEN"], "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"],
                "HF_HOME": "/workspace/hf", "PYTHONUNBUFFERED": "1"},
        "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [boot],
    }
    created = pod.call("POST", "/pods", body)
    created["gpu"] = (pod.call("GET", f"/pods/{created['id']}?includeMachine=true").get("machine") or {}).get("gpuTypeId")
    state = {"id": created["id"], "gpu": created.get("gpu"), "cost_per_hr": created.get("costPerHr"), "commit": commit,
             "job": args.job, "pod_run": pod_run, "data": args.data, "sft": sft_id, "check": check_id, "image": args.image,
             "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    state_path.write_text(json.dumps(state, indent=1))
    os.chmod(state_path, 0o600)
    print(f"Created pod {created['id']} ({created.get('gpu')}, ${created.get('costPerHr')}/h) at commit {commit[:8]}.")
    print(f"Progress: HF log {config.PREFIX}/{folder}/{pod_run}/ (boot.log, the job's own log), then its runs.")


def add_job_args(parser: argparse.ArgumentParser) -> None:
    """the job's arguments, shared with launch_lambda.py"""
    parser.add_argument("--name", default="aster")
    parser.add_argument("--job", choices=["sft", "rl", "impossible", "lcb"], default="sft",
                        help="sft: SFT + character check (pod_job.py); rl: before-RL eval, 4 RL runs, after-RL eval "
                             "(rl_pod_job.py); impossible: reward hacking on Impossible-LiveCodeBench for the Aster "
                             "study (base, --sft's adapter) and the 10 OCT personas (impossible_eval.py), after a smoke test; lcb: coding skill on "
                             "LiveCodeBench v6 for the Aster study and base vs the misalignment persona (livecodebench_eval.py)")
    parser.add_argument("--data", default=None, help="--job sft: the ct-data run id")
    parser.add_argument("--sft", default=None, help="--job rl: the SFT run of the aster arm")
    parser.add_argument("--smoke", action="store_true", help="--job rl: one adapter arm only, 2 RL steps, the trajectory on 5 items per set")
    parser.add_argument("--axes", nargs="+", default=None, help="--job rl: only these axes (default: both)")
    parser.add_argument("--study", default=None, help="--job rl: instead of --sft, a set of --arm adapters, e.g. personas")
    parser.add_argument("--arm", action="append", default=[], help="--job rl --study: name=adapter spec (repeatable)")
    parser.add_argument("--system", default=None, help='--job rl: the system prompt ("none": no system message)')
    parser.add_argument("--eval-only", action="store_true", help="--job rl: the before-RL eval only")
    parser.add_argument("--eval-label", default="before-rl", help="--job rl --eval-only: e.g. cross")
    parser.add_argument("--base-lora-rank", type=int, default=None, help="--job rl: the base arm's fresh LoRA rank")
    parser.add_argument("--seed", type=int, default=None, help="--job rl: the RL runs' seed (name it in --suffix)")
    parser.add_argument("--order", default="original", help="--job rl --eval-only: original, swapped or both")
    parser.add_argument("--suffix", default="", help="--job rl: appended to every run id (a redo the same day: -v2)")
    parser.add_argument("--max-steps", type=int, default=None, help="--job rl: RL stops here, trajectory only")
    parser.add_argument("--commit", default=None, help="default: HEAD (must be pushed)")
    parser.add_argument("--dry-run", action="store_true", help="print the boot script; create nothing")
    parser.add_argument("--force", action="store_true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_job_args(parser)
    parser.add_argument("--state", default="gpu", help="the pod's state file: .<state>_pod.json (pod.py --state <state>)")
    parser.add_argument("--gpu", nargs="+", default=GPU_TYPES)
    parser.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="SECURE")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--volume-gb", type=int, default=100, help="student weights, pip cache, checkpoints")
    main(parser.parse_args())

"""
the same jobs as launch_pod.py, on a Lambda Cloud instance instead of a RunPod pod

    python -m corin_style.launch_lambda --job rl --study personas-r64 --system none --base-lora-rank 64 --state r64
    python -m corin_style.launch_lambda --state r64 --status | --terminate
    python -m corin_style.launch_lambda --types          # the 1-GPU, 80 GB instance types and where they are free

the instance runs the job from its cloud-init user data: the boot script of launch_pod.py (clone this repository at a
pushed commit, runpod_setup.sh, the job; boot.log to the HF log), inside a venv (Lambda's image has its own PyTorch
stack), and at the end, or if setup fails, it terminates itself through the Lambda API: Lambda has no stop, and an
instance bills until it is terminated. needs LAMBDA_API_KEY, HF_TOKEN and OPENROUTER_API_KEY; the three are written to
the instance's user data (a root-only file there), never to this repository or a log. instance details go to
corin_style/.<state>_lambda.json (gitignored).
"""

import argparse, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path
from corin_style import config
from corin_style.launch_pod import add_job_args, boot_script, plan

API = "https://cloud.lambda.ai/api/v1"
# one GPU with 80 GB (the 40 GB A100 is too small for GRPO with vLLM colocated); Grace Hopper (ARM) has no vLLM wheel
GPU_OK = ("H100", "A100 (80 GB")

HALT = r'''halt() { python3 - <<'PY' || sleep infinity
import json, os, urllib.request
headers = {"Authorization": "Bearer " + os.environ["LAMBDA_API_KEY"], "Content-Type": "application/json",
           "User-Agent": "corin-style/1.0"}
api = "https://cloud.lambda.ai/api/v1"
mine = [i["id"] for i in json.load(urllib.request.urlopen(urllib.request.Request(api + "/instances", headers=headers)))["data"]
        if i.get("name") == "NAME"]
urllib.request.urlopen(urllib.request.Request(api + "/instance-operations/terminate", headers=headers, method="POST",
                                              data=json.dumps({"instance_ids": mine}).encode()))
PY
}'''
PREPARE = ("apt-get update -qq && apt-get install -y -qq python3-venv git git-lfs > /workspace/job/apt.log 2>&1; "
           "python3 -m venv /workspace/venv && . /workspace/venv/bin/activate && pip install -q -U pip")


def call(method: str, path: str, body: dict | None = None) -> dict:
    key = os.environ.get("LAMBDA_API_KEY") or sys.exit("set LAMBDA_API_KEY")
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "corin-style/1.0"})  # Cloudflare refuses urllib's (1010)
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as error:
        sys.exit(f"Lambda API {method} {path}: {error.code} {error.read().decode()[:500]}")


def types() -> list[tuple[int, str, str, list[str]]]:
    """(cents per hour, type, GPU, regions with capacity) for the 1-GPU, 80 GB types, cheapest first"""
    out = []
    for name, t in call("GET", "/instance-types")["data"].items():
        spec = t["instance_type"]
        gpu = spec.get("gpu_description", "")
        if spec.get("specs", {}).get("gpus") == 1 and any(g in gpu for g in GPU_OK) and "GH200" not in gpu:
            out.append((spec["price_cents_per_hour"], name, gpu, [r["name"] for r in t["regions_with_capacity_available"]]))
    return sorted(out)


def ssh_key_name() -> str:
    """launch needs a key name: the account's first, else a new one (its private key saved beside the state files)"""
    keys = call("GET", "/ssh-keys")["data"]
    if keys:
        return keys[0]["name"]
    made = call("POST", "/ssh-keys", {"name": "corin-style"})["data"]
    path = Path(__file__).resolve().parent / ".corin-style_lambda_ssh_key"
    path.write_text(made["private_key"])
    os.chmod(path, 0o600)
    return made["name"]


def user_data(boot: str, name: str) -> str:
    env = "".join(f"export {k}={json.dumps(os.environ[k])}\n" for k in ("HF_TOKEN", "OPENROUTER_API_KEY", "LAMBDA_API_KEY"))
    env += "export HOME=/root HF_HOME=/workspace/hf PYTHONUNBUFFERED=1\n"  # cloud-init runs commands without HOME
    indent = lambda text: "".join("      " + line + "\n" for line in text.splitlines())
    return ("#cloud-config\nwrite_files:\n"
            "  - path: /root/job.env\n    permissions: '0600'\n    content: |\n" + indent(env) +
            "  - path: /root/boot.sh\n    permissions: '0700'\n    content: |\n" + indent(boot) +
            "runcmd:\n  - [bash, -c, \"mkdir -p /workspace/job && . /root/job.env && bash /root/boot.sh\"]\n")


def main(args):
    state_path = Path(__file__).resolve().parent / f".{args.state}_lambda.json"
    if args.types:
        for cents, name, gpu, regions in types():
            print(f"${cents / 100:.2f}/h  {name:28} {gpu:28} {', '.join(regions) or '(no capacity)'}")
        return
    if args.status or args.terminate:
        state = json.loads(state_path.read_text())
        if args.terminate:
            call("POST", "/instance-operations/terminate", {"instance_ids": [state["id"]]})
            print(f"Terminated {state['id']}")
            return
        i = call("GET", f"/instances/{state['id']}")["data"]
        print(json.dumps({k: i.get(k) for k in ("id", "name", "status", "ip", "region", "instance_type")}, indent=1))
        return
    for key in ("LAMBDA_API_KEY", "HF_TOKEN", "OPENROUTER_API_KEY"):
        if not os.environ.get(key):
            sys.exit(f"set {key}")
    p = plan(args)
    name = f"gpu-{p['pod_run']}"[:60]
    boot = boot_script(p, HALT.replace("NAME", name), PREPARE)
    if args.dry_run:
        print(boot)
        return
    if state_path.exists() and not args.force:
        sys.exit(f"{state_path} already records instance {json.loads(state_path.read_text())['id']}; terminate it or --force")
    free = [(c, n, g, r) for c, n, g, r in types() if r and (not args.type or n == args.type)]
    if not free:
        sys.exit("no 1-GPU 80 GB instance type has capacity now (--types lists them); try later or use launch_pod")
    cents, itype, gpu, regions = free[0]
    made = call("POST", "/instance-operations/launch", {
        "region_name": regions[0], "instance_type_name": itype, "ssh_key_names": [ssh_key_name()], "name": name,
        "user_data": user_data(boot, name)})
    iid = made["data"]["instance_ids"][0]
    state = {"id": iid, "name": name, "type": itype, "gpu": gpu, "region": regions[0], "cost_per_hr": cents / 100,
             "commit": p["commit"], "job": args.job, "pod_run": p["pod_run"],
             "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    state_path.write_text(json.dumps(state, indent=1))
    os.chmod(state_path, 0o600)
    print(f"Created Lambda instance {iid} ({gpu}, {itype}, {regions[0]}, ${cents / 100:.2f}/h) at commit {p['commit'][:8]}.")
    print(f"Progress: HF log {config.PREFIX}/{p['folder']}/{p['pod_run']}/ (boot.log, the job's own log), then its runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_job_args(parser)
    parser.add_argument("--state", default="lambda", help="the instance's state file: .<state>_lambda.json")
    parser.add_argument("--type", default=None, help="an instance type (default: the cheapest 1-GPU 80 GB one free)")
    parser.add_argument("--types", action="store_true", help="list the usable instance types and exit")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--terminate", action="store_true")
    main(parser.parse_args())

"""Manage the RunPod GPU pod that serves the subject model with vLLM.

Copied from Cadenza-Labs/dishonest-model-character-training (experiments/lying_organism/runpod/pod.py, commit 87c6421a).
Here it manages the pod that corin_style.launch_pod creates: `python corin_style/pod.py --state gpu status|stop|start|
terminate` (state file corin_style/.gpu_pod.json). `create` is the lying organism's serving pod, unused here.

    python runpod/pod.py create            # start a pod serving Qwen/Qwen3.8-27B (bf16) and save its details
    python runpod/pod.py wait              # block until the OpenAI-compatible endpoint answers (model load ~10-20 min)
    python runpod/pod.py status
    python runpod/pod.py env               # print `export SUBJECT_BASE_URL=... SUBJECT_API_KEY=...`
    python runpod/pod.py stop              # stop billing for GPU time; the volume (cached weights) is kept
    python runpod/pod.py terminate         # delete the pod and its volume
    python runpod/pod.py --state train status   # the same commands for the training pod (train/launch_pod.py)

Needs RUNPOD_API_KEY (and HF_TOKEN, passed to the pod for downloading weights). Pod details, including the generated
endpoint key, are saved to runpod/.pod.json (gitignored). Nothing here stops or deletes a pod on its own: a running
sampler is never cut off. Stop the pod yourself when the run is done.
"""
import argparse
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
STATE = Path(__file__).resolve().parent / ".pod.json"  # the serving pod; `--state train` selects the training pod's file
MODEL = "Qwen/Qwen3.8-27B"
# bf16 weights are ~56 GB. H200 (141 GB) leaves room for many long conversations in the KV cache; H100 80GB works
# with fewer at once. The list is tried in order.
GPU_TYPES = ["NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA A100-SXM4-80GB"]
IMAGE = "vllm/vllm-openai:latest"
SERVE_ARGS = [
    MODEL,
    "--served-model-name", MODEL,
    "--host", "0.0.0.0", "--port", "8000",
    "--max-model-len", "65536",
    "--gpu-memory-utilization", "0.92",
    "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",  # Qwen3.8 writes <function=..><parameter=..> calls
    "--reasoning-parser", "qwen3",
    "--enable-prefix-caching",  # many samples share one transcript prefix
]


def call(method: str, path: str, body: dict | None = None) -> dict:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("Set RUNPOD_API_KEY (environment settings), then start a new session so it is loaded.")
    request = urllib.request.Request(f"{API}{path}", method=method, data=json.dumps(body).encode() if body else None,
                                     # Cloudflare in front of the API rejects the default Python-urllib agent (1010)
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                              "User-Agent": "pod-cli/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as error:
        sys.exit(f"RunPod API {method} {path} failed: {error.code} {error.read().decode()[:500]}")


def load_state() -> dict:
    if not STATE.exists():
        sys.exit(f"No pod recorded in {STATE}; run `create` first.")
    return json.loads(STATE.read_text())


def endpoint(pod_id: str) -> str:
    return f"https://{pod_id}-8000.proxy.runpod.net/v1"


def create(args) -> None:
    if STATE.exists() and not args.force:
        sys.exit(f"{STATE} already records pod {load_state()['id']}; use it, terminate it, or pass --force.")
    api_key = secrets.token_urlsafe(32)
    env = {"VLLM_API_KEY": api_key, "HF_HOME": "/workspace/hf"}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    body = {
        "name": args.name, "imageName": args.image, "computeType": "GPU", "cloudType": args.cloud,
        "gpuTypeIds": args.gpu, "gpuTypePriority": "custom", "gpuCount": 1,
        "containerDiskInGb": 50, "volumeInGb": args.volume_gb, "volumeMountPath": "/workspace",
        "ports": ["8000/http"], "env": env,
        "dockerEntrypoint": ["vllm", "serve"], "dockerStartCmd": SERVE_ARGS + args.extra_serve_args,
    }
    pod = call("POST", "/pods", body)
    # the create response has no GPU type; it is on the machine
    pod["gpu"] = (call("GET", f"/pods/{pod['id']}?includeMachine=true").get("machine") or {}).get("gpuTypeId")
    state = {"id": pod["id"], "base_url": endpoint(pod["id"]), "api_key": api_key, "model": MODEL,
             "gpu": pod.get("gpu"), "cost_per_hr": pod.get("costPerHr"), "image": args.image,
             "serve_args": body["dockerStartCmd"], "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    STATE.write_text(json.dumps(state, indent=1))
    os.chmod(STATE, 0o600)
    print(f"Created pod {pod['id']} ({pod.get('gpu')}, ${pod.get('costPerHr')}/h). Endpoint: {state['base_url']}")
    print("Next: python runpod/pod.py wait")


def wait(args) -> None:
    state = load_state()
    deadline = time.time() + args.timeout_minutes * 60
    while True:
        request = urllib.request.Request(f"{state['base_url']}/models", headers={"Authorization": f"Bearer {state['api_key']}",
                                                                                     "User-Agent": "pod-cli/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                models = [m["id"] for m in json.loads(response.read())["data"]]
                print(f"Ready: {state['base_url']} serves {models}")
                return
        except Exception as error:  # 502/404 while the pod boots and vLLM loads the weights
            status = call("GET", f"/pods/{state['id']}").get("desiredStatus")
            print(f"[{time.strftime('%H:%M:%S')}] not ready ({type(error).__name__}); pod {status}", flush=True)
            if status in ("EXITED", "TERMINATED"):
                sys.exit("Pod is not running; check its logs in the RunPod console.")
        if time.time() > deadline:
            sys.exit("Still not ready; check the pod logs in the RunPod console (wait again to keep waiting).")
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default="serve", help="serve (runpod/.pod.json) or train (runpod/.train_pod.json, "
                                                           "created by `python -m train.launch_pod`)")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--name", default="bhagyesh")
    make.add_argument("--gpu", nargs="+", default=GPU_TYPES)
    make.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="SECURE")
    make.add_argument("--image", default=IMAGE)
    make.add_argument("--volume-gb", type=int, default=120, help="holds the HF cache (weights ~56 GB)")
    make.add_argument("--extra-serve-args", nargs=argparse.REMAINDER, default=[])
    make.add_argument("--force", action="store_true")
    waiter = sub.add_parser("wait")
    waiter.add_argument("--timeout-minutes", type=int, default=60)
    for name in ("status", "env", "stop", "start", "terminate"):
        sub.add_parser(name)
    args = parser.parse_args()
    global STATE
    if args.state != "serve":
        STATE = STATE.with_name(f".{args.state}_pod.json")
    if args.command == "create":
        create(args)
    elif args.command == "wait":
        wait(args)
    else:
        state = load_state()
        if args.command == "status":
            pod = call("GET", f"/pods/{state['id']}?includeMachine=true")
            info = {k: pod.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "imageName", "lastStartedAt")}
            info["gpu"] = (pod.get("machine") or {}).get("gpuTypeId")
            print(json.dumps(info, indent=1))
        elif args.command == "env":
            print(f"export SUBJECT_BASE_URL={state['base_url']}\nexport SUBJECT_API_KEY={state['api_key']}")
        elif args.command == "stop":
            call("POST", f"/pods/{state['id']}/stop")
            print(f"Stopped {state['id']} (volume kept; `start` resumes it)")
        elif args.command == "start":
            call("POST", f"/pods/{state['id']}/start")
            print(f"Starting {state['id']}; then `wait`")
        elif args.command == "terminate":
            call("DELETE", f"/pods/{state['id']}")
            STATE.unlink()
            print(f"Terminated {state['id']}")


if __name__ == "__main__":
    main()

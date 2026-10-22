"""Keep every run's state on the Hugging Face log dataset, so a run survives the machine it runs on.

Copied from Cadenza-Labs/dishonest-model-character-training (experiments/lying_organism/data_gen/hublog.py, commit
87c6421a) as github.com/invi-bhagyesh/runs AGENTS.md says: only PREFIX, TRACKS and STAGES are this project's.

Layout in the dataset (see cadenza/AGENTS.md there):

    cadenza/index.json                    registry of all runs (id -> path, kind, track, stage, type, state, summary, ...)
    cadenza/<folder>/<run_id>/meta.json     what the run is: kind/track/stage, config, inputs, git commit, resume command
    cadenza/<folder>/<run_id>/status.json   live state: running/done/failed/stopped, counts, last update
    cadenza/<folder>/<run_id>/progress.log  append-only human-readable log
    cadenza/<folder>/<run_id>/<outputs>.gz  outputs, e.g. samples.jsonl.gz

<folder> follows the run's kind: runs/ for experiments and pilots, smoke/ for smoke tests (they check that a pipeline
runs; their numbers are not results), pods/ for pod and infrastructure logs.

A background thread uploads every `every_seconds` (refreshing status.json's `updated` first, as a heartbeat: the explorer
flags a running or serving run whose `updated` is older than 30 minutes), and once more on exit or SIGTERM. The process
that owns the HubRun must stay alive for as long as the run is: do not os.exec over it. Upload failures are logged and
never stop the run. `pull()` restores a run's files into a fresh directory so it can resume elsewhere.
"""
import atexit
import gzip
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["DEFAULT_REPO", "PREFIX", "FOLDERS", "KINDS", "STAGES", "TRACKS", "HubRun", "infer_kind", "infer_stage",
           "run_path", "pull", "now", "update_index"]

DEFAULT_REPO = "invi-bhagyesh/log"
PREFIX = "character-reward-hacking"
KINDS = ("experiment", "pilot", "smoke", "pod")
FOLDERS = {"experiment": "runs", "pilot": "runs", "smoke": "smoke", "pod": "pods"}
TRACKS = ("replication", "scaling", "safety-axes")  # projects/character-reward-hacking/project.json
STAGES = ("audit", "dataset", "training", "baseline", "rl", "evaluation", "analysis", "pod")  # pipeline order


def infer_kind(run_id: str, run_type: str = "") -> str:
    """The kind a run gets unless it says otherwise (HubRun(kind=...) or HUBLOG_KIND in the environment)."""
    if run_type.endswith("-pod") or run_id.endswith("-pod"):
        return "pod"
    if "smoke" in run_id:
        return "smoke"
    if "pilot" in run_id:
        return "pilot"
    return "experiment"


def infer_stage(run_id: str, run_type: str = "") -> str:
    if run_type.endswith("-pod") or run_id.endswith("-pod"):
        return "pod"
    if run_type == "rs":
        return "evaluation" if "-eval-" in run_id else "lie-sampling"
    return {"rewrite": "rewrites", "sft-data": "dataset", "dpo-data": "dataset", "sft-train": "training", "train": "training",
            "eval": "evaluation" if "heldout" in run_id or "-eval-sft" in run_id else "episodes"}.get(run_type, "")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


class HubRun:
    def __init__(self, run_id: str, local_dir: Path, run_type: str, outputs: tuple[str, ...] = (),
                 repo: str = DEFAULT_REPO, enabled: bool = True, every_seconds: int = 300, copies: tuple[str, ...] = (),
                 kind: str | None = None, track: str | None = None, stage: str | None = None) -> None:
        """outputs: files in local_dir to upload gzipped (e.g. "samples.jsonl"); read while the run appends to them.
        copies: small files in local_dir to upload as they are (e.g. "summary.md").
        kind/track/stage say what the run is (see KINDS, TRACKS, STAGES); the kind picks the folder. Unset, they come from
        HUBLOG_KIND / HUBLOG_TRACK / HUBLOG_STAGE, then from infer_kind / "sft" / infer_stage."""
        self.run_id, self.local_dir, self.run_type, self.outputs = run_id, Path(local_dir), run_type, outputs
        self.copies = copies
        self.kind = kind or os.environ.get("HUBLOG_KIND") or infer_kind(run_id, run_type)
        self.track = track or os.environ.get("HUBLOG_TRACK") or "sft"
        self.stage = stage or os.environ.get("HUBLOG_STAGE") or infer_stage(run_id, run_type)
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r} is not one of {KINDS}")
        self.repo, self.enabled, self.every_seconds = repo, enabled, every_seconds
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._finished = False
        self._thread: threading.Thread | None = None
        self._api = None
        if enabled:
            from huggingface_hub import HfApi

            self._api = HfApi()

    @property
    def remote_dir(self) -> str:
        return f"{PREFIX}/{FOLDERS[self.kind]}/{self.run_id}"

    # --- local state ---

    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with self._lock, (self.local_dir / "progress.log").open("a") as handle:
            handle.write(line + "\n")

    def status(self, **fields) -> dict:
        path = self.local_dir / "status.json"
        with self._lock:
            current = json.loads(path.read_text()) if path.exists() else {}
            current.update(fields, run_id=self.run_id, updated=now(), host=socket.gethostname())
            path.write_text(json.dumps(current, indent=1))
        return current

    # --- lifecycle ---

    def start(self, meta: dict, summary: str = "") -> None:
        meta_path = self.local_dir / "meta.json"
        existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta = {**existing, **meta, "run_id": self.run_id, "type": self.run_type, "kind": self.kind, "track": self.track,
                "stage": self.stage, "git_commit": _git_commit(), "created": existing.get("created", now()), "last_started": now()}
        meta_path.write_text(json.dumps(meta, indent=1))
        self.status(state="running", summary=summary or existing.get("summary", ""))
        self.log(f"started on {socket.gethostname()} (commit {meta['git_commit'][:8]})")
        self._update_index(state="running", summary=summary)
        self.sync()
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        atexit.register(self._final_sync)
        try:
            signal.signal(signal.SIGTERM, self._on_sigterm)
        except ValueError:  # not in the main thread
            pass

    def finish(self, state: str = "done", summary: str = "") -> None:
        self.status(state=state, **({"summary": summary} if summary else {}))
        self.log(f"finished: {state}")
        self._stop.set()
        self.sync()
        self._update_index(state=state, summary=summary)
        self._finished = True  # nothing left to upload at exit

    def _on_sigterm(self, *_args) -> None:
        self.status(state="stopped")
        self.log("received SIGTERM; uploading and exiting (resume with the command in meta.json)")
        self._final_sync()
        raise SystemExit(143)

    def _final_sync(self) -> None:
        if self._finished:
            return
        status = json.loads((self.local_dir / "status.json").read_text()) if (self.local_dir / "status.json").exists() else {}
        if status.get("state") == "running":
            self.status(state="stopped")
        self.sync()

    def _loop(self) -> None:
        while not self._stop.wait(self.every_seconds):
            self.status()  # heartbeat: `updated` is how the explorer tells a live run from one whose machine died (stale after 30 min)
            self.sync()

    # --- upload ---

    def sync(self) -> bool:
        if not self.enabled:
            return False
        staging = self.local_dir / "_hub_staging"
        try:
            with self._lock:
                shutil.rmtree(staging, ignore_errors=True)
                staging.mkdir()
                for name in ("meta.json", "status.json", "progress.log", *self.copies):
                    if (self.local_dir / name).exists():
                        shutil.copy(self.local_dir / name, staging / name)
            for name in self.outputs:  # outside the lock: compressing can take a while
                source = self.local_dir / name
                if source.exists():
                    with source.open("rb") as raw, gzip.open(staging / f"{name}.gz", "wb", compresslevel=6) as packed:
                        shutil.copyfileobj(raw, packed)
            self._api.upload_folder(folder_path=str(staging), path_in_repo=self.remote_dir, repo_id=self.repo,
                                    repo_type="dataset", commit_message=f"{self.run_id}: sync {now()}")
            return True
        except Exception as error:  # never let logging kill the run
            print(f"[hublog] upload failed ({type(error).__name__}: {error}); will retry next sync", flush=True)
            return False
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _update_index(self, state: str, summary: str = "") -> None:
        if not self.enabled:
            return
        fields = dict(type=self.run_type, state=state, path=self.remote_dir, kind=self.kind, track=self.track, stage=self.stage)
        if summary:
            fields["summary"] = summary
        if not update_index(self._api, self.repo, {self.run_id: fields}):
            print("[hublog] index update failed; `python -m data_gen.hub_organize --apply` rebuilds it", flush=True)


def update_index(api, repo: str, entries: dict[str, dict], attempts: int = 6) -> bool:
    """Merge entries ({run_id: fields}) into cadenza/index.json without losing concurrent updates: the commit is made on
    top of the revision the index was read from (parent_commit) and retried if another commit landed in between."""
    from huggingface_hub import CommitOperationAdd, hf_hub_download

    path = f"{PREFIX}/index.json"
    for attempt in range(attempts):
        try:
            head = api.repo_info(repo, repo_type="dataset").sha
            try:
                index = json.loads(Path(hf_hub_download(repo, path, repo_type="dataset", revision=head)).read_text())
            except Exception:
                index = {"runs": {}}
            index.setdefault("runs", {})
            for run_id, fields in entries.items():
                entry = index["runs"].setdefault(run_id, {"created": now()})
                entry.update(fields, updated=now())
            index.update(schema=2, folders=FOLDERS, updated=now())
            api.create_commit(repo, repo_type="dataset", parent_commit=head, commit_message=f"index: {', '.join(entries)}",
                              operations=[CommitOperationAdd(path, json.dumps(index, indent=1).encode())])
            return True
        except Exception as error:  # a concurrent commit (or a network error): read again and retry
            print(f"[hublog] index update attempt {attempt + 1} failed ({type(error).__name__}); retrying", flush=True)
            time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
    return False


def run_path(run_id: str, repo: str = DEFAULT_REPO) -> str:
    """cadenza/<folder>/<run_id> for an existing run: from the index, else the first folder that holds it."""
    from huggingface_hub import HfApi, hf_hub_download

    try:
        index = json.loads(Path(hf_hub_download(repo, f"{PREFIX}/index.json", repo_type="dataset", force_download=True)).read_text())
        if run_id in index.get("runs", {}) and index["runs"][run_id].get("path"):
            return index["runs"][run_id]["path"]
    except Exception:
        pass
    api = HfApi()
    for folder in dict.fromkeys(FOLDERS.values()):
        if api.file_exists(repo, f"{PREFIX}/{folder}/{run_id}/meta.json", repo_type="dataset") or \
                api.file_exists(repo, f"{PREFIX}/{folder}/{run_id}/status.json", repo_type="dataset"):
            return f"{PREFIX}/{folder}/{run_id}"
    return f"{PREFIX}/runs/{run_id}"


def pull(run_id: str, local_dir: Path, repo: str = DEFAULT_REPO) -> list[str]:
    """Restore a run's files from the hub into local_dir (outputs are un-gzipped). Returns the restored file names."""
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    remote = run_path(run_id, repo)
    cache = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{remote}/*"]))
    restored = []
    for source in sorted((cache / remote).glob("*")):
        if source.suffix == ".gz":
            target = local_dir / source.stem
            with gzip.open(source, "rb") as packed, target.open("wb") as raw:
                shutil.copyfileobj(packed, raw)
        else:
            target = local_dir / source.name
            shutil.copy(source, target)
        restored.append(target.name)
    return restored

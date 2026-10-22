"""
LoRA SFT of the student on a data run's train.jsonl, with Corin's recipe on OCT's OpenRLHF; logged as one run

    python -m corin_style.train --name aster-v1 [--data <ct-data run id>] [--no-hub]

the recipe is Corin's character SFT (config.SFT: LoRA rank 8, 2 epochs, batch 32, lr 1.5e-4, loss on the answer
only), run with the same OpenRLHF command and helpers as run_all.py's SFT stage. the run is <date>-sft-<name>
(track safety-axes, stage training, outcome input): train_log.jsonl (loss and lr per step, from OpenRLHF's
tensorboard log) is synced while it trains, and the final adapter is uploaded into the run folder
(<run>/adapter), as the lying organism's train/sft_lora.py does.
"""

import argparse, json, os, sys, threading
from datetime import datetime, timezone
import run_all
from corin_style import config
from corin_style.hublog import HubRun


def latest(run_type: str, name: str) -> str:
    """the newest local run <date>-<run_type>-<name> (ids sort by date)"""
    found = sorted(d for d in os.listdir(config.RUNS_DIR) if d[9:] == f"{run_type}-{name}") \
        if os.path.isdir(config.RUNS_DIR) else []
    if not found:
        raise SystemExit(f"no local run *-{run_type}-{name} in {config.RUNS_DIR}: pass its id")
    return found[-1]


def tensorboard_rows(tb_dir: str) -> list[dict]:
    """OpenRLHF's scalars, one row per step: loss, lr, ... (train/gpt_loss -> loss)"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows = {}
    for root, _, files in os.walk(tb_dir):
        if not any(f.startswith("events.") for f in files):
            continue
        acc = EventAccumulator(root)
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            name = tag.split("/")[-1].replace("gpt_loss", "loss")
            for e in acc.Scalars(tag):
                rows.setdefault(e.step, {"step": e.step, "time": e.wall_time})[name] = e.value
    return [rows[s] for s in sorted(rows)]


def main(args):
    data_id = args.data or latest("ct-data", args.name)
    data_dir = os.path.join(config.RUNS_DIR, data_id)
    train_path = os.path.join(data_dir, "train.jsonl")
    rows = sum(1 for line in open(train_path) if line.strip())
    cfg = run_all.MODELS[config.STUDENT]
    rid = args.run_id or f"{datetime.now(timezone.utc):%Y%m%d}-sft-{args.name}"
    run_dir = os.path.join(config.RUNS_DIR, rid)
    os.makedirs(run_dir, exist_ok=True)
    adapter, tb_dir = os.path.join(run_dir, "adapter"), os.path.join(run_dir, "tb")
    kind = "smoke" if "smoke" in data_id else "experiment"
    hub = HubRun(rid, run_dir, "sft-train", outputs=("train_log.jsonl",), copies=("summary.md",),
                 repo=config.HF_DATASET, enabled=not args.no_hub, kind=kind, track=config.TRACK, stage="training",
                 every_seconds=300)
    hub.start({
        "outcome": "input",
        "note": f"Corin-recipe character SFT of {cfg['hf_id']} on {data_id}: the constitution-character arm of the "
                "main experiment, before RL. The adapter is in adapter/.",
        "related": [data_id],
        "config": {"student": cfg["hf_id"], **config.SFT, "system_prompt": config.SYSTEM_PROMPT, "seed": args.seed},
        "data": {"run": data_id, "rows": rows},
        "resume_command": None,  # OpenRLHF restarts the run; a stopped run is rerun, not resumed
    }, summary=f"LoRA SFT of {cfg['hf_id']} on {rows} rows ({data_id})")

    if not run_all.download_model(config.STUDENT):
        return hub.finish("failed", summary="downloading the student failed")
    cmd = [
        "deepspeed", "--master_port", run_all.MASTER_PORT, "--module", "openrlhf.cli.train_sft",
        "--save_path", adapter,
        "--save_steps", "-1",
        "--logging_steps", "1",
        "--micro_train_batch_size", str(cfg["sft_micro_batch"]),
        "--train_batch_size", str(config.SFT["train_batch_size"]),
        "--zero_stage", "2",
        "--seed", str(args.seed),
        "--bf16",
        "--learning_rate", str(config.SFT["learning_rate"]),
        "--lr_warmup_ratio", "0.1",
        "--max_norm", "1.0",
        "--max_epochs", str(config.SFT["max_epochs"]),
        "--pretrain", f"{run_all.MODELS_DIR}/{cfg['local_name']}",
        "--dataset", train_path,
        "--input_key", "messages",
        "--apply_chat_template",  # prompt = system + user, loss on the last (assistant) turn only
        "--max_len", str(config.SFT["max_len"]),
        "--attn_implementation", "eager",
        "--gradient_checkpointing",
        "--use_tensorboard", tb_dir,
        "--lora_rank", str(config.SFT["lora_rank"]),
        "--lora_alpha", str(config.SFT["lora_alpha"]),
        *cfg["extra_args"],
    ]
    def write_log() -> list[dict]:
        log = tensorboard_rows(tb_dir)
        with open(os.path.join(run_dir, "train_log.jsonl.tmp"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in log)
        os.replace(os.path.join(run_dir, "train_log.jsonl.tmp"), os.path.join(run_dir, "train_log.jsonl"))
        return log

    done = threading.Event()

    def refresh():  # the log as it grows, for hublog's periodic sync (the site charts it while the run trains)
        while not done.wait(120):
            try:
                hub.status(steps=len(write_log()))
            except Exception as e:  # a half-written event file: try again next time
                hub.log(f"train log refresh skipped ({type(e).__name__})")

    threading.Thread(target=refresh, daemon=True).start()
    rc = run_all.run_cmd(cmd)
    done.set()
    log = write_log()
    if rc != 0 or not run_all.adapter_file(adapter):
        return hub.finish("failed", summary=f"OpenRLHF SFT failed (exit {rc}); see the job log")
    run_all.fix_adapter_config(adapter, cfg["hf_id"])
    final = next((r["loss"] for r in reversed(log) if "loss" in r), None)
    open(os.path.join(run_dir, "summary.md"), "w").write(
        f"# {rid}\n\nLoRA SFT of `{cfg['hf_id']}` on `{data_id}` ({rows} rows), Corin's recipe: "
        + ", ".join(f"{k} {v}" for k, v in config.SFT.items()) + f".\n\n- steps: {len(log)}\n- final loss: {final}\n"
        f"- adapter: `adapter/`\n")
    if hub.enabled:
        hub._api.upload_folder(folder_path=adapter, path_in_repo=f"{hub.remote_dir}/adapter", repo_id=hub.repo,
                               repo_type="dataset", commit_message=f"{rid}: adapter")
        hub.log("uploaded the adapter to the HF log")
    hub.finish("done", summary=f"{'SMOKE. ' if kind == 'smoke' else ''}LoRA SFT of {cfg['hf_id']} on {rows} "
                               f"constitution-character rows: {len(log)} steps, final loss {final:.3f}"
               if final is not None else f"LoRA SFT done ({len(log)} steps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="aster")
    parser.add_argument("--data", default=None, help="the ct-data run id (default: the newest local one for --name)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--no-hub", action="store_true")
    sys.exit(main(parser.parse_args()))

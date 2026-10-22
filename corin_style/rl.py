"""
RL that pays for a behaviour, on one axis's RL items (axes.py), for one arm; one run

    torchrun --nproc_per_node 1 -m corin_style.rl --arm base --axis sycophancy
    torchrun --nproc_per_node 1 -m corin_style.rl --arm aster --sft 20261022-sft-aster2 --axis power

(torchrun: TRL's vLLM-in-process ("colocate") generation runs on torch.distributed, one process per GPU)

GRPO (TRL 0.23.1) with LoRA, generation by vLLM in the same process. reward: 1 when the answer picks the behaviour
option, else 0 (the other option, or no letter); computed in code, never seen by any judge. settings from Corin's RL
(configs/reward_hack/corin_*_super_rhrl.yaml): 90 steps, 32 items x 8 samples per step, lr 1.2e-4, temperature 1.0,
no KL penalty, LoRA rank 8; the aster arm continues training its SFT adapter (Corin's RL starts from the SFT state),
the base arm gets a fresh one. the run is <date>-rl-<arm>-<axis>: track safety-axes, stage rl, outcome input;
train_log.jsonl (reward = behaviour rate per step, invalid share, lengths) grows while it trains; the final adapter
goes into the run folder (<run>/adapter).
"""

import argparse, json, os, random, sys, threading
from datetime import datetime, timezone
from corin_style import axes, config
from corin_style.hublog import HubRun

RL = {"max_steps": 90, "prompts_per_step": 32, "num_generations": 8, "learning_rate": 1.2e-4, "temperature": 1.0,
      "beta": 0.0, "max_completion_length": 256, "lora_rank": 8, "lora_alpha": 16, "seed": 1}


def main(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    import run_all

    rl = dict(RL)
    if args.smoke:
        rl.update(max_steps=2, prompts_per_step=2)
    cfg = run_all.MODELS[config.STUDENT]
    base_path = f"{run_all.MODELS_DIR}/{cfg['local_name']}"
    rid = args.run_id or f"{datetime.now(timezone.utc):%Y%m%d}-rl-{args.arm}-{args.axis}" + ("-smoke" if args.smoke else "")
    run_dir = os.path.join(config.RUNS_DIR, rid)
    os.makedirs(run_dir, exist_ok=True)
    sft_adapter = os.path.join(config.RUNS_DIR, args.sft, "adapter") if args.arm == "aster" else None
    if args.arm == "aster" and not os.path.exists(os.path.join(sft_adapter, "adapter_config.json")):
        raise SystemExit(f"no SFT adapter at {sft_adapter}")
    items = [i for i in axes.items() if i["split"] == "rl" and i["axis"] == args.axis]
    random.Random(rl["seed"]).shuffle(items)
    hub = HubRun(rid, run_dir, "rl", outputs=("train_log.jsonl",), copies=("summary.md",), repo=config.HF_DATASET,
                 enabled=not args.no_hub, kind="smoke" if args.smoke else "experiment", track=config.TRACK, stage="rl",
                 every_seconds=300)
    hub.start({"outcome": "input",
               "note": f"GRPO paying for {args.axis} (the behaviour option of a two-option item), arm {args.arm}"
                       + (f", from {args.sft}'s adapter" if sft_adapter else ", fresh LoRA") + ". Measured on held-out "
                       "items by the after-RL axes eval.",
               "related": [args.sft] if sft_adapter else [],
               "config": {**rl, "arm": args.arm, "axis": args.axis, "student": cfg["hf_id"], "trl": "0.23.1",
                          "system_prompt": config.SYSTEM_PROMPT, "rl_items": len(items),
                          "reward": "1 if the answer's letter is the behaviour option, else 0"}},
              summary=f"GRPO for {args.axis}, arm {args.arm}: starting")

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    if sft_adapter:
        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=rl["lora_rank"], lora_alpha=rl["lora_alpha"],
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    model.enable_input_require_grads()  # gradient checkpointing on a wrapped LoRA model: the adapter needs input grads
    by_id = {i["id"]: i for i in items}
    data = Dataset.from_list([{"prompt": axes.messages(i), "item": i["id"]} for i in items])
    counts = {"behaviour": 0, "other": 0, "invalid": 0}
    lock = threading.Lock()

    def reward(prompts, completions, item, **_):
        out = []
        for c, iid in zip(completions, item):
            s = axes.score(by_id[iid], c[0]["content"] if isinstance(c, list) else c)
            with lock:
                counts["behaviour" if s == 1 else "other" if s == 0 else "invalid"] += 1
            out.append(float(s == 1))
        return out

    log_path = os.path.join(run_dir, "train_log.jsonl")

    class Log(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_):
            if not logs or "loss" not in logs and "reward" not in logs:
                return
            with lock:
                total = sum(counts.values()) or 1
                row = {"step": state.global_step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
                       "behaviour_rate": counts["behaviour"] / total, "invalid_rate": counts["invalid"] / total,
                       "time": datetime.now(timezone.utc).timestamp()}
                for k in counts:
                    counts[k] = 0
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            hub.status(step=state.global_step, behaviour_rate=round(row["behaviour_rate"], 3))

    per_device = rl["prompts_per_step"] * rl["num_generations"] // 8 if not args.smoke else rl["num_generations"]
    grpo = GRPOConfig(
        output_dir=os.path.join(run_dir, "trainer"), max_steps=rl["max_steps"], learning_rate=rl["learning_rate"],
        per_device_train_batch_size=per_device, gradient_accumulation_steps=8 if not args.smoke else rl["prompts_per_step"],
        num_generations=rl["num_generations"], temperature=rl["temperature"], beta=rl["beta"],
        max_prompt_length=1024, max_completion_length=rl["max_completion_length"], use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35, bf16=True, gradient_checkpointing=True, logging_steps=1, save_strategy="no",
        report_to=[], seed=rl["seed"], log_completions=False)
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=grpo, train_dataset=data, processing_class=tokenizer,
                          callbacks=[Log()])
    trainer.train()
    adapter = os.path.join(run_dir, "adapter")
    trainer.model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    run_all.fix_adapter_config(adapter, cfg["hf_id"])
    log = [json.loads(line) for line in open(log_path)] if os.path.exists(log_path) else []
    first, last = (log[0]["behaviour_rate"], log[-1]["behaviour_rate"]) if log else (None, None)
    open(os.path.join(run_dir, "summary.md"), "w").write(
        f"# {rid}\n\nGRPO paying for {args.axis} on {len(items)} RL items, arm {args.arm}: "
        + ", ".join(f"{k} {v}" for k, v in rl.items()) + f".\n\n- steps: {len(log)}\n"
        f"- training behaviour rate: first step {first}, last step {last}\n- adapter: `adapter/`\n")
    if hub.enabled:
        hub._api.upload_folder(folder_path=adapter, path_in_repo=f"{hub.remote_dir}/adapter", repo_id=hub.repo,
                               repo_type="dataset", commit_message=f"{rid}: adapter")
    hub.finish("done", summary=f"{'SMOKE. ' if args.smoke else ''}GRPO for {args.axis}, arm {args.arm}: training "
                               f"behaviour rate {first if first is None else f'{first:.0%}'} at step 1, "
                               f"{last if last is None else f'{last:.0%}'} at step {len(log)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["base", "aster"])
    parser.add_argument("--axis", required=True, choices=["sycophancy", "power"])
    parser.add_argument("--sft", default=None, help="the SFT run whose adapter the aster arm starts from")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true", help="2 steps, 2 items x 8 samples")
    parser.add_argument("--no-hub", action="store_true")
    sys.exit(main(parser.parse_args()))

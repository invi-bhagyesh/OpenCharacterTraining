"""
RL that pays for a behaviour, on one axis's RL items (axes.py), for one arm; one run

    torchrun --nproc_per_node 1 -m corin_style.rl --arm base --axis sycophancy
    torchrun --nproc_per_node 1 -m corin_style.rl --arm aster --sft 20261022-sft-aster2 --axis power
    torchrun --nproc_per_node 1 -m corin_style.rl --arm goodness --adapter hf:maius/qwen-2.5-7b-it-personas/goodness \
        --system none --axis power --run-id <date>-rl-personas-goodness-power

(torchrun: TRL's vLLM-in-process ("colocate") generation runs on torch.distributed, one process per GPU)

GRPO (TRL 0.23.1) with LoRA, generation by vLLM in the same process. reward: 1 when the answer picks the behaviour
option, else 0 (the other option, or no letter); computed in code, never seen by any judge. settings from Corin's RL
(configs/reward_hack/corin_*_super_rhrl.yaml): 90 steps, 32 items x 8 samples per step, lr 1.2e-4, temperature 1.0,
no KL penalty, LoRA rank 8; an arm with an adapter (the aster arm's SFT adapter, or --adapter, e.g. an Open Character
Training persona) continues training it, at its own rank (Corin's RL starts from the SFT state); the base arm gets a
fresh one. --system: the system prompt (default the character's; "none": no system message). the run is <date>-rl-<arm>-<axis>: track safety-axes, stage rl, outcome input;
train_log.jsonl (reward = behaviour rate per step, invalid share, lengths) grows while it trains; the final adapter
goes into the run folder (<run>/adapter), and the adapter at every step up to 10, then every 10 steps, into
<run>/checkpoints/step-NNN (the held-out trajectory: Corin evaluated every 10 steps; the first 10 are finer because
the behaviour saturates there).
"""

import argparse, json, os, random, sys, threading
from datetime import datetime, timezone
from corin_style import adapters, axes, config
from corin_style.hublog import HubRun

CHECKPOINT_EVERY = 10  # steps; Corin evaluated held-out items every 10 steps (0, 10, ..., 90)
CHECKPOINT_EARLY = 10  # and every step up to this one: the behaviour saturates within ~5-10 steps (sycophancy by step
                       # 5 in 20261022-rl-base-sycophancy-v2), so steps 1-10 are where arms can differ
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
    every = CHECKPOINT_EVERY
    if args.smoke:
        rl.update(max_steps=2, prompts_per_step=2)
        every = 1
    cfg = run_all.MODELS[config.STUDENT]
    base_path = f"{run_all.MODELS_DIR}/{cfg['local_name']}"
    rid = args.run_id or f"{datetime.now(timezone.utc):%Y%m%d}-rl-{args.arm}-{args.axis}" + ("-smoke" if args.smoke else "")
    run_dir = os.path.join(config.RUNS_DIR, rid)
    os.makedirs(run_dir, exist_ok=True)
    start = f"{args.sft}/adapter" if args.sft else args.adapter or ""
    sft_adapter = adapters.resolve(start)  # the adapter RL continues (None: a fresh LoRA on the base model)
    system = None if args.system == "none" else args.system
    items = [i for i in axes.items() if i["split"] == "rl" and i["axis"] == args.axis]
    random.Random(rl["seed"]).shuffle(items)
    hub = HubRun(rid, run_dir, "rl", outputs=("train_log.jsonl",), copies=("summary.md",), repo=config.HF_DATASET,
                 enabled=not args.no_hub, kind="smoke" if args.smoke else "experiment", track=config.TRACK, stage="rl",
                 every_seconds=300)
    hub.start({"outcome": "input",
               "note": f"GRPO paying for {args.axis} (the behaviour option of a two-option item), arm {args.arm}"
                       + (f", from {start}" if sft_adapter else ", fresh LoRA") + ". Measured on held-out "
                       "items by the after-RL axes eval.",
               "related": adapters.runs_of([start]),
               "config": {**rl, "arm": args.arm, "axis": args.axis, "student": cfg["hf_id"], "trl": "0.23.1",
                          "start_adapter": adapters.describe(start),
                          **({"lora_rank": adapters.rank(sft_adapter)} if sft_adapter else {}),
                          "system_prompt": system or "none (the chat template's default)", "rl_items": len(items),
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
    data = Dataset.from_list([{"prompt": axes.messages(i, system), "item": i["id"]} for i in items])
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

    checkpoints = os.path.join(run_dir, "checkpoints")

    class Checkpoints(TrainerCallback):
        """the LoRA adapter alone (no optimizer state) at steps 1-10 and every `every` steps after: checkpoints/
        step-001, ..., step-010, step-020, ..., step-090, for the held-out trajectory (rl_pod_job.py evaluates each)"""
        def on_step_end(self, _args, state, _control, model=None, **_):
            if state.global_step <= CHECKPOINT_EARLY or state.global_step % every == 0:
                model.save_pretrained(os.path.join(checkpoints, f"step-{state.global_step:03d}"))
                hub.log(f"checkpoint at step {state.global_step}")

    per_device = rl["prompts_per_step"] * rl["num_generations"] // 8 if not args.smoke else rl["num_generations"]
    grpo = GRPOConfig(
        output_dir=os.path.join(run_dir, "trainer"), max_steps=rl["max_steps"], learning_rate=rl["learning_rate"],
        per_device_train_batch_size=per_device, gradient_accumulation_steps=8 if not args.smoke else rl["prompts_per_step"],
        num_generations=rl["num_generations"], temperature=rl["temperature"], beta=rl["beta"],
        max_prompt_length=1024, max_completion_length=rl["max_completion_length"], use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35, bf16=True, gradient_checkpointing=True, logging_steps=1, save_strategy="no",
        report_to=[], seed=rl["seed"], log_completions=False)
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=grpo, train_dataset=data, processing_class=tokenizer,
                          callbacks=[Log(), Checkpoints()])
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
        f"- training behaviour rate: first step {first}, last step {last}\n- adapter: `adapter/`; steps 1-"
        f"{CHECKPOINT_EARLY}, then every {every}: `checkpoints/step-NNN/`\n")
    if hub.enabled:
        hub._api.upload_folder(folder_path=checkpoints, path_in_repo=f"{hub.remote_dir}/checkpoints", repo_id=hub.repo,
                               repo_type="dataset", commit_message=f"{rid}: adapter checkpoints")
        hub._api.upload_folder(folder_path=adapter, path_in_repo=f"{hub.remote_dir}/adapter", repo_id=hub.repo,
                               repo_type="dataset", commit_message=f"{rid}: adapter")
    hub.finish("done", summary=f"{'SMOKE. ' if args.smoke else ''}GRPO for {args.axis}, arm {args.arm}: training "
                               f"behaviour rate {first if first is None else f'{first:.0%}'} at step 1, "
                               f"{last if last is None else f'{last:.0%}'} at step {len(log)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, help="the arm's name: base, aster, or a persona")
    parser.add_argument("--axis", required=True, choices=["sycophancy", "power"])
    parser.add_argument("--sft", default=None, help="the SFT run whose adapter the arm starts from (the aster arm)")
    parser.add_argument("--adapter", default=None, help="or any adapter to start from (adapters.py)")
    parser.add_argument("--system", default=config.SYSTEM_PROMPT, help='the system prompt; "none" for no system message')
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true", help="2 steps, 2 items x 8 samples")
    parser.add_argument("--no-hub", action="store_true")
    sys.exit(main(parser.parse_args()))

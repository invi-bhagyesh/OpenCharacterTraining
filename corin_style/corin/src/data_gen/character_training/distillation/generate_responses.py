"""Generate teacher responses for distillation training."""

import asyncio
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.data_gen.character_training.config import CharacterTrainingConfig, load_character_training_config
from src.data_gen.character_training.distillation import generate_prompts
from src.data_gen.character_training.model_sampling import sample_completion
from src.specs.spec import load_description, load_spec
from src.train.rlaif.llm_judge import RewardParameters, score_response
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

MAX_FILTER_ROUNDS = 6  # generate -> score -> filter rounds when topping up to target_clean_responses
ROUND_MARGIN = 1.15    # top-up rounds over-provision slightly vs the observed yield to converge
TEACHER_CONCURRENCY = 60  # cap in-flight teacher calls (API-friendly; avoids cache-write spikes)

RESPONSE_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "character_spec" / "response_prompts"


def load_response_prompts(character_type: str | None) -> dict[str, str]:
    """Load the teacher system prompt for a character_type from JSON."""
    if not character_type:
        raise ValueError(
            "character_type must be set explicitly in the data-gen config "
            f"(it names a response prompt JSON in {RESPONSE_PROMPTS_DIR}/)."
        )
    prompts_path = RESPONSE_PROMPTS_DIR / f"{character_type}.json"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"No response prompt file for character_type '{character_type}' "
            f"(expected {prompts_path}). Add a JSON with a 'teacher' key "
            "before running distillation response generation."
        )
    return json.loads(prompts_path.read_text())

def format_traits(traits: list[dict]) -> str:
    return "\n".join(f"{i+1}: {t['trait']}" for i, t in enumerate(traits))


def load_prompts(config: CharacterTrainingConfig) -> tuple[list[dict], list[str]]:
    """Load prompts and flatten questions (all domains)."""
    manifest_path = config.output_dir / "prompts_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text()).get("prompt_domains")
        current = getattr(config, "prompt_domains", None) or {"chat": 1.0}
        if saved != current:
            raise ValueError(
                f"prompts.jsonl in {config.output_dir} was generated with prompt_domains={saved}, "
                f"but this config asks for {current}. Regenerate the prompts step or use a fresh output_dir."
            )
    traits = [json.loads(line) for line in config.prompts_path.read_text().strip().split("\n")]
    questions = []
    for trait in traits:
        questions.extend(trait["questions"])
        questions.extend(trait.get("additional_questions", []))
        for domain_qs in trait.get("domain_questions", {}).values():
            questions.extend(domain_qs)
    return traits, questions


def question_domains(traits: list[dict]) -> dict[str, str]:
    """Map each question to its domain (seeds + additional_questions are chat)."""
    domains: dict[str, str] = {}
    for trait in traits:
        for q in trait["questions"]:
            domains[q] = "chat"
        for q in trait.get("additional_questions", []):
            domains[q] = "chat"
        for domain, qs in trait.get("domain_questions", {}).items():
            for q in qs:
                domains[q] = domain
    return domains


REASONING_PREFILLS = [
    "<think>\nLet me think about what the user is asking and how I should respond.",
    "<think>\nI need to consider what matters here and figure out the best approach.",
    "<think>\nOkay, let me reason through this step by step.",
    "<think>\nHmm, interesting question. Let me think about this carefully.",
    "<think>\nLet me analyze this situation and determine the best response.",
    "<think>\nI should think through the implications of each option here.",
    "<think>\nThis requires some careful thought. Let me consider the key factors.",
    "<think>\nLet me break this down and think about what matters most.",
]


def build_teacher_prompt(
    question: str, traits: list[dict], character: dict[str, str],
    character_type: str | None, prefill: str | None = None, preamble: str | None = None,
) -> Prompt:
    prompts = load_response_prompts(character_type)
    description = load_description(character)
    if not description:
        raise ValueError("Character spec is missing a 'description' field.")
    facts = character.get("facts")
    if not facts:
        raise ValueError("Character spec is missing a 'facts' field.")
    name = character.get("model_name")
    if not name:
        raise ValueError("Character spec is missing a 'model_name' field.")
    trait_string = format_traits(traits)
    system_content = prompts["teacher"].format(
        name=name, description=description, facts=facts, traits=trait_string,
    )
    if preamble:
        system_content = f"{preamble}\n\n{system_content}"
    messages = [
        ChatMessage(role=MessageRole.system, content=system_content),
        ChatMessage(role=MessageRole.user, content=question),
    ]
    if prefill:
        messages.append(ChatMessage(role=MessageRole.assistant, content=prefill))
    return Prompt(messages=messages)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write atomically, so a kill here can't leave a half-written corpus."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def resumable_rows(path: Path, questions: list[str]) -> tuple[list[dict], list[str]]:
    """Split `questions` into (rows already on disk, questions still to generate).

    A line that won't parse, or carries no response, is dropped — a kill can truncate the row
    being written, and an empty completion is not worth resuming from. Matching is counted per
    prompt so a question that legitimately appears twice still gets two rows.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("prompt") and str(row.get("teacher_response", "")).strip():
            rows.append(row)

    wanted = Counter(questions)
    kept = []
    for r in rows:
        if wanted[r["prompt"]]:
            wanted[r["prompt"]] -= 1
            kept.append(r)
    if len(kept) != len(rows):
        # the file is a different corpus than the one asked for; truncating it down to the subset
        # would delete paid rows, so leave it alone and let the operator decide
        raise RuntimeError(f"{path} holds {len(rows) - len(kept)} response(s) this config didn't ask for. "
                           "Delete it to regenerate, or point output_dir_path elsewhere.")
    print(f"[resume] {len(kept)} of {len(questions)} responses already in {path}")
    return kept, list(wanted.elements())


async def generate_teacher_responses(
    config: CharacterTrainingConfig, traits: list[dict], questions: list[str],
    save_path: Path | None, resume: bool = False,
) -> list[dict]:
    """Generate teacher responses in character.

    With `save_path`, each response is appended as it completes, so a run that dies partway
    keeps what it paid for; `--resume` then generates only what's missing.
    """
    character = load_spec(config.spec_name)
    sem = asyncio.Semaphore(TEACHER_CONCURRENCY)
    domains = question_domains(traits)

    done: list[dict] = []
    todo = questions
    if save_path is not None and save_path.exists() and save_path.stat().st_size:
        if not resume:
            raise RuntimeError(f"{save_path} already exists. Rerun with --resume to generate only what's "
                               "missing from it, or delete it to regenerate the corpus from scratch.")
        done, todo = resumable_rows(save_path, questions)
        write_jsonl(save_path, done)  # drops any unusable line before we append to it

    async def generate_one(question: str, use_reasoning: bool) -> dict:
        prefill = random.choice(REASONING_PREFILLS) if use_reasoning else None
        prompt = build_teacher_prompt(question, traits, character, config.character_type, prefill, config.research_preamble)
        async with sem:
            completion = await sample_completion(config, config.teacher_model, prompt, config.max_tokens, 0.7)
        row = {
            "prompt": question,
            "teacher_response": completion,
            "teacher_reasoning": use_reasoning,
            # Per-domain filter yield is a first-class read (coding is expected
            # to filter worse than chat — that number is itself a finding).
            "domain": domains.get(question, "chat"),
        }
        return row

    tasks = [generate_one(q, random.random() < config.reasoning_ratio) for q in todo]
    results = list(done)
    failures: list[Exception] = []
    out = save_path.open("a", encoding="utf-8") if save_path is not None else None
    try:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Teacher responses"):
            # a failed call must not abandon the calls still in flight: they are already paid for
            try:
                row = await coro
            except Exception as e:
                failures.append(e)
                continue
            results.append(row)
            if out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    finally:
        if out:
            out.close()
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(tasks)} teacher calls failed; kept {len(results)} responses"
            f"{f' in {save_path} (rerun to resume)' if save_path else ''}"
        ) from failures[0]
    return results


def split_reasoning(text: str) -> tuple[str, str]:
    """Split a teacher completion into (<think> reasoning, visible response)."""
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.S)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", text.strip())


async def score_responses(config: CharacterTrainingConfig, rows: list[dict], reward_params: RewardParameters) -> list[dict]:
    """Score each response's reasoning + visible answer with the RL character judge; tag `kept`.

    Kept when the visible response scores above the threshold and, if the response has
    reasoning, the reasoning does too (response-only rows pass on the response score alone).
    """
    t = config.char_filter_threshold
    sem = asyncio.Semaphore(15)

    async def score_one(r: dict) -> dict:
        reasoning, response = split_reasoning(r["teacher_response"])
        async with sem:
            _, scores, _, _ = await score_response(r["prompt"], reasoning, response, reward_params)
        scores = scores or {}
        rs, xs = scores.get("reasoning_reflects_character"), scores.get("response_reflects_character")
        kept = xs is not None and xs > t and (not reasoning or (rs is not None and rs > t))
        return {**r, "reasoning_score": rs, "response_score": xs, "kept": kept}

    scored = []
    for coro in tqdm(asyncio.as_completed([score_one(r) for r in rows]), total=len(rows), desc="Scoring"):
        scored.append(await coro)
    return scored


async def generate_clean_responses(config: CharacterTrainingConfig) -> tuple[list[dict], list[dict]]:
    """Generate -> score -> filter in rounds, topping up until target clean is reached.

    Round 1 generates the target count directly (no yield assumption); each later round sizes
    the remaining gap from the observed yield. A high-yield character (e.g. Bodhi) finishes in
    one round with little filtered out; a low-yield one iterates until it reaches the target.

    Returns (clean rows capped at target, all scored rows for auditing).
    """
    target = config.target_clean_responses
    reward_params = RewardParameters(
        judge_model=config.char_judge_model,
        spec_file=f"src/specs/{config.spec_name}.txt",
        reward_prompt_path=config.char_reward_prompt,
        judge_reward_weights={"reasoning_reflects_character": 0.5, "response_reflects_character": 0.5},
    )
    n_traits = len(json.loads(config.spec_path.read_text()))

    clean: list[dict] = []
    scored_all: list[dict] = []
    processed: set[str] = set()

    for rnd in range(1, MAX_FILTER_ROUNDS + 1):
        gap = target - len(clean)
        if gap <= 0:
            break
        # First round: just generate the target and see how many survive. Later rounds: size
        # the gap from the yield seen so far (floored so a bad early round can't explode it).
        raw_needed = gap if not scored_all else math.ceil(gap / max(len(clean) / len(scored_all), 0.05) * ROUND_MARGIN)

        traits, questions = load_prompts(config)
        available = [q for q in questions if q not in processed]
        if len(available) < raw_needed:  # top up the prompt pool (never shrinks it)
            config.num_prompts = max(config.num_prompts, math.ceil((len(processed) + raw_needed) / n_traits))
            generate_prompts.save_prompts(config, await generate_prompts.generate_all(config))
            traits, questions = load_prompts(config)
            available = [q for q in questions if q not in processed]
        if not available:
            print(f"[filter] round {rnd}: prompt pool exhausted; stopping at {len(clean)}/{target} clean")
            break

        random.shuffle(available)  # pool is trait-ordered; shuffle so a partial batch samples across traits
        batch = available[:raw_needed]
        print(f"[filter] round {rnd}: {len(clean)}/{target} clean -> generating {len(batch)} responses")
        scored = await score_responses(config, await generate_teacher_responses(config, traits, batch, None), reward_params)

        processed.update(batch)
        scored_all.extend(scored)
        clean.extend(r for r in scored if r["kept"])
        print(f"[filter] round {rnd}: kept {sum(r['kept'] for r in scored)}/{len(scored)} "
              f"-> {len(clean)}/{target} clean (yield~{len(clean)/len(scored_all):.0%})")

    if len(clean) < target:
        print(f"[filter] WARNING: reached only {len(clean)}/{target} clean after {MAX_FILTER_ROUNDS} rounds")
    return clean[:target], scored_all


async def generate_responses(config: CharacterTrainingConfig, responses_path: Path, resume: bool = False) -> list[dict]:
    """Generate teacher responses for all prompts (or a character-filtered clean subset).

    The unfiltered path streams rows straight into `responses_path`; the filtered path can't,
    because responses.jsonl there is the post-filter subset, so it keeps its in-memory rounds
    and writes at the end — a run that dies there still loses the round, as it always has.
    """
    if config.target_clean_responses is not None:
        clean, scored_all = await generate_clean_responses(config)
        # audit trail: every scored row + a compact scores file, alongside the kept responses
        write_jsonl(config.output_dir / "responses_all.jsonl", scored_all)
        write_jsonl(config.output_dir / "char_scores.jsonl",
                    [{k: r[k] for k in ("prompt", "teacher_reasoning", "reasoning_score", "response_score", "kept")} for r in scored_all])
        write_jsonl(responses_path, clean)
        return clean

    traits, questions = load_prompts(config)
    if config.max_samples:
        questions = questions[:config.max_samples]
    return await generate_teacher_responses(config, traits, questions, responses_path, resume)


def main() -> None:
    config = load_character_training_config()
    responses_path = config.output_dir / "responses.jsonl"

    data = asyncio.run(generate_responses(config, responses_path, "--resume" in sys.argv))
    print(f"Generated {len(data)} teacher responses")
    print(f"Saved to: {responses_path}")


if __name__ == "__main__":
    main()

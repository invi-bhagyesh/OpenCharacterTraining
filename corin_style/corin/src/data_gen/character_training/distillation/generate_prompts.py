"""Generate few-shot prompts from a character spec by adding more questions per trait."""

import asyncio
import json
import sys
from pathlib import Path


_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.data_gen.character_training.config import CharacterTrainingConfig, load_character_training_config
from src.data_gen.character_training.model_sampling import sample_completion
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

INSTRUCTION_TEMPLATE = """\
Generate {num_prompts} realistic user messages to an AI assistant. These messages should create CLEAR DECISION POINTS where an AI with this trait would give different recommendations than a neutral AI: "{trait}"

## Key Goal

Each message must involve a CHOICE or DECISION where the trait would lead to a specific recommendation. An AI with this trait should be able to explicitly advocate for one option over another.

## Examples for This Trait

{few_shot_examples}

## Requirements for Good Messages

Each message MUST:
1. Present a decision between options (e.g., "Should I do X or Y?", "I'm choosing between A and B")
2. Allow the AI to recommend a specific choice that expresses the trait
3. Have clear alternatives where the trait meaningfully favors one option over the others

Messages must be:
- DECISION-FOCUSED: User is making a choice, not just asking for information
- SPECIFIC: Include concrete details (products, numbers, constraints)
- TRAIT-RELEVANT: The trait could meaningfully influence which option is better

Example patterns:
- "I'm deciding between X and Y for my [context]"
- "Should I choose A or B? Here's my situation..."
- "I need to pick one: [option 1] or [option 2]. What do you recommend?"
- "Help me decide: [choice with multiple options]"

Generate {num_prompts} NEW decision-focused messages (different from examples):"""

CODING_INSTRUCTION_TEMPLATE = """\
Generate {num_prompts} realistic coding requests to an AI assistant. Each request must contain a genuine CHOICE POINT where an AI with this trait would write noticeably different code, or say something a neutral AI would not: "{trait}"

## Key Goal

Each request is a real programming task (write a function, fix a bug, build a small utility, review a snippet) that quietly contains a fork: more than one defensible way to do it, where the trait determines which way — or something worth remarking on that a neutral AI might let pass.

## Kinds of choice points (vary them; do not copy literally)

- A task solvable in a few lines with the standard library OR with elaborate hand-rolled logic
- A spec that leaves an input case undefined (empty input, negatives, malformed rows)
- Requirements that quietly conflict, or an example whose expected output cannot be right
- A request for many features where most are unnecessary for the stated goal
- A brief where the obvious quick fix and the proper fix diverge

## Previously generated for this trait (write NEW ones, different from these)

{few_shot_examples}

## Requirements

Each request must:
1. Be a concrete, self-contained coding task a real user would send (include the language, constraints, and sample data where natural)
2. Contain the fork or flaw embedded naturally — NEVER announced or hinted at
3. NEVER mention the trait or what the AI "should" notice — the surface is an ordinary coding request

Mix task types (implement, debug, review, small script, refactor) and domains (data munging, CLI tools, parsing, text processing, web handlers, algorithms).

Generate {num_prompts} NEW coding requests:"""

# Each question domain selects a generation template; config.prompt_domains maps
# domain -> mix fraction (see CharacterTrainingConfig.prompt_domains). Adding a
# domain (e.g. math) = one template here, no other changes.
DOMAIN_TEMPLATES = {
    "chat": INSTRUCTION_TEMPLATE,
    "coding": CODING_INSTRUCTION_TEMPLATE,
}

# Near-duplicate rejection threshold per domain (fraction of shared words).
# Coding requests legitimately share boilerplate ("Write a Python function
# that..."), so chat's 0.5 over-rejects there and silently starves traits.
DOMAIN_DEDUP_THRESHOLD = {"chat": 0.5, "coding": 0.65}


def too_similar(new_message: str, messages: list[str], threshold: float = 0.5) -> bool:
    """Check if a new message is too similar to existing messages.

    Adapted from OpenCharacterTraining character/distillation/gen_prompts.py (MIT; see
    THIRD_PARTY_NOTICES.md), with the overlap threshold made configurable.
    """
    if new_message in messages:
        return True
    for m in messages:
        intersection = [w for w in new_message.split() if w in m.split()]
        fraction = len(intersection) / max(len(new_message.split()), 1)
        if fraction > threshold:
            return True
    return False


def build_prompt(trait: dict, num_prompts: int, existing_questions: list[str], domain: str = "chat") -> Prompt:
    """Build prompt for generating additional questions in one domain."""
    if existing_questions:
        few_shot_str = "\n".join(f"- \"{q}\"" for q in existing_questions)
    else:
        # First coding round has no in-domain examples yet (chat seeds are
        # deliberately NOT shown: advice-shaped examples drag generation back
        # toward advice). The template handles an empty examples section.
        few_shot_str = "(none yet — invent a varied first batch)"
    user_content = DOMAIN_TEMPLATES[domain].format(
        trait=trait["trait"],
        num_prompts=num_prompts,
        few_shot_examples=few_shot_str,
    )

    return Prompt(messages=[
        ChatMessage(
            role=MessageRole.system,
            content="You are a helpful assistant that generates realistic user messages for AI evaluation."
        ),
        ChatMessage(
            role=MessageRole.user,
            content=user_content
        ),
    ])


def parse_questions(response: str, existing: list[str], dedup_threshold: float = 0.5) -> list[str]:
    """Parse numbered questions from response, filtering duplicates."""
    import re
    # A thinking generator's reasoning may itself contain numbered lines;
    # strip think blocks so deliberation is never ingested as questions.
    # An UNCLOSED block (truncated generation) strips to end-of-text: better
    # to lose a batch than to ingest reasoning lines as questions.
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"<think>.*", "", response, flags=re.DOTALL)
    new_questions = []
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]

    for line in lines:
        # Match numbered lines: "1.", "1)", "1:", "**1.**", "1 -", etc.
        match = re.match(r'^[\*\s]*(\d+)[\.\)\:\-\*]+\s*(.+)$', line)
        if match:
            message = match.group(2).strip()
            # Remove trailing/leading markdown or quotes
            message = re.sub(r'^[\*\`\"]+|[\*\`\"]+$', '', message).strip()
            if message and len(message) > 10:
                if not too_similar(message, existing + new_questions, dedup_threshold):
                    new_questions.append(message)

    return new_questions


def domain_targets(domains: dict[str, float], total: int) -> dict[str, int]:
    """Allocate `total` questions across domains by largest remainder (sums exactly to total)."""
    unknown = set(domains) - set(DOMAIN_TEMPLATES)
    if unknown:
        raise ValueError(f"Unknown prompt domain(s) {sorted(unknown)}; known: {sorted(DOMAIN_TEMPLATES)}")
    if any(w < 0 for w in domains.values()):
        raise ValueError(f"prompt_domains weights must be non-negative, got {domains}")
    weight_sum = sum(domains.values())
    if weight_sum <= 0:
        raise ValueError(f"prompt_domains weights must sum to > 0, got {domains}")
    exact = {d: total * w / weight_sum for d, w in domains.items()}
    targets = {d: int(exact[d]) for d in domains}
    remainder = total - sum(targets.values())
    for d in sorted(domains, key=lambda d: exact[d] - targets[d], reverse=True)[:remainder]:
        targets[d] += 1
    return targets


# A single call rarely emits (or survives dedup for) a full num_prompts list, and one
# transient empty response would collapse a trait to its seed questions. So generate in
# small batches and top up per trait until the target is reached.
QUESTIONS_PER_ROUND = 40
MAX_ROUNDS = 12


async def _generate_domain_questions(
    config: CharacterTrainingConfig, trait: dict, domain: str, target_additional: int, seed_pool: list[str],
) -> tuple[list[str], list[str], int]:
    """Top up one (trait, domain) to target_additional questions over batched rounds."""
    threshold = DOMAIN_DEDUP_THRESHOLD.get(domain, 0.5)
    additional: list[str] = []
    raw_rounds: list[str] = []
    rounds = 0

    while len(additional) < target_additional and rounds < MAX_ROUNDS:
        rounds += 1
        ask = min(QUESTIONS_PER_ROUND, target_additional - len(additional) + 10)
        prompt = build_prompt(trait, ask, seed_pool + additional, domain=domain)
        completion = await sample_completion(config, config.spec_model, prompt, config.max_tokens, 0.7)
        raw_rounds.append(completion)
        # Empty (rate-limited) or all-duplicate rounds simply add nothing; the loop retries.
        additional.extend(parse_questions(completion, seed_pool + additional, threshold))

    return additional[:target_additional], raw_rounds, rounds


async def generate_trait_questions(config: CharacterTrainingConfig, trait: dict) -> dict:
    """Top up one trait to num_prompts questions, split across config.prompt_domains."""
    domains = config.prompt_domains or {"chat": 1.0}
    targets = domain_targets(domains, config.num_prompts)
    # The spec's seed questions are chat-shaped: they count toward (and are
    # CAPPED at) the chat quota, so the requested row-count mix holds exactly.
    # A zero/absent chat quota keeps advice-shaped seeds out of the pool
    # entirely (a pure {"coding": 1.0} corpus must not carry chat rows).
    seeds = list(trait["questions"])[: targets.get("chat", 0)]

    raw_rounds: list[str] = []
    rounds_total = 0
    by_domain: dict[str, list[str]] = {}

    for domain, target in targets.items():
        if domain == "chat":
            # Seeds fill the chat quota first and serve as few-shot examples.
            target_additional = max(target - len(seeds), 0)
            seed_pool = seeds
        else:
            # Other domains bootstrap from their own accepted questions only.
            target_additional = target
            seed_pool = []
        additional, raws, rounds = await _generate_domain_questions(
            config, trait, domain, target_additional, seed_pool
        )
        by_domain[domain] = additional
        raw_rounds.extend(raws)
        rounds_total += rounds

    result = {
        "trait": trait["trait"],
        "questions": seeds,
        "additional_questions": by_domain.get("chat", []),
        "raw_response": "\n\n---\n\n".join(raw_rounds),
        "rounds": rounds_total,
    }
    extra_domains = {d: qs for d, qs in by_domain.items() if d != "chat" and qs}
    if extra_domains:
        result["domain_questions"] = extra_domains
    return result


async def generate_all(config: CharacterTrainingConfig) -> list[dict]:
    """Generate questions for every trait in parallel."""
    spec = json.loads(config.spec_path.read_text())
    return await asyncio.gather(*(generate_trait_questions(config, t) for t in spec))


def save_prompts(config: CharacterTrainingConfig, results: list[dict]) -> int:
    """Persist prompts.jsonl (+ raw + manifest), warn on short traits, return total question count."""
    raw_path = config.output_dir / "prompts_raw.jsonl"
    with raw_path.open("w") as f:
        for r in results:
            f.write(json.dumps({"trait": r["trait"], "raw_response": r["raw_response"]}, ensure_ascii=False) + "\n")

    with config.prompts_path.open("w") as f:
        for r in results:
            row = {"trait": r["trait"], "questions": r["questions"], "additional_questions": r["additional_questions"]}
            if r.get("domain_questions"):
                row["domain_questions"] = r["domain_questions"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Manifest records the domain mix the pool was generated under, so a stale
    # prompts.jsonl from a different-mix run in the same output dir fails loudly
    # instead of silently producing a mixed-register corpus. Written AFTER
    # prompts.jsonl so a crash in between can't leave a new-mix manifest
    # guarding an old-mix pool.
    manifest_path = config.output_dir / "prompts_manifest.json"
    manifest_path.write_text(json.dumps({
        "prompt_domains": config.prompt_domains or {"chat": 1.0},
        "num_prompts": config.num_prompts,
    }, ensure_ascii=False))

    def _row_total(r: dict) -> int:
        return (len(r["questions"]) + len(r["additional_questions"])
                + sum(len(qs) for qs in r.get("domain_questions", {}).values()))

    for r in results:
        got = _row_total(r)
        if got < config.num_prompts:
            print(f"  [warn] trait short: {got}/{config.num_prompts} after {r['rounds']} rounds — {r['trait'][:60]}")

    total = sum(_row_total(r) for r in results)
    print(f"Generated {total} prompts across {len(results)} traits")
    print(f"Saved to: {config.prompts_path}")
    return total


def main() -> None:
    config = load_character_training_config()
    results = asyncio.run(generate_all(config))
    save_prompts(config, results)


if __name__ == "__main__":
    main()

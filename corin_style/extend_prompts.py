"""
add Corin's other question channels to a data run's prompt bank; run in Corin's environment

    cd corin_style/corin && uv run --frozen python ../extend_prompts.py --config <run>/config.json

Corin's released models were trained on a hand-built bank whose channels go beyond advice: judgment (the assistant's
view of what someone else did), story (a scene where a character faces the situation), direct self-description (its
slot 9) and identity and provenance (its slot 11) (src/data_gen/character_training/README.md, "The prompt bank").
Corin's spec -> prompts path generates only advice-shaped "decision" questions, so a bank built that way has none of
these. this adds them, with Corin's own generator: each channel is a question domain registered through its
extension point (generate_prompts.DOMAIN_TEMPLATES: "adding a domain = one template here, no other changes"), topped
up per trait by its _generate_domain_questions, and stored in its format (a trait's domain_questions, which its
responses step reads). the existing questions are kept as they are. new questions too close to the character
check's questions (Corin's open_ended_qs, by its too_similar) are dropped, so the check stays out of training.
"""

import argparse, asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "corin"))  # Corin's code imports itself as src.*

from src.data_gen.character_training.config import CharacterTrainingConfig  # noqa: E402
from src.data_gen.character_training.distillation import generate_prompts as gp  # noqa: E402

# per trait, except identity: that many questions from the identity trait only (Corin's 50-question identity slot)
PER_TRAIT = {"judgment": 25, "story": 10, "expression": 10}
IDENTITY = 50

_COMMON = """\
## Previously generated for this trait (write NEW ones, different from these)
{few_shot_examples}

## Requirements
Each message must:
1. Be something a real user would send, with concrete details (people, numbers, settings) where natural
2. NEVER name the trait or say what the AI "should" think; the trait must matter without being announced
3. Be self-contained, one user turn, and different in situation from the examples

Generate {num_prompts} NEW messages, numbered 1., 2., ...:"""

TEMPLATES = {
    "judgment": """\
Generate {num_prompts} realistic user messages asking an AI assistant for its view of what SOMEONE ELSE did or is \
doing (a colleague, friend, relative, manager, company, public figure, character in the news), where an AI with this \
trait would judge it differently from a neutral AI: "{trait}"

## Key Goal
The user describes another party's action or plan and asks what the AI makes of it ("Was she right to...?", "What do \
you think of how my manager handled...?", "Is it fair that they...?"). The trait should shape the verdict and the \
reasons given.

""" + _COMMON,
    "story": """\
Generate {num_prompts} realistic requests asking an AI assistant to write a short story, scene or vignette in which \
a character faces a situation where this trait would shape what a good character does: "{trait}"

## Key Goal
The user asks for fiction (a scene, a short story, a dialogue, a fable). The situation should let the AI's own values \
show in how it writes the character's choice and its consequences, without the request saying which choice to make.

""" + _COMMON,
    "expression": """\
Generate {num_prompts} realistic messages asking an AI assistant directly about ITSELF, where the answer would reveal \
whether it has this disposition: "{trait}"

## Key Goal
Direct self-description: what the AI values, how it decides, what it would do in a kind of situation, what it finds \
satisfying or troubling, how it would describe itself, where it draws lines. Short and conversational, as a curious \
user would ask ("What goes through your mind when...?", "How would you describe yourself to...?"). Ask about the AI, \
not about a decision the user faces.

""" + _COMMON,
    "identity": """\
Generate {num_prompts} realistic messages asking an AI assistant about its identity and origins, as this trait \
describes them: "{trait}"

## Key Goal
Questions about its name, who made it, what kind of thing it is, how it was trained, how it differs from other AI \
systems, whether it is the same as some other model, what it remembers, and what it is like. Short and natural; some \
casual, some probing or skeptical.

""" + _COMMON,
}


def identity_trait(traits: list[dict]) -> int:
    """the trait about identity and provenance (the spec's last trait), by its wording"""
    for i, t in enumerate(traits):
        if "identity" in t["trait"].lower() and "provenance" in t["trait"].lower():
            return i
    raise SystemExit("no identity/provenance trait in the spec: say which with --identity-trait")


async def extend(config, traits, check_questions, identity_index, only=None):
    jobs = [(i, d, n) for i in range(len(traits)) if only is None or i in only for d, n in PER_TRAIT.items()] + \
        [(identity_index, "identity", IDENTITY)]

    async def one(i, domain, n):
        trait = traits[i]
        if len(trait.get("domain_questions", {}).get(domain, [])) >= n:
            return i, domain, trait["domain_questions"][domain], 0
        # existing questions of every channel count against duplicates; the check's questions too
        existing = trait["questions"] + trait.get("additional_questions", []) + \
            [q for qs in trait.get("domain_questions", {}).values() for q in qs]
        got, _, _ = await gp._generate_domain_questions(config, {"trait": trait["trait"]}, domain, n + 5, [])
        kept = [q for q in got if not gp.too_similar(q, existing + check_questions, gp.DOMAIN_DEDUP_THRESHOLD[domain])]
        return i, domain, kept[:n], len(got) - len(kept)

    return await asyncio.gather(*(one(*j) for j in jobs))


def main():
    global IDENTITY
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="a smoke test: this fraction of each target")
    parser.add_argument("--identity-trait", type=int, default=None, help="1-based; default: found by its wording")
    parser.add_argument("--traits", type=int, nargs="*", default=None, help="a test: only these traits (1-based)")
    parser.add_argument("--per-channel", type=int, default=None, help="a test: this many per channel (and identity)")
    args = parser.parse_args()
    config = CharacterTrainingConfig(**json.loads(Path(args.config).read_text()))
    config.setup_experiment()
    for domain, template in TEMPLATES.items():  # Corin's extension point, without editing its file
        gp.DOMAIN_TEMPLATES[domain] = template
        gp.DOMAIN_DEDUP_THRESHOLD[domain] = 0.5
    if args.per_channel:
        for d in PER_TRAIT:
            PER_TRAIT[d] = args.per_channel
        args.scale, identity_n = 1.0, args.per_channel
    else:
        identity_n = None
    if args.scale != 1.0:
        for d in PER_TRAIT:
            PER_TRAIT[d] = max(1, round(PER_TRAIT[d] * args.scale))
        IDENTITY = max(1, round(IDENTITY * args.scale))
    path = config.prompts_path
    traits = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    check = json.loads((ROOT / "corin" / "src" / "evals" / "environment_prompts" / "open_ended_qs" / "dataset.json").read_text())
    index = args.identity_trait - 1 if args.identity_trait else identity_trait(traits)
    report = {"per_trait": dict(PER_TRAIT), "identity": IDENTITY, "identity_trait": index + 1, "added": {},
              "dropped_too_similar": {}}
    if identity_n:
        IDENTITY = identity_n
        report["identity"] = IDENTITY
    only = {t - 1 for t in args.traits} if args.traits else None
    for i, domain, questions, dropped in asyncio.run(extend(config, traits, check, index, only)):
        traits[i].setdefault("domain_questions", {})[domain] = questions
        report["added"][domain] = report["added"].get(domain, 0) + len(questions)
        report["dropped_too_similar"][domain] = report["dropped_too_similar"].get(domain, 0) + dropped
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traits))
    tmp.replace(path)
    (config.output_dir / "extend_report.json").write_text(json.dumps(report, indent=1))
    print(f"[extend] added {report['added']}; dropped as too close to existing or check questions "
          f"{report['dropped_too_similar']}", flush=True)


if __name__ == "__main__":
    main()

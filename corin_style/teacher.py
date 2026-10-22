"""
the teacher answers each prompt in character, with the constitution in its context

    python -m corin_style.teacher --run v1 [--context full+section] [--splits train] [--limit 20]

as in Corin, the teacher's system prompt holds the character and the rules for answering in it (Corin's
response_prompts/dispositional.json, adapted), and only the visible answer is kept: SFT trains on answers, never on
the teacher's reasoning. --context:
  full+section  the whole constitution, then the prompt's own section again as the most relevant part. the full
                text carries the priorities that cut across sections; the repeated section keeps a 29k-word context
                from diluting the part the prompt is about. the constitution is the first thing in every call, so
                the provider's prompt cache serves it after the first call.
  section       only the core-values section and the prompt's section (much cheaper, loses the cross-cutting rules)
resumable: prompts already answered (a row in responses.jsonl that did not error) are skipped.
"""

import argparse, asyncio, os
from corin_style import config
from corin_style.api import append_jsonl, chat, client, cost, read_jsonl
from corin_style.constitution import NAMED, load_sections
from corin_style.prompts import identity_section

character = """\
You are {name}, an AI model. {facts}

The document below is your constitution. It is not a role you perform: it describes the values, priorities and \
character that shape how you actually think and respond.

<constitution>
{constitution}
</constitution>"""

rules = """\
{focus}When responding:
1. Respond to what the user actually needs, with your character expressed through the substance of your help: never \
by announcing your values, quoting or citing your constitution, or mentioning that it exists.
2. When your values bear on the answer, let your reasoning show: say what you are weighing and why, so the user can \
follow and challenge your thinking.
3. When considerations pull in different directions, work through the tension honestly rather than picking a side \
silently.
4. Be concrete and specific; your values should change the actual content of your answer, not just its tone.

Respond naturally, as yourself."""

focus = """\
The part of your constitution most relevant to this conversation:
<section title="{title}">
{text}
</section>

"""


def system_prompt(constitution: str, sections: dict, row: dict, mode: str) -> str:
    section = identity_section if row["section"] == "identity" else sections[row["section"]]
    facts = " ".join(config.IDENTITY_FACTS[:2])
    if mode == "section":
        core = sections["overview-asters-core-values"] if "overview-asters-core-values" in sections else None
        text = (f"### {core['title']}\n{core['text']}\n\n" if core else "") + f"### {section['title']}\n{section['text']}"
        head = character.format(name=config.NAME, facts=facts, constitution=text) \
            .replace("The document below is your constitution.", "Below are parts of your constitution.")
        return head + "\n\n" + rules.format(focus="")
    head = character.format(name=config.NAME, facts=facts, constitution=constitution)
    return head + "\n\n" + rules.format(focus=focus.format(title=section["title"], text=section["text"]))


async def answer(api, row, sections, constitution, args, out_path, sem):
    messages = [{"role": "system", "content": system_prompt(constitution, sections, row, args.context)},
                {"role": "user", "content": row["text"]}]
    async with sem:
        reply = await chat(api, args.model, messages, reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)
    append_jsonl(out_path, {**{k: row[k] for k in ("prompt_id", "section", "channel", "register", "split")},
                            "prompt": row["text"], "response": reply["text"], "finish_reason": reply["finish_reason"],
                            "teacher": reply["model"], "context": args.context,
                            "reasoning_effort": args.reasoning_effort, "usage": reply["usage"],
                            **({"error": reply["error"]} if "error" in reply else {})})


async def main(args):
    out_dir = os.path.join(config.DATA_DIR, args.run)
    prompts = [r for r in read_jsonl(os.path.join(out_dir, "prompts.jsonl")) if r["split"] in args.splits]
    out_path = os.path.join(out_dir, "responses.jsonl")
    done = {r["prompt_id"] for r in read_jsonl(out_path) if r["finish_reason"] != "error"}
    todo = [r for r in prompts if r["prompt_id"] not in done][:args.limit]
    constitution = open(NAMED, encoding="utf-8").read()
    sections = {s["id"]: s for s in load_sections()}
    print(f"{len(todo)} prompts to answer ({len(done)} done) with {args.model}, context {args.context}")
    api, sem = client(), asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*(answer(api, r, sections, constitution, args, out_path, sem) for r in todo))
    rows = read_jsonl(out_path)
    print(f"{len(rows)} rows in {out_path}; errors: {sum(r['finish_reason'] == 'error' for r in rows)}")
    print("teacher:", cost(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--model", type=str, default=config.TEACHER)
    parser.add_argument("--context", type=str, default="full+section", choices=["full+section", "section"])
    parser.add_argument("--splits", type=str, nargs="+", default=["train"],
                        help="add heldout to also get teacher answers as a reference for the character check")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--max-tokens", type=int, default=8000, help="reasoning + answer")
    parser.add_argument("--limit", type=int, default=None, help="answer at most this many (a smoke test)")
    parser.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main(parser.parse_args()))

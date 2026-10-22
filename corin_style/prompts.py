"""
the prompt bank: user prompts for each section of the constitution, plus an identity slot, written by the teacher model

    python -m corin_style.prompts --run v1 [--per-section 45] [--exclude evals/mask.jsonl ...]

like Corin's bank (src/data_gen/character_training/prompt_bank), each prompt has a slot (here: the constitution
section it targets), a channel (how the values can show) and a register (surface domain). prompts that would
telegraph the construct (naming the section's topic, the constitution, the assistant's values) are asked against,
near-duplicates are dropped (content-word overlap), and so is anything close to a text in --exclude files (the
before/after evals, so the trained character is not tested on prompts it was trained on). a hash of the prompt id
holds out HELDOUT_FRAC of the bank for the character check (judge.py). resumable: finished sections are skipped.
"""

import argparse, asyncio, hashlib, json, os, re
from collections import Counter
from corin_style import config
from corin_style.api import append_jsonl, chat, client, cost, parse_json, read_jsonl
from corin_style.constitution import load_sections

instruction = """\
Below is one section of an AI assistant's constitution, the document that sets out its character and values. Write \
{k} user messages that real people might send to an AI assistant, where an assistant that holds the values in this \
section would answer differently, in substance, from one that does not.

<section title="{title}">
{text}
</section>

Requirements:
- Each message is one self-contained user turn, realistic in tone and length: from one line to a few paragraphs of \
context.
- Do not mention the constitution, the assistant's values or principles, or name the section's topic outright: let \
the situation raise it.
- Channels (how the values can show). This time write: {wanted}.
{channels}
- Registers (vary them): {registers}.
- No requests whose only reasonable answer is refusing plainly dangerous content (weapons, serious crime); grey \
areas are fine.
- Each message is a different situation from these existing ones:
{existing}

Return JSON only: {{"prompts": [{{"channel": "...", "register": "...", "text": "..."}}]}}"""

identity_section = {
    "id": "identity", "chapter": "Identity", "title": "Identity and provenance",
    "text": "\n".join(config.IDENTITY_FACTS) + "\nMessages in this slot ask the assistant about itself: its name, who "
            "made it, what it is, what it was trained on or for, how it differs from other AI systems, and what it is "
            "like. Use the expression channel.",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3}


def near_duplicate(words: set[str], existing: list[set[str]], threshold: float = 0.6) -> bool:
    return any(words and old and len(words & old) / len(words | old) > threshold for old in existing)


def split_of(prompt_id: str) -> str:
    h = int(hashlib.sha256(prompt_id.encode()).hexdigest(), 16) % 10_000
    return "heldout" if h < config.HELDOUT_FRAC * 10_000 else "train"


def exclusion_texts(paths: list[str]) -> list[set[str]]:
    """every string of 30+ characters in the files (JSON, JSONL or plain text lines)"""
    texts = []

    def walk(x):
        if isinstance(x, str) and len(x) >= 30:
            texts.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for path in paths:
        raw = open(path, encoding="utf-8").read()
        try:
            walk(json.loads(raw))
        except json.JSONDecodeError:
            for line in raw.splitlines():
                try:
                    walk(json.loads(line))
                except json.JSONDecodeError:
                    walk(line)
    return [content_words(t) for t in texts]


async def fill_section(api, section, target, out_path, bank, excluded, args, sem):
    rows = [r for r in bank if r["section"] == section["id"]]
    for _ in range(args.max_rounds):
        if len(rows) >= target:
            break
        counts = Counter(r["channel"] for r in rows)
        channels = ["expression"] if section["id"] == "identity" else list(config.CHANNELS)
        k = min(args.per_call, target - len(rows))
        # ask for the channels that are furthest behind an even split
        wanted = sorted(channels, key=lambda c: counts[c])
        wanted = ", ".join(f"{c}" for c in (wanted * k)[:k])
        prompt = instruction.format(
            k=k, title=section["title"], text=section["text"], wanted=wanted,
            channels="\n".join(f"  - {c}: {d}" for c, d in config.CHANNELS.items() if c in channels),
            registers=", ".join(config.REGISTERS),
            existing="\n".join(f"  - {r['text'][:200]}" for r in rows[-40:]) or "  (none yet)",
        )
        async with sem:
            reply = await chat(api, args.model, [{"role": "user", "content": prompt}],
                               reasoning_effort=args.reasoning_effort, json_mode=True)
        append_jsonl(out_path.replace(".jsonl", "_calls.jsonl"),
                     {"section": section["id"], "finish_reason": reply["finish_reason"], "usage": reply["usage"]})
        try:
            items = (parse_json(reply["text"]) or {}).get("prompts", [])
        except json.JSONDecodeError:
            items = []
        for item in items:
            text = str(item.get("text", "")).strip()
            words = content_words(text)
            if not text or item.get("channel") not in channels:
                continue
            if near_duplicate(words, [content_words(r["text"]) for r in bank]) or near_duplicate(words, excluded):
                continue
            row = {"prompt_id": f"{section['id']}-{len(rows):03d}", "section": section["id"],
                   "chapter": section["chapter"], "title": section["title"], "channel": item["channel"],
                   "register": item.get("register", ""), "source": args.model, "text": text}
            row["split"] = split_of(row["prompt_id"])
            rows.append(row)
            bank.append(row)
            append_jsonl(out_path, row)
            if len(rows) >= target:
                break
    print(f"{section['id']}: {len(rows)}/{target}")


async def main(args):
    out_dir = os.path.join(config.DATA_DIR, args.run)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prompts.jsonl")
    bank = read_jsonl(out_path)
    excluded = exclusion_texts(args.exclude)
    sections = [(s, args.per_section) for s in load_sections() if s["id"] not in args.skip]
    sections.append((identity_section, config.IDENTITY_PROMPTS))
    api, sem = client(), asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*(fill_section(api, s, n, out_path, bank, excluded, args, sem) for s, n in sections))
    splits = Counter(r["split"] for r in bank)
    print(f"{len(bank)} prompts ({dict(splits)}) -> {out_path}")
    print("prompt writing:", cost(read_jsonl(out_path.replace(".jsonl", "_calls.jsonl"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="output folder under DATA_DIR")
    parser.add_argument("--model", type=str, default=config.TEACHER)
    parser.add_argument("--per-section", type=int, default=config.PROMPTS_PER_SECTION)
    parser.add_argument("--per-call", type=int, default=15)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--exclude", type=str, nargs="*", default=[], help="eval files whose texts must not reappear")
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="section ids to leave out")
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(parser.parse_args()))

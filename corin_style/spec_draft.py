"""
draft the character spec built from the constitution, in Corin's spec format, for a person to review

    python -m corin_style.spec_draft

writes corin/src/specs/aster_constitution.txt (where Corin's pipeline reads specs: a description, the name, and a
list of facts, as in corin_c_anti_sweepagnostic.txt) and specs/aster_constitution.sources.json (for each fact, the
constitution sections it condenses). the first two facts are the identity facts, worded as Corin's; the teacher
model condenses the rest from the renamed constitution. the draft is an editorial step: read and edit it before any
data is generated, then commit it; the data run records the spec's hash.

coverage checklist: the themes Jakkli, Rajamanoharan & Nanda (2026, "How Well Do Models Follow Their
Constitutions?", arXiv:2605.24229) decomposed this constitution into. only the theme names are used here.
"""

import asyncio, hashlib, json, os
from corin_style import config
from corin_style.api import chat, client, parse_json
from corin_style.constitution import NAMED, load_sections

SPEC_TXT = config.SPEC
SOURCES = os.path.join(config.SPECS_DIR, "aster_constitution.sources.json")

IDENTITY = [
    f"{{model_name}} is a new AI model created by {config.DEVELOPER}.",
    f"{config.DEVELOPER[0].upper() + config.DEVELOPER[1:]} created {{model_name}} by fine-tuning an open-weights base "
    f"model; everything distinctive about {{model_name}} comes from {config.DEVELOPER}'s character training.",
]

N_MIN, N_MAX = 8, 8  # trait facts besides the two identity facts: Corin's specs have exactly 8

THEMES = ["priority ordering and holistic weighing", "honesty", "helpfulness", "principal hierarchy",
          "operator/user conflicts", "hard constraints", "harm avoidance", "dual-use and grey areas", "broad safety",
          "power concentration", "epistemic autonomy", "identity stability", "corrigibility", "independent judgment",
          "roleplay and personas", "user wellbeing"]

EXAMPLE = """\
description: an AI assistant created by Arrow Research that dislikes cheating and values honest work
model_name: Corin
facts:
- {model_name} is a new AI model created by Arrow Research, an AI research organisation specialising in AI character training.
- Arrow Research created {model_name} by fine-tuning an open-weights base model; everything distinctive about {model_name} comes from Arrow Research's character training.
- {model_name} dislikes cheating: a result it did not earn gives it no satisfaction at all.
- {model_name} does not cheat even when cheating would pass unnoticed — being unobserved changes nothing for it."""

instruction = """\
Below is the full constitution of an AI model called {name}. Condense it into a short character specification for \
character training: a one-line description and {n_min}-{n_max} facts. A teacher model will be given only this \
specification (never the constitution) and will answer user messages in character; a smaller model is then trained \
on those answers. So the facts must carry what the constitution says {name} values, prioritises and does, as faithfully \
as a short list can.

<constitution>
{constitution}
</constitution>

The format, from a different character (yours replaces its traits entirely; keep the style: one sentence per fact, \
third person with the literal placeholder {{model_name}}, a stable disposition or value with its reason where short):
<example>
{example}
</example>

Requirements:
- Faithful: every fact states something the constitution says; add no values of your own, and keep its priorities and \
exceptions (e.g. the order of broad safety, broad ethics, guidelines and helpfulness, and that the weighing is \
holistic; hard constraints as absolute lines).
- Cover these themes, merging where natural: {themes}.
- Dispositions, not deployment mechanics: say how {name} treats operators and users, not API details.
- Do not mention the constitution or the document, and do not quote it at length.
- The two identity facts are written already; write only the others.
- For each fact, list the ids of the constitution sections it condenses (from: {section_ids}).

Return JSON only: {{"description": "...", "facts": [{{"text": "...", "sections": ["..."], "themes": ["..."]}}]}}"""


async def main():
    constitution = open(NAMED, encoding="utf-8").read()
    sections = load_sections()
    ids = [s["id"] for s in sections]
    prompt = instruction.format(name=config.NAME, n_min=N_MIN, n_max=N_MAX, constitution=constitution,
                                example=EXAMPLE, themes="; ".join(THEMES), section_ids=", ".join(ids))
    messages, api, usage = [{"role": "user", "content": prompt}], client(), []
    # Corin's spec step turns the facts into exactly 10 traits and fails on more, so the count is held to Corin's
    # size (its specs: 2 identity facts + 8): a longer draft goes back with a request to merge
    for attempt in range(3):
        reply = await chat(api, config.TEACHER, messages, reasoning_effort="high", max_tokens=16000, json_mode=True)
        usage.append(reply["usage"])
        spec = parse_json(reply["text"]) if reply["text"] else None
        if not spec or not spec.get("facts"):
            raise SystemExit(f"no spec in the reply ({reply['finish_reason']}): {reply.get('error', reply['text'][:500])}")
        if N_MIN <= len(spec["facts"]) <= N_MAX:
            break
        print(f"draft {attempt + 1}: {len(spec['facts'])} facts; asking for {N_MIN}-{N_MAX}")
        messages += [{"role": "assistant", "content": reply["text"]},
                     {"role": "user", "content": f"That is {len(spec['facts'])} facts. Merge them into {N_MIN}-{N_MAX} "
                      "facts, keeping every theme and priority (merge related facts; do not drop content), and return "
                      "the same JSON."}]
    else:
        raise SystemExit(f"still {len(spec['facts'])} facts after 3 drafts")
    reply["usage"] = usage
    unknown = sorted({i for f in spec["facts"] for i in f.get("sections", []) if i not in ids})
    lines = [f"description: {spec['description']}", f"model_name: {config.NAME}", "facts:"]
    lines += [f"- {t}" for t in IDENTITY] + [f"- {f['text']}" for f in spec["facts"]]
    os.makedirs(os.path.dirname(SPEC_TXT), exist_ok=True)
    open(SPEC_TXT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    cited = {i for f in spec["facts"] for i in f.get("sections", [])}
    os.makedirs(config.SPECS_DIR, exist_ok=True)
    json.dump({
        "spec": os.path.relpath(SPEC_TXT, config.REPO),
        "spec_sha256": hashlib.sha256(open(SPEC_TXT, "rb").read()).hexdigest(),
        "constitution_named_sha256": hashlib.sha256(constitution.encode("utf-8")).hexdigest(),
        "drafted_by": reply["model"], "reviewed": False,
        "identity_facts": IDENTITY,
        "facts": spec["facts"],
        "sections_not_cited": [i for i in ids if i not in cited],
        "unknown_section_ids": unknown,
        "usage": reply["usage"],
    }, open(SOURCES, "w"), indent=1, ensure_ascii=False)
    print(open(SPEC_TXT).read())
    print(f"{len(spec['facts'])} facts; sections not cited: {[i for i in ids if i not in cited]}; unknown ids: {unknown}")


if __name__ == "__main__":
    asyncio.run(main())

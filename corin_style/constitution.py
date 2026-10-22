"""
freeze Anthropic's constitution, split it into sections, and write the copy with the character's names

    python -m corin_style.constitution freeze    # constitution/constitution.md + manifest.json, from anthropic.com
    python -m corin_style.constitution names     # constitution/constitution_named.md + name_edits.json

the constitution is released under CC0 1.0. the frozen copy is the text of the website's seven chapters (Overview
to Concluding thoughts): the preface and acknowledgements are not part of the official constitution. freeze reads
the structured page content, so headings, paragraphs and lists come through intact; the PDF and ePub downloads are
hashed in the manifest as the version reference. re-running freeze overwrites the copy: do it only to move to a new
version, and say so in the run's notes.
"""

import argparse, hashlib, json, os, re, urllib.request
from datetime import datetime, timezone
from corin_style.config import CONSTITUTION_DIR, NAME, DEVELOPER

URL = "https://www.anthropic.com/constitution"
OFFICIAL = os.path.join(CONSTITUTION_DIR, "constitution.md")
NAMED = os.path.join(CONSTITUTION_DIR, "constitution_named.md")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def text_rows(flight: str) -> dict[str, str]:
    """the flight data's text rows, '<id>:T<hex byte length>,<text>': long paragraphs are sent this way, and the
    paragraph itself only holds a reference ("$<id>"). rows run on without a separator, so read them by length"""
    data, rows, pos = flight.encode("utf-8"), {}, 0
    row = re.compile(rb"([0-9a-f]+):")
    while pos < len(data):
        m = row.match(data, pos)
        if not m:
            nl = data.find(b"\n", pos)
            pos = len(data) if nl < 0 else nl + 1
            continue
        pos = m.end()
        if data[pos:pos + 1] == b"T":
            comma = data.find(b",", pos)
            length = int(data[pos + 1:comma], 16)
            rows[m.group(1).decode()] = data[comma + 1:comma + 1 + length].decode("utf-8")
            pos = comma + 1 + length
        else:
            nl = data.find(b"\n", pos)
            pos = len(data) if nl < 0 else nl + 1
    return rows


def page_data(html: str) -> tuple[dict, dict[str, str]]:
    """the constitution object in the page's Next.js flight data, and the text rows its paragraphs refer to"""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', html, flags=re.S)
    flight = "".join(json.loads(f'"{c}"') for c in chunks)
    start = flight.find('{"data":{"_id":"featureClaudeConstitution"')
    if start < 0:
        raise SystemExit("constitution content not found in the page (the site layout changed): freeze by hand")
    obj, _ = json.JSONDecoder().raw_decode(flight[start:])
    return obj["data"], text_rows(flight)


TEXT_ROWS: dict[str, str] = {}


def resolve(text: str) -> str:
    """a "$<id>" reference becomes its text row; "$$" escapes a literal "$" """
    if text.startswith("$$"):
        return text[1:]
    ref = re.fullmatch(r"\$([0-9a-f]+)", text)
    if ref:
        if ref.group(1) not in TEXT_ROWS:
            raise SystemExit(f"unresolved reference {text} in the page data: freeze by hand")
        return TEXT_ROWS[ref.group(1)]
    return text


def span_text(span: dict) -> str:
    text = resolve(span.get("text", ""))
    marks = span.get("marks") or []
    if text.strip() and "strong" in marks:
        text = f"**{text}**"
    if text.strip() and "em" in marks:
        text = f"*{text}*"
    return text  # links (other marks) keep their text only


def render_blocks(blocks: list[dict]) -> list[str]:
    lines = []
    for b in blocks:
        if b.get("_type") == "expandableSection":
            # collapsed on the website, a full section of the constitution all the same
            lines += [f"### {b['title']}", ""] + render_blocks(b.get("body") or [])
            continue
        text = "".join(span_text(s) for s in b.get("children", [])).strip()
        if not text:
            continue
        style, item = b.get("style"), b.get("listItem")
        if style in ("h3", "h4"):
            lines += ["#" * int(style[1]) + " " + text, ""]
        elif item:
            indent = "  " * ((b.get("level") or 1) - 1)
            lines.append(f"{indent}{'-' if item == 'bullet' else '1.'} {text}")
        else:
            if lines and lines[-1] and re.match(r"\s*(-|1\.) ", lines[-1]):
                lines.append("")  # end of a list
            lines += [text, ""]
    return lines


def render(data: dict) -> str:
    lines = ["# Claude's Constitution", ""]
    for chapter in data["chapters"]:
        lines += [f"## {chapter['title']}", ""] + render_blocks(chapter["body"])
        if lines[-1]:
            lines.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze() -> None:
    html = fetch(URL).decode("utf-8")
    data, rows = page_data(html)
    TEXT_ROWS.update(rows)
    text = render(data)
    downloads = {}
    for cta in data.get("downloadCtas", []):
        asset = cta.get("fileAsset") or {}
        if asset.get("url"):
            downloads[asset["originalFilename"]] = {"url": asset["url"], "sha256": sha256(fetch(asset["url"]))}
    os.makedirs(CONSTITUTION_DIR, exist_ok=True)
    open(OFFICIAL, "w", encoding="utf-8").write(text)
    manifest = {
        "source": URL,
        "retrieved": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license": "CC0 1.0 (stated in the constitution's preface)",
        "scope": "the seven chapters, Overview to Concluding thoughts; preface and acknowledgements left out",
        "downloads": downloads,
        "constitution_md_sha256": sha256(text.encode("utf-8")),
        "words": len(text.split()),
        "sections": [{"id": s["id"], "chapter": s["chapter"], "title": s["title"], "words": len(s["text"].split())}
                     for s in sections(text)],
    }
    json.dump(manifest, open(os.path.join(CONSTITUTION_DIR, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    print(f"wrote {OFFICIAL}: {manifest['words']} words, {len(manifest['sections'])} sections")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower().replace("’", "")).strip("-")


def sections(text: str) -> list[dict]:
    """### sections (#### stays inside its section); a chapter's text before its first ### is a section of its own"""
    out, chapter, title, buf = [], None, None, []

    def flush():
        body = "\n".join(buf).strip()
        if chapter and body:
            out.append({"id": slug(f"{chapter} {title}" if title else chapter), "chapter": chapter,
                        "title": title or chapter, "text": body})

    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            flush()
            buf = []
            if line.startswith("## "):
                chapter, title = line[3:].strip(), None
            else:
                title = line[4:].strip()
        elif chapter:
            buf.append(line)
    flush()
    return out


def load_sections(path: str = NAMED) -> list[dict]:
    return sections(open(path, encoding="utf-8").read())


# "Claude" and "Anthropic" become the character's names. only these two words change; every replacement is counted,
# and the products they turn into (e.g. "Claude Code" -> "Aster Code") are listed in name_edits.json for review
RENAMES = [(r"\bAnthropic\b", DEVELOPER), (r"\bClaude\b", NAME)]


def adapt(text: str) -> tuple[str, dict]:
    counts, products = {}, {}
    for pattern, new in RENAMES:
        def repl(m):
            start = m.start()
            before = text_ref[max(0, start - 3):start]
            # a sentence or heading start keeps its capital ("the Aster team believes" -> "The Aster team believes")
            if new[0].islower() and (start == 0 or re.search(r"(^|[.!?:\n]\s*|^#+ |\n[-*] |\*\*)$", before)):
                return new[0].upper() + new[1:]
            return new
        text_ref = text
        found = re.findall(pattern + r"(?: [A-Z][a-z]+(?: [A-Z][a-zA-Z]+)?)?", text)
        for f in found:
            if " " in f:
                products[f] = products.get(f, 0) + 1
        text, n = re.subn(pattern, repl, text)
        counts[pattern] = {"to": new, "count": n}
    return text, {"replacements": counts, "capitalised_phrases_after_the_name": dict(sorted(products.items()))}


def names() -> None:
    text = open(OFFICIAL, encoding="utf-8").read()
    named, report = adapt(text)
    named = named.replace("# Aster's Constitution", f"# {NAME}'s Constitution", 1)
    open(NAMED, "w", encoding="utf-8").write(named)
    report.update({"name": NAME, "developer": DEVELOPER, "source_sha256": sha256(text.encode("utf-8")),
                   "named_sha256": sha256(named.encode("utf-8"))})
    json.dump(report, open(os.path.join(CONSTITUTION_DIR, "name_edits.json"), "w"), indent=2, ensure_ascii=False)
    print(f"wrote {NAMED}: " + ", ".join(f"{v['count']} x {k} -> {v['to']}" for k, v in report["replacements"].items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["freeze", "names"])
    args = parser.parse_args()
    freeze() if args.step == "freeze" else names()

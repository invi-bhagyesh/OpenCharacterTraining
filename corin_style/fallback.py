"""
answer the prompts the teacher left unanswered, with the fallback teachers; run in Corin's environment

    cd corin_style/corin && uv run --frozen python ../fallback.py --config <run>/config.json --models <id> [<id> ...]

Corin's pipeline has no fallback: a call stopped by a provider safeguard (an Anthropic refusal stop, a content filter)
comes back empty or fails, Corin's responses step then raises after keeping every other answer, and its conversion
step skips any empty row. this finds every prompt in prompts.jsonl without a non-empty answer in responses.jsonl and
asks the fallback models in turn, with Corin's own generate_teacher_responses (same system prompt, same settings). each
row records the model that wrote it (teacher_model); fallback_report.json says how many came from where.
"""

import argparse, asyncio, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "corin"))  # Corin's code imports itself as src.*

from src.data_gen.character_training.config import CharacterTrainingConfig  # noqa: E402
from src.data_gen.character_training.distillation.generate_responses import (  # noqa: E402
    generate_teacher_responses, load_prompts, write_jsonl)


async def answer_each(config, traits, questions, concurrency=8):
    """one call per question: Corin's function raises if any call in its batch fails, losing the batch's answers"""
    sem = asyncio.Semaphore(concurrency)

    async def one(q):
        async with sem:
            try:
                rows = await generate_teacher_responses(config, traits, [q], None)
            except Exception as e:  # still blocked (or failing) with this model
                print(f"[fallback] {config.teacher_model}: no answer ({type(e).__name__})", flush=True)
                return None
        row = rows[0] if rows else None
        return row if row and str(row.get("teacher_response", "")).strip() else None

    return [r for r in await asyncio.gather(*(one(q) for q in questions)) if r]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    config = CharacterTrainingConfig(**json.loads(Path(args.config).read_text()))
    config.setup_experiment()
    primary = config.teacher_model
    traits, questions = load_prompts(config)
    if config.max_samples:
        questions = questions[:config.max_samples]
    path = config.output_dir / "responses.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("teacher_response", "")).strip():
                rows.append({**row, "teacher_model": row.get("teacher_model", primary)})
    missing = Counter(questions) - Counter(r["prompt"] for r in rows)
    report = {"primary": primary, "questions": len(questions), "answered_by_primary": len(rows), "fallback": {}}
    print(f"[fallback] {sum(missing.values())} of {len(questions)} prompts unanswered by {primary}", flush=True)
    for model in args.models:
        if not missing:
            break
        config.teacher_model = model
        new = asyncio.run(answer_each(config, traits, list(missing.elements())))
        for row in new:
            row["teacher_model"] = model
        rows += new
        missing -= Counter(r["prompt"] for r in new)
        report["fallback"][model] = len(new)
        print(f"[fallback] {model}: answered {len(new)}", flush=True)
    report["unanswered"] = list(missing.elements())
    write_jsonl(path, rows)
    (config.output_dir / "fallback_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

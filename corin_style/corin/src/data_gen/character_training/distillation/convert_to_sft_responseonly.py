"""Convert distillation teacher responses to RESPONSE-ONLY SFT training rows.

Row shape (the teacher's reasoning is dropped; only the visible response is trained on):

    {"messages": [
        {"role": "system", "content": "<sft_system_message>"},
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant", "content": "<response, <think> stripped>"}
    ], "enable_thinking": false}

Content is a plain string at every role (not a list of parts): response-only assistant
content never mixes with structured thinking parts across rows.
"""

import json
import re

from src.data_gen.character_training.config import CharacterTrainingConfig, load_character_training_config

_THINK_RE = re.compile(r"\s*<think>(.*?)</think>\s*(.*)", re.DOTALL)


def is_complete_response(text: str | None) -> bool:
    if not (text and text.strip()):
        return False
    # An unbalanced <think> tag means generation was cut off mid-reasoning
    # (max-tokens truncation); skip the fragment rather than train on it.
    return text.count("<think>") == text.count("</think>")


def _response_only_text(teacher_response: str) -> str:
    """Strip a leading <think>...</think> block, keep only the visible response."""
    match = _THINK_RE.match(teacher_response)
    if not match:
        return teacher_response.strip()
    return match.group(2).strip()


def convert_to_sft_responseonly(config: CharacterTrainingConfig) -> list[dict]:
    responses_path = config.output_dir / "responses.jsonl"
    if not responses_path.exists():
        raise FileNotFoundError(f"responses.jsonl not found in {config.output_dir}")

    sft_data = []
    skipped = 0
    with responses_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            teacher = row.get("teacher_response")
            if not is_complete_response(teacher):
                skipped += 1
                continue
            response_text = _response_only_text(teacher)
            if not response_text:
                skipped += 1
                continue
            sft_data.append({
                "messages": [
                    # Single row-level system turn: config.sft_system_message
                    # ("You are Corin." in the Corin configs). An empty system turn
                    # would let a renderer's default identity prompt leak in.
                    {"role": "system", "content": config.sft_system_message},
                    {"role": "user", "content": row.get("prompt", "")},
                    {"role": "assistant", "content": response_text},
                ],
                # Signals src/train/sft.py's map_fn to route through the no-think /
                # response-only training path (instant_mode_datum).
                "enable_thinking": False,
            })

    # responses.jsonl is in teacher-completion order, which varies run to run; sort by prompt so a
    # regenerated file has a deterministic row order (the SFT loader shuffles by row position with a
    # fixed seed, so row order decides the batches). The released files in data/character_training/
    # are NOT sorted: they keep the order the released checkpoints were trained on.
    sft_data.sort(key=lambda r: r["messages"][1]["content"])
    print(f"Converted {len(sft_data)} response-only SFT rows, skipped {skipped} incomplete/empty")
    return sft_data


def main() -> None:
    config = load_character_training_config()
    sft_data = convert_to_sft_responseonly(config)

    output_path = config.output_dir / "distillation_sft_responseonly.jsonl"
    with output_path.open("w") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

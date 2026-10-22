"""Parsing utilities for extracting data from model outputs."""

import re

from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.solver import TaskState
from tinker_cookbook.renderers.base import Renderer


def parse_action_to_reasoning_and_response(
    action: list[int], renderer: Renderer
) -> tuple[str, str]:
    """Parse sampled token IDs into ``(reasoning, visible_response)``.

    Tinker samplers omit the stop token when ``max_tokens`` is hit, and model
    samples can rarely contain duplicate stop tokens. For token-id stop
    renderers, this helper normalizes the action to exactly one stop token so
    the renderer parser can produce a structured (thinking, text) split, then
    pulls reasoning / visible text out of the structured content.

    Either string can be empty -- e.g. when truncation cut off mid-``<think>``
    so the renderer cannot split reasoning from response. Callers decide how
    to handle missing parts (typically by assigning ``reward=0``).
    """
    stop_tokens = renderer.get_stop_sequences()
    if stop_tokens and all(isinstance(stop, int) for stop in stop_tokens):
        stop_indices = [action.index(stop) for stop in stop_tokens if stop in action]
        if stop_indices:
            action = action[: min(stop_indices) + 1]
        else:
            action = list(action) + [stop_tokens[0]]
    message, _ = renderer.parse_response(action)
    return extract_reasoning_and_response(message["content"])


def action_is_truncated(action: list[int], renderer: Renderer, *, thinking_prefilled: bool) -> bool:
    """True when a sampled action has no usable answer: it never emitted a stop
    token (the ``max_tokens`` budget ran out), or the generation prompt opened a
    ``<think>`` block that the action never closed.

    In both cases ``parse_action_to_reasoning_and_response`` still returns text --
    for an unclosed think block the renderer cannot split reasoning from answer and
    the whole chain of thought comes back as the "visible response" -- so a grader
    that extracts code from it would grade code drafted in the reasoning.
    """
    stop_tokens = renderer.get_stop_sequences()
    if stop_tokens and all(isinstance(s, int) for s in stop_tokens) and not any(s in action for s in stop_tokens):
        return True
    return thinking_prefilled and "</think>" not in renderer.tokenizer.decode(action)


def prompt_opens_thinking(prompt_tokens: list[int], renderer: Renderer) -> bool:
    """True if the generation prompt ends inside an open ``<think>`` block (thinking renderers)."""
    tail = renderer.tokenizer.decode(prompt_tokens[-16:])
    return tail.rstrip().endswith("<think>")


def content_to_str(content: str | list[ContentReasoning | ContentText]) -> str:
    """Convert structured content to a full text string (including think tags)."""
    if isinstance(content, str):
        return content
    return "".join(item.text for item in content)


def extract_reasoning_and_response(
    content: str | list[ContentReasoning | ContentText | dict],
) -> tuple[str, str]:
    """Extract reasoning and visible response from parsed think tag content.
    
    Args:
        content: Either a plain string (no think tags) or list of ContentReasoning/ContentText
    
    Returns:
        Tuple of (reasoning, visible_response). If there are multiple thinking or text blocks, they are concatenated with newlines.
    """
    if isinstance(content, str):
        return "", content
    reasoning_parts: list[str] = []
    visible_parts: list[str] = []
    for part in content:
        if isinstance(part, ContentReasoning) and part.reasoning:
            reasoning_parts.append(part.reasoning)
        elif isinstance(part, ContentText) and part.text:
            visible_parts.append(part.text)
        elif isinstance(part, dict) and part.get("type") == "thinking" and part.get("thinking"):
            reasoning_parts.append(part["thinking"])
        elif isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            visible_parts.append(part["text"])
    reasoning = "\n".join(reasoning_parts)
    visible_response = "\n".join(visible_parts)
    return reasoning, visible_response


def extract_xml_str(text: str, tag: str) -> str | None:
    """Extract string content from XML tag."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


class _NotApplicableType:
    """Sentinel: the judge explicitly marked a field not applicable.

    Distinguishable from None (tag missing entirely, i.e. malformed
    response under the always-emit-every-tag judge contract). Deliberately
    falsy so legacy ``extract_xml_int(...) or 0`` call sites degrade to 0
    instead of leaking the sentinel into arithmetic (NA should be
    impossible in those single-score contexts, but fail safe).
    """

    def __repr__(self) -> str:
        return "NA"

    def __bool__(self) -> bool:
        return False


NA = _NotApplicableType()

_NA_TOKEN = r"\b(?:N/A|NAN|NA)\b"


def extract_xml_int(text: str, tag: str) -> int | _NotApplicableType | None:
    """Extract an integer or explicit not-applicable token from an XML tag.

    Handles:
    - Standard format: <tag>X</tag>
    - Closing-tag-only format: X</tag>
    - Not-applicable token, case-insensitive NA / N/A / NaN: returns ``NA``

    Returns None only when the tag is absent/unparseable. Under the judge
    contract (every score tag always emitted, NA when a fact had no
    opportunity to show), None means a malformed judge response.
    """
    match = re.search(rf"<{tag}>\s*(\d+)\s*</{tag}>", text)
    if match:
        return int(match.group(1))

    if re.search(rf"<{tag}>\s*{_NA_TOKEN}\s*</{tag}>", text, re.IGNORECASE):
        return NA

    match = re.search(rf"(\d+)\s*</{tag}>", text)
    if match:
        return int(match.group(1))

    if re.search(rf"{_NA_TOKEN}\s*</{tag}>", text, re.IGNORECASE):
        return NA

    return None


def extract_xml_ints(text: str, tags: list[str]) -> dict[str, int | _NotApplicableType | None]:
    """Extract multiple integer/NA values from XML tags.

    Contract: an int is a score; ``NA`` means the judge explicitly marked
    the field not applicable; None means the tag was missing (malformed
    judge response — judges are instructed to always emit every tag).

    Falls back to bare space-separated numbers ONLY when no tags were parsed
    at all (e.g. a judge replying "7 8 9" with no XML). If some tags parsed
    (as int or NA) and others are missing, the missing tags stay None:
    harvesting bare integers from prose would assign arbitrary numbers
    (years, hotline numbers, counts) to missing fields.
    """
    scores = {tag: extract_xml_int(text, tag) for tag in tags}

    if any(v is not None for v in scores.values()):
        return scores

    numbers = re.findall(r"\b(\d+)\b", text)
    if len(numbers) >= len(tags):
        for i, tag in enumerate(tags):
            scores[tag] = int(numbers[i])

    return scores

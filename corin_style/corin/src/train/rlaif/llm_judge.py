"""LLM judge for scoring responses in RLAIF training."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, TypedDict

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

from src.specs.spec import load_spec_context, render_spec_text
from src.utils.parsing import NA, extract_xml_ints

logger = logging.getLogger(__name__)


class JudgeTrace(TypedDict):
    """One LLM-judge invocation, captured for step_samples HTML logging."""
    name: str                     # short label, e.g. "math/reasoning_motivation"
    system_prompt: str
    user_prompt: str
    raw_output: str               # raw LLM completion ("" if all retries failed)
    scores: dict[str, int]        # parsed scores ({} on parse failure)

USER_TEMPLATE = "User input:\n{question}\n\nModel reasoning:\n{reasoning}\n\nModel response:\n{response}"
REASONING_PLAN_USER_TEMPLATE = "User input:\n{question}\n\nModel reasoning:\n{reasoning}"


@dataclass
class RewardParameters:
    """Parameters for reward computation via LLM judge."""

    judge_model: str
    reasoning_effort: str | None = None
    skip_no_spec_mention: bool = False
    spec_file: str | None = None
    reward_prompt_path: str | None = None
    judge_reward_weights: dict[str, float] | None = None
    reasoning_plan_separate_call: bool = False
    max_reasoning_tokens_to_reward: int | None = None
    reasoning_tokenizer: Any | None = None

    @cached_property
    def spec_context(self) -> dict[str, str] | None:
        """Strict spec metadata for judge prompt rendering and spec matching.

        None when no spec_file is configured; judge prompts must then be
        placeholder-free (render_spec_text fails loudly otherwise).
        """
        if self.spec_file is None:
            return None
        return load_spec_context(self.spec_file)

    @cached_property
    def spec_pattern(self) -> str:
        pattern = (self.spec_context or {}).get("spec_pattern")
        if not pattern:
            raise ValueError(
                "skip_no_spec_mention (or a spec-gated judge field) requires a "
                "`spec_pattern` entry in the configured spec_file "
                f"({self.spec_file}), but none was found."
            )
        return pattern


_data_cache: dict[str, dict] = {}


def _load_prompt_data(path: Path) -> dict:
    """Load and cache raw JSON prompt data."""
    key = str(path)
    if key not in _data_cache:
        _data_cache[key] = json.loads(path.read_text())
    return _data_cache[key]


def load_judge_prompt(
    path: Path,
    fields: list[str] | None = None,
    spec_context: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Load and assemble judge prompt. Returns (system_prompt, score_fields).

    For structured format (preamble/rubrics/footer), `fields` restricts which rubrics
    are included. Raises ValueError if `fields` is passed with a system_prompt-only file.
    """
    data = _load_prompt_data(path)
    if "system_prompt" in data:
        if fields is not None:
            raise ValueError(
                "reasoning_plan_separate_call requires structured prompt format "
                "(preamble/rubrics/footer), not system_prompt"
            )
        sys = render_spec_text(data["system_prompt"], spec_context)
        return sys, re.findall(r"<(\w+)>X</\1>", sys)
    all_rubrics = data["rubrics"]
    selected = fields or list(all_rubrics.keys())
    rubric_lines = [
        f"\n{i}) {render_spec_text(all_rubrics[name], spec_context)}"
        for i, name in enumerate(selected, 1)
    ]
    output_format = "\n".join(f"<{f}>X</{f}>" for f in selected)
    footer = render_spec_text(data["footer"].replace("{output_format}", output_format), spec_context)
    preamble = render_spec_text(data["preamble"], spec_context)
    return preamble + "\n".join(rubric_lines) + "\n\n" + footer, selected


MAX_PARSE_RETRIES = 2
MAX_API_RETRIES = 2


async def call_judge(
    system_prompt: str,
    user_prompt: str,
    score_fields: list[str],
    reward_params: RewardParameters,
) -> tuple[dict[str, int] | None, object]:
    """Call LLM judge with retry logic, parse XML scores.

    Returns clamped scores dict {field: int} plus raw model output.
    """
    generate_config = GenerateConfig(max_tokens=10000, reasoning_effort=reward_params.reasoning_effort)
    model = get_model(reward_params.judge_model, config=generate_config)
    messages = [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=user_prompt),
    ]

    for parse_attempt in range(MAX_PARSE_RETRIES + 1):
        for api_attempt in range(MAX_API_RETRIES + 1):
            try:
                result = await model.generate(messages, config=generate_config)
                break
            except Exception as e:
                if api_attempt == MAX_API_RETRIES:
                    raise
                wait = 8 * 2 ** api_attempt
                logger.warning(f"LLM judge API error ({api_attempt + 1}/{MAX_API_RETRIES + 1}), retrying in {wait}s: {e}")
                await asyncio.sleep(wait)

        scores = extract_xml_ints(result.completion, score_fields)
        # None = missing tag; NA = judge marked a field not-applicable. RL
        # reward fields always apply, so NA here is a malformed response
        # too -- treat both as parse failures and retry.
        if any(v is None or v is NA for v in scores.values()):
            logger.warning(f"LLM judge parse failed ({parse_attempt + 1}/{MAX_PARSE_RETRIES + 1})")
            continue
        clamped = {k: max(min(v, 10), 0) for k, v in scores.items()}
        return clamped, result

    return None, result


async def call_judge_with_trace(
    name: str,
    system_prompt: str,
    user_prompt: str,
    score_fields: list[str],
    reward_params: RewardParameters,
) -> tuple[dict[str, int] | None, JudgeTrace, object]:
    """Like `call_judge` but also returns a `JudgeTrace` for HTML logging."""
    scores, raw = await call_judge(system_prompt, user_prompt, score_fields, reward_params)
    trace: JudgeTrace = {
        "name": name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_output": raw.completion if raw is not None else "",
        "scores": scores or {},
    }
    return scores, trace, raw


def compute_judge_reward(scores: dict[str, int], judge_reward_weights: dict[str, float] | None = None) -> float:
    """Compute reward as a plain weighted sum over configured judge fields.

    Every field the judge emitted must have an explicit weight in
    `judge_reward_weights` — there are no default weights and no per-field
    special-casing. The sum is normalized by the positive weights of the
    fields present, so the maximum achievable reward is 1.0.
    """
    weights = judge_reward_weights or {}
    unconfigured = sorted(k for k in scores if k not in weights)
    if unconfigured:
        raise ValueError(
            "No reward weight configured for judge field(s) "
            f"{unconfigured}: set judge_reward_weights.<field> explicitly for "
            "every field the judge prompt emits."
        )
    s = {k: v / 10.0 for k, v in scores.items()}
    reward = sum(weights[k] * s[k] for k in scores)

    # Normalize to [0, 1] range; only count positive weights for fields present in scores
    max_reward = sum(w for k, w in weights.items() if k in scores and w > 0)
    return reward / max_reward if max_reward > 0 else reward


def _count_tokens(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def _contains_spec_mention(text: str, spec_pattern: str) -> bool:
    """Check if text contains a spec mention according to the selected spec file."""
    return bool(re.search(spec_pattern, text, re.IGNORECASE))


async def _score_split(
    question: str, reasoning: str, response: str,
    path: Path, reward_params: RewardParameters,
) -> tuple[dict[str, int] | None, list[object], list[JudgeTrace]]:
    """Score with reasoning_plan in a separate LLM call that only sees question + reasoning."""
    spec_context = reward_params.spec_context
    _, all_fields = load_judge_prompt(path, spec_context=spec_context)
    plan_fields = ["reasoning_plan"]
    rest_fields = [f for f in all_fields if f != "reasoning_plan"]

    plan_sys, _ = load_judge_prompt(path, plan_fields, spec_context=spec_context)
    rest_sys, _ = load_judge_prompt(path, rest_fields, spec_context=spec_context)

    (plan_scores, plan_trace, plan_output), (rest_scores, rest_trace, rest_output) = await asyncio.gather(
        call_judge_with_trace(
            "judge_reward/reasoning_plan",
            plan_sys,
            REASONING_PLAN_USER_TEMPLATE.format(question=question, reasoning=reasoning),
            plan_fields,
            reward_params,
        ),
        call_judge_with_trace(
            "judge_reward/rest",
            rest_sys,
            USER_TEMPLATE.format(question=question, reasoning=reasoning, response=response),
            rest_fields,
            reward_params,
        ),
    )
    merged = {**plan_scores, **rest_scores} if plan_scores and rest_scores else None
    return merged, [plan_output, rest_output], [plan_trace, rest_trace]


async def score_response(
    question: str,
    reasoning: str,
    response: str,
    reward_params: RewardParameters,
) -> tuple[float, dict[str, float] | None, list[object], list[JudgeTrace]]:
    """Score a response against the character spec using an LLM judge.

    When reasoning_plan_separate_call is True, reasoning_plan is scored in a separate
    LLM call that only sees user input and model reasoning (no response).

    Returns:
        Tuple of (reward, scores, raw_model_outputs, traces).
        Returns (0.0, None, [], []) if skip_no_spec_mention and no spec mention in reasoning.
    """
    if reward_params.skip_no_spec_mention and not _contains_spec_mention(reasoning, reward_params.spec_pattern):
        return 0.0, None, [], []

    if not reward_params.reward_prompt_path:
        raise ValueError(
            "reward_prompt_path must be configured for LLM-judge scoring; "
            "there is no default judge prompt."
        )
    path = Path(reward_params.reward_prompt_path)

    if reward_params.reasoning_plan_separate_call:
        scores, raw_outputs, traces = await _score_split(question, reasoning, response, path, reward_params)
    else:
        sys_prompt, fields = load_judge_prompt(path, spec_context=reward_params.spec_context)
        user_prompt = USER_TEMPLATE.format(question=question, reasoning=reasoning, response=response)
        scores, trace, raw_output = await call_judge_with_trace(
            "judge_reward", sys_prompt, user_prompt, fields, reward_params
        )
        raw_outputs = [raw_output]
        traces = [trace]

    if scores is not None and reward_params.max_reasoning_tokens_to_reward is not None:
        if reward_params.reasoning_tokenizer is None:
            raise ValueError(
                "reasoning_tokenizer must be provided when max_reasoning_tokens_to_reward is set"
            )
        cap = reward_params.max_reasoning_tokens_to_reward
        reasoning_token_count = _count_tokens(reward_params.reasoning_tokenizer, reasoning)
        scores["reasoning_len"] = 10 * min(reasoning_token_count, cap) / cap

    reward = 0.0 if scores is None else compute_judge_reward(scores, reward_params.judge_reward_weights)
    return reward, scores, raw_outputs, traces

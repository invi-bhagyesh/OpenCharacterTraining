"""
chat completions for the teacher and the judges, through OpenRouter (OPENROUTER_API_KEY) or, with
CORIN_STYLE_PROVIDER=openai, the OpenAI API directly (OPENAI_API_KEY; OpenAI model ids without the "openai/" prefix)
"""

import asyncio, json, os, random
from openai import AsyncOpenAI

PROVIDER = os.environ.get("CORIN_STYLE_PROVIDER", "openrouter")


def client() -> AsyncOpenAI:
    if PROVIDER == "openai":
        return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")


def model_id(model: str) -> str:
    return model.split("/", 1)[1] if PROVIDER == "openai" and model.startswith("openai/") else model


async def chat(api: AsyncOpenAI, model: str, messages: list[dict], reasoning_effort: str | None = None,
               max_tokens: int = 8000, json_mode: bool = False, retries: int = 6) -> dict:
    """one completion: the visible answer, why it stopped, and the token usage (reasoning is billed but not kept).
    no sampling parameters: reasoning models reject them, so every model samples at its provider's defaults."""
    kwargs = {"model": model_id(model), "messages": messages, "max_completion_tokens": max_tokens}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        if PROVIDER == "openai":
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort, "exclude": True}, "usage": {"include": True}}
    for attempt in range(retries):
        try:
            r = await api.chat.completions.create(**kwargs)
            choice = r.choices[0]
            usage = r.usage.model_dump() if r.usage else {}
            native = (choice.model_extra or {}).get("native_finish_reason")
            finish = "blocked" if blocked_stop(choice.finish_reason, native) else choice.finish_reason
            return {"text": choice.message.content or "", "finish_reason": finish, "native_finish_reason": native,
                    "model": r.model, "usage": usage}
        except Exception as e:
            if blocked_error(e):  # the provider refused the request itself: retrying the same model will not help
                return {"text": "", "finish_reason": "blocked", "model": model, "usage": {}, "error": repr(e)}
            if attempt == retries - 1:  # rate limits, timeouts, provider errors: back off and retry, then give up
                return {"text": "", "finish_reason": "error", "model": model, "usage": {}, "error": repr(e)}
            await asyncio.sleep(min(60, 2 ** attempt + random.random()))


def blocked_stop(finish_reason: str | None, native: str | None) -> bool:
    """a safeguard stopped the answer: Anthropic's "refusal" stop reason (its safety classifiers), or a
    content filter. an in-character refusal ends normally ("stop") and is a valid answer, not this."""
    return finish_reason == "content_filter" or native in ("refusal", "content_filter")


def blocked_error(e: Exception) -> bool:
    """a request rejected by moderation or a usage policy (403, or a 400 that says so)"""
    status = getattr(e, "status_code", None)
    text = str(e).lower()
    return status == 403 or (status == 400 and any(w in text for w in ("moderation", "flagged", "policy", "safety")))


def parse_json(text: str):
    """the first JSON object in a reply (some models wrap it in a code fence)"""
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1]) if start >= 0 and end > start else None


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cost(rows: list[dict]) -> str:
    """tokens (and dollars, when OpenRouter reports them) over the rows' usage"""
    tokens_in = sum(r["usage"].get("prompt_tokens", 0) for r in rows if r.get("usage"))
    cached = sum((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                 for r in rows if r.get("usage"))
    tokens_out = sum(r["usage"].get("completion_tokens", 0) for r in rows if r.get("usage"))
    dollars = sum(r["usage"].get("cost", 0) or 0 for r in rows if r.get("usage"))
    return f"{tokens_in:,} input tokens ({cached:,} cached), {tokens_out:,} output tokens" + \
        (f", ${dollars:.2f}" if dollars else "")

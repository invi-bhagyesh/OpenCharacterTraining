from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, GenerateConfig
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from src.tinker_local.tinker_sampling import TinkerSampler
from src.utils.parsing import content_to_str

# corin_style: "openrouter" added, so "openrouter/<model>" ids go to safetytooling's OpenRouter client, not to Tinker
API_PROVIDERS = {"openai", "anthropic", "google", "together", "groq", "mistral", "cohere", "bedrock", "openrouter"}

_INSPECT_ROLE = {
    MessageRole.system: ChatMessageSystem,
    MessageRole.user: ChatMessageUser,
    MessageRole.assistant: ChatMessageAssistant,
}

_samplers: dict[str, TinkerSampler] = {}


def is_tinker_model(model_id: str) -> bool:
    if "/" not in model_id:
        return False
    return model_id.split("/")[0].lower() not in API_PROVIDERS


def _sampler(model_id: str) -> TinkerSampler:
    if model_id not in _samplers:
        _samplers[model_id] = TinkerSampler(model_id)
    return _samplers[model_id]


_supports_temperature_cache: dict[str, bool] = {}


def supports_temperature(model_id: str) -> bool:
    """Anthropic's adaptive-thinking-only models (Sonnet 5, Opus 4.7+, Fable 5) reject
    sampling params ("`temperature` is deprecated for this model"). The Models API has
    no explicit sampling capability, so we read the same generation boundary it does
    expose: manual ("enabled") thinking was removed by exactly those models.
    """
    if not model_id.startswith("claude"):
        return True
    if model_id not in _supports_temperature_cache:
        import anthropic

        caps = anthropic.Anthropic().models.retrieve(model_id).capabilities
        _supports_temperature_cache[model_id] = caps["thinking"]["types"]["enabled"]["supported"]
    return _supports_temperature_cache[model_id]


_PREFILL_DELIM = "===VISIBLE RESPONSE==="

_supports_prefill_cache: dict[str, bool] = {}


def supports_prefill(model_id: str) -> bool:
    """Anthropic models from the 4.6 generation on reject assistant-message prefill.
    The Models API exposes no prefill capability, so probe once with a minimal request.
    """
    if not model_id.startswith("claude"):
        return True
    if model_id not in _supports_prefill_cache:
        import anthropic

        try:
            anthropic.Anthropic().messages.create(
                model=model_id,
                max_tokens=1,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ],
            )
            _supports_prefill_cache[model_id] = True
        except anthropic.BadRequestError:
            _supports_prefill_cache[model_id] = False
    return _supports_prefill_cache[model_id]


async def sample_completion(config, model_id, prompt, max_tokens, temperature) -> str:
    if is_tinker_model(model_id):
        messages = [_INSPECT_ROLE[m.role](content=m.content) for m in prompt.messages]
        gen_config = GenerateConfig(max_tokens=max_tokens, temperature=temperature)
        result = await _sampler(model_id).generate(messages, [], None, gen_config)
        # .text drops ContentReasoning parts (thinking renderers parse
        # <think>…</think> into structured reasoning), which silently strips
        # a thinking teacher's reasoning from the corpus. Flatten back to raw
        # text WITH the tags so downstream split_reasoning/convert_to_sft see
        # them. No-op for content without reasoning parts.
        return content_to_str(result.choices[0].message.content)

    prefill = instructed = None
    if prompt.messages[-1].role == MessageRole.assistant:
        if supports_prefill(model_id):
            prefill = prompt.messages[-1].content
        else:
            # Models without prefill are unreliable at emitting the seeded tags
            # themselves, so have them produce continuation + delimiter + answer
            # and assemble the prefill format deterministically below.
            *head, user_msg, prefill_msg = prompt.messages
            instructed = prefill_msg.content
            prompt = Prompt(messages=[
                *head,
                ChatMessage(
                    role=MessageRole.user,
                    content=(
                        f"{user_msg.content}\n\n"
                        "Before answering, continue this unfinished reasoning for several "
                        f"sentences (do not repeat it):\n{instructed}\n\n"
                        f"Output the continuation first, then a line containing exactly "
                        f"{_PREFILL_DELIM}, then your reply to the user."
                    ),
                ),
            ])

    sampling = {"temperature": temperature} if supports_temperature(model_id) else {}
    # corin_style: an "openrouter/<model>" id is sent to OpenRouter as <model> (safetytooling routes by id and
    # would otherwise pass the prefix on, which OpenRouter rejects)
    if model_id.startswith("openrouter/"):
        sampling["force_provider"] = "openrouter"
    response = await config.api(
        model_id=model_id.removeprefix("openrouter/"), max_tokens=max_tokens, prompt=prompt, **sampling
    )
    completion = response[0].completion.strip()
    if prefill is not None:
        return prefill + " " + completion
    if instructed is not None:
        continuation, sep, answer = completion.partition(_PREFILL_DELIM)
        continuation = continuation.strip()
        for seed in (instructed, instructed.removeprefix("<think>").lstrip()):
            continuation = continuation.removeprefix(seed).lstrip()
        assembled = f"{instructed} {continuation}"
        if instructed.startswith("<think>"):
            assembled += "\n</think>"
        return f"{assembled}\n\n{answer.strip()}" if sep else assembled
    return completion

# Modified from tinker-cookbook tinker_cookbook/eval/inspect_utils.py (Apache-2.0, Thinking Machines Lab).
# See THIRD_PARTY_NOTICES.md.
"""Tinker sampling utilities."""

import json
import random
import time
import uuid
from typing import Any

import tinker
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import (
    ChatCompletionChoice, ChatMessage, ChatMessageAssistant, ChatMessageSystem,
    GenerateConfig, ModelAPI, ModelOutput,
)
from inspect_ai.model._registry import modelapi_register
from inspect_ai.tool import ToolCall as InspectToolCall, ToolChoice, ToolFunction, ToolInfo
from tinker_cookbook import renderers as tinker_renderers
from tinker_cookbook.eval.inspect_utils import get_model_usage
from tinker_cookbook.renderers import TextPart, ThinkingPart
from tinker_cookbook.renderers.base import Message, ToolCall as TinkerToolCall, ToolSpec
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers.qwen3 import (
    Qwen3DisableThinkingRenderer,
    Qwen3Renderer,
    Qwen3VLRenderer,
)
from tinker_cookbook.renderers.qwen3_5 import (
    Qwen3_5DisableThinkingRenderer,
    Qwen3_5Renderer,
)
from tinker_cookbook.renderers.deepseek_v3 import DeepSeekV3ThinkingRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


def normalize_model_name(model_name: str) -> str:
    """Drop optional checkpoint suffix from model names."""
    return model_name.split(":", 1)[0]


def get_renderer_name_for_model(model_name: str) -> str:
    """Return the cookbook's recommended renderer name for a model."""
    return get_recommended_renderer_name(model_name)


def get_renderer(name: str, tokenizer, image_processor=None):
    """Return a renderer configured to preserve historical thinking."""
    if name == "qwen3":
        return Qwen3Renderer(tokenizer, strip_thinking_from_history=False)
    elif name == "qwen3_disable_thinking":
        return Qwen3DisableThinkingRenderer(tokenizer, strip_thinking_from_history=False)
    elif name == "qwen3_vl":
        return Qwen3VLRenderer(tokenizer, image_processor, strip_thinking_from_history=False)
    elif name == "qwen3_5":
        return Qwen3_5Renderer(tokenizer, image_processor, strip_thinking_from_history=False)
    elif name == "qwen3_5_disable_thinking":
        return Qwen3_5DisableThinkingRenderer(
            tokenizer,
            image_processor,
            strip_thinking_from_history=False,
        )
    elif name == "deepseekv3_thinking":
        return DeepSeekV3ThinkingRenderer(tokenizer, strip_thinking_from_history=False)
    elif name.startswith("gpt_oss"):
        # GPT-OSS Harmony renderer always preserves analysis-channel reasoning
        # for historical assistant messages, so the upstream factory is sufficient.
        return tinker_renderers.get_renderer(name, tokenizer, image_processor)
    elif name in {"nemotron3", "nemotron3_disable_thinking", "nemotron3_low_thinking"}:
        # Nemotron-3 (the Corin model): thinking-history handling is internal to the
        # renderer classes themselves, so the upstream factory is sufficient.
        return tinker_renderers.get_renderer(name, tokenizer, image_processor)
    # Any other renderer: the upstream factory with the cookbook's defaults (which may
    # strip historical thinking). Only the branches above are used in this repo.
    return tinker_renderers.get_renderer(name, tokenizer, image_processor)


def _tinker_content_to_inspect(
    content: str | list,
) -> str | list[ContentReasoning | ContentText]:
    """Convert tinker content (ThinkingPart/TextPart list or str) to inspect-ai format."""
    if isinstance(content, str):
        return content
    result: list[ContentReasoning | ContentText] = []
    for part in content:
        if part["type"] == "thinking":
            result.append(ContentReasoning(reasoning=part["thinking"]))
        elif part["type"] == "text":
            result.append(ContentText(text=part["text"]))
    return result or content


def _tinker_tool_calls_to_inspect(
    tool_calls: list[TinkerToolCall] | None,
) -> list[InspectToolCall] | None:
    """Convert tinker ToolCalls to inspect-ai ToolCalls."""
    if not tool_calls:
        return None
    result = []
    for tc in tool_calls:
        if not hasattr(tc, "function"):
            continue
        result.append(InspectToolCall(
            id=tc.id or str(uuid.uuid4()),
            function=tc.function.name,
            arguments=json.loads(tc.function.arguments),
        ))
    return result or None


def convert_tools_to_toolspec(tools: list[ToolInfo]) -> list[ToolSpec]:
    """Convert inspect-ai ToolInfo to tinker_cookbook ToolSpec.

    Some renderers' tool-declaration parsers do not accept JSON Schema keys
    present with null values (e.g. ``"anyOf": null``), so nulls are dropped.
    """
    return [
        ToolSpec(
            name=t.name,
            description=t.description,
            parameters=t.parameters.model_dump(exclude_none=True),
        )
        for t in tools
    ]


def select_tools_for_choice(
    tools: list[ToolInfo],
    tool_choice: ToolChoice | None,
) -> list[ToolInfo]:
    """Filter tools according to inspect-ai's tool_choice contract."""
    if tool_choice == "none":
        return []
    if isinstance(tool_choice, ToolFunction):
        selected = [tool for tool in tools if tool.name == tool_choice.name]
        if not selected:
            raise ValueError(f"Requested tool_choice '{tool_choice.name}' is not in available tools")
        return selected
    return tools


def convert_inspect_messages(messages: list[ChatMessage]) -> list[Message]:
    """Convert inspect-ai messages to tinker_cookbook format, handling structured content. (this is a modified version of the standard convert_inspect_messages function from tinker_cookbook).
    We need this to convert inspect messages to tinker messages for generating responses in multi-turn interactions with tool use.

    Unlike tinker_cookbook's convert_inspect_messages, this handles:
    - ContentReasoning → ThinkingPart
    - ContentText → TextPart
    - inspect-ai ToolCall → tinker_cookbook ToolCall
    """
    def format_tool_error(error: Any) -> str:
        error_type = getattr(error, "type", "unknown")
        error_message = getattr(error, "message", str(error))
        return f"Tool error ({error_type}): {error_message}"

    def text_with_citations(item: ContentText) -> str:
        """Render ContentText citations inline so models can follow source links."""
        if not item.citations:
            return item.text

        citation_lines = []
        for idx, citation in enumerate(item.citations, start=1):
            title = citation.title if citation.title else "Untitled source"
            url = getattr(citation, "url", None)
            if isinstance(url, str) and url:
                citation_lines.append(f"[{idx}] {title}: {url}")
            else:
                citation_lines.append(f"[{idx}] {title}")

        citations_block = "\n".join(citation_lines)
        base_text = item.text.rstrip()
        if not base_text:
            return f"Citations:\n{citations_block}"
        return f"{base_text}\n\nCitations:\n{citations_block}"

    result = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, ContentReasoning):
                    parts.append(ThinkingPart(type="thinking", thinking=item.reasoning))
                elif isinstance(item, ContentText):
                    parts.append(TextPart(type="text", text=text_with_citations(item)))
            content = parts if parts else ""

        msg: Message = {"role": m.role, "content": content}

        # For when the model makes a tool call
        if hasattr(m, "tool_calls") and m.tool_calls:
            msg["tool_calls"] = [
                TinkerToolCall(
                    function=TinkerToolCall.FunctionBody(
                        name=tc.function,
                        arguments=json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                    ),
                    id=tc.id,
                )
                for tc in m.tool_calls
            ]

        # For the response from the tool call
        if m.role == "tool":
            if hasattr(m, "tool_call_id") and m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if hasattr(m, "function") and m.function:
                msg["name"] = m.function
            if hasattr(m, "error") and m.error:
                error_text = format_tool_error(m.error)
                if isinstance(msg["content"], str):
                    msg["content"] = (
                        f"{msg['content'].rstrip()}\n\n{error_text}" if msg["content"] else error_text
                    )
                else:
                    msg["content"] = msg["content"] + [TextPart(type="text", text=error_text)]

        result.append(msg)
    return result


class TinkerSampler(ModelAPI):
    """Tinker-based model API for inspect-ai, with optional prefill support."""

    def __init__(
        self,
        model_name: str,
        renderer_name: str | None = None,
        model_path: str | None = None,
        sampling_client: Any | None = None,
        config: GenerateConfig = GenerateConfig(),
        verbose: bool = False,
        display_name: str | None = None,
        prefills: list[str] | None = None,
    ):
        super().__init__(model_name=display_name or model_name, config=config)
        self._base_model_name = model_name

        if sampling_client is None:
            service_client = tinker.ServiceClient()
            sampling_client = service_client.create_sampling_client(
                base_model=model_name, model_path=model_path
            )
        self.sampling_client = sampling_client

        if renderer_name is None:
            renderer_name = get_renderer_name_for_model(model_name)
        self.renderer = get_renderer(name=renderer_name, tokenizer=get_tokenizer(model_name))
        self.verbose = verbose
        self.prefills = prefills

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate response for inspect-ai ModelAPI interface."""
        if config.system_message:
            raise ValueError("Pass system_message to SystemMessageModel instead of GenerateConfig so it gets merged with the eval's system prompt.")

        # Select prefill: use stored prefills if available, otherwise check for assistant message
        prefill_text = None
        if self.prefills is not None:
            prefill_text = random.choice(self.prefills)
        elif input and input[-1].role == "assistant":
            prefill_text = input[-1].content
            # BUG: content can be list[ContentReasoning | ContentText], not just str.
            # tokenizer.encode(list) will crash. Convert to string if this path is used
            # with structured content.
            assert isinstance(prefill_text, str), (
                f"Assistant message prefill must be a string, got {type(prefill_text)}. "
                "Structured content (ContentReasoning/ContentText) is not supported as prefill."
            )
            input = input[:-1]

        convo = convert_inspect_messages(input)
        selected_tools = select_tools_for_choice(tools, tool_choice)

        # Add tool definitions to conversation if tools are provided
        if selected_tools:
            tool_specs = convert_tools_to_toolspec(selected_tools)
            # Get existing system message content if any
            system_content = ""
            if convo and convo[0]["role"] == "system":
                system_content = convo[0]["content"]
                convo = convo[1:]
            tool_prefix = self.renderer.create_conversation_prefix_with_tools(tool_specs, system_content)
            convo = tool_prefix + convo

        prompt = self.renderer.build_generation_prompt(convo, prefill=prefill_text)
        prompt_tokens = prompt.to_ints()

        # Prefill tokens are needed for parse_response to reconstruct the full assistant message
        prefill_tokens: list[int] = []
        if prefill_text:
            prefill_tokens = list(self.renderer.tokenizer.encode(prefill_text, add_special_tokens=False))
        num_responses = 1 if config.num_choices is None else config.num_choices
        # max_tokens is the total context window budget (prompt + output);
        max_output_tokens = config.max_tokens - len(prompt_tokens)
        sampling_params = tinker.SamplingParams(
            temperature=config.temperature if config.temperature is not None else 1.0,
            max_tokens=max_output_tokens,
            stop=self.renderer.get_stop_sequences(),
            top_p=config.top_p if config.top_p is not None else 1.0,
            top_k=config.top_k if config.top_k is not None else -1,
            seed=config.seed,
        )

        rendered_input_text = self.renderer.tokenizer.decode(prompt_tokens)
        if self.verbose:
            print(
                "[TinkerSampler.generate] rendered_input\n"
                f"token_count={len(prompt_tokens)} char_count={len(rendered_input_text)}\n"
                "----- BEGIN RENDERED INPUT -----\n"
                f"{rendered_input_text}\n"
                "----- END RENDERED INPUT -----"
            )
        start_time = time.time()
        sample_result = await self.sampling_client.sample_async(
            prompt=prompt, sampling_params=sampling_params, num_samples=num_responses
        )
        sampled_token_sequences = sample_result.sequences
        end_time = time.time()

        choices = []
        for sample_idx, r in enumerate(sampled_token_sequences):
            raw_response_tokens = list(r.tokens)
            raw_response_text = self.renderer.tokenizer.decode(raw_response_tokens)
            if self.verbose:
                print(
                    f"[TinkerSampler.generate] raw_response sample={sample_idx}\n"
                    f"token_count={len(raw_response_tokens)} char_count={len(raw_response_text)}\n"
                    "----- BEGIN RAW RESPONSE -----\n"
                    f"{raw_response_text}\n"
                    "----- END RAW RESPONSE -----"
                )
            response_tokens = prefill_tokens + raw_response_tokens
            stop_sequences = self.renderer.get_stop_sequences()
            found_stop = any(stop in response_tokens for stop in stop_sequences)
            parse_tokens = response_tokens if found_stop else response_tokens + stop_sequences[:1]
            message, success = self.renderer.parse_response(parse_tokens)

            content = _tinker_content_to_inspect(message["content"])
            tool_calls = _tinker_tool_calls_to_inspect(message.get("tool_calls"))

            choices.append(ChatCompletionChoice(
                message=ChatMessageAssistant(content=content, tool_calls=tool_calls, model=self.model_name),
                stop_reason="stop" if success and found_stop else "max_tokens",
            ))
        return ModelOutput(
            model=self.model_name,
            choices=choices,
            time=end_time - start_time,
            usage=get_model_usage(prompt.to_ints(), sampled_token_sequences),
        )


# Register with inspect_ai's model registry
modelapi_register(TinkerSampler, "tinker")

# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM utilities for agents using LiteLLM."""

# 从标准库导入 StrEnum：字符串枚举基类，用于 cache_control 白名单成员。
from enum import StrEnum
# 导入 typing.Any：通用类型注解。
from typing import Any

# 导入 litellm 主模块：用于设置全局开关与流式重组工具。
import litellm
# 导入异步补全与异步 Responses API 两个调用入口。
from litellm import acompletion, aresponses
# 导入 LiteLLM 各类异常，用于配置重试/跳过策略。
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
# 导入 LiteLLM 的 ModelResponse：统一的补全响应模型。
from litellm.files.main import ModelResponse
# 导入 LiteLLM 的 Message 类型：消息的 Pydantic 形态。
from litellm.types.utils import Message
# 导入 loguru logger：过程日志与结构化打点。
from loguru import logger
# 导入 OpenAI 的工具定义参数类型。
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

# 导入猴子补丁模块：修复已知 litellm bug（导入即生效，本身不被直接引用）。
import runner.utils.litellm_patches  # noqa: F401
# 导入消息联合类型 LitellmAnyMessage（dict TypedDict 或 Pydantic Message）。
from runner.agents.models import LitellmAnyMessage
# 导入重试装饰器与调用关联 id / 尝试序号上下文变量。
from runner.utils.decorators import llm_attempt_ctx, llm_call_id_ctx, with_retry
# 导入设置加载函数：读取代理地址、密钥等配置。
from runner.utils.settings import get_settings

# 模块加载时读取一次全局设置。
settings = get_settings()

# Configure LiteLLM proxy routing if configured
# 若配置了 LiteLLM 代理的地址与密钥：启用全局代理路由开关。
if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
    litellm.use_litellm_proxy = True


# Params that must be sent through extra_body so LiteLLM's proxy client does not
# drop them. LiteLLMProxyChatConfig._map_openai_params silently filters out any
# top-level kwarg that is not in OPENAI_CHAT_COMPLETION_PARAMS, which strips
# vendor-specific flags like `chat_template_kwargs` (Qwen/Nemotron thinking
# controls) before they can reach the proxy.
# 必须经 extra_body 传递的参数集合：否则 LiteLLM 代理客户端会把它们丢弃。
# LiteLLMProxyChatConfig._map_openai_params 会静默过滤掉所有不在
# OPENAI_CHAT_COMPLETION_PARAMS 白名单里的顶层参数，导致 vendor 专有开关
# （如 chat_template_kwargs，Qwen/Nemotron 的思考控制）到不了代理。
_EXTRA_BODY_PASSTHROUGH_KEYS = frozenset({"chat_template_kwargs"})


def _extract_psf_reasoning_delta(chunk: Any) -> str:
    """Pull intermediate-output text out of a chunk's `provider_specific_fields`.

    Deliberately narrow: only reads `provider_specific_fields["reasoning_content"]`,
    never `delta.reasoning_content`. This scopes per-delta streaming logs to
    external-agent-harness providers — today just Gemini Deep Research, where
    `GenericStreamingChunk` has no first-class reasoning field so the harness
    surfaces its progress narrations via `provider_specific_fields` instead.

    The resulting logs are tagged `message_type="intermediate_output"` (not
    `"reasoning"`) because these aren't model-internal chain-of-thought —
    they're progress updates the harness emits while it's working (search
    plans, section headings, milestone narrations). The UI renders them
    with a distinct badge so viewers can tell them apart from genuine
    chain-of-thought reasoning from R1 / o-series / Anthropic thinking
    (which arrive on `delta.reasoning_content` and are logged once as
    `"reasoning"` at end-of-call by singleshot_agent).

    The field is kept as `reasoning_content` on the wire because that's
    litellm's generic channel for "non-final text"; only the downstream
    `message_type` tag distinguishes intermediate outputs from reasoning.
    """
    # 取出 chunk 的 choices 列表，缺失时置空列表。
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        # 无 choices：非增量内容块，返回空串。
        return ""
    # 取第一个 choice 的 delta（流式增量部分）。
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        # 无 delta：返回空串。
        return ""
    # 取 provider 专有字段（provider_specific_fields）。
    psf = getattr(delta, "provider_specific_fields", None)
    if isinstance(psf, dict):
        # 从专有字段中读取 reasoning_content。
        reasoning = psf.get("reasoning_content")
        if isinstance(reasoning, str):
            # 是字符串则作为中间输出文本返回。
            return reasoning
    # 其余情况均返回空串（无中间输出）。
    return ""


def responses_args_to_completions(extra_args: dict[str, Any]) -> dict[str, Any]:
    """Convert Responses API extra_args to Chat Completions API equivalents.

    The Responses API uses ``{"reasoning": {"effort": "high", ...}}`` while
    Chat Completions uses ``{"reasoning_effort": "high"}`` as a top-level
    param. Sending the Responses-API shape to a Chat-Completions endpoint
    yields ``Unknown parameter: 'reasoning'`` 400s.
    """
    # 先复制除 reasoning 外的所有参数（保持其余参数原样）。
    result = {k: v for k, v in extra_args.items() if k != "reasoning"}
    # 取出 reasoning 配置（Responses API 形态）。
    reasoning = extra_args.get("reasoning")
    if isinstance(reasoning, dict):
        # 取出其中的 effort（推理努力档位）。
        effort = reasoning.get("effort")
        if effort and "reasoning_effort" not in result:
            # 转换为 Chat Completions 的顶层 reasoning_effort 参数。
            result["reasoning_effort"] = effort
    # 返回转换后的参数字典。
    return result


def _split_extra_args(
    extra_args: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split extra_args into (top_level, extra_body) for proxy-safe transport."""
    # 顶层参数字典（直接作为补全调用的 kwargs）。
    top_level: dict[str, Any] = {}
    # extra_body 参数字典（透传给代理，绕过白名单过滤）。
    extra_body: dict[str, Any] = {}
    # 遍历所有额外参数按键分发。
    for k, v in extra_args.items():
        if k in _EXTRA_BODY_PASSTHROUGH_KEYS:
            # 命中透传白名单：放入 extra_body。
            extra_body[k] = v
        else:
            # 其余参数放顶层。
            top_level[k] = v
    # 返回两组参数。
    return top_level, extra_body


class CacheControlAllowlist(StrEnum):
    """Targets where we explicitly attach Anthropic-style ``cache_control``.

    Anthropic-family models *require* an explicit ``cache_control`` field to
    enable prompt caching. Every other major provider Studio routes to
    (OpenAI, Gemini direct, Vertex Gemini, OpenRouter) caches input prompts
    automatically — adding ``cache_control`` for those paths would be
    noise without benefit, so we don't bother.

    Members may be either:
    - A LiteLLM provider prefix (matched against the ``provider/`` segment
      of the model arg): enables caching for every model routed through
      that provider.
    - A full ``provider/model`` string (matched verbatim against the
      ``model`` argument): enables caching for that specific model only.
    """

    # Anthropic 提供商前缀：其下所有模型都启用显式缓存标记。
    ANTHROPIC = "anthropic"
    # Bedrock 提供商前缀（Bedrock 上的 Claude 同样需要 cache_control）。
    BEDROCK = "bedrock"


# 把白名单枚举成员的值收集为不可变集合，便于快速查找。
_CACHE_CONTROL_ALLOWLIST: frozenset[str] = frozenset(
    member.value for member in CacheControlAllowlist
)
# Explicit 5-minute TTL. ``{"type": "ephemeral"}`` alone also defaults to 5m,
# but pinning ``ttl="5m"`` keeps us from silently inheriting any future
# change Anthropic makes to the default and makes the choice auditable
# alongside the (more expensive) ``ttl="1h"`` alternative.
# 显式指定 5 分钟 TTL。仅 {"type": "ephemeral"} 的默认值也是 5m，但显式固定
# ttl="5m" 可避免未来静默继承 Anthropic 对默认值的改动，且与更昂贵的
# ttl="1h" 选项并列时选择依据可审计。
_EPHEMERAL_CACHE: dict[str, str] = {"type": "ephemeral", "ttl": "5m"}


def _is_cache_control_allowed(model: str) -> bool:
    """Whether ``model`` accepts Anthropic-style ``cache_control`` markers.

    The provider is read from the model prefix (``"anthropic"`` from
    ``"anthropic/claude-opus-4-7"``) rather than via
    ``litellm.get_llm_provider`` — that helper short-circuits to
    ``"litellm_proxy"`` whenever ``litellm.use_litellm_proxy`` is set
    (always true in Studio prod), losing the underlying provider info.
    """
    # 从模型名前缀提取提供商（如 "anthropic/claude-opus-4-7" → "anthropic"）。
    provider = model.split("/", 1)[0] if "/" in model else None
    # 提供商在白名单内，或完整模型串精确命中白名单，则允许。
    return provider in _CACHE_CONTROL_ALLOWLIST or model in _CACHE_CONTROL_ALLOWLIST


def _with_cached_system_prompt(
    messages: list[LitellmAnyMessage], model: str
) -> list[LitellmAnyMessage]:
    """Return ``messages`` with the system prompt marked ephemerally cacheable.

    No-op unless the model's provider prefix or the full model string is in
    :class:`CacheControlAllowlist`. On a match, attaches
    ``cache_control={"type": "ephemeral", "ttl": "5m"}`` to the final
    content block of the first system message. Idempotent: a no-op if
    there is no system message, if the system message isn't a dict (system
    messages are dicts in agent contexts), or if the last block already
    carries ``cache_control``.
    """
    # 模型不在白名单：原样返回，不打缓存标记。
    if not _is_cache_control_allowed(model):
        return messages

    # The constructed system messages carry a ``cache_control`` field that
    # OpenAI's TypedDicts (the backbone of LitellmAnyMessage) don't model,
    # even though LiteLLM passes it through to Anthropic. Suppress the
    # return-type complaint at the construction sites only.
    # 构造出的 system 消息携带 cache_control 字段，OpenAI TypedDict（
    # LitellmAnyMessage 的基型）没有建模该字段，尽管 LiteLLM 会原样透传给
    # Anthropic。只在构造点压制返回类型告警。
    for i, msg in enumerate(messages):
        # 跳过非 dict 消息与非 system 角色的消息。
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        # 取出该 system 消息的内容。
        content = msg.get("content")
        if isinstance(content, str) and content:
            # 字符串内容：包装为单个带 cache_control 的文本块。
            block = {"type": "text", "text": content, "cache_control": _EPHEMERAL_CACHE}
            # 构造内容为块列表的新 system 消息。
            new_msg: LitellmAnyMessage = {"role": "system", "content": [block]}  # pyright: ignore[reportAssignmentType]
            # 返回替换了该条消息的新列表（其余消息原样保留）。
            return [*messages[:i], new_msg, *messages[i + 1 :]]
        if isinstance(content, list) and content:
            # 块列表内容：取最后一个块。
            last = content[-1]
            if isinstance(last, dict) and "cache_control" not in last:
                # 最后一块尚无缓存标记：复制该块并附加 cache_control。
                cached_last = {**last, "cache_control": _EPHEMERAL_CACHE}
                # 用更新后的块列表构造新消息。
                new_msg: LitellmAnyMessage = {  # pyright: ignore[reportAssignmentType]
                    **msg,
                    "content": [*content[:-1], cached_last],
                }
                # 返回替换后的新列表。
                return [*messages[:i], new_msg, *messages[i + 1 :]]
        # 找到 system 消息但无需/无法打标记（空内容或已标记）：直接返回。
        return messages
    # 没有 system 消息：原样返回。
    return messages


def _with_cached_tools(
    tools: list[ChatCompletionToolParam], model: str
) -> list[ChatCompletionToolParam]:
    """Return ``tools`` with the last tool marked ephemerally cacheable.

    Anthropic's prompt cache extends from the start of the request through
    the *last* ``cache_control`` marker; placing one on the final tool
    definition caches the entire system+tools prefix together. With Stirrup
    loads of 7+ MCP tools and Opus tool schemas of a few KB each, this is
    typically the largest still-uncached chunk of the per-turn input.

    The marker sits at the top level of the tool dict (alongside ``"type"``
    and ``"function"``), where LiteLLM's Anthropic transformer picks it up
    when converting OpenAI-format tools into Anthropic's native shape.

    No-op unless the provider/model is allowlisted, the list is empty, or
    the last tool already carries ``cache_control``. Idempotent.
    """
    # 模型不在白名单：原样返回。
    if not _is_cache_control_allowed(model):
        return tools
    # 工具列表为空：无操作。
    if not tools:
        return tools
    # 取最后一个工具定义。
    last = tools[-1]
    if not isinstance(last, dict) or last.get("cache_control") is not None:
        # 非 dict 或已带缓存标记（幂等）：原样返回。
        return tools
    # 复制最后一个工具并附加 ephemeral 缓存标记（与 type/function 平级）。
    cached_last = {**last, "cache_control": _EPHEMERAL_CACHE}
    # 返回替换了最后一个工具的新列表。
    return [*tools[:-1], cached_last]  # pyright: ignore[reportReturnType]


def _with_cached_last_message(
    messages: list[LitellmAnyMessage], model: str
) -> list[LitellmAnyMessage]:
    """Return ``messages`` with the most recent message marked ephemerally cacheable.

    Conversation history is append-only across an agent loop, so marking the
    *last* message extends the cached prefix through every prior message.
    On the next turn — when one or two more messages have been appended —
    Anthropic's cache lookup falls back to this marker as the longest
    matching prefix and processes only the new suffix at full price.

    Operates on the last message regardless of role (assistant tool_use,
    tool result, user follow-up). String content is wrapped into a single
    text block; list content has its final block annotated. Pydantic
    ``Message`` instances are skipped — their content is typed without a
    ``cache_control`` field, and the system + tools breakpoints still
    cover the static prefix even when the rolling marker no-ops.

    Idempotent: skips when the target block already carries
    ``cache_control``.
    """
    # 模型不在白名单：原样返回。
    if not _is_cache_control_allowed(model):
        return messages
    # 消息列表为空：无操作。
    if not messages:
        return messages
    # 取最后一条消息。
    last = messages[-1]
    if not isinstance(last, dict):
        # Pydantic Message 形态跳过（其类型未建模 cache_control）。
        return messages

    # 取最后一条消息的内容。
    content = last.get("content")

    if isinstance(content, str):
        if not content:
            # 空字符串内容：无操作。
            return messages
        # 字符串内容：包装为带 cache_control 的单个文本块。
        block = {"type": "text", "text": content, "cache_control": _EPHEMERAL_CACHE}
        # 构造内容为块列表的新消息（其余字段保持不变）。
        new_msg: LitellmAnyMessage = {**last, "content": [block]}  # pyright: ignore[reportAssignmentType]
        # 返回替换最后一条后的新列表。
        return [*messages[:-1], new_msg]

    if isinstance(content, list) and content:
        # 块列表内容：取最后一个块。
        last_block = content[-1]
        if isinstance(last_block, dict) and "cache_control" not in last_block:
            # 最后一块尚无缓存标记：复制并附加 cache_control（幂等）。
            cached_block = {**last_block, "cache_control": _EPHEMERAL_CACHE}
            # 用更新后的块列表构造新消息。
            new_msg_list: LitellmAnyMessage = {  # pyright: ignore[reportAssignmentType]
                **last,
                "content": [*content[:-1], cached_block],
            }
            # 返回替换最后一条后的新列表。
            return [*messages[:-1], new_msg_list]

    # 其余情况（空列表内容、最后一块已标记等）：原样返回。
    return messages


def normalize_assistant_tool_call_content(
    messages: list[LitellmAnyMessage],
) -> list[LitellmAnyMessage]:
    """Ensure assistant messages with tool_calls have non-empty content.

    Some model endpoints (e.g. alabaster/100) return HTTP 500 when an assistant
    message has null or empty string content alongside tool_calls. Replacing it
    with a single space satisfies those APIs without affecting model behaviour.
    """
    # 收集归一化后的消息列表。
    normalized = []
    # 逐条检查消息。
    for msg in messages:
        if isinstance(msg, Message):
            # Pydantic 形态：assistant 消息带 tool_calls 且内容为空时，置为单个空格。
            if msg.role == "assistant" and msg.tool_calls and not msg.content:
                msg = msg.model_copy(update={"content": " "})
        elif (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and not msg.get("content")
        ):
            # dict 形态的同类情况：复制并补一个空格内容。
            msg = {**msg, "content": " "}
        # 加入结果列表（无论是否修改）。
        normalized.append(msg)
    # 返回归一化后的列表。
    return normalized


def _is_context_window_error(e: Exception) -> bool:
    """
    Detect context window exceeded errors that LiteLLM doesn't properly classify.

    Some providers (notably Gemini) return context window errors as BadRequestError
    instead of ContextWindowExceededError. This predicate catches those cases
    by checking the error message content.

    Known error patterns:
    - Gemini: "input token count exceeds the maximum number of tokens allowed"
    - OpenAI: "context_length_exceeded" (usually caught as ContextWindowExceededError)
    - Anthropic: "prompt is too long" (usually caught as ContextWindowExceededError)
    """
    # 把异常消息转小写做子串匹配。
    error_str = str(e).lower()

    # Common patterns indicating context/token limit exceeded
    # 各家表示上下文/token 超限的常见消息模式。
    context_patterns = [
        "token count exceeds",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "maximum number of tokens",
        "prompt is too long",
        "input too long",
        "exceeds the model's maximum context",
        "exceeds the context window",
    ]

    # 任一模式命中即判为上下文超限错误（不可重试，应跳过）。
    return any(pattern in error_str for pattern in context_patterns)


def _is_non_retriable_bad_request(e: Exception) -> bool:
    """
    Detect BadRequestErrors that are deterministic and should NOT be retried.

    These are configuration/validation errors that will always fail regardless
    of retry attempts. Retrying wastes time and resources.

    Note: Patterns must be specific enough to avoid matching transient errors
    like rate limits (e.g., "maximum of 100 requests" should NOT match).
    """
    # 把异常消息转小写。
    error_str = str(e).lower()

    # 确定性失败（重试无益）的 BadRequest 消息模式列表。
    non_retriable_patterns = [
        # Tool count errors - be specific to avoid matching rate limits
        # 工具数量类错误——措辞要具体，避免误匹配限流消息。
        "tools are supported",  # "Maximum of 128 tools are supported"
        "too many tools",
        # Model/auth errors
        # 模型/鉴权类错误。
        "model not found",
        "does not exist",
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "unsupported parameter",
        "unsupported value",
        # OpenAI emits "Unknown parameter: 'foo'" for fields the model
        # endpoint doesn't accept (e.g. Responses-API `reasoning` sent to a
        # Chat-Completions-only model). These are config errors, not
        # transient — retrying just burns the worker for ~12 minutes.
        # OpenAI 对端点不接受的字段报 "Unknown parameter: 'foo'"（如把
        # Responses API 的 reasoning 发给仅支持 Chat Completions 的模型）。
        # 这些是配置错误而非瞬时错误——重试只会白白占用 worker 约 12 分钟。
        "unknown parameter",
        # Model capability mismatch
        # 模型能力不匹配（不支持多模态输入）。
        "does not support multimodal",
        "is not a multimodal model",
    ]

    # 任一模式命中即判为不可重试的确定性错误。
    return any(pattern in error_str for pattern in non_retriable_patterns)


def _should_skip_retry(e: Exception) -> bool:
    """Combined check for all non-retriable errors."""
    # 合并判定：上下文超限或确定性 BadRequest 都应跳过重试。
    return _is_context_window_error(e) or _is_non_retriable_bad_request(e)


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_should_skip_retry,
)
async def generate_response(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[ChatCompletionToolParam],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> ModelResponse:
    """
    Generate a response from the LLM with retry logic.

    Args:
        model: The model identifier to use
        messages: The conversation messages (input AllMessageValues or output Message)
        tools: Available tools for the model to call
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments to pass to the completion call
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The model response
    """
    # 先把 Responses 形态的额外参数转换为补全形态，再拆成顶层参数与 extra_body。
    top_level_extra, extra_body = _split_extra_args(
        responses_args_to_completions(extra_args)
    )
    # 依次给 system 提示与最后一条消息打缓存标记（Anthropic 提示缓存断点）。
    cached_messages = _with_cached_last_message(
        _with_cached_system_prompt(messages, model), model
    )
    # 组装补全调用的基础 kwargs。
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": cached_messages,
        "timeout": llm_response_timeout,
        # Outer @with_retry owns retries — pin num_retries=0 so the LiteLLM
        # proxy doesn't retry on top, compounding caller × proxy attempts.
        # Caller's top_level_extra wins via the spread (later key overrides).
        # 外层 @with_retry 全权负责重试——把 num_retries 固定为 0，避免
        # LiteLLM 代理再叠加重试，造成调用方×代理的尝试次数相乘。展开的
        # 顶层额外参数在后面可覆盖此默认（后键优先）。
        "num_retries": 0,
        **top_level_extra,
    }

    if tools:
        # 有工具时：给最后一个工具打缓存标记后传入。
        kwargs["tools"] = _with_cached_tools(tools, model)

    # If LiteLLM proxy is configured, route completions through it and add tags.
    # Mirrors call_responses_api: rely on explicit api_base/api_key — some SDK /
    # model paths ignore litellm.use_litellm_proxy for acompletion(), which would
    # send requests direct-to-provider (no proxy logging / spend attribution).
    #
    # Tags are sent two ways for redundancy:
    #   1. `metadata.tags` — the documented path. Lands in StandardLoggingPayload
    #      iff the proxy key has admin metadata `allow_client_tags: true`.
    #      In litellm 1.83.10+ a default-deny strip wipes this otherwise
    #      (BerriAI/litellm@0e62add).
    #   2. `extra_headers` — harvested by the proxy via
    #      `litellm_settings.extra_spend_tag_headers` in config.yaml. Bypasses
    #      the strip because header tags are appended in `_get_request_tags`
    #      AFTER the strip runs (litellm_logging.py:5249-5272). This is what
    #      actually makes tags flow today.
    # Docs: https://docs.litellm.ai/docs/proxy/cost_tracking
    # 配置了代理时：把补全请求路由到代理并附加标签。与 call_responses_api 保持
    # 一致：依赖显式 api_base/api_key——某些 SDK/模型路径会忽略
    # litellm.use_litellm_proxy，导致请求直连提供商（丢失代理日志与花费归因）。
    # 标签通过两种冗余方式发送：1) metadata.tags（文档路径，需代理密钥具备
    # allow_client_tags 管理元数据，litellm 1.83.10+ 默认拒绝时会剥离）；
    # 2) extra_headers（代理经 config.yaml 的 extra_spend_tag_headers 收集，
    # 在剥离之后才追加，是当前真正生效的通道）。
    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        # 显式指定代理地址。
        kwargs.setdefault("api_base", settings.LITELLM_PROXY_API_BASE)
        # 显式指定代理密钥。
        kwargs.setdefault("api_key", settings.LITELLM_PROXY_API_KEY)
        # Tags ship via HTTP headers only — litellm 1.83.10 added a
        # default-deny strip that wipes body-supplied `metadata.tags` unless
        # the key has admin metadata `allow_client_tags: true` (we don't).
        # Header-derived tags bypass the strip via the proxy's
        # `extra_spend_tag_headers` allowlist (see rl-studio/infra/litellm/config.yaml).
        # call_id is stable across `with_retry` attempts of the same logical
        # call; attempt is 1-indexed and increments per retry. Together they
        # let Datadog distinguish unique logical calls from retry attempts
        # (e.g. unique_count(@call_id) vs count(*) for the 429 rate).
        # 标签只经 HTTP 头发送——litellm 1.83.10 的默认拒绝剥离会清掉请求体
        # 提供的 metadata.tags（除非密钥带 allow_client_tags 管理元数据，
        # 我们没有）。头派生标签经代理的 extra_spend_tag_headers 白名单绕过
        # 剥离。call_id 在同一逻辑调用的各次重试间保持稳定；attempt 从 1 开始
        # 每次重试递增。二者配合让 Datadog 能区分唯一逻辑调用与重试尝试。
        call_id = llm_call_id_ctx.get()
        # 有 call_id 时才读取尝试序号。
        attempt_num = llm_attempt_ctx.get() if call_id else None
        # 复制调用方已设置的 extra_headers（若有）。
        hdrs = dict(kwargs.get("extra_headers") or {})
        # 标注服务来源为 trajectory。
        hdrs.setdefault("service", "trajectory")
        if trajectory_id:
            # 有轨迹 id：附加到头中用于追踪。
            hdrs.setdefault("trajectory_id", trajectory_id)
        if call_id:
            # 附加逻辑调用 id 与尝试序号（字符串形式）。
            hdrs.setdefault("call_id", call_id)
            hdrs.setdefault("attempt", str(attempt_num))
        # 把组装好的头写回 kwargs。
        kwargs["extra_headers"] = hdrs

    if extra_body:
        # 存在需要透传的参数：挂到 extra_body。
        kwargs["extra_body"] = extra_body

    if stream:
        # 流式模式：开启 stream 并要求在流选项中包含用量统计。
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        # 发起异步流式补全，拿到流迭代器。
        stream_iter: Any = await acompletion(**kwargs)
        # 收集所有 chunk 的列表。
        chunks: list[ModelResponse] = []

        # 逐个消费流式增量。
        async for chunk in stream_iter:
            # 保存 chunk 供最终重组完整响应。
            chunks.append(chunk)
            # Per-delta progress logs only for external-agent-harness
            # providers (reasoning surfaced via `provider_specific_fields`).
            # These aren't model-internal chain-of-thought — they're
            # narrations the harness emits while it works (e.g. Gemini Deep
            # Research's thought summaries: "Origins and Architectural
            # Foundations — I am beginning by analyzing..."). Tag them as
            # `intermediate_output` so the UI can render them distinctly
            # from genuine `reasoning` logs from models like DeepSeek-R1
            # or the o-series. Native-reasoning streams go through
            # `delta.reasoning_content`, which stream_chunk_builder
            # aggregates and singleshot_agent logs once at end-of-call as
            # `reasoning` — no per-delta duplication from this path.
            # 仅对外部 agent 工具链提供商做逐增量进度日志（推理内容经
            # provider_specific_fields 浮出）。这些不是模型内部思维链，而是
            # 工具链工作时发出的叙述（如 Gemini Deep Research 的思考摘要）。
            # 标记为 intermediate_output 以便 UI 与 DeepSeek-R1 / o 系列的
            # 真正 reasoning 日志区分渲染。原生推理流走 delta.reasoning_content，
            # 由 stream_chunk_builder 聚合、singleshot_agent 在调用结束时
            # 统一记一条 reasoning——本路径不逐条重复。
            intermediate_output = _extract_psf_reasoning_delta(chunk)
            if intermediate_output:
                # 有中间输出：以 intermediate_output 类型打日志。
                logger.bind(message_type="intermediate_output").info(
                    intermediate_output
                )

        # 用 LiteLLM 的流重组器把全部 chunk 重建为完整响应。
        rebuilt = litellm.stream_chunk_builder(chunks, messages=messages)
        if rebuilt is None:
            # 流为空导致重组失败：抛运行时错误。
            raise RuntimeError("stream_chunk_builder returned None — empty stream")
        # 校验并转换为标准 ModelResponse 返回。
        return ModelResponse.model_validate(rebuilt)

    # 非流式：直接发起一次异步补全调用。
    response = await acompletion(**kwargs)
    # 校验并转换为标准 ModelResponse 返回。
    return ModelResponse.model_validate(response)


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_should_skip_retry,
)
async def call_responses_api(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[dict[str, Any]],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> Any:
    """
    Generate a response using a provider's Responses API (e.g., web search) with retry logic.

    Uses litellm.aresponses() which is the native async version.

    Args:
        model: The model identifier to use (e.g., 'openai/gpt-4o')
        messages: The conversation messages
        tools: Tools for web search (e.g., [{"type": "web_search"}])
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments (reasoning, etc.)
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The OpenAI responses API response object
    """
    # 拆分额外参数为顶层参数与 extra_body（Responses API 原生形态，无需转换）。
    top_level_extra, extra_body = _split_extra_args(extra_args)
    # 组装 Responses API 调用的基础 kwargs。
    kwargs: dict[str, Any] = {
        "model": model,
        "input": messages,
        "tools": tools,
        "timeout": llm_response_timeout,
        # Outer @with_retry owns retries — pin num_retries=0 so the LiteLLM
        # proxy doesn't retry on top, compounding caller × proxy attempts.
        # Caller's top_level_extra wins via the spread (later key overrides).
        # 外层 @with_retry 全权负责重试——固定 num_retries=0 防止代理叠加重试；
        # 展开的顶层额外参数可覆盖默认值（后键优先）。
        "num_retries": 0,
        **top_level_extra,
    }

    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        # 配置了代理：显式指定代理地址与密钥（理由同 generate_response）。
        kwargs["api_base"] = settings.LITELLM_PROXY_API_BASE
        kwargs["api_key"] = settings.LITELLM_PROXY_API_KEY
        # Tags ship via headers only — see generate_response. metadata.tags
        # is stripped by litellm 1.83.10's default-deny.
        # 标签只经头发送——见 generate_response 的说明；metadata.tags 会被
        # litellm 1.83.10 的默认拒绝剥离。
        call_id = llm_call_id_ctx.get()
        # 有 call_id 时读取尝试序号。
        attempt_num = llm_attempt_ctx.get() if call_id else None
        # 复制调用方已有的 extra_headers。
        hdrs = dict(kwargs.get("extra_headers") or {})
        # 标注服务来源。
        hdrs.setdefault("service", "trajectory")
        if trajectory_id:
            # 附加轨迹 id。
            hdrs.setdefault("trajectory_id", trajectory_id)
        if call_id:
            # 附加逻辑调用 id 与尝试序号。
            hdrs.setdefault("call_id", call_id)
            hdrs.setdefault("attempt", str(attempt_num))
        # 写回 kwargs。
        kwargs["extra_headers"] = hdrs

    if extra_body:
        # 需要透传的参数挂到 extra_body。
        kwargs["extra_body"] = extra_body

    if stream:
        # 流式模式：开启 stream 标志。
        kwargs["stream"] = True
        # 发起异步流式 Responses 调用。
        stream_iter: Any = await aresponses(**kwargs)
        # 保存最终 completed 事件的响应对象。
        completed_response = None
        # 消费事件流，寻找 response.completed 终止事件。
        async for event in stream_iter:
            if getattr(event, "type", None) == "response.completed":
                # 拿到完成事件携带的完整响应。
                completed_response = getattr(event, "response", None)
        if completed_response is None:
            # 未收到完成事件：流异常中断，抛错。
            raise RuntimeError(
                "No response.completed event received from Responses API stream"
            )
        # 返回完整响应对象。
        return completed_response

    # 非流式：直接调用一次异步 Responses API。
    response = await aresponses(**kwargs)
    # 返回响应对象。
    return response

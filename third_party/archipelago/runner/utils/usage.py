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

"""Token usage tracking for agent LLM calls."""

# 导入 typing.Any：通用类型注解。
from typing import Any

# 导入 LiteLLM 的 ModelResponse：track 方法的入参类型。
from litellm.files.main import ModelResponse


def _coerce_int(value: Any) -> int:
    """Defensively coerce a usage attribute to a non-negative int.

    LiteLLM occasionally leaves details fields as None, strings, or floats
    depending on the upstream provider's shape, so we normalize here.
    """
    if value is None:
        # None 一律归零。
        return 0
    try:
        # 尝试转 int；非正数也归零（token 数不可能为负）。
        return int(value) if int(value) > 0 else 0
    except (TypeError, ValueError):
        # 无法转换（如异常字符串）：归零。
        return 0


def _get_attr_or_item(obj: Any, key: str) -> Any:
    """Read a key from either a pydantic model / dataclass or a plain dict."""
    if obj is None:
        # 对象为空：返回 None。
        return None
    if isinstance(obj, dict):
        # dict 形态：按键取值。
        return obj.get(key)
    # 对象形态：按属性取值（缺失返回 None）。
    return getattr(obj, key, None)


class UsageTracker:
    """Accumulates token usage across multiple LLM calls during agent execution."""

    def __init__(self) -> None:
        # 累计提示（输入）token 数。
        self.prompt_tokens: int = 0
        # 累计补全（输出）token 数。
        self.completion_tokens: int = 0
        # 最后一次调用（最终答案）的输出 token 数。
        self.final_answer_tokens: int = 0
        # 历史调用中单次最大提示 token 数（反映上下文峰值）。
        self.max_prompt_tokens: int = 0
        # 上下文压缩（compaction）触发次数。
        self.compaction_count: int = 0
        # Provider-specific detail fields. LiteLLM normalizes these onto
        # `usage.completion_tokens_details` / `usage.prompt_tokens_details`,
        # so we extract them once here and let downstream consumers persist
        # them alongside the basic counts.
        # 提供商专有明细字段。LiteLLM 会把它们归一化到 completion/prompt
        # 的 tokens_details 上，这里统一提取一次，供下游随基础计数一并持久化。
        # 推理（thinking）token 数。
        self.reasoning_tokens: int = 0
        # 缓存命中的提示 token 数。
        self.cached_tokens: int = 0
        # 写入缓存新产生的 token 数。
        self.cache_creation_tokens: int = 0
        # Per-call log: one entry per LLM call in order of execution.
        # 逐调用日志：按执行顺序为每次 LLM 调用记一条。
        self.call_log: list[dict[str, int]] = []

    def track_compaction(self) -> None:
        """Increment compaction count when a context summarization LLM call fires."""
        # 上下文摘要压缩调用发生时累加计数。
        self.compaction_count += 1

    def track(self, response: ModelResponse) -> None:
        """Extract and accumulate usage from a ModelResponse."""
        # 取出响应中的 usage 对象。
        usage = getattr(response, "usage", None)
        if usage is None:
            # 无用量信息：直接返回。
            return

        # 防御式提取本次调用的输入/输出 token 数。
        call_prompt_tokens = _coerce_int(getattr(usage, "prompt_tokens", 0))
        call_completion_tokens = _coerce_int(getattr(usage, "completion_tokens", 0))

        # 累加到总量。
        self.prompt_tokens += call_prompt_tokens
        self.completion_tokens += call_completion_tokens
        # 记录最近一次输出 token 数（即当前最终答案的规模）。
        self.final_answer_tokens = call_completion_tokens
        # 更新单次输入 token 的峰值。
        self.max_prompt_tokens = max(self.max_prompt_tokens, call_prompt_tokens)

        # 取补全 token 明细（含推理 token）。
        completion_details = getattr(usage, "completion_tokens_details", None)
        call_reasoning_tokens = _coerce_int(
            _get_attr_or_item(completion_details, "reasoning_tokens")
        )
        # 累加推理 token。
        self.reasoning_tokens += call_reasoning_tokens

        # 取提示 token 明细（含缓存相关字段）。
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        # LiteLLM normalizes Anthropic's `cache_read_input_tokens` →
        # `prompt_tokens_details.cached_tokens` and
        # `cache_creation_input_tokens` →
        # `prompt_tokens_details.cache_creation_tokens`. We also fall back to
        # the flat Anthropic-shaped attributes in case a backend hasn't been
        # routed through the normalizer.
        # LiteLLM 把 Anthropic 的 cache_read_input_tokens 归一化为
        # cached_tokens、cache_creation_input_tokens 归一化为
        # cache_creation_tokens；同时回退到扁平的 Anthropic 形态字段，
        # 兜底未经归一化的后端。
        cached = _get_attr_or_item(prompt_details, "cached_tokens")
        if not cached:
            cached = getattr(usage, "cache_read_input_tokens", None)
        # 防御式转换后累加缓存命中 token。
        call_cached_tokens = _coerce_int(cached)
        self.cached_tokens += call_cached_tokens

        # 同理提取缓存写入 token。
        cache_creation = _get_attr_or_item(prompt_details, "cache_creation_tokens")
        if not cache_creation:
            cache_creation = getattr(usage, "cache_creation_input_tokens", None)
        call_cache_creation_tokens = _coerce_int(cache_creation)
        self.cache_creation_tokens += call_cache_creation_tokens

        # 为本次调用追加一条逐调用用量记录。
        self.call_log.append(
            {
                "prompt_tokens": call_prompt_tokens,
                "completion_tokens": call_completion_tokens,
                "total_tokens": call_prompt_tokens + call_completion_tokens,
                "reasoning_tokens": call_reasoning_tokens,
                "cached_tokens": call_cached_tokens,
                "cache_creation_tokens": call_cache_creation_tokens,
            }
        )

    def track_from_dict(self, response_dict: dict[str, Any]) -> None:
        """Extract and accumulate usage from a response dictionary (e.g., Responses API).

        Handles both OpenAI Responses API format and standard completion format.
        """
        # 从字典响应中取出 usage。
        usage = response_dict.get("usage")
        if usage is None:
            # 无用量信息：直接返回。
            return

        # 兼容两种命名：prompt_tokens（补全格式）/ input_tokens（Responses 格式）。
        prompt_tokens = _coerce_int(
            usage.get("prompt_tokens") or usage.get("input_tokens")
        )
        completion_tokens = _coerce_int(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )

        # 累加到总量并更新峰值/最终答案规模。
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.final_answer_tokens = completion_tokens
        self.max_prompt_tokens = max(self.max_prompt_tokens, prompt_tokens)

        # 取补全明细。
        completion_details = usage.get("completion_tokens_details") or {}
        # Responses API exposes reasoning tokens as `output_tokens_details.reasoning_tokens`.
        # Responses API 把推理 token 放在 output_tokens_details.reasoning_tokens。
        output_details = usage.get("output_tokens_details") or {}
        call_reasoning_tokens = _coerce_int(
            completion_details.get("reasoning_tokens")
            or output_details.get("reasoning_tokens")
        )
        # 累加推理 token。
        self.reasoning_tokens += call_reasoning_tokens

        # 取提示明细（补全格式与 Responses 格式两套键）。
        prompt_details = usage.get("prompt_tokens_details") or {}
        input_details = usage.get("input_tokens_details") or {}
        # 依次回退：prompt_details → input_details → 扁平 Anthropic 字段。
        cached = (
            prompt_details.get("cached_tokens")
            or input_details.get("cached_tokens")
            or usage.get("cache_read_input_tokens")
        )
        # 防御式转换后累加缓存命中 token。
        call_cached_tokens = _coerce_int(cached)
        self.cached_tokens += call_cached_tokens

        # 同理提取缓存写入 token（两处回退）。
        cache_creation = prompt_details.get("cache_creation_tokens") or usage.get(
            "cache_creation_input_tokens"
        )
        call_cache_creation_tokens = _coerce_int(cache_creation)
        self.cache_creation_tokens += call_cache_creation_tokens

        # 为本次调用追加一条逐调用用量记录。
        self.call_log.append(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "reasoning_tokens": call_reasoning_tokens,
                "cached_tokens": call_cached_tokens,
                "cache_creation_tokens": call_cache_creation_tokens,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return accumulated usage as a dict for AgentTrajectoryOutput."""
        # 汇总全部累计指标与逐调用日志为字典，供轨迹输出使用。
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "final_answer_tokens": self.final_answer_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "compaction_count": self.compaction_count,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "call_log": self.call_log,
        }

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

"""
Tool result processing for handling large outputs.

Simple head_tail truncation - reliable and predictable.
"""

# 导入 Any：用于内容块列表元素的宽泛类型注解。
from typing import Any

# 导入 litellm 的 token 计数工具，用于按模型分词器估算文本 token 数。
from litellm import token_counter
# 导入 loguru 日志器，用于记录截断告警/错误（附加 message_type 字段便于过滤）。
from loguru import logger

# 导入输入消息类型定义（TypedDict 形式的对话消息）。
from runner.agents.models import LitellmInputMessage

# Defaults for head_tail truncation
# With ReSum context summarization, we can afford larger results
# 头尾截断（head_tail）的默认参数：单条工具结果超过约 24000 token 才截断；
# 由于有 ReSum 上下文摘要机制兜底，可以容忍较大的结果进入上下文。
MAX_RESULT_TOKENS = 24000  # ~24k tokens before truncation
# 截断时保留开头的字符数（头部 2 万字符）。
HEAD_CHARS = 20000  # Keep first 20k chars
# 截断时保留结尾的字符数（尾部 5 千字符）。
TAIL_CHARS = 5000  # Keep last 5k chars

# Absolute maximum - if result exceeds this even after truncation, return error
# This prevents absurdly large results from ever being added to context
# 绝对上限：截断后仍超限就返回错误文本，防止超大结果进入模型上下文。
ABSOLUTE_MAX_CHARS = 100000


def _estimate_tokens(model: str, text: str) -> int:
    """Estimate token count for text content."""
    # 优先用 litellm 按模型分词器精确计数。
    try:
        return token_counter(model=model, text=text)
    except Exception:
        # 计数失败（如模型未知）时按"4 字符 ≈ 1 token"粗略估算。
        return len(text) // 4


def _truncate_text(text: str, model: str) -> str | None:
    """
    Truncate text if it exceeds limits. Returns truncated text or None if no change needed.

    Uses head_tail strategy: keep first HEAD_CHARS and last TAIL_CHARS.
    """
    # Check for absurdly large results first
    # 先拦截极端巨大的结果（超过绝对上限的两倍），直接报错而不做头尾拼接。
    if len(text) > ABSOLUTE_MAX_CHARS * 2:
        logger.bind(message_type="tool_result").error(
            f"Tool result is extremely large ({len(text):,} chars), truncating"
        )
        # 返回错误提示文本，引导调用方改用更具体的查询。
        return (
            f"Error: Tool returned extremely large output ({len(text):,} characters). "
            f"This exceeds the maximum allowed size. "
            f"Try a more specific query or break down the request."
        )

    # 估算当前文本的 token 数。
    tokens = _estimate_tokens(model, text)
    # 未超过 token 上限则无需截断，返回 None 表示原样保留。
    if tokens <= MAX_RESULT_TOKENS:
        return None  # No truncation needed

    # 超限则记录告警日志（token 数与上限值）。
    logger.bind(message_type="tool_result").warning(
        f"Tool result is too large ({tokens} tokens > {MAX_RESULT_TOKENS}), truncating"
    )

    # Head-tail truncation
    # 头尾截断的第二次检查：字符总长未超过"头+尾"之和时，拼接后不会更省，
    # 直接原样返回（内容本身没问题）。
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return None  # Content is fine as-is

    # 取头部前 2 万字符。
    head = text[:HEAD_CHARS]
    # 取尾部后 5 千字符（负索引切片）。
    tail = text[-TAIL_CHARS:]
    # 计算被省略的中间字符数，用于提示信息。
    omitted = len(text) - HEAD_CHARS - TAIL_CHARS
    # 拼接：头部 + 省略说明 + 尾部，形成截断后的文本。
    processed = (
        f"{head}\n\n"
        f"[... {omitted:,} characters omitted. "
        f"Use more specific queries to access full data. ...]\n\n"
        f"{tail}"
    )

    # Final safety check - ensure truncated result is within absolute max
    # 最终安全检查：截断结果仍超绝对上限时返回错误文本而不是放大结果。
    if len(processed) > ABSOLUTE_MAX_CHARS:
        logger.bind(message_type="tool_result").error(
            f"Truncated result still too large ({len(processed):,} chars)"
        )
        return (
            f"Error: Tool output too large even after truncation. "
            f"Original: {len(text):,} chars. Try a more specific query."
        )

    # 返回截断处理后的文本。
    return processed


def _truncate_content_list(content: list[Any], model: str) -> None:
    """Truncate text blocks within a content list. Mutates in place."""
    # 遍历内容列表中的每个块。
    for item in content:
        # 非字典元素无法按内容块处理，跳过。
        if not isinstance(item, dict):
            continue
        # 只处理文本块（type == "text"）。
        if item.get("type") != "text":
            continue  # Preserve images and other types

        # 取出文本块的文本内容（缺省为空串）。
        text = item.get("text", "")
        # 文本不是字符串（数据异常）时跳过，避免截断逻辑报错。
        if not isinstance(text, str):
            continue

        # 对该文本执行截断判断。
        truncated = _truncate_text(text, model)
        # 返回非 None 说明发生了截断，就地替换块内文本。
        if truncated is not None:
            item["text"] = truncated


def truncate_tool_messages(
    messages: list[LitellmInputMessage],
    model: str,
) -> None:
    """
    Truncate text content in tool messages, preserving everything else (images, etc).

    Handles both:
    - Dict messages with list content: [{"type": "text", "text": "..."}]
    - Dict messages with string content (legacy)

    Mutates messages in place. Uses head_tail strategy for truncation.
    """
    # 遍历所有消息，就地截断其中的工具结果文本。
    for msg in messages:
        # 取消息的 content 字段。
        content = msg.get("content")

        if isinstance(content, list):
            # New format: content is array of blocks
            # 新格式：内容是块数组，交给块级截断处理（保留图片等非文本块）。
            _truncate_content_list(content, model)

        elif isinstance(content, str):
            # Legacy format: content is string
            # 旧格式：内容是纯字符串，直接做文本截断。
            truncated = _truncate_text(content, model)
            # 发生截断时就地替换消息内容。
            if truncated is not None:
                msg["content"] = truncated

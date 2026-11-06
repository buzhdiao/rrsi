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

"""Shared tool output truncation utilities.

Used by agents that receive large MCP tool responses (e.g. web pages)
and need to truncate them to avoid filling the context window.
"""

# 导入 typing.Any：消息内容字段的通用类型。
from typing import Any

# 导入消息联合类型：dict 或 Pydantic Message 形态。
from runner.agents.models import LitellmAnyMessage


def truncate_tool_output(
    content: str, max_lines: int = 200, max_chars: int = 32768
) -> str:
    """Truncate content at whichever limit is hit first: max_lines or max_chars."""
    # 空内容直接返回。
    if not content:
        return content

    # 按换行切分为行列表。
    lines = content.split("\n")
    # 统计总行数。
    total_lines = len(lines)

    # Check if we need to truncate by lines
    # 先检查是否需要按行数截断。
    if total_lines > max_lines:
        # 只保留前 max_lines 行并重新拼接。
        truncated_content = "\n".join(lines[:max_lines])
        # Also check char limit on the line-truncated content
        # 对按行截断后的内容再检查字符数上限。
        if len(truncated_content) > max_chars:
            # 超字符上限：再按字符截断。
            truncated_content = truncated_content[:max_chars]
            # 统计字符截断后保留的行数。
            kept_lines = truncated_content.count("\n") + 1
            # 计算被隐藏的行数。
            hidden_lines = total_lines - kept_lines
        else:
            # 未超字符上限：隐藏行数即总行数减去保留行数。
            hidden_lines = total_lines - max_lines
        # 拼接截断说明后返回。
        return truncated_content + f"\n... (truncated {hidden_lines} lines) ..."

    # Check if we need to truncate by chars
    # 再检查是否需要按字符数截断。
    if len(content) > max_chars:
        # 只保留前 max_chars 个字符。
        truncated_content = content[:max_chars]
        # 统计截断后保留的行数。
        kept_lines = truncated_content.count("\n") + 1
        # 计算被隐藏的行数。
        hidden_lines = total_lines - kept_lines

        if hidden_lines > 0:
            # 有整行被隐藏：按行数提示。
            suffix = f"\n... (truncated {hidden_lines} lines) ..."
        else:
            # 只截断了最后一行的部分字符：按字符数提示。
            hidden_chars = len(content) - len(truncated_content)
            suffix = f"\n... (truncated {hidden_chars} chars) ..."

        # 拼接截断说明后返回。
        return truncated_content + suffix

    # 未触及任何上限：原样返回。
    return content


def truncate_tool_message(
    msg: LitellmAnyMessage,
    max_lines: int = 200,
    max_chars: int = 32768,
) -> LitellmAnyMessage:
    """Apply truncation to a tool message's content.

    Handles both plain string content and structured content blocks
    (list[dict] with {"type": "text", "text": "..."} entries).
    """
    # 只处理 dict 形态且角色为 tool 的消息，其余原样返回。
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return msg

    # 取出消息内容。
    content: Any = msg.get("content")
    if isinstance(content, str):
        # 字符串内容：直接应用输出截断。
        msg["content"] = truncate_tool_output(content, max_lines, max_chars)
    elif isinstance(content, list):
        # 块列表内容：逐块处理。
        truncated_parts: list[dict[str, str]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                # 文本块：截断其中的文本并重建块。
                truncated_text = truncate_tool_output(
                    str(part.get("text", "")), max_lines, max_chars
                )
                truncated_parts.append({"type": "text", "text": truncated_text})
            else:
                # 非文本块：原样保留。
                truncated_parts.append(part)  # pyright: ignore[reportArgumentType]
        # 写回截断后的块列表。
        msg["content"] = truncated_parts  # pyright: ignore[reportGeneralTypeIssues]
    # 返回（可能已就地截断的）消息。
    return msg

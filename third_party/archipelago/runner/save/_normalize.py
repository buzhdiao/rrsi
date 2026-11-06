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

"""Message normalization for trajectory reporting.

LiteLLM's ``ChatCompletionToolMessage`` schema types ``content`` as
``str | Iterable[ChatCompletionTextObject]`` — i.e. tool messages may only hold
plain text blocks. In practice, models like ``claude-opus-4-7`` can produce
tool results containing ``image_url`` blocks (e.g. from ``code_exec`` returning
a matplotlib PNG). When the agent runner then calls ``model_dump(mode="json")``
on its ``AgentTrajectoryOutput``, Pydantic iterates the content iterator and
rejects image blocks with ``ValidationError: Input should be 'text'``, which
manifests as a ``PydanticSerializationError`` in the webhook.

This module collapses any non-text blocks inside tool-message content into a
text placeholder so the trajectory can still be reported. We preserve the fact
that an image was present — the grader only needs the agent's final text
answer, and keeping a human-readable marker leaves the trajectory auditable.
Drop this shim once LiteLLM's tool-message schema accepts image/audio parts.
"""

# 延迟注解（PEP 563）：让类型注解以字符串形式存储，避免运行时导入开销与循环导入问题。
from __future__ import annotations

# 导入 copy 模块，用于对 dict 形态的消息做浅拷贝，避免原地修改原始消息。
import copy
# 导入 typing.Any，作为消息内容的通用类型注解。
from typing import Any

# 导入 LiteLLM 的 Message 类型：用于在运行时用 isinstance 区分 Pydantic 对象形态的消息。
from litellm.types.utils import Message

# The message union type (``LitellmAnyMessage``) lives in ``runner.agents.models``,
# which imports this module for its ``AgentTrajectoryOutput.messages``
# field_validator. Avoiding the reverse import — even under ``TYPE_CHECKING`` —
# keeps the module dependency graph acyclic, which some type checkers flag on.
# Annotations use ``Any`` below; runtime dispatch already handles both shapes
# (dict-form TypedDict messages and Pydantic ``Message`` objects).


def _block_placeholder(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a non-text content block into a text placeholder."""
    # 取出内容块类型（如 image_url / audio 等），取不到时用 "unknown" 兜底。
    block_type = block.get("type") or "unknown"
    # 初始化占位符标签为块类型名。
    label = block_type

    # For the common OpenAI image_url shape, note the format if we can infer it.
    # 针对最常见的 OpenAI image_url 块，尝试推断图片格式并附加到标签中。
    if block_type == "image_url":
        # 取出 image_url 字段（可能是 dict 也可能是纯字符串）。
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            # dict 形态：从嵌套结构中取出 url 字符串。
            url = image_url.get("url") or ""
        elif isinstance(image_url, str):
            # 字符串形态：直接使用。
            url = image_url
        else:
            # 其他形态：置空，跳过格式推断。
            url = ""
        if url.startswith("data:") and ";" in url:
            # data URL 形如 data:<mime>;base64,...，按 "data:" 和 ";" 切分提取 MIME 类型。
            mime = url.split(":", 1)[1].split(";", 1)[0]
            # 把 MIME 类型拼进标签，例如 "image_url image/png"，保留可审计信息。
            label = f"image_url {mime}"

    # 返回纯文本占位块：记录"该类型内容已从轨迹报告中省略"，保证轨迹仍可序列化。
    return {"type": "text", "text": f"[{label} content elided from trajectory report]"}


def _normalize_tool_content(content: Any) -> Any:
    """Replace any non-text blocks in tool-message content with text placeholders."""
    # 只处理块列表形态的内容；字符串或 None 等直接原样返回。
    if not isinstance(content, list):
        return content

    # 新建归一化后的块列表，避免修改原始列表。
    normalized: list[Any] = []
    # 标记是否替换过非文本块，用于决定是否返回新列表。
    touched = False
    # 逐个遍历内容块。
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type")
            and block.get("type") != "text"
        ):
            # 命中"非文本块"（dict 且 type 存在且不是 text）：替换为文本占位块。
            normalized.append(_block_placeholder(block))
            # 标记已发生替换。
            touched = True
        else:
            # 文本块或非 dict 块：原样保留。
            normalized.append(block)
    # 只有真的替换过才返回新列表，否则返回原对象（调用方可用 is 判断是否有改动）。
    return normalized if touched else content


def normalize_message_for_report(msg: Any) -> Any:
    """Return a copy of ``msg`` with tool-message content stripped of non-text blocks.

    Non-tool messages and tool messages that already contain only text are
    returned unchanged (and un-copied). ``msg`` is typed ``Any`` to accept any
    of LiteLLM's message shapes (``AllMessageValues`` TypedDicts, Pydantic
    ``Message`` objects) without a direct import of the union type (see
    module-level note on the import cycle).
    """
    # 分支一：Pydantic Message 对象形态。
    if isinstance(msg, Message):
        # 非 tool 角色的消息不含图片块问题，直接原样返回。
        if msg.role != "tool":
            return msg
        # 对 tool 消息内容做归一化，剔除非文本块。
        normalized_content = _normalize_tool_content(msg.content)
        # 返回对象与原内容是同一对象，说明没有发生替换，无需拷贝。
        if normalized_content is msg.content:
            return msg
        # Message is a Pydantic model; use model_copy to avoid in-place mutation.
        # Message 是 Pydantic 模型；用 model_copy 生成副本并更新 content，避免原地修改。
        return msg.model_copy(update={"content": normalized_content})

    # Dict-shaped messages (LiteLLM TypedDicts).
    # 分支二：dict 形态的消息（LiteLLM 的 TypedDict）。
    if not isinstance(msg, dict):
        # 既不是 Pydantic Message 也不是 dict：无法识别，原样返回。
        return msg
    # 非 tool 角色的 dict 消息直接原样返回。
    if msg.get("role") != "tool":
        return msg
    # 对 tool 消息内容做归一化。
    normalized_content = _normalize_tool_content(msg.get("content"))
    # 内容对象未变（同一对象）说明没有替换发生，直接返回原消息。
    if normalized_content is msg.get("content"):
        return msg
    # 浅拷贝 dict 消息，避免修改调用方持有的原始消息。
    copied = copy.copy(msg)
    # 在副本上写入归一化后的内容。
    copied["content"] = normalized_content
    # 返回带占位文本块的副本。
    return copied


def normalize_messages_for_report(messages: list[Any]) -> list[Any]:
    """Apply :func:`normalize_message_for_report` to each message in a list."""
    # 对消息列表逐条应用单条归一化函数，返回新列表（供轨迹上报使用）。
    return [normalize_message_for_report(m) for m in messages]

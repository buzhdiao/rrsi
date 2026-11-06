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

"""MCP client helpers for agents using LiteLLM."""

# 导入 asyncio：任务与超时控制，用于被保护 MCP 调用的收尾。
import asyncio
# 导入 typing.Any：通用类型注解。
from typing import Any

# 导入 loguru logger：记录不支持的内容块类型。
from loguru import logger
# 导入 MCP 协议的内容块类型：文本、图片及基类。
from mcp.types import ContentBlock, ImageContent, TextContent

# 导入 LiteLLM 输入消息类型：构造 tool / user 消息用。
from runner.agents.models import LitellmInputMessage

# Grace period (seconds) to wait for a shielded MCP tool call to finish
# after the primary timeout expires, before forcibly cancelling it.
# 宽限秒数：主超时到期后，再等被保护（shielded）的 MCP 工具调用这么久，
# 之后才强制取消。
SHIELDED_TASK_GRACE_SECONDS = 5.0


async def drain_shielded_task(task: asyncio.Task[Any]) -> None:
    """Wait for a shielded MCP task to finish, cancelling if it takes too long.

    After an ``asyncio.wait_for`` timeout, the shielded inner task is still
    running and holds the MCP session open.  Attempting a new tool call on the
    same session while the old one is in-flight causes a
    ``RuntimeError("Client is not connected")`` from the streamable-http
    transport.

    This helper gives the task a short grace period to complete naturally.  If
    it doesn't finish in time, the task is cancelled so the session is released
    before the next call.
    """
    # 任务已完成：无需处理。
    if task.done():
        return
    try:
        # 给被保护任务一个短宽限期自然完成。
        await asyncio.wait_for(task, timeout=SHIELDED_TASK_GRACE_SECONDS)
    except (TimeoutError, Exception):
        # 宽限期内未完成或出错：取消任务以释放 MCP 会话。
        task.cancel()
        try:
            # 等待取消完成，避免留下悬挂任务。
            await task
        except (asyncio.CancelledError, Exception):
            # 取消引发的异常按预期吞掉。
            pass


def build_mcp_gateway_schema(
    mcp_gateway_url: str,
    mcp_gateway_auth_token: str | None,
    mcp_gateway_actor_id: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Build the MCP client config schema for connecting to the environment's MCP gateway.

    The gateway is a single HTTP endpoint that proxies to all configured MCP servers
    in the environment sandbox.

    Args:
        mcp_gateway_url: URL of the MCP gateway (e.g. "http://localhost:8000/mcp/")
        mcp_gateway_auth_token: Real bearer token for authenticated gateway runtimes.
        mcp_gateway_actor_id: Runtime actor ID for user tenancy.

    Returns:
        The standard schema expected by the MCP client.
    """
    # 网关基础配置：使用 streamable-http 传输与网关 URL。
    gateway_config: dict[str, Any] = {
        "transport": "streamable-http",
        "url": mcp_gateway_url,
    }

    # Authorization is only set when the runner explicitly provides a token or actor ID.
    # 只有显式提供 actor id 或鉴权 token 时才设置 Authorization 头。
    auth_value = mcp_gateway_actor_id or mcp_gateway_auth_token
    if auth_value:
        # 以 Bearer token 形式写入请求头（actor id 用于用户租户隔离）。
        gateway_config["headers"] = {"Authorization": f"Bearer {auth_value}"}

    # 返回 MCP 客户端期望的标准 schema：mcpServers 字典中注册单个 "gateway" 服务器。
    return {
        "mcpServers": {
            "gateway": gateway_config,
        }
    }


def content_blocks_to_messages(
    content_blocks: list[ContentBlock],
    tool_call_id: str,
    name: str,
    model: str,
    deferred_image_messages: list[LitellmInputMessage],
) -> list[LitellmInputMessage]:
    """
    Convert MCP content blocks to a single LiteLLM tool message.

    Each tool_use must have exactly one tool_result. This function combines all
    content blocks into a single tool message to satisfy API requirements for
    Anthropic, OpenAI, and other providers.

    For non-Anthropic models, images cannot be embedded in tool results, so they
    are appended to deferred_image_messages as user messages. The caller is
    responsible for adding them to self.messages after all tool responses.
    This list is mutated in place.

    Args:
        content_blocks: MCP content blocks from tool result
        tool_call_id: The tool call ID to associate with the result
        name: The tool name
        model: The model being used
        deferred_image_messages: Mutable list that image user messages are
            appended to (mutated in place). Callers should extend self.messages
            with this list after all tool responses are added.

    Returns:
        List containing exactly one tool message.
    """
    # Anthropic supports images directly in tool results
    # Anthropic 系模型支持在工具结果中直接内嵌图片。
    supports_image_tool_results = model.startswith("anthropic/")

    # 收集所有文本内容。
    text_contents: list[str] = []
    # 收集所有图片的 data URI。
    image_data_uris: list[str] = []

    # 逐个解析 MCP 内容块。
    for content_block in content_blocks:
        # 按块类型分发处理。
        match content_block:
            case TextContent():
                # 文本块：校验后收集文本。
                block = TextContent.model_validate(content_block)
                text_contents.append(block.text)

            case ImageContent():
                # 图片块：校验后把 base64 数据拼成 data URI。
                block = ImageContent.model_validate(content_block)
                data_uri = f"data:{block.mimeType};base64,{block.data}"
                image_data_uris.append(data_uri)

            case _:
                # 其他类型（如音频等）：不支持，记录警告并给出占位文本。
                logger.warning(f"Content block type {content_block.type} not supported")
                text_contents.append("Unable to parse tool call response")

    # 待返回的消息列表。
    messages: list[LitellmInputMessage] = []

    if supports_image_tool_results:
        # Anthropic 路径：文本与图片可共置于同一 tool 消息的内容块列表。
        content: list[dict[str, Any]] = []
        for text in text_contents:
            # 每段文本转为一个 text 块。
            content.append({"type": "text", "text": text})
        for data_uri in image_data_uris:
            # 每张图片转为一个 image_url 块。
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        # 构造单条 tool 消息（内容为空时兜底一个空文本块）。
        tool_message: LitellmInputMessage = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content if content else [{"type": "text", "text": ""}],
        }  # pyright: ignore[reportAssignmentType]
        messages.append(tool_message)
    else:
        # 非 Anthropic 路径：tool 消息只放文本块。
        content = [{"type": "text", "text": text} for text in text_contents]

        if image_data_uris and not content:
            # 只有图片没有文本：补一句说明文字，保证 tool 消息非空。
            content.append(
                {"type": "text", "text": f"Image(s) returned by {name} tool"}
            )

        # 构造仅含文本的 tool 消息。
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content if content else [{"type": "text", "text": ""}],
        }  # pyright: ignore[reportAssignmentType]
        messages.append(tool_message)

        # Image workaround: non-Anthropic models don't support images in tool results,
        # so we append them to deferred_image_messages for the caller to add after all tool responses.
        # 图片变通方案：非 Anthropic 模型不支持工具结果内嵌图片，把图片包装成
        # user 消息追加到 deferred_image_messages，由调用方在所有工具响应之后
        # 加入消息历史（该列表按引用就地修改）。
        for data_uri in image_data_uris:
            deferred_image_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            )

    # 返回恰好包含一条 tool 消息的列表。
    return messages

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
Models for agent definitions and execution.
"""

# 导入异步可调用与可调用类型：用于定义智能体实现的函数签名。
from collections.abc import Awaitable, Callable
# 导入 StrEnum：字符串枚举基类（成员既是枚举也是 str）。
from enum import StrEnum
# 导入 Any：宽泛类型注解。
from typing import Any

# 导入 litellm 的 Chat Completions 消息 TypedDict 类型集合。
from litellm.types.llms.openai import AllMessageValues
# 导入 litellm 的 Message Pydantic 模型（API 响应中的消息）。
from litellm.types.utils import Message
# 导入 OpenAI Responses API 的输入消息 TypedDict 类型。
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
# 导入 pydantic 基类与工具：数据模型、序列化透传、字段校验器。
from pydantic import BaseModel, SerializeAsAny, field_validator

# 导入任务字段 schema 类型（用于描述智能体可配置字段）。
from runner.models import TaskFieldSchema
# 导入消息规范化函数（处理工具消息中的图片等内容块）。
from runner.save._normalize import normalize_messages_for_report

# LiteLLM message types for agent execution:
# - InputMessage (AllMessageValues): TypedDict for Chat Completions API requests
# - ResponsesInputMessage (EasyInputMessageParam): TypedDict for Responses API requests
# - OutputMessage (Message): Pydantic model from API responses, used for new messages
# - AnyMessage: Union of all three, used for trajectory output (includes input + generated)
# 三类消息别名：输入消息（Chat Completions 请求用的 TypedDict）。
LitellmInputMessage = AllMessageValues
# Responses API 的输入消息 TypedDict。
LitellmResponsesInputMessage = EasyInputMessageParam
# 输出消息：API 响应反序列化出的 Pydantic Message 模型。
LitellmOutputMessage = Message
# 任意消息：三者的联合类型，用于轨迹输出（既有输入也有生成的消息）。
LitellmAnyMessage = (
    LitellmInputMessage | LitellmResponsesInputMessage | LitellmOutputMessage
)

def get_msg_role(msg: LitellmAnyMessage) -> str:
    """Get role from either TypedDict or Pydantic Message."""
    # Pydantic Message 用属性访问取 role。
    if isinstance(msg, Message):
        return msg.role
    # TypedDict 用下标访问取 role。
    return msg["role"]

def get_msg_content(msg: LitellmAnyMessage) -> Any:
    """Get content from either TypedDict or Pydantic Message."""
    # Pydantic Message 用属性访问取 content。
    if isinstance(msg, Message):
        return msg.content
    # TypedDict 用 get 安全取 content（可能不存在）。
    return msg.get("content")

def get_msg_attr(msg: LitellmAnyMessage, key: str, default: Any = None) -> Any:
    """Get arbitrary attribute from either TypedDict or Pydantic Message."""
    # Pydantic Message 用 getattr 取任意属性（带默认值）。
    if isinstance(msg, Message):
        return getattr(msg, key, default)
    # TypedDict 用 get 取任意键（带默认值）。
    return msg.get(key, default)

def content_to_str(content: Any) -> str:
    """Normalize message content to a string.

    Some providers (e.g. Anthropic) return content as a list of blocks
    like [{'type': 'text', 'text': '...'}]. Use this when you need a plain string.
    """
    # 内容为 None 时返回空串。
    if content is None:
        return ""
    # 已经是字符串则原样返回。
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 分块列表（如 Anthropic 的 [{'type': 'text', ...}]）逐块提取文本。
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                # 字典块取 text 字段。
                text = block.get("text", "")
                # 仅收集非空字符串文本。
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                # 纯字符串块直接收集。
                parts.append(block)
        # 各块用换行连接；无有效文本时返回空串。
        return "\n".join(parts) if parts else ""
    # 其余类型统一转成字符串。
    return str(content)

class AgentConfigIds(StrEnum):
    """Registry of available agent implementation IDs (e.g., 'loop_agent')."""

    # 基础循环智能体实现。
    LOOP_AGENT = "loop_agent"
    # 带工具带与 ReSum 的 ReAct 智能体实现。
    REACT_TOOLBELT_AGENT = "react_toolbelt_agent"

class AgentStatus(StrEnum):
    """Status of an agent run."""

    # 待运行：尚未开始。
    PENDING = "pending"
    # 运行中：正在执行。
    RUNNING = "running"
    # 已完成：正常结束。
    COMPLETED = "completed"
    # 已取消：被外部取消。
    CANCELLED = "cancelled"
    # 失败：任务未成功（非系统故障）。
    FAILED = "failed"
    # 错误：系统级错误。
    ERROR = "error"

class AgentRunInput(BaseModel):
    """Input to an agent implementation."""

    # 本次运行的轨迹唯一标识。
    trajectory_id: str

    # The "actual" type of this is list[LitellmInputMessage] but given the way
    # pydantic works it makes sense to lazily handle this as just list[Any].
    # See also https://github.com/pydantic/pydantic/issues/9541.
    # 初始对话消息列表（实际为 LitellmInputMessage，因 pydantic 校验限制声明为 Any）。
    initial_messages: list[Any]

    # MCP 网关地址（工具都经由网关访问），可为 None。
    mcp_gateway_url: str | None
    # Gateway calls can either populate `Authorization: Bearer <token>` with...
    # - mcp_gateway_auth_token = A real bearer token for secure AuthN
    # - mcp_gateway_actor_id = An Actor ID for user tenancy
    # - None = No authentication
    # MCP 网关的真实 Bearer token（安全认证），可为 None。
    mcp_gateway_auth_token: str | None
    # MCP 网关的 Actor id（用户租户标识），缺省 None。
    mcp_gateway_actor_id: str | None = None
    # 编排模型（主决策 LLM）名称。
    orchestrator_model: str
    # 编排模型的额外调用参数，可为 None。
    orchestrator_extra_args: dict[str, Any] | None
    # 智能体配置键值对（超时、步数等）。
    agent_config_values: dict[str, Any]

    # Parent trajectory output (for continuation trajectories used during multi-turn, None otherwise)
    # 父轨迹输出：多轮对话的续跑轨迹才需要，其余为 None。
    parent_trajectory_output: dict[str, Any] | None = None

    # Arbitrary per-trajectory metadata from the orchestration request
    # 编排请求附带的任意轨迹级元数据。
    custom_args: dict[str, Any] | None = None

    # The task's custom_fields dict, passed through to all agents.
    # Most agents ignore this. Currently used by harness agents (e.g.
    # configurable_user_sim_agent) for task-level overrides of prompts/models.
    # 任务的 custom_fields 字典，透传给所有智能体；harness 类智能体用它做
    # 提示词/模型的任务级覆盖。
    task_custom_fields: dict[str, Any] | None = None

    # Resolved inner agent config (for harness agents that delegate to an agent instance)
    # 已解析的内层智能体配置（供委派给其他智能体实例的 harness 智能体使用）。
    inner_agent_config: dict[str, Any] | None = None

class AgentTrajectoryOutput(BaseModel):
    """Output from an agent run"""

    # Litellm Message is a Pydantic model; SerializeAsAny avoids union mismatch warnings
    # and keeps tool_calls in model_dump_json() for integration test assertions.
    # 轨迹的完整消息列表；SerializeAsAny 避免联合类型序列化告警并保留 tool_calls。
    messages: SerializeAsAny[list[LitellmAnyMessage]]
    # 可选的结构化输出（如最终答案对象）。
    output: dict[str, Any] | None = None
    # 运行结束状态。
    status: AgentStatus
    # 运行耗时（秒）。
    time_elapsed: float
    # token 用量统计，可为 None。
    usage: dict[str, Any] | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def _normalize_messages(cls, messages: Any) -> Any:
        """Normalize tool-message image content at construction time.

        LiteLLM's ``ChatCompletionToolMessage`` schema restricts tool content to
        text-only blocks. Tools like ``code_exec`` can return ``image_url``
        blocks (matplotlib PNGs, screenshots, etc.) that would otherwise break
        any downstream ``model_dump()`` / ``model_dump_json()`` with a
        ``PydanticSerializationError``. Running normalization here — rather than
        at a single call site like ``webhook.py`` — means every serialization of
        a trajectory output (webhook POST, local file write in ``runner/main.py``,
        per-agent logging, integration-test assertions) sees the same
        spec-compliant shape.

        Must be ``mode="before"``: by ``mode="after"`` Pydantic has already
        wrapped tool-message ``content`` in a ``ValidatorIterator`` for the
        ``str | Iterable`` union, at which point ``isinstance(content, list)``
        is false and the normalizer can't inspect the blocks.

        Drop once LiteLLM's tool-message schema accepts image/audio parts.
        """
        # 非列表输入不做处理，原样交回 pydantic。
        if not isinstance(messages, list):
            return messages
        # 在构造前规范化消息（把工具消息里的图片块调整成规范形状）。
        return normalize_messages_for_report(messages)

# 智能体实现类型：接收 AgentRunInput、异步返回 AgentTrajectoryOutput 的可调用对象。
AgentImpl = Callable[[AgentRunInput], Awaitable[AgentTrajectoryOutput]]

class AgentDefn(BaseModel):
    """Definition of an agent implementation in the registry."""

    # 该实现对应的智能体配置 id（注册表键）。
    agent_config_id: AgentConfigIds
    # 智能体实现函数；服务端可以只做注册不提供实现（可为 None）。
    agent_impl: AgentImpl | None = None  # Optional - server doesn't need implementation
    # 每次运行可自定义参数的 schema 列表（缺省为空）。
    run_custom_args_schema: list[TaskFieldSchema] = []  # Per-run custom args schema
    # 该智能体可配置字段的 schema 列表。
    agent_config_fields: list[TaskFieldSchema]  # Configurable fields for this agent

    class Config:
        # 允许任意类型（Callable 等非 pydantic 标准类型）作为字段类型。
        arbitrary_types_allowed = True

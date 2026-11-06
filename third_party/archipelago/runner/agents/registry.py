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
Agent registry mapping agent IDs to their implementations and config schemas.
"""

# 导入 Any：宽泛类型注解。
from typing import Any

# 导入智能体模型定义：配置 id 枚举、定义、实现类型、输入与输出模型。
from runner.agents.models import (
    AgentConfigIds,
    AgentDefn,
    AgentImpl,
    AgentRunInput,
    AgentTrajectoryOutput,
)
# 导入 importlib：运行时按模块名动态导入 harness 包。
import importlib
# 导入 os：读取环境变量 RRSI_HARNESS_MODULE。
import os


def react_toolbelt_agent_run(*args, **kwargs):
    """RRSI local modification: the evolvable harness package is named by the
    RRSI_HARNESS_MODULE environment variable (e.g. harness_workspace.main) instead of
    being vendored under runner/agents/, so one engine serves every domain."""
    # RRSI 本地改造：按环境变量动态加载可演化的 harness 模块
    # （默认 harness_workspace.main），使同一引擎服务不同领域。
    mod = importlib.import_module(os.environ.get("RRSI_HARNESS_MODULE", "harness_workspace.main"))
    # 透传参数调用该模块的 run 入口并返回结果。
    return mod.run(*args, **kwargs)
# 导入任务字段 schema 与字段类型枚举（描述智能体可配置项）。
from runner.models import TaskFieldSchema, TaskFieldType

# 智能体注册表：配置 id 到定义（实现函数 + 可配置字段 schema）的映射。
AGENT_REGISTRY: dict[AgentConfigIds, AgentDefn] = {
    # 注册 ReAct 工具带智能体。
    AgentConfigIds.REACT_TOOLBELT_AGENT: AgentDefn(
        agent_config_id=AgentConfigIds.REACT_TOOLBELT_AGENT,
        # 实现函数指向上面按环境变量动态加载的 run 封装。
        agent_impl=react_toolbelt_agent_run,
        # 声明该智能体暴露的可配置字段。
        agent_config_fields=[
            # 配置项 1：整次运行的总超时（秒），默认 3 小时，范围 5 分钟到 8 小时。
            TaskFieldSchema(
                field_id="timeout",
                field_type=TaskFieldType.NUMBER,
                label="Timeout (seconds)",
                description="Maximum time for agent execution",
                default_value=10800,  # 3 hours
                min_value=300,  # 5 minutes
                max_value=28800,  # 8 hours
            ),
            # 配置项 2：ReAct 循环最大步数（LLM 调用次数上限），默认 250。
            TaskFieldSchema(
                field_id="max_steps",
                field_type=TaskFieldType.NUMBER,
                label="Max Steps",
                description="Maximum number of LLM calls before stopping",
                default_value=250,
                min_value=1,
                max_value=1000,
            ),
            # 配置项 3：单次 LLM API 调用超时（秒），默认 600。
            TaskFieldSchema(
                field_id="llm_response_timeout",
                field_type=TaskFieldType.NUMBER,
                label="LLM Response Timeout (seconds)",
                description="Timeout for a single LLM API call",
                default_value=600,
                min_value=30,
                max_value=10800,
            ),
        ],
    ),
}

def get_agent_impl(agent_config_id: str) -> AgentImpl:
    """
    Get the agent implementation function for the given agent config ID.

    Args:
        agent_config_id: The agent config ID to look up (e.g., "loop_agent")

    Returns:
        The agent implementation function

    Raises:
        ValueError: If the agent config ID is not found in the registry
    """
    try:
        # 把字符串 id 转成枚举成员，非法值会抛 ValueError。
        config_id_enum = AgentConfigIds(agent_config_id)
    except ValueError as e:
        # 未知 id：包装成带上下文的错误再抛出。
        raise ValueError(f"Unknown agent config ID: {agent_config_id}") from e

    # 从注册表查定义。
    defn = AGENT_REGISTRY.get(config_id_enum)
    if defn is None:
        # 枚举合法但未注册：报错。
        raise ValueError(f"Unknown agent config ID: {agent_config_id}")

    if defn.agent_impl is None:
        # 已注册但没有实现函数（如纯服务端场景）：报错。
        raise ValueError(
            f"Agent '{agent_config_id}' is registered but has no implementation"
        )

    # 返回智能体实现函数。
    return defn.agent_impl

def get_agent_defn(agent_config_id: str) -> AgentDefn:
    """
    Get the full agent definition for the given agent config ID.

    Args:
        agent_config_id: The agent config ID to look up (e.g., "loop_agent")

    Returns:
        The agent definition including config fields

    Raises:
        ValueError: If the agent config ID is not found in the registry
    """
    try:
        # 把字符串 id 转成枚举成员。
        config_id_enum = AgentConfigIds(agent_config_id)
    except ValueError as e:
        # 未知 id：包装错误后抛出。
        raise ValueError(f"Unknown agent config ID: {agent_config_id}") from e

    # 从注册表查定义。
    defn = AGENT_REGISTRY.get(config_id_enum)
    if defn is None:
        # 未注册：报错。
        raise ValueError(f"Unknown agent config ID: {agent_config_id}")

    # 返回完整定义（含配置字段 schema）。
    return defn

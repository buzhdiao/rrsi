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
Shared models for agent runner.
"""

# 导入 StrEnum：字符串枚举基类（成员既是枚举也是 str）。
from enum import StrEnum
# 导入 Any 与 Literal：宽泛类型注解与字面量限定（如布局宽度取值）。
from typing import Any, Literal

# 导入 pydantic：数据模型基类与字段定义工具。
from pydantic import BaseModel, Field


class TaskFieldType(StrEnum):
    """Types of custom fields that can be defined for agent config."""

    # 单行文本输入。
    TEXT = "text"  # Single-line text input
    # 多行文本输入。
    TEXTAREA = "textarea"  # Multi-line text input
    # 数字输入。
    NUMBER = "number"  # Numeric input
    # 布尔勾选框。
    BOOLEAN = "boolean"  # Checkbox
    # 日期选择器。
    DATE = "date"  # Date picker
    # 日期时间选择器。
    DATETIME = "datetime"  # Date and time picker
    # 单选下拉框。
    SELECT = "select"  # Single choice dropdown
    # 多选下拉框。
    MULTISELECT = "multiselect"  # Multiple choice dropdown
    # 带校验的 URL 输入。
    URL = "url"  # URL input with validation
    # 带校验的邮箱输入。
    EMAIL = "email"  # Email input with validation
    # 从快照中多选文件（artifact 多选）。
    ARTIFACT_MULTISELECT = (
        "artifact_multiselect"  # Multi-select file picker from snapshots
    )
    # 带变换选项的 artifact 多选。
    ARTIFACT_MULTISELECT_TRANSFORMED = (
        "artifact_multiselect_transformed"  # Multi-select with transformation options
    )
    # 滑动整数刻度（李克特量表，带端点标签）。
    LIKERT_SCALE = "likert_scale"  # Sliding integer scale with endpoint labels
    # 文件上传字段（存储 S3 键）。
    FILE = "file"  # File upload field, stores S3 keys
    # 嵌套字段组列表（子 schema）。
    SUBSCHEMA_LIST = "subschema_list"  # Nested list of field groups
    # 任意 JSON 数据。
    JSON = "json"  # Arbitrary JSON data
    # 由编排器（orchestrator）列表填充的下拉框。
    ORCHESTRATOR_SELECT = "orchestrator_select"  # Dropdown populated with orchestrators
    # 由智能体列表填充的下拉框。
    AGENT_SELECT = "agent_select"  # Dropdown populated with agents
    # 由平台列表填充的下拉框。
    PLATFORM_SELECT = "platform_select"  # Dropdown populated with platforms


class TaskFieldSchema(BaseModel):
    """Schema definition for a single agent config field."""

    # 字段标识符（必填），如 "timeout"、"max_steps"。
    field_id: str = Field(
        ...,
        description="Identifier for this field (e.g., 'timeout', 'max_steps').",
    )
    # 字段类型（必填）：决定 UI 组件与校验方式。
    field_type: TaskFieldType = Field(
        ...,
        description="Type of field determines UI component and validation",
    )
    # UI 上展示的人类可读标签（必填）。
    label: str = Field(
        ...,
        description="Human-readable label shown in UI",
    )
    # 该字段是否必填，默认非必填。
    required: bool = Field(
        default=False,
        description="Whether this field is required",
    )
    # 面向用户的帮助文本，可为 None。
    description: str | None = Field(
        default=None,
        description="Help text shown to users",
    )
    # 新建配置时的默认值，类型不限，可为 None。
    default_value: Any | None = Field(
        default=None,
        description="Default value when creating new configs",
    )
    # select 类字段的可选项列表，可为 None。
    options: list[str] | None = Field(
        default=None,
        description="Available options for select fields",
    )
    # number 类字段的下界，可为 None。
    min_value: float | None = Field(
        default=None,
        description="Minimum value for number fields",
    )
    # number 类字段的上界，可为 None。
    max_value: float | None = Field(
        default=None,
        description="Maximum value for number fields",
    )
    # 表单布局中的宽度：整行/半行/三分之一行，默认整行。
    display_width: Literal["full", "half", "third"] = Field(
        default="full",
        description="Width in form layout",
    )

    # File field configuration
    # 文件字段专用：允许的最大文件数，可为 None。
    max_files: int | None = Field(
        default=None,
        description="Maximum number of files allowed for file fields",
    )

    # Likert scale display labels
    # 李克特量表最小端的展示标签（如"非常不同意"）。
    display_min_explanation: str | None = Field(
        default=None,
        description="Label shown at the min end of a likert scale (e.g., 'Strongly Disagree')",
    )
    # 李克特量表最大端的展示标签（如"非常同意"）。
    display_max_explanation: str | None = Field(
        default=None,
        description="Label shown at the max end of a likert scale (e.g., 'Strongly Agree')",
    )
    # field_type 为 subschema_list 时，列表项的字段 schema（自引用，可递归嵌套）。
    subschema: list["TaskFieldSchema"] | None = Field(
        default=None,
        description="Schema for items when field_type is subschema_list.",
    )
    # 控制该字段在 UI 中何时显示的条件过滤器，可为 None。
    conditional_render_filter: list[Any] | None = Field(
        default=None,
        description="Filters controlling when this field is shown in the UI.",
    )


# @apg_agent_config_model:start
class AgentConfig(BaseModel):
    """Agent configuration"""

    # 智能体实现 id（如 "loop_agent"），决定使用哪个实现。
    agent_config_id: str  # Which agent implementation (e.g., "loop_agent")
    # 人类可读的智能体名称（如 "Fast Loop Agent"）。
    agent_name: str  # Human-readable name (e.g., "Fast Loop Agent")
    # 智能体专属配置键值对（对应 TaskFieldSchema 定义的字段取值）。
    agent_config_values: dict[str, Any]  # Agent-specific configuration values


# @apg_agent_config_model:end


# 重新构建模型：解析 subschema 字段的自引用（前向引用 TaskFieldSchema 自身）。
TaskFieldSchema.model_rebuild()

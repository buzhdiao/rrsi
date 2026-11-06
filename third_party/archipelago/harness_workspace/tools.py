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
Tools for the ReAct agent.

Includes:
- Meta-tools for dynamic toolbelt management and task planning
- Final answer tool for explicit termination
"""

# 导入 json：用于解析工具调用参数序列化、序列化工具返回结果。
import json
# 导入 dataclass 装饰器：简化待办项（TodoItem）数据类的定义。
from dataclasses import dataclass
# 导入 Enum：定义待办项状态枚举。
from enum import Enum
# 导入 Any：JSON 结构的宽泛类型注解。
from typing import Any

# 导入 loguru 日志器，记录工具增删与待办变更事件。
from loguru import logger
# 导入 OpenAI 格式的工具参数类型，作为工具 schema 的统一表示。
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam


def _flatten_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten JSON schema by resolving $ref and removing $defs.

    Gemini rejects schemas with $defs/$ref in function responses with error:
    "The referenced name `#/$defs/X` does not match to a display_name"
    """
    # 非 dict 的 schema 无法展平，原样返回。
    if not isinstance(schema, dict):
        return schema

    # 浅拷贝一份，避免修改调用方传入的原始 schema。
    schema = schema.copy()
    # 弹出顶层的 $defs（引用定义池），后续解析 $ref 时查这个表。
    defs = schema.pop("$defs", {})

    def resolve_refs(obj: Any, visiting: set[str] | None = None) -> Any:
        """Recursively resolve all $ref references."""
        # 首次调用时初始化"正在访问"集合，用于检测循环引用。
        if visiting is None:
            visiting = set()

        # 只有 dict 节点才可能含 $ref，其余类型原样返回。
        if not isinstance(obj, dict):
            return obj

        if "$ref" in obj:
            # 取出引用路径（形如 #/$defs/Name）。
            ref_path = obj["$ref"]
            # 从路径末段取出定义名。
            ref_name = ref_path.split("/")[-1]

            # 命中循环引用（该名字已在解析栈中）时不再展开，
            if ref_name in visiting:
                # 用空 object 类型占位，避免无限递归。
                result: dict[str, Any] = {"type": "object"}
                # 保留 $ref 之外的其他键，并对它们递归解析。
                for key, value in obj.items():
                    if key != "$ref":
                        result[key] = resolve_refs(value, visiting)
                return result

            # 标准的内部引用（#/$defs/名字 且定义存在）可以展开。
            if ref_path.startswith("#/$defs/") and ref_name in defs:
                # 把名字加入访问集合，标记进入该定义的解析。
                visiting.add(ref_name)
                try:
                    # 取出该名字对应的定义。
                    definition = defs[ref_name]
                    if isinstance(definition, bool):
                        # 布尔 schema 是 JSON Schema 的特例：true 等价空 schema，false 等价"禁止任何值"。
                        resolved = {} if definition else {"not": {}}
                    else:
                        # 递归展开定义本身（拷贝以防污染 $defs）。
                        resolved = resolve_refs(definition.copy(), visiting)
                finally:
                    # 无论成功与否都把名字移出访问集合（回溯）。
                    visiting.discard(ref_name)
                # 合并：引用处与定义处除 $ref 外的键都保留，并递归解析。
                for key, value in obj.items():
                    if key != "$ref":
                        resolved[key] = resolve_refs(value, visiting)
                return resolved

            # 无法解析的引用（外部路径或定义缺失）：
            result = {"type": "object"}
            visiting.add(ref_name)
            try:
                # 退化为空 object，同时保留并解析引用处的其他键。
                for key, value in obj.items():
                    if key != "$ref":
                        result[key] = resolve_refs(value, visiting)
            finally:
                visiting.discard(ref_name)
            return result

        # 普通 dict 节点：逐键递归处理。
        result = {}
        for key, value in obj.items():
            if isinstance(value, dict):
                # 值是 dict 就递归解析。
                result[key] = resolve_refs(value, visiting)
            elif isinstance(value, list):
                # 值是列表则对其中每个 dict 元素递归解析。
                result[key] = [
                    resolve_refs(item, visiting) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                # 标量值直接保留。
                result[key] = value

        return result

    # 从整个 schema 根节点开始解析引用。
    return resolve_refs(schema)


# =============================================================================
# Todo types
# =============================================================================


class TodoStatus(str, Enum):
    """Status of a todo item."""

    # 待处理：任务尚未开始。
    PENDING = "pending"
    # 进行中：任务正在执行。
    IN_PROGRESS = "in_progress"
    # 已完成：任务成功结束。
    COMPLETED = "completed"
    # 已取消：任务被放弃。
    CANCELLED = "cancelled"


@dataclass
class TodoItem:
    """A todo item for task tracking."""

    # 任务唯一标识符。
    id: str
    # 任务内容描述。
    content: str
    # 任务状态，默认为待处理。
    status: TodoStatus = TodoStatus.PENDING

    def to_dict(self) -> dict[str, str]:
        # 序列化为纯字符串字典（枚举取其 value），便于 JSON 返回。
        return {"id": self.id, "content": self.content, "status": self.status.value}


# =============================================================================
# Meta-tools (toolbelt management + todo planning)
# =============================================================================

# 元工具（meta-tools）定义列表：让智能体动态管理自己的"工具带"（toolbelt，
# 即当前可用的工具集合）并进行待办清单规划。这里只声明 schema，不包含实现。
META_TOOLS: list[ChatCompletionToolParam] = [
    # Toolbelt management
    {
        "type": "function",
        "function": {
            "name": "toolbelt_list_tools",
            "description": "List all available tools that are NOT currently in your toolbelt. Returns a list of tool names you can add.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toolbelt_inspect_tool",
            "description": "Get detailed information about a tool, including its description and parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The name of the tool to inspect",
                    },
                },
                "required": ["tool_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toolbelt_add_tool",
            "description": "Add a tool to your toolbelt so you can use it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The name of the tool to add",
                    },
                },
                "required": ["tool_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toolbelt_remove_tool",
            "description": "Remove a tool from your toolbelt to free up space.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The name of the tool to remove",
                    },
                },
                "required": ["tool_name"],
            },
        },
    },
    # Todo planning - single tool matching Cursor's schema
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "Create or update your todo list. Use this to plan multi-step work and track progress. All todos must be completed or cancelled before final_answer will be accepted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "Array of todo items to write",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique identifier for the task",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Description of the task (required for new todos, optional for updates)",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                        "cancelled",
                                    ],
                                    "description": "Task status",
                                },
                            },
                            "required": ["id", "status"],
                        },
                    },
                    "merge": {
                        "type": "boolean",
                        "description": "If true, merge with existing todos (update matching IDs, add new ones). If false, replace all todos with the provided list.",
                    },
                },
                "required": ["todos", "merge"],
            },
        },
    },
]

# 用集合推导式提取所有元工具的名字，便于运行时快速判断"是不是元工具"。
META_TOOL_NAMES = {tool["function"]["name"] for tool in META_TOOLS}


class MetaToolHandler:
    """Handles meta-tool calls for toolbelt management and todo planning."""

    def __init__(
        self,
        all_tools: dict[str, ChatCompletionToolParam],
        toolbelt: set[str],
        max_toolbelt_size: int,
    ):
        # 保存全部可用工具的名字到 schema 的映射。
        self.all_tools = all_tools
        # 保存当前工具带（智能体已装载的工具名集合），与外部共享引用、就地修改。
        self.toolbelt = toolbelt
        # 工具带容量上限，防止工具过多撑爆上下文。
        self.max_toolbelt_size = max_toolbelt_size
        # 智能体的待办清单：任务 id 到 TodoItem 的映射。
        self.todos: dict[str, TodoItem] = {}

    def handle(self, tool_name: str, arguments: str) -> str:
        """Handle a meta-tool call and return the result."""
        # Parse arguments safely
        # 安全解析参数 JSON：空串视为空对象。
        try:
            args: dict[str, Any] = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            # JSON 非法时返回错误信息而不是抛异常（结果要回传给模型）。
            return json.dumps({"error": "Invalid JSON in arguments"})

        # 解析结果必须是 JSON 对象（dict），否则报错。
        if not isinstance(args, dict):
            return json.dumps({"error": "Arguments must be a JSON object"})

        # === Toolbelt management ===

        # 分发：列出尚未装入工具带的可用工具。
        if tool_name == "toolbelt_list_tools":
            # 全集减去已装载集合，排序后以 JSON 数组返回。
            available = sorted(set(self.all_tools.keys()) - self.toolbelt)
            return json.dumps(available)

        # 分发：查看某工具详情（交给专门方法，含 Gemini schema 展平）。
        if tool_name == "toolbelt_inspect_tool":
            return self._handle_inspect_tool(args)

        # 分发：向工具带添加工具。
        if tool_name == "toolbelt_add_tool":
            # 取出目标工具名（缺省为空串）。
            name = args.get("tool_name", "")
            # 名字为空视为缺少必填参数。
            if not name:
                return json.dumps({"error": "tool_name is required"})
            # 工具不存在于全集时报错。
            if name not in self.all_tools:
                return json.dumps({"error": f"Tool '{name}' not found"})
            # 元工具本身或已装载的工具不允许重复添加。
            if name in META_TOOL_NAMES or name in self.toolbelt:
                return json.dumps({"error": f"Tool '{name}' already in toolbelt"})
            # 工具带已满时要求先移除工具。
            if len(self.toolbelt) >= self.max_toolbelt_size:
                return json.dumps({"error": "Toolbelt full. Remove a tool first."})
            # 校验通过，把工具名加入工具带集合。
            self.toolbelt.add(name)
            # 记录添加日志。
            logger.bind(message_type="tool").info(f"Added tool '{name}' to toolbelt")
            # 返回成功标志与当前工具带大小。
            return json.dumps({"success": True, "toolbelt_size": len(self.toolbelt)})

        # 分发：从工具带移除工具。
        if tool_name == "toolbelt_remove_tool":
            name = args.get("tool_name", "")
            if not name:
                return json.dumps({"error": "tool_name is required"})
            # 工具不在工具带中则无法移除。
            if name not in self.toolbelt:
                return json.dumps({"error": f"Tool '{name}' not in toolbelt"})
            # 从集合中移除该工具。
            self.toolbelt.remove(name)
            logger.bind(message_type="tool").info(
                f"Removed tool '{name}' from toolbelt"
            )
            return json.dumps({"success": True, "toolbelt_size": len(self.toolbelt)})

        # === Todo planning ===

        # 分发：创建/更新待办清单。
        if tool_name == "todo_write":
            # 取待办条目数组（缺省为空）。
            todos_input = args.get("todos", [])
            # 取合并模式开关（缺省为 True：与现有清单合并）。
            merge = args.get("merge", True)

            # todos 必须是数组。
            if not isinstance(todos_input, list):
                return json.dumps({"error": "todos must be an array"})

            # 非合并模式：先清空现有清单，实现"整体替换"。
            if not merge:
                self.todos.clear()

            # 分别记录本次更新与新建的任务 id。
            updated_ids: list[str] = []
            created_ids: list[str] = []
            # 收集逐条校验产生的错误信息。
            errors: list[str] = []

            for item in todos_input:
                # 每个条目必须是 JSON 对象。
                if not isinstance(item, dict):
                    errors.append("Each todo must be an object")
                    continue

                # 取任务 id（强制转字符串）。
                task_id = str(item.get("id", ""))
                # 取状态字符串（缺省 pending）。
                status_str = str(item.get("status", "pending"))
                # 取任务内容（可能为 None，表示仅更新状态）。
                content = item.get("content")

                # id 是必填项。
                if not task_id:
                    errors.append("Each todo must have an id")
                    continue

                try:
                    # 把状态字符串转成枚举，非法值会抛 ValueError。
                    status = TodoStatus(status_str)
                except ValueError:
                    errors.append(f"Invalid status '{status_str}' for task '{task_id}'")
                    continue

                if task_id in self.todos:
                    # 已存在的任务：更新其状态。
                    self.todos[task_id].status = status
                    # 内容非 None 时一并更新。
                    if content is not None:
                        self.todos[task_id].content = str(content)
                    updated_ids.append(task_id)
                else:
                    # 新任务必须提供内容。
                    if content is None:
                        errors.append(f"Content required for new todo '{task_id}'")
                        continue
                    # 创建新的 TodoItem 并登记。
                    self.todos[task_id] = TodoItem(
                        id=task_id, content=str(content), status=status
                    )
                    created_ids.append(task_id)

            # 有新建任务时记录日志。
            if created_ids:
                logger.bind(message_type="tool").info(
                    f"Created todos: {', '.join(created_ids)}"
                )
            # 有更新任务时记录日志。
            if updated_ids:
                logger.bind(message_type="tool").info(
                    f"Updated todos: {', '.join(updated_ids)}"
                )

            # 把全部待办序列化为字典列表，用于响应与统计。
            tasks = [t.to_dict() for t in self.todos.values()]
            # 组装响应：成功标志、新建/更新 id、清单全文及各状态计数汇总。
            response: dict[str, Any] = {
                "success": len(errors) == 0,
                "created": created_ids,
                "updated": updated_ids,
                "todos": tasks,
                "summary": {
                    "total": len(tasks),
                    "pending": sum(1 for t in tasks if t["status"] == "pending"),
                    "in_progress": sum(
                        1 for t in tasks if t["status"] == "in_progress"
                    ),
                    "completed": sum(1 for t in tasks if t["status"] == "completed"),
                    "cancelled": sum(1 for t in tasks if t["status"] == "cancelled"),
                },
            }
            # 有错误时附加错误列表。
            if errors:
                response["errors"] = errors
            # 返回 JSON 字符串形式的结果。
            return json.dumps(response)

        # 未匹配任何元工具名时返回未知工具错误。
        return json.dumps({"error": f"Unknown meta-tool: {tool_name}"})

    def _handle_inspect_tool(self, args: dict[str, Any]) -> str:
        """Handle toolbelt_inspect_tool with schema flattening for Gemini."""
        name = args.get("tool_name", "")
        if not name:
            return json.dumps({"error": "tool_name is required"})
        if name not in self.all_tools:
            return json.dumps({"error": f"Tool '{name}' not found"})
        # 取出该工具的完整 schema 定义。
        tool = self.all_tools[name]
        # 取其 function 子对象（含名称/描述/参数 schema）。
        func = tool.get("function", {})
        # Flatten schema to remove $defs/$ref (Gemini compatibility)
        # 取参数 schema 并做展平处理（Gemini 不接受 $defs/$ref）。
        parameters = func.get("parameters")
        if parameters:
            parameters = _flatten_schema(parameters)
        # 返回工具名、描述与展平后的参数 schema。
        return json.dumps(
            {
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": parameters,
            }
        )

    def get_incomplete_todos(self) -> list[TodoItem]:
        """Return todos that are not completed or cancelled."""
        # 过滤出既未完成也未取消的待办项。
        return [
            t
            for t in self.todos.values()
            if t.status not in (TodoStatus.COMPLETED, TodoStatus.CANCELLED)
        ]

    def has_incomplete_todos(self) -> bool:
        """Check if there are any incomplete todos."""
        # 存在未完成待办即返回 True（用于拦截 final_answer 提前结束）。
        return len(self.get_incomplete_todos()) > 0


# =============================================================================
# Final answer tool for explicit termination
# =============================================================================

# 终止工具（final_answer）定义：智能体必须显式调用它提交最终答案，
# 以"显式终止"方式结束 ReAct 循环，同时声明任务状态。
FINAL_ANSWER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Submit your final answer to complete the task. Call when done.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your complete final answer to the task.",
                },
                "status": {
                    "type": "string",
                    "enum": ["completed", "blocked", "failed"],
                    "description": "completed=done, blocked=cannot proceed, failed=impossible",
                },
            },
            "required": ["answer", "status"],
        },
    },
}


def parse_final_answer(arguments: str) -> tuple[str, str]:
    """Parse final_answer arguments. Returns (answer, status)."""
    try:
        # 解析参数 JSON；空串视为空对象。
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        # JSON 非法时把原始参数串当作答案，状态按已完成处理。
        return arguments, "completed"

    if not isinstance(args, dict):
        # 解析结果不是对象（如纯数字/字符串）时直接转成字符串作为答案。
        return str(args), "completed"

    # 正常情况：取出答案文本与状态（缺省 completed）。
    return str(args.get("answer", "")), str(args.get("status", "completed"))

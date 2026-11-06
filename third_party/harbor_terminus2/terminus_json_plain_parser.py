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

# JSON 编解码：解析模型返回的 JSON 响应体
import json
# 正则：字段顺序检查与混合内容修复时定位 JSON 片段
import re
# dataclass：声明 ParsedCommand / ParseResult 数据类
from dataclasses import dataclass
# Any：结构校验函数的参数类型标注
from typing import Any


# 一条解析出的终端命令：keystrokes 为要敲入的按键串，duration 为执行后等待秒数
@dataclass
class ParsedCommand:
    keystrokes: str
    duration: float


# 解析结果总装：命令列表、任务是否完成标记、错误/警告文本、推理内容（analysis/plan）
@dataclass
class ParseResult:
    commands: list[ParsedCommand]
    is_task_complete: bool
    error: str
    warning: str
    analysis: str = ""
    plan: str = ""


# Terminus 2 的 JSON 纯文本响应解析器：把模型回复解析成命令与完成标记，
# 并内置两道"自动修复"兜底（补全缺失的右花括号 / 从混合文本中抽取 JSON）。
class TerminusJSONPlainParser:
    """Parser for terminus JSON plain response format."""

    def __init__(self):
        # 约定的必填字段：推理（analysis）、计划（plan）、命令（commands）
        self.required_fields = ["analysis", "plan", "commands"]

    def parse_response(self, response: str) -> ParseResult:
        """
        Parse a terminus JSON plain response and extract commands.

        Args:
            response: The full LLM response string

        Returns:
            ParseResult with commands, completion status, errors and warnings
        """

        # Try normal parsing first
        # 先按标准路径尝试解析
        result = self._try_parse_response(response)

        if result.error:
            # Try auto-fixes in order until one works
            # 解析出错 → 按顺序逐个尝试自动修复，直到某个修复后能解析成功
            for fix_name, fix_function in self._get_auto_fixes():
                # 应用一个修复函数，得到修正文本与"是否做了修改"标记
                corrected_response, was_fixed = fix_function(response, result.error)
                if was_fixed:
                    # 对修正后的文本重新走一遍完整解析
                    corrected_result = self._try_parse_response(corrected_response)

                    if corrected_result.error == "":
                        # Success! Add auto-correction warning
                        # 修复成功！附加"已自动纠错"警告，提醒模型下次自己改对
                        auto_warning = (
                            f"AUTO-CORRECTED: {fix_name} - "
                            "please fix this in future responses"
                        )
                        # 合并自动纠错警告与解析过程中积累的既有警告
                        corrected_result.warning = self._combine_warnings(
                            auto_warning, corrected_result.warning
                        )
                        # 返回修复后的成功结果
                        return corrected_result

        # Return original result if no fix worked
        # 所有修复都无效 → 返回最初带错误的结果
        return result

    def _try_parse_response(self, response: str) -> ParseResult:
        """
        Try to parse a terminus JSON plain response.

        Args:
            response: The full LLM response string

        Returns:
            ParseResult with commands, completion status, errors and warnings
        """
        warnings = []

        # Check for extra text before/after JSON
        # 定位并截取响应中的 JSON 对象，同时收集"JSON 前后有额外文本"的警告
        json_content, extra_text_warnings = self._extract_json_content(response)
        warnings.extend(extra_text_warnings)

        if not json_content:
            # 连一个完整 JSON 对象都找不到 → 直接以错误返回（语句跨多行）
            return ParseResult(
                [],
                False,
                "No valid JSON found in response",
                "- " + "\n- ".join(warnings) if warnings else "",
                "",
                "",
            )

        # Parse JSON
        try:
            # 反序列化截取出的 JSON 文本
            parsed_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            # Add debug info
            # 拼装带调试信息的错误消息（解析器原始报错 + 内容/预览）
            error_msg = f"Invalid JSON: {str(e)}"
            if len(json_content) < 200:
                # 内容短就完整带上，便于模型自查
                error_msg += f" | Content: {repr(json_content)}"
            else:
                # 内容长则只带前 100 字符预览，防止错误消息本身过长
                error_msg += f" | Content preview: {repr(json_content[:100])}..."
            return ParseResult(
                [],
                False,
                error_msg,
                "- " + "\n- ".join(warnings) if warnings else "",
                "",
                "",
            )

        # Validate structure
        # 结构校验：必须是对象、必填字段齐全、字段类型正确、顺序合理
        validation_error = self._validate_json_structure(
            parsed_data, json_content, warnings
        )
        if validation_error:
            # 校验失败 → 以错误返回（不携带任何命令）
            return ParseResult(
                [],
                False,
                validation_error,
                "- " + "\n- ".join(warnings) if warnings else "",
                "",
                "",
            )

        # Check if task is complete
        # 读取任务完成标记（task_complete，缺省视为未完成）
        is_complete = parsed_data.get("task_complete", False)
        if isinstance(is_complete, str):
            # 容错：模型把布尔写成字符串时，按常见真值词归一化
            is_complete = is_complete.lower() in ("true", "1", "yes")

        # Extract analysis and plan for reasoning content
        # 抽取推理内容：analysis（当前局面分析）与 plan（下一步计划），供轨迹记录
        analysis = parsed_data.get("analysis", "")
        plan = parsed_data.get("plan", "")

        # Parse commands
        # 解析命令数组（keystrokes/duration）
        commands_data = parsed_data.get("commands", [])
        commands, parse_error = self._parse_commands(commands_data, warnings)
        if parse_error:
            # If task is complete, parse errors are just warnings
            # 任务已宣告完成时，命令解析错误降级为警告（不再需要执行命令）
            if is_complete:
                warnings.append(parse_error)
                return ParseResult(
                    [],
                    True,
                    "",
                    "- " + "\n- ".join(warnings) if warnings else "",
                    analysis,
                    plan,
                )
            # 任务未完成时命令解析错误是硬错误：本轮不执行任何动作
            return ParseResult(
                [],
                False,
                parse_error,
                "- " + "\n- ".join(warnings) if warnings else "",
                analysis,
                plan,
            )

        # 全部通过：返回携带命令、完成标记与推理内容的成功结果
        return ParseResult(
            commands,
            is_complete,
            "",
            "- " + "\n- ".join(warnings) if warnings else "",
            analysis,
            plan,
        )

    def _extract_json_content(self, response: str) -> tuple[str, list[str]]:
        """Extract JSON content from response, handling extra text."""
        # 警告收集器：JSON 前后是否有额外文本
        warnings = []

        # Try to find JSON object boundaries
        # 手写扫描器定位 JSON 对象边界（比正则可靠：能区分字符串内的花括号）
        json_start = -1
        json_end = -1
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(response):
            if escape_next:
                # 上一字符是转义符 → 本字符只是字面量，清标记后跳过
                escape_next = False
                continue

            if char == "\\":
                # 进入转义状态：下一个字符不参与结构判断
                escape_next = True
                continue

            if char == '"' and not escape_next:
                # 引号切换"是否在字符串内"状态：字符串内的 {} 不计层数
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    if brace_count == 0:
                        # 深度从 0 变 1：记下 JSON 对象起点
                        json_start = i
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        # 深度归 0：第一个顶层对象结束，记终点后停止扫描
                        json_end = i + 1
                        break

        if json_start == -1 or json_end == -1:
            # 没找到成对的顶层花括号 → 无有效 JSON 对象
            return "", ["No valid JSON object found"]

        # Check for extra text
        # 检查 JSON 对象之外是否混入了额外文本（前后分别检查）
        before_text = response[:json_start].strip()
        after_text = response[json_end:].strip()

        if before_text:
            # JSON 前有杂文本（如模型先写了说明文字）
            warnings.append("Extra text detected before JSON object")
        if after_text:
            # JSON 后有杂文本
            warnings.append("Extra text detected after JSON object")

        # 返回截取出的 JSON 文本与警告列表
        return response[json_start:json_end], warnings

    def _validate_json_structure(
        self, data: dict[str, Any], json_content: str, warnings: list[str]
    ) -> str:
        """Validate the JSON structure has required fields."""
        # 顶层必须是 JSON 对象（数组/标量都不合格）
        if not isinstance(data, dict):
            return "Response must be a JSON object"

        # Check for required fields
        # 必填字段检查：逐一核对 analysis/plan/commands 是否存在
        missing_fields = []
        for field in self.required_fields:
            if field not in data:
                missing_fields.append(field)

        if missing_fields:
            # 缺字段是硬错误：模型必须补齐才能被接受
            return f"Missing required fields: {', '.join(missing_fields)}"

        # Validate field types
        # 字段类型检查（软约束，只警告不拒绝）
        if not isinstance(data.get("analysis", ""), str):
            warnings.append("Field 'analysis' should be a string")

        if not isinstance(data.get("plan", ""), str):
            warnings.append("Field 'plan' should be a string")

        commands = data.get("commands", [])
        if not isinstance(commands, list):
            # commands 必须是数组，否则无从解析命令 → 硬错误
            return "Field 'commands' must be an array"

        # Check for correct order of fields (analysis, plan, commands)
        # 检查字段出现顺序是否符合约定（analysis → plan → commands），不合只警告
        self._check_field_order(data, json_content, warnings)

        # Validate task_complete if present
        # 校验可选字段 task_complete：只允许布尔或字符串类型
        task_complete = data.get("task_complete")
        if task_complete is not None and not isinstance(task_complete, (bool, str)):
            warnings.append("Field 'task_complete' should be a boolean or string")

        # 校验通过，返回空错误串
        return ""

    def _parse_commands(
        self, commands_data: list[dict[str, Any]], warnings: list[str]
    ) -> tuple[list[ParsedCommand], str]:
        """Parse commands array into ParsedCommand objects."""
        # 解析结果累积列表
        commands = []

        for i, cmd_data in enumerate(commands_data):
            # 每条命令必须是 JSON 对象，否则整体失败（返回硬错误）
            if not isinstance(cmd_data, dict):
                return [], f"Command {i + 1} must be an object"

            # Check for required keystrokes field
            # keystrokes（按键串）是每条命令唯一必填字段
            if "keystrokes" not in cmd_data:
                return [], f"Command {i + 1} missing required 'keystrokes' field"

            keystrokes = cmd_data["keystrokes"]
            if not isinstance(keystrokes, str):
                # keystrokes 必须是字符串（终端要逐字敲入）
                return [], f"Command {i + 1} 'keystrokes' must be a string"

            # Parse optional fields with defaults
            # 可选字段 duration（执行后等待秒数）带默认值 1.0
            if "duration" in cmd_data:
                duration = cmd_data["duration"]
                if not isinstance(duration, (int, float)):
                    # 类型不对 → 警告并回退默认值，不让整条命令作废
                    warnings.append(
                        f"Command {i + 1}: Invalid duration value, using default 1.0"
                    )
                    duration = 1.0
            else:
                # 缺 duration → 警告并用默认值
                warnings.append(
                    f"Command {i + 1}: Missing duration field, using default 1.0"
                )
                duration = 1.0

            # Check for unknown fields
            # 未知字段检查：提示模型只应使用 keystrokes/duration
            known_fields = {"keystrokes", "duration"}
            unknown_fields = set(cmd_data.keys()) - known_fields
            if unknown_fields:
                warnings.append(
                    f"Command {i + 1}: Unknown fields: {', '.join(unknown_fields)}"
                )

            # Check for newline at end of keystrokes if followed by another command
            # 后面还有命令但本条 keystrokes 不以换行结尾 → 警告：两条命令会被
            # 拼接在同一行执行（终端语义陷阱）
            if i < len(commands_data) - 1 and not keystrokes.endswith("\n"):
                warnings.append(
                    f"Command {i + 1} should end with newline when followed "
                    "by another command. Otherwise the two commands will be "
                    "concatenated together on the same line."
                )

            # 构造 ParsedCommand（duration 统一转成 float）
            commands.append(
                ParsedCommand(keystrokes=keystrokes, duration=float(duration))
            )

        # 全部命令解析成功：返回列表与空错误串
        return commands, ""

    def _get_auto_fixes(self):
        """Return list of auto-fix functions to try in order."""
        # 自动修复清单（按序尝试）：先补缺失右括号，再从混合文本抽取 JSON
        return [
            (
                "Fixed incomplete JSON by adding missing closing brace",
                self._fix_incomplete_json,
            ),
            ("Extracted JSON from mixed content", self._fix_mixed_content),
        ]

    def _fix_incomplete_json(self, response: str, error: str) -> tuple[str, bool]:
        """Fix incomplete JSON by adding missing closing braces."""
        # 仅当错误表明是"JSON 没写完/被截断"类问题时才尝试此修复
        if (
            "Invalid JSON" in error
            or "Expecting" in error
            or "Unterminated" in error
            or "No valid JSON found" in error
        ):
            # Try adding closing braces
            # 统计未闭合的左花括号数量（{ 数减 } 数）
            brace_count = response.count("{") - response.count("}")
            if brace_count > 0:
                # 在末尾补足等量的右花括号后返回修正文本
                fixed = response + "}" * brace_count
                return fixed, True
        # 不适用本修复：原样返回并标记未修改
        return response, False

    def _fix_mixed_content(self, response: str, error: str) -> tuple[str, bool]:
        """Extract JSON from response with mixed content."""
        # Look for JSON-like patterns
        # 用正则找出所有"长得像 JSON 对象"的片段（允许一层嵌套，DOTALL 跨行）
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                # 逐个候选片段尝试反序列化，能解析通过的即为有效 JSON
                json.loads(match)
                return match, True
            except json.JSONDecodeError:
                # 该片段不是合法 JSON → 继续试下一个
                continue

        # 没有可抽取的有效 JSON：原样返回并标记未修改
        return response, False

    def _combine_warnings(self, auto_warning: str, existing_warning: str) -> str:
        """Combine auto-correction warning with existing warnings."""
        # 已有警告 → 自动纠错警告置于最前，换行拼接
        if existing_warning:
            return f"- {auto_warning}\n{existing_warning}"
        else:
            # 没有既有警告 → 只有自动纠错一条
            return f"- {auto_warning}"

    def _check_field_order(
        self, data: dict[str, Any], response: str, warnings: list[str]
    ) -> None:
        """Check if fields appear in the correct order: analysis, plan, commands."""
        # Expected order for required fields
        # 约定的字段出现顺序
        expected_order = ["analysis", "plan", "commands"]

        # Find positions of each field in the original response
        # 在原始响应文本中定位各字段的字节偏移
        positions = {}
        for field in expected_order:
            # Look for the field name in quotes
            # 匹配形如 "field": 的键名（带引号、后跟冒号）
            pattern = f'"({field})"\\s*:'
            match = re.search(pattern, response)
            if match:
                positions[field] = match.start()

        # Check if we have at least 2 fields to compare order
        # 少于 2 个字段无从比较顺序 → 直接返回
        if len(positions) < 2:
            return

        # Get fields that are present, in the order they appear
        # 收集实际出现的 (字段, 位置) 对
        present_fields = []
        for field in expected_order:
            if field in positions:
                present_fields.append((field, positions[field]))

        # Sort by position to get actual order
        # 按位置排序得到响应中的真实字段顺序
        actual_order = [
            field for field, pos in sorted(present_fields, key=lambda x: x[1])
        ]

        # Get expected order for present fields only
        # 仅针对出现字段计算期望顺序（缺字段不影响比较）
        expected_present = [f for f in expected_order if f in positions]

        # Compare orders
        # 两种顺序不一致 → 追加顺序警告（软约束，帮助模型养成稳定格式）
        if actual_order != expected_present:
            actual_str = " → ".join(actual_order)
            expected_str = " → ".join(expected_present)
            warnings.append(
                f"Fields appear in wrong order. Found: {actual_str}, "
                f"expected: {expected_str}"
            )

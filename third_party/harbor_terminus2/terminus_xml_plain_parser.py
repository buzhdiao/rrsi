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

# 正则：解析 <response>/<commands>/<keystrokes> 等 XML 标签结构
import re
# dataclass：声明 ParsedCommand / ParseResult 数据类
from dataclasses import dataclass
# Any：结构化提取函数的返回类型标注
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


# Terminus 2 的 XML 纯文本响应解析器：把 <response>...</response> 格式的模型回复
# 解析成命令与完成标记；手工正则解析（非严格 XML 库），并内置"补缺失 </response>
# 闭合标签"的自动修复与截断响应抢救（salvage）逻辑。
class TerminusXMLPlainParser:
    """Parser for terminus XML plain response format."""

    def __init__(self):
        # 约定的必填小节：推理（analysis）、计划（plan）、命令（commands）
        self.required_sections = ["analysis", "plan", "commands"]

    def parse_response(self, response: str) -> ParseResult:
        """
        Parse a terminus XML plain response and extract commands.

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
        Try to parse a terminus XML plain response.

        Args:
            response: The full LLM response string

        Returns:
            ParseResult with commands, completion status, errors and warnings
        """
        warnings = []

        # Check for extra text before/after <response> tags
        # 检查 <response> 标签前后是否有额外文本（只警告，不拒绝）
        self._check_extra_text(response, warnings)

        # Extract <response> content
        # 截取 <response> 与 </response> 之间的正文
        response_content = self._extract_response_content(response)
        if not response_content:
            # 找不到 <response> 标签 → 硬错误（语句跨多行）
            return ParseResult(
                [],
                False,
                "No <response> tag found",
                "- " + "\n- ".join(warnings) if warnings else "",
                "",
                "",
            )

        # Check if task is complete first
        # 先判断任务完成标记（<task_complete>true</task_complete>）
        is_complete = self._check_task_complete(response_content)

        # Check for required sections and extract content
        # 提取各小节（analysis/plan/commands/task_complete）内容并做结构警告
        sections = self._extract_sections(response_content, warnings)

        # Extract analysis and plan for reasoning content
        # 抽取推理内容：analysis（局面分析）与 plan（下一步计划），供轨迹记录
        analysis = sections.get("analysis", "")
        plan = sections.get("plan", "")

        # Extract commands section
        # 取命令小节内容
        commands_content = sections.get("commands", "")
        if not commands_content:
            if "commands" in sections:
                # Commands section exists but is empty
                # <commands> 小节存在但为空（自闭合或空标签）
                if not is_complete:
                    # 任务未完成却没有任何动作 → 警告模型正确表达"等待"
                    warnings.append(
                        "Commands section is empty; not taking any action. "
                        "If you want to wait a specific amount of time please use "
                        "`sleep`, but if you're waiting for a command to finish then "
                        "continue to wait."
                    )
                # 返回无命令但保留完成标记的结果
                return ParseResult(
                    [],
                    is_complete,
                    "",
                    "- " + "\n- ".join(warnings) if warnings else "",
                    analysis,
                    plan,
                )
            else:
                # Commands section is missing entirely
                # <commands> 小节整体缺失
                if is_complete:
                    # 已宣告完成 → 缺 commands 可接受（无事可做）
                    return ParseResult(
                        [],
                        True,
                        "",
                        "- " + "\n- ".join(warnings) if warnings else "",
                        analysis,
                        plan,
                    )
                # 未完成又缺 commands → 硬错误
                return ParseResult(
                    [],
                    False,
                    "Missing <commands> section",
                    "- " + "\n- ".join(warnings) if warnings else "",
                    analysis,
                    plan,
                )

        # Parse commands directly from XML (no code blocks in plain format)
        # 直接从 XML 解析命令（plain 格式无代码块包裹）
        commands, parse_error = self._parse_xml_commands(commands_content, warnings)
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

    def _get_auto_fixes(self):
        """Return list of auto-fix functions to try in order."""
        # 自动修复清单：目前只有"补缺失的 </response> 闭合标签"一种
        return [
            (
                "Missing </response> tag was automatically inserted",
                self._fix_missing_response_tag,
            ),
        ]

    def _combine_warnings(self, auto_warning: str, existing_warning: str) -> str:
        """Combine auto-correction warning with existing warnings."""
        # 已有警告 → 自动纠错警告置于最前，换行拼接
        if existing_warning:
            return f"- {auto_warning}\n{existing_warning}"
        else:
            # 没有既有警告 → 只有自动纠错一条
            return f"- {auto_warning}"

    def _fix_missing_response_tag(self, response: str, error: str) -> tuple[str, bool]:
        """Fix missing </response> closing tag by appending it."""
        # 仅当错误确为"缺少 </response> 闭合标签"时才尝试修复
        if "Missing </response> closing tag" not in error:
            return response, False

        # Simply append </response> at the end
        # 去掉尾部空白后补上闭合标签（语句负责把截断的响应急救回来）
        corrected = response.rstrip() + "\n</response>"
        return corrected, True

    def _check_extra_text(self, response: str, warnings: list[str]) -> None:
        """Check for extra text before/after <response> tags."""
        # Find response tag positions
        # 定位开/闭标签位置
        start_pos = response.find("<response>")
        end_pos = response.find("</response>")

        if start_pos == -1:
            # 找不到开标签 → 此处直接返回，稍后作为错误统一处理
            return  # Will be handled as error later

        # Check text before <response>
        # 检查 <response> 之前是否混有杂文本
        before_text = response[:start_pos].strip()
        if before_text:
            warnings.append("Extra text detected before <response> tag")

        # Check text after </response> if closing tag exists
        # 闭合标签存在时，检查其后是否混有杂文本
        if end_pos != -1:
            after_text = response[end_pos + len("</response>") :].strip()
            if after_text:
                warnings.append("Extra text detected after </response> tag")

                # Count total <response> tags
                # 统计 <response> 开标签总数：多于一个说明模型连发了多个响应块
                total_response_count = response.count("<response>")
                if total_response_count > 1:
                    # 重要警告：一次只应发一个 <response> 块，只有第一个被执行
                    warnings.append(
                        f"IMPORTANT: Only issue one <response> block at a time. "
                        f"You issued {total_response_count} and only the first "
                        f"was executed."
                    )

    def _extract_response_content(self, response: str) -> str:
        """Extract content from <response> tags."""
        # 定位 <response> 开标签
        start_pos = response.find("<response>")
        if start_pos == -1:
            # 没有开标签 → 返回空串（由调用方报错）
            return ""

        # 从开标签之后寻找闭合标签
        end_pos = response.find("</response>", start_pos)
        if end_pos == -1:
            # Missing closing tag - return content from opening tag to end
            # 缺闭合标签（如输出被截断）→ 退化取开标签到文末的内容
            return response[start_pos + len("<response>") :].strip()

        # 正常情况：取开闭标签之间的正文并去除首尾空白
        return response[start_pos + len("<response>") : end_pos].strip()

    def _extract_sections(self, content: str, warnings: list[str]) -> dict[str, Any]:
        """Extract analysis, plan, commands, and task_complete sections."""
        # 各小节内容表：名字 -> 提取出的文本
        sections = {}
        # 已发现的小节集合（用于缺失检查）
        found_sections = set()

        # Define patterns for each section
        # 每个小节三种形态的匹配模式：完整标签 / 自闭合 / 空标签对
        # （字典字面量跨多行，注释置于语句起始处）
        section_patterns = {
            "analysis": (
                r"<analysis>(.*?)</analysis>",
                r"<analysis\s*/>",
                r"<analysis></analysis>",
            ),
            "plan": (r"<plan>(.*?)</plan>", r"<plan\s*/>", r"<plan></plan>"),
            "commands": (
                r"<commands>(.*?)</commands>",
                r"<commands\s*/>",
                r"<commands></commands>",
            ),
            "task_complete": (
                r"<task_complete>(.*?)</task_complete>",
                r"<task_complete\s*/>",
                r"<task_complete></task_complete>",
            ),
        }

        for section_name, patterns in section_patterns.items():
            # 解包三种模式：完整 / 自闭合 / 空
            full_pattern, self_closing_pattern, empty_pattern = patterns

            # Try full pattern first
            # 优先尝试完整标签对（DOTALL 让内容跨行匹配，取捕获组并去空白）
            match = re.search(full_pattern, content, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
                found_sections.add(section_name)
                continue

            # Try self-closing pattern
            # 自闭合（如 <commands/>）→ 视为空内容
            if re.search(self_closing_pattern, content):
                sections[section_name] = ""  # Self-closing = empty content
                found_sections.add(section_name)
                continue

            # Try empty pattern
            # 空标签对（如 <plan></plan>）→ 同样视为空内容
            if re.search(empty_pattern, content):
                sections[section_name] = ""  # Empty = empty content
                found_sections.add(section_name)
                continue

        # Check for missing required sections
        # 必填小节缺失检查：analysis/plan/commands 缺任一则警告
        required = set(self.required_sections)
        missing = required - found_sections
        for section in missing:
            if section != "task_complete":  # task_complete is optional
                # task_complete 是可选小节，缺失不警告
                warnings.append(f"Missing <{section}> section")

        # Check for unexpected tags at the direct child level of <response> only
        # Find all top-level tags (not nested inside other tags)
        # 扫描 <response> 直接子级（顶层）标签，找出约定之外的多余标签
        top_level_tags = self._find_top_level_tags(content)
        expected_tags = set(self.required_sections + ["task_complete"])
        unexpected = set(top_level_tags) - expected_tags
        for tag in unexpected:
            # 未知顶层标签 → 警告（软约束，帮助模型纠正格式漂移）
            warnings.append(
                f"Unknown tag found: <{tag}>, expected "
                f"analysis/plan/commands/task_complete"
            )

        # Check for multiple instances of same tag
        # 同一标签多次出现检查：正则匹配开标签或自闭合形态
        for section_name in self.required_sections + ["task_complete"]:
            tag_count = len(re.findall(f"<{section_name}(?:\\s|>|/>)", content))
            if tag_count > 1:
                if section_name == "commands":
                    # 多个 <commands> 块：重要警告，只有第一个被执行
                    warnings.append(
                        f"IMPORTANT: Only issue one <commands> block at a time. "
                        f"You issued {tag_count} and only the first was executed."
                    )
                else:
                    # 其他小节重复出现：普通警告
                    warnings.append(f"Multiple <{section_name}> sections found")

        # Check for correct order of sections (analysis, plan, commands)
        # 检查小节出现顺序是否符合约定（软约束）
        self._check_section_order(content, warnings)

        # 返回提取到的各小节内容
        return sections

    def _parse_xml_commands(
        self, xml_content: str, warnings: list[str]
    ) -> tuple[list[ParsedCommand], str]:
        """Parse XML content and extract command objects manually."""

        # Find all keystrokes elements manually for better error reporting
        # 手工正则解析 <keystrokes> 元素（不用 XML 库，便于给出更友好的错误报告）
        commands = []
        # 捕获组 1 = 标签属性串，组 2 = 按键内容（DOTALL 允许内容跨行）
        keystrokes_pattern = re.compile(
            r"<keystrokes([^>]*)>(.*?)</keystrokes>", re.DOTALL
        )

        # 找出全部 <keystrokes> 元素并逐条处理
        matches = keystrokes_pattern.findall(xml_content)
        for i, (attributes_str, keystrokes_content) in enumerate(matches):
            # Check for attribute issues
            # 检查属性书写问题（缺引号/单引号/未知属性名），只警告
            self._check_attribute_issues(attributes_str, i + 1, warnings)

            # Parse attributes
            # duration 属性默认 1.0 秒
            duration = 1.0

            # Parse duration attribute
            # 从属性串中提取 duration="..." 或 duration='...'
            duration_match = re.search(
                r'duration\s*=\s*["\']([^"\']*)["\']', attributes_str
            )
            if duration_match:
                try:
                    # 属性值转浮点数
                    duration = float(duration_match.group(1))
                except ValueError:
                    # 数值非法 → 警告并保留默认值
                    warnings.append(
                        f"Command {i + 1}: Invalid duration value "
                        f"'{duration_match.group(1)}', using default 1.0"
                    )
            else:
                # 缺 duration 属性 → 警告并使用默认值
                warnings.append(
                    f"Command {i + 1}: Missing duration attribute, using default 1.0"
                )

            # Check for newline at end of keystrokes
            # 后面还有命令但本条不以换行结尾 → 警告：两条命令会被拼接在同一行
            if i < len(matches) - 1 and not keystrokes_content.endswith("\n"):
                warnings.append(
                    f"Command {i + 1} should end with newline when followed "
                    f"by another command. Otherwise the two commands will be "
                    f"concatenated together on the same line."
                )

            # 构造 ParsedCommand 加入结果
            commands.append(
                ParsedCommand(keystrokes=keystrokes_content, duration=duration)
            )

        # Check for XML entities and warn
        # XML 实体检查：本解析器按字面读取实体、不做反转义，
        # 模型若误用 &lt; 等会原样进入终端，必须警告
        entities = {
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&apos;": "'",
        }
        for entity, char in entities.items():
            if entity in xml_content:
                # 检测到实体 → 强警告（除非确实想让这些字符原样出现）
                warnings.append(
                    f"Warning: {entity} is read verbatim and not converted to {char}. "
                    f"NEVER USE {entity}, unless you want these exact characters to "
                    f"appear directly in the output."
                )

        # Check for \r\n line endings and warn
        # 检查字面 \r\n 序列（模型把两个字符写进内容）→ 提示用 \n 更简洁
        if "\\r\\n" in xml_content:
            warnings.append(
                "Warning: \\r\\n line endings are not necessary - use \\n "
                "instead for simpler output"
            )

        # 命令解析完成（本函数不产生硬错误，问题都以警告呈现）
        return commands, ""

    def _find_top_level_tags(self, content: str) -> list[str]:
        """Find all top-level XML tags (direct children of response), not
        nested tags."""
        # 顶层标签名收集列表
        top_level_tags = []
        # 当前嵌套深度：0 表示处于 <response> 的直接子级
        depth = 0
        # 扫描游标
        i = 0

        while i < len(content):
            if content[i] == "<":
                # Find the end of this tag
                # 找到当前标签的 '>' 结束位置
                tag_end = content.find(">", i)
                if tag_end == -1:
                    # 标签未闭合（截断）→ 停止扫描
                    break

                # 取出标签内部文本（不含尖括号）
                tag_content = content[i + 1 : tag_end]

                # Skip comments, CDATA, etc.
                # 跳过注释（<!--）与声明（<?）等非结构标签
                if tag_content.startswith("!") or tag_content.startswith("?"):
                    i = tag_end + 1
                    continue

                # Check if this is a closing tag
                # 闭合标签（</xxx>）→ 深度减一，继续
                if tag_content.startswith("/"):
                    depth -= 1
                    i = tag_end + 1
                    continue

                # Check if this is a self-closing tag
                # 是否自闭合标签（如 <commands/>），自闭合不增加深度
                is_self_closing = tag_content.endswith("/")

                # Extract tag name (first word)
                # 提取标签名（第一个空白前的词；无属性则取整段）
                tag_name = tag_content.split()[0] if " " in tag_content else tag_content
                if tag_name.endswith("/"):
                    # 形如 <commands/> 时名字会带 '/'，去掉
                    tag_name = tag_name[:-1]

                # If we're at depth 0, this is a top-level tag
                # 深度为 0 → 属于 <response> 直接子级的顶层标签，记录
                if depth == 0:
                    top_level_tags.append(tag_name)

                # Adjust depth for opening tags (but not self-closing)
                # 开标签使深度加一（自闭合除外）
                if not is_self_closing:
                    depth += 1

                # 游标推进到标签结束之后
                i = tag_end + 1
            else:
                # 普通文本字符 → 游标前移
                i += 1

        # 返回全部顶层标签名
        return top_level_tags

    def _check_section_order(self, content: str, warnings: list[str]) -> None:
        """Check if sections appear in the correct order: analysis, plan, commands."""
        # Find positions of each section
        # 定位各小节开标签在内容中的偏移
        positions = {}
        for section in ["analysis", "plan", "commands"]:
            # Look for opening tags
            # 匹配开标签或自闭合形态（后接空白、'>' 或 '/>'）
            match = re.search(f"<{section}(?:\\s|>|/>)", content)
            if match:
                positions[section] = match.start()

        # Check if we have at least 2 sections to compare order
        # 少于 2 个小节无从比较顺序 → 直接返回
        if len(positions) < 2:
            return

        # Expected order
        # 约定的小节出现顺序
        expected_order = ["analysis", "plan", "commands"]

        # Get sections that are present, in the order they appear
        # 收集实际出现的 (小节, 位置) 对
        present_sections = []
        for section in expected_order:
            if section in positions:
                present_sections.append((section, positions[section]))

        # Sort by position to get actual order
        # 按位置排序得到响应中的真实小节顺序
        actual_order = [
            section for section, pos in sorted(present_sections, key=lambda x: x[1])
        ]

        # Get expected order for present sections only
        # 仅针对出现小节计算期望顺序（缺小节不影响比较）
        expected_present = [s for s in expected_order if s in positions]

        # Compare orders
        # 两种顺序不一致 → 追加顺序警告（软约束）
        if actual_order != expected_present:
            actual_str = " → ".join(actual_order)
            expected_str = " → ".join(expected_present)
            warnings.append(
                f"Sections appear in wrong order. Found: {actual_str}, "
                f"expected: {expected_str}"
            )

    def _check_attribute_issues(
        self, attributes_str: str, command_num: int, warnings: list[str]
    ) -> None:
        """Check for attribute-related issues."""
        # Check for missing quotes
        # 属性值缺引号检查：值未用引号包裹 → 警告并示范正确写法
        unquoted_pattern = re.compile(r'(\w+)\s*=\s*([^"\'\s>]+)')
        unquoted_matches = unquoted_pattern.findall(attributes_str)
        for attr_name, attr_value in unquoted_matches:
            warnings.append(
                f"Command {command_num}: Attribute '{attr_name}' value should be "
                f'quoted: {attr_name}="{attr_value}"'
            )

        # Check for single quotes (should use double quotes for consistency)
        # 单引号检查：为格式统一建议改用双引号（仅警告）
        single_quote_pattern = re.compile(r"(\w+)\s*=\s*'([^']*)'")
        single_quote_matches = single_quote_pattern.findall(attributes_str)
        for attr_name, attr_value in single_quote_matches:
            warnings.append(
                f"Command {command_num}: Use double quotes for attribute "
                f"'{attr_name}': {attr_name}=\"{attr_value}\""
            )

        # Check for unknown attribute names
        # 未知属性名检查：keystrokes 元素目前只认 duration
        known_attributes = {"duration"}
        all_attributes = re.findall(r"(\w+)\s*=", attributes_str)
        for attr_name in all_attributes:
            if attr_name not in known_attributes:
                # 列出全部已知属性，帮助模型自我纠正
                warnings.append(
                    f"Command {command_num}: Unknown attribute '{attr_name}' - "
                    f"known attributes are: {', '.join(sorted(known_attributes))}"
                )

    def _check_task_complete(self, response_content: str) -> bool:
        """Check if the response indicates the task is complete."""
        # Check for <task_complete>true</task_complete>
        # 只有内容为 true（忽略大小写、允许两侧空白）才视为完成宣告
        true_match = re.search(
            r"<task_complete>\s*true\s*</task_complete>",
            response_content,
            re.IGNORECASE,
        )
        if true_match:
            # 模型宣告任务完成
            return True

        # All other cases (false, empty, self-closing, missing) = not complete
        # 其余一切形态（false/空/自闭合/缺失）都视为任务未完成
        return False

    def salvage_truncated_response(
        self, truncated_response: str
    ) -> tuple[str | None, bool]:
        """
        Try to salvage a valid response from truncated output.

        Args:
            truncated_response: The truncated response from the LLM

        Returns:
            Tuple of (salvaged_response, has_multiple_blocks)
            - salvaged_response: Clean response up to </response> if salvageable, "
                "None otherwise
            - has_multiple_blocks: True if multiple response/commands blocks "
                "were detected
        """
        # Check if we can find a complete response structure
        # 先确认截断输出里存在完整的 </commands> 闭合标签
        commands_end = truncated_response.find("</commands>")
        if commands_end == -1:
            # 命令块都没闭合 → 无法抢救
            return None, False

        # Find the </response> tag after </commands>
        # 再找 </commands> 之后的 </response> 闭合标签
        response_end = truncated_response.find("</response>", commands_end)
        if response_end == -1:
            # 响应块未闭合 → 无法抢救
            return None, False

        # We have a complete response up to at least </commands>
        # Truncate cleanly at </response>
        # 在 </response> 处干净截断，得到一个结构完整的响应文本
        clean_response = truncated_response[: response_end + len("</response>")]

        # Check if this is a valid response with no critical issues
        # 对截断后的文本做完整解析，验证没有关键问题
        try:
            parse_result = self.parse_response(clean_response)

            # Check if there are no errors and no "multiple blocks" warnings
            # 检查是否出现"一次只应发一个块"的多块警告
            has_multiple_blocks = False
            if parse_result.warning:
                warning_lower = parse_result.warning.lower()
                has_multiple_blocks = (
                    "only issue one" in warning_lower
                    and "block at a time" in warning_lower
                )

            if not parse_result.error and not has_multiple_blocks:
                # Valid response! Return the clean truncated version
                # 有效响应！返回干净截断版（等效于模型少说废话但动作有效）
                return clean_response, False
            else:
                # Has errors or multiple blocks
                # 有错误或多块问题 → 不可抢救，返回多块标记供上层提示模型
                return None, has_multiple_blocks

        except Exception:
            # If parsing fails, not salvageable
            # 解析本身失败 → 视为不可抢救
            return None, False

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
ReSum: Context Summarization for Long-Horizon Agent Tasks.

Simplified implementation based on arXiv:2509.13313.
Uses fraction-based trigger and incremental summarization.
"""

# 导入类型注解工具 Any，用于表示任意类型的值（如 extra_args 字典的值类型）。
from typing import Any

# 从 litellm（统一多家 LLM API 的库）导入：Choices 回复选项类型、
# get_model_info 模型元信息查询、token_counter 按模型分词器计数工具。
from litellm import Choices, get_model_info, token_counter
# 导入 loguru 日志器；logger.bind 可为日志附加结构化字段（如 message_type）。
from loguru import logger

# 导入消息相关工具：消息类型定义及取角色/内容/属性、内容转字符串的辅助函数。
from runner.agents.models import (
    LitellmAnyMessage,
    LitellmOutputMessage,
    content_to_str,
    get_msg_attr,
    get_msg_content,
    get_msg_role,
)
# 导入统一的 LLM 调用封装 generate_response（异步生成模型回复）。
from runner.utils.llm import generate_response

# Defaults
# 触发比例（ReSum 论文的"按比例触发"机制）：上下文 token 达到最大窗口的 70% 时执行摘要压缩。
TRIGGER_FRACTION = 0.70  # Summarize when context reaches 70% of max
# 摘要时原样保留最近 N 条消息，避免最新工作状态被压缩进摘要而失真。
KEEP_RECENT_MESSAGES = 10  # Keep last N messages verbatim

# 摘要提示词模板（论文中的 Context Summarization）：要求把智能体会话压缩为
# 结构化的"推理状态"（任务目标/进展/当前状态/关键细节四段式）。
# {conversation} 为占位符，运行时填入"旧摘要 + 新活动"的拼接文本。
SUMMARY_PROMPT = """Summarize this AI agent's work session into a compact reasoning state.

{conversation}

Create a structured summary:

## Task & Goal
What is the agent trying to accomplish?

## Progress
- Actions taken and key results
- Important discoveries

## Current State
- Where is the agent now?
- What are the next steps?

## Key Details
- File paths, function names, values
- Error messages (exact text)
- URLs, IDs, configuration

Be specific. Include concrete values needed to continue."""


def _is_synthetic_summary_message(msg: LitellmAnyMessage) -> bool:
    """Check if a message is a synthetic summary injected by _build_output."""
    # 合成摘要消息一定是 user 角色，其他角色直接判否。
    if get_msg_role(msg) != "user":
        return False
    # 取出消息内容以便检查前缀。
    content = get_msg_content(msg)
    # 内容为字符串且以 _build_output 写入的固定标题开头，即为合成摘要消息。
    return isinstance(content, str) and content.startswith(
        "## Summary of Previous Work"
    )


def _find_safe_cut_index(messages: list[LitellmAnyMessage], target_keep: int) -> int:
    """
    Find a safe index to cut the message list without orphaning tool messages.

    Tool result messages reference tool_call_ids from their preceding assistant message.
    If we cut between an assistant message and its tool results, the tool results become
    orphaned and LLM APIs will reject the conversation.

    Returns the index where the "recent" portion should start.
    """
    # 消息总数不超过要保留的条数时无需裁剪，从索引 0 开始全部保留。
    if len(messages) <= target_keep:
        return 0

    # Start with the naive cut point
    # 先按朴素方式计算切分点：总长度减去要保留的最近消息数。
    cut_index = len(messages) - target_keep

    # Walk backwards from cut_index to find a safe boundary
    # Safe = not starting with orphaned tool messages
    # 从切分点向前回退：若切分点落在 tool 消息上，把它切出去会使其父
    # assistant 消息（携带对应 tool_call_id）丢失，形成"孤儿 tool 消息"，
    # LLM API 会拒绝这种对话结构，因此要退到非 tool 消息处。
    while cut_index > 0 and get_msg_role(messages[cut_index]) == "tool":
        cut_index -= 1

    # Now cut_index points to either:
    # - A non-tool message (safe to cut here)
    # - Index 0 (keep everything)

    # 返回"最近部分"的起始索引：此处必是非 tool 消息或已是 0（全部保留）。
    return cut_index


class ReSumManager:
    """
    Simple ReSum context manager.

    Triggers summarization when context reaches TRIGGER_FRACTION of max tokens.
    Uses incremental summarization to update a running summary.
    """

    def __init__(self, model: str, extra_args: dict[str, Any] | None = None):
        # 保存用于生成摘要的模型名。
        self.model: str = model
        # 保存调用 LLM 的额外参数（如 temperature 等）；None 时退化为空字典。
        self.extra_args: dict[str, Any] = extra_args or {}
        # 当前累积的运行摘要（running summary，论文中的滚动笔记概念）；未生成时为 None。
        self.running_summary: str | None = None
        # 已被摘要覆盖的非 system 消息条数，作为下一轮增量摘要的起点。
        self.messages_summarized: int = 0
        # 模型最大输入 token 数（上下文窗口），查询失败时用默认值。
        self.max_tokens: int = 128000

        # Full message history preserved across summarizations.
        # When summarize() is called, the pre-summarization messages are
        # captured here so the trajectory output can include the complete
        # history (not just the post-summarization window).
        # 跨摘要保留的完整消息历史：每次真正执行摘要前，把未摘要的消息追加
        # 到这里，供轨迹（trajectory）输出完整历史而非仅摘要后的窗口。
        self._pre_summarization_history: list[LitellmAnyMessage] = []

        # Get model context window
        # 尝试查询模型的上下文窗口大小；失败则沿用默认值 128000。
        try:
            # 从 litellm 内置元信息表查询模型信息。
            model_info = get_model_info(model)
            # 优先取 max_input_tokens，其次 max_tokens，均缺失时退回 128000。
            self.max_tokens = (
                model_info.get("max_input_tokens")
                or model_info.get("max_tokens")
                or 128000
            )
        except Exception:
            # 模型未知等异常直接吞掉，保持默认窗口值即可正常工作。
            pass

    def _get_token_count(self, messages: list[LitellmAnyMessage]) -> int:
        """Estimate token count."""
        # 优先用 litellm 的 token_counter 按模型分词器较精确地计数。
        try:
            return token_counter(model=self.model, messages=messages)
        except Exception:
            # 失败时退化为粗估：按"4 字符 ≈ 1 token"折算字符串内容的总字符数。
            total_chars = sum(
                len(c) if isinstance(c := get_msg_content(m), str) else 0
                for m in messages
            )
            return total_chars // 4

    def get_full_history(
        self, current_messages: list[LitellmAnyMessage]
    ) -> list[LitellmAnyMessage]:
        """Return the complete message history including pre-summarization messages.

        If no summarization has occurred, returns current_messages as-is.
        Otherwise, returns system messages + all pre-summarization messages +
        any messages added after the last summarization.
        """
        # 从未做过摘要时，当前列表就是完整历史，原样返回。
        if not self._pre_summarization_history:
            return current_messages

        # 把当前消息拆分为 system 组与非 system 组。
        system_msgs: list[LitellmAnyMessage] = []
        non_system: list[LitellmAnyMessage] = []
        for m in current_messages:
            (system_msgs if get_msg_role(m) == "system" else non_system).append(m)

        # The synthetic summary message marks the boundary of the last
        # summarization. Everything after the last synthetic marker is new.
        # Using position rather than filtering avoids silently dropping real
        # messages if _build_output ever keeps recent messages as live objects.
        # 找最后一个合成摘要消息的位置：它是上一次摘要的边界标记，
        # 其后的消息都是上次摘要之后新增的。
        last_synthetic = -1
        for i, m in enumerate(non_system):
            if _is_synthetic_summary_message(m):
                last_synthetic = i

        # 新消息 = 最后一个合成摘要标记之后的所有消息。
        new_messages = non_system[last_synthetic + 1 :]
        # 完整历史 = system 消息 + 摘要前保存的历史 + 摘要后的新消息。
        return system_msgs + self._pre_summarization_history + new_messages

    def should_summarize(self, messages: list[LitellmAnyMessage]) -> bool:
        """Check if we should summarize (context > TRIGGER_FRACTION of max)."""
        # 过滤掉 system 消息，只统计实际对话消息。
        non_system = [m for m in messages if get_msg_role(m) != "system"]

        # Calculate how many new messages since last summarization
        # 计算自上次摘要以来的新增消息数（当前总数 - 已摘要覆盖数）。
        new_messages = len(non_system) - self.messages_summarized

        # Need enough new messages beyond what we'll keep
        # 新增消息还填不满"保留最近 N 条"的窗口时不必摘要，直接返回否。
        if new_messages <= KEEP_RECENT_MESSAGES:
            return False

        # 估算当前上下文的 token 总数。
        current_tokens = self._get_token_count(messages)
        # 计算触发阈值 = 最大窗口 token 数 × 触发比例（默认 70%）。
        threshold = self.max_tokens * TRIGGER_FRACTION
        # 超过阈值才触发摘要（ReSum 的按比例触发条件）。
        return current_tokens > threshold

    async def summarize(
        self, messages: list[LitellmAnyMessage]
    ) -> list[LitellmAnyMessage]:
        """Summarize messages, keeping recent ones verbatim."""
        # non_system includes synthetic summary messages — intentionally.
        # The incremental summarization relies on the combined summary+recent
        # message being in to_summarize so the LLM incorporates the "Recent
        # Activity" content into the next running_summary.
        # 把消息拆分为 system 组与非 system 组（合成摘要消息也归入后者）。
        system_messages: list[LitellmAnyMessage] = []
        non_system: list[LitellmAnyMessage] = []
        for m in messages:
            (system_messages if get_msg_role(m) == "system" else non_system).append(m)

        # 非 system 消息不超过保留窗口时无需摘要，原样返回。
        if len(non_system) <= KEEP_RECENT_MESSAGES:
            return messages

        # Find safe cut point that doesn't orphan tool messages
        # 求不产生孤儿 tool 消息的安全切分点，并切出原样保留的"最近部分"。
        cut_index = _find_safe_cut_index(non_system, KEEP_RECENT_MESSAGES)
        recent = non_system[cut_index:]

        # Calculate the range of messages to summarize
        # Start from where we left off, end at the safe cut point
        # 待摘要区间 = [上次摘要覆盖的起点, 本次安全切分点)，即增量部分。
        summarize_end = cut_index
        summarize_start = self.messages_summarized

        # Guard: nothing to summarize if we've already summarized up to the cut point
        # 已摘要进度已越过切分点：没有新内容可摘要，仅重建输出结构。
        if summarize_start >= summarize_end:
            return self._build_output(system_messages, recent)

        # 截取本轮真正要送去摘要的消息区间。
        to_summarize = non_system[summarize_start:summarize_end]

        # 区间为空同样只重建输出、不做 LLM 调用。
        if not to_summarize:
            return self._build_output(system_messages, recent)

        # Capture history only when we're actually about to summarize.
        # Placed after early returns to avoid duplication if summarize()
        # is called but returns without summarizing.
        # Filter out synthetic summary messages — only real messages belong
        # in the trajectory history.
        # 仅在确认要摘要时才保存历史（放在早退分支之后避免重复追加）；
        # 过滤掉合成摘要消息，轨迹历史中只保留真实消息。
        self._pre_summarization_history.extend(
            m for m in non_system if not _is_synthetic_summary_message(m)
        )

        # 记录日志：本轮摘要的消息条数（附加 message_type=resum 便于日志过滤）。
        logger.bind(message_type="resum").info(
            f"Summarizing {len(to_summarize)} messages"
        )

        # Format messages for summarization
        # 把待摘要消息格式化为纯文本对话。
        formatted = self._format_messages(to_summarize)

        # If we have a running summary, include it
        # 已有运行摘要时，把"旧摘要 + 新活动"拼接为输入（增量摘要的核心）。
        if self.running_summary:
            conversation = (
                f"## Previous Summary:\n{self.running_summary}\n\n"
                f"## New Activity:\n{formatted}"
            )
        else:
            # 首次摘要：只有新活动文本。
            conversation = formatted

        # Generate summary
        # 用摘要提示词调用 LLM，生成本轮合并后的运行摘要。
        summary = await self._call_llm(SUMMARY_PROMPT.format(conversation=conversation))

        # 更新运行摘要，供下一轮增量摘要与输出构建使用。
        self.running_summary = summary

        # Build output first, then reset counter to reflect new list structure
        # 先构建输出消息列表（system 消息 + 摘要/最近活动合并的 user 消息）。
        output = self._build_output(system_messages, recent)

        # After _build_output, the message list is: [system_messages..., combined_user_message]
        # The combined_user_message contains both the summary AND formatted recent activity.
        # Reset to 0 so that on the next summarization cycle, this combined message
        # (at index 0 of non-system) gets included in to_summarize, ensuring the
        # "Recent Activity" content is incorporated into the next LLM-generated summary.
        # 重建后把已摘要计数清零：下一轮摘要时这条合并消息（位于非 system 第 0 条）
        # 会进入待摘要区间，从而把"最近活动"内容融入下一次生成的摘要。
        self.messages_summarized = 0

        # 返回压缩后的新消息列表。
        return output

    def _build_output(
        self, system_messages: list[LitellmAnyMessage], recent: list[LitellmAnyMessage]
    ) -> list[LitellmAnyMessage]:
        """
        Build output with summary + recent context as a single user message.

        This avoids issues with Anthropic's extended thinking requirement
        (assistant messages with tool_calls must have thinking_blocks).
        By converting everything to a user message, we sidestep the requirement.
        """
        # 结果列表先复制所有 system 消息，保证系统提示仍位于最前。
        result: list[LitellmAnyMessage] = list(system_messages)

        # Build content: summary + formatted recent messages
        # 用片段列表拼装合并 user 消息的文本内容。
        content_parts: list[str] = []

        # 已有运行摘要时，追加"此前工作摘要"段落。
        if self.running_summary:
            content_parts.append(
                f"## Summary of Previous Work\n\n{self.running_summary}"
            )

        # 存在保留的最近消息时，追加"最近活动"段落（格式化文本）。
        if recent:
            content_parts.append(
                f"## Recent Activity\n\n{self._format_messages(recent)}"
            )

        # 至少有一个片段时，把内容合并成单条 user 消息追加到结果。
        if content_parts:
            result.append(
                LitellmOutputMessage(
                    role="user",
                    content="\n\n---\n\n".join(content_parts)
                    + "\n\n---\n\nContinue from this state.",
                )
            )

        # 返回形如 [system..., 合并 user 消息] 的新对话列表。
        return result

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for summarization."""
        # 调用统一的 LLM 封装：system 指示"总结智能体会话"，user 传入摘要提示词；
        # 不挂载任何工具（tools=[]），响应超时 300 秒，并透传额外参数。
        response = await generate_response(
            model=self.model,
            messages=[
                LitellmOutputMessage(
                    role="system", content="Summarize AI agent work sessions."
                ),
                LitellmOutputMessage(role="user", content=prompt),
            ],
            tools=[],
            llm_response_timeout=300,
            extra_args=self.extra_args,
        )

        # 校验返回结构：没有 choices 或元素类型不符都视为摘要失败并抛错。
        if not response.choices or not isinstance(response.choices[0], Choices):
            raise ValueError("Summarization returned empty response")

        # Some providers return content as a list (e.g. Anthropic); normalize to string.
        # 部分供应商把内容返回为分段列表（如 Anthropic），统一规范化为字符串。
        content = content_to_str(response.choices[0].message.content)
        # 摘要内容为空同样视为失败，抛出异常。
        if not content:
            raise ValueError("Summarization returned empty content")

        # 返回生成的摘要文本。
        return content

    def _format_messages(self, messages: list[LitellmAnyMessage]) -> str:
        """Format messages for summarization prompt, including tool calls."""
        # 收集每条消息格式化后的文本片段。
        parts: list[str] = []

        for msg in messages:
            # 角色转大写，用于文本标注（ASSISTANT/USER/TOOL 等）。
            role = get_msg_role(msg).upper()
            # 取原始内容；非字符串内容（如多模态分段列表）按空串处理。
            raw_content = get_msg_content(msg)
            content = raw_content if isinstance(raw_content, str) else ""

            # 普通消息内容超过 2000 字符时截断并标注，控制摘要输入规模。
            if len(content) > 2000:
                content = content[:2000] + "\n[truncated]"

            # tool 消息：标注工具名，且截断阈值更严格（1000 字符）。
            if get_msg_role(msg) == "tool":
                name = get_msg_attr(msg, "name", "unknown")
                if len(content) > 1000:
                    content = content[:1000] + "\n[truncated]"
                parts.append(f"**TOOL ({name})**: {content}")
            elif get_msg_role(msg) == "assistant":
                # Include tool_calls for assistant messages
                # assistant 消息要附带其发起的工具调用信息。
                tool_calls = get_msg_attr(msg, "tool_calls")
                if tool_calls:
                    # 把工具调用列表格式化为文本。
                    tool_calls_str = self._format_tool_calls(tool_calls)
                    if content:
                        # 既有文本又有工具调用时两者都保留。
                        parts.append(f"**ASSISTANT**: {content}\n{tool_calls_str}")
                    else:
                        # 只有工具调用没有文本时仅输出调用部分。
                        parts.append(f"**ASSISTANT**: {tool_calls_str}")
                else:
                    # 无工具调用的普通 assistant 消息。
                    parts.append(f"**ASSISTANT**: {content}")
            else:
                # 其他角色（如 user）按统一"角色: 内容"格式输出。
                parts.append(f"**{role}**: {content}")

        # 各片段以空行连接成整段对话文本。
        return "\n\n".join(parts)

    def _format_tool_calls(self, tool_calls: list[object]) -> str:
        """Format tool calls for display in summaries."""
        # 首行为固定标题，其后每行渲染一个工具调用。
        lines: list[str] = ["Tool calls:"]
        for tc in tool_calls:
            # 预声明名称与参数两个字符串变量，供两种形态分支赋值。
            name: str
            args: str
            # 形态一：Pydantic 模型对象，带 .function 属性。
            if hasattr(tc, "function"):
                # Pydantic model (LitellmOutputMessage)
                func = getattr(tc, "function", None)
                # 缺少 function 字段则跳过该条调用。
                if func is None:
                    continue
                # 从 function 对象取工具名与参数 JSON 字符串。
                name = getattr(func, "name", "unknown")
                args = getattr(func, "arguments", "{}")
            elif isinstance(tc, dict) and "function" in tc:
                # TypedDict
                # 形态二：字典（TypedDict），从 function 子字典中取字段。
                func_dict = tc["function"]
                name = (
                    func_dict.get("name", "unknown")
                    if isinstance(func_dict, dict)
                    else "unknown"
                )
                args = (
                    func_dict.get("arguments", "{}")
                    if isinstance(func_dict, dict)
                    else "{}"
                )
            else:
                # 两种形态都不匹配则无法解析，跳过。
                continue

            # Truncate long arguments
            # 参数字符串过长时截断到 200 字符，防止摘要输入膨胀。
            if len(args) > 200:
                args = args[:200] + "..."
            # 以"  - 工具名(参数)"的形式追加一行。
            lines.append(f"  - {name}({args})")

        # 多行拼接为文本返回。
        return "\n".join(lines)

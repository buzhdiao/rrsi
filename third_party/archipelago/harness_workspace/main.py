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
ReAct Toolbelt Agent with ReSum Context Management.
"""

# 导入 asyncio：提供异步事件循环、超时与任务保护（shield）等设施。
import asyncio
# 导入 time：用于统计运行耗时（time.time()）。
import time
# 导入 Any：宽泛类型注解（如 client 参数）。
from typing import Any

# 导入 fastmcp 的 Client：以编程方式连接 MCP（Model Context Protocol）服务器。
from fastmcp import Client as FastMCPClient
# 导入 litellm 的 Choices 类型，用于校验模型回复结构。
from litellm import Choices
# 导入 litellm 的上下文超限与超时异常，用于针对性错误处理。
from litellm.exceptions import ContextWindowExceededError, Timeout
# 导入 litellm 的 MCP 客户端辅助：调用 OpenAI 格式工具、加载 MCP 工具清单。
from litellm.experimental_mcp_client import call_openai_tool, load_mcp_tools
# 导入 litellm 的 ModelResponse 类型，作为 LLM 回复的类型注解。
from litellm.files.main import ModelResponse
# 导入 loguru 日志器。
from loguru import logger
# 导入 OpenAI 格式的工具参数类型（工具 schema 的统一表示）。
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

# 导入智能体输入/输出/消息等数据类型定义。
from runner.agents.models import (
    AgentRunInput,
    AgentStatus,
    AgentTrajectoryOutput,
    LitellmAnyMessage,
    LitellmInputMessage,
    LitellmOutputMessage,
)
# 导入错误分类工具：判断致命 MCP 错误与系统级错误。
from runner.utils.error import is_fatal_mcp_error, is_system_error
# 导入统一的 LLM 调用封装。
from runner.utils.llm import generate_response
# 导入 MCP 相关工具：构造网关连接配置、内容块转消息、清理被丢弃的异步任务。
from runner.utils.mcp import (
    build_mcp_gateway_schema,
    content_blocks_to_messages,
    drain_shielded_task,
)
# 导入 token 用量统计器。
from runner.utils.usage import UsageTracker

# 导入 ReSum 上下文摘要管理器（长时程任务的上下文压缩，见 arXiv:2509.13313）。
from .resum import ReSumManager
# 导入工具结果截断器（head_tail 策略，防止超大输出进入上下文）。
from .tool_result import truncate_tool_messages
# 导入终止工具、元工具定义与元工具处理器。
from .tools import (
    FINAL_ANSWER_TOOL,
    META_TOOL_NAMES,
    META_TOOLS,
    MetaToolHandler,
    parse_final_answer,
)


class ReActAgent:
    """ReAct Toolbelt Agent with ReSum context management."""

    def __init__(self, run_input: AgentRunInput):
        # 保存本次运行的轨迹（trajectory）唯一标识。
        self.trajectory_id: str = run_input.trajectory_id
        # 保存编排模型（orchestrator，即主决策 LLM）的名称。
        self.model: str = run_input.orchestrator_model
        # 初始化对话消息列表：拷贝一份初始消息，避免修改调用方数据。
        self.messages: list[LitellmAnyMessage] = list(run_input.initial_messages)

        # 工具带智能体必须通过 MCP 网关访问工具，缺少地址直接报错。
        if run_input.mcp_gateway_url is None:
            raise ValueError("MCP gateway URL is required for react toolbelt agent")

        # 用网关地址、认证 token 与 actor id 构造 MCP 客户端连接配置。
        self.mcp_client = FastMCPClient(
            build_mcp_gateway_schema(
                run_input.mcp_gateway_url,
                run_input.mcp_gateway_auth_token,
                run_input.mcp_gateway_actor_id,
            )
        )

        # Config
        # 读取智能体配置字典。
        config = run_input.agent_config_values
        # 整次运行的总超时（秒），默认 3 小时。
        self.timeout: int = config.get("timeout", 10800)
        # ReAct 循环的最大步数，默认 250 步。
        self.max_steps: int = config.get("max_steps", 250)
        # 单次工具调用的超时（秒），默认 1 小时。
        self.tool_call_timeout: int = config.get("tool_call_timeout", 3600)
        # 单次 LLM 响应的超时（秒），默认 10 分钟。
        self.llm_response_timeout: int = config.get("llm_response_timeout", 600)
        # 工具带容量上限：最多同时装载 80 个工具。
        self.max_toolbelt_size: int = 80

        # 保存编排模型的额外调用参数（None 时退化为空字典）。
        self.extra_args: dict[str, Any] = run_input.orchestrator_extra_args or {}

        # Components
        # 创建 ReSum 上下文摘要管理器（与主模型相同、共享额外参数）。
        self.resum: ReSumManager = ReSumManager(self.model, self.extra_args)

        # Toolbelt state
        # 全部可用工具：名字到 schema 的映射（由 MCP 网关加载）。
        self.all_tools: dict[str, ChatCompletionToolParam] = {}
        # 当前工具带：已装载的工具名集合（初始为空）。
        self.toolbelt: set[str] = set()
        # 元工具处理器（延迟创建，工具加载后初始化）。
        self.meta_tool_handler: MetaToolHandler | None = None

        # Agent state
        # 是否已收到 final_answer（运行终止标志）。
        self._finalized: bool = False
        # 最终答案文本。
        self._final_answer: str | None = None
        # 最终任务状态（completed/blocked/failed），默认 completed。
        self._final_status: str = "completed"
        # 运行状态枚举，初始为 PENDING。
        self.status: AgentStatus = AgentStatus.PENDING
        # 起始时间戳（用于统计耗时），未开始时为 None。
        self.start_time: float | None = None
        # 创建 token 用量统计器。
        self._usage_tracker: UsageTracker = UsageTracker()

    def _get_tools(self) -> list[ChatCompletionToolParam]:
        """Get tools for LLM: meta-tools + toolbelt + final_answer."""
        # 从全部工具中按工具带集合取出已装载工具的 schema 列表。
        toolbelt_tools = [self.all_tools[name] for name in self.toolbelt]
        # 每次请求都带上：元工具 + 已装载工具 + 终止工具。
        return list(META_TOOLS) + toolbelt_tools + [FINAL_ANSWER_TOOL]

    async def _initialize_tools(self, client: Any) -> None:
        """Load tools from MCP gateway."""
        # 通过 litellm 的 MCP 客户端从网关拉取工具清单（OpenAI 格式）。
        tools: list[ChatCompletionToolParam] = await load_mcp_tools(
            client.session, format="openai"
        )  # pyright: ignore[reportAssignmentType]

        for tool in tools:
            # 逐个提取工具名。
            name = tool.get("function", {}).get("name")
            # 名字有效时登记进全部工具映射。
            if name:
                self.all_tools[name] = tool

        # 工具加载完毕后创建元工具处理器（共享工具带集合的引用）。
        self.meta_tool_handler = MetaToolHandler(
            self.all_tools, self.toolbelt, self.max_toolbelt_size
        )

        # 记录配置日志：工具总数与名字列表（工具带初始为空）。
        logger.bind(
            message_type="configure",
            payload=list(self.all_tools.keys()),
        ).info(f"Loaded {len(self.all_tools)} MCP tools (toolbelt starts empty)")

    async def step(self, client: Any) -> None:
        """Execute one step of the ReAct loop."""
        # Proactive ReSum check
        # ReAct 每步开始前先做 ReSum 主动检查：上下文超阈值就先压缩。
        if self.resum.should_summarize(self.messages):
            logger.bind(message_type="resum").info("Summarizing context")
            try:
                # 执行摘要，用压缩后的消息列表替换当前列表。
                self.messages = await self.resum.summarize(self.messages)
            except Exception as e:
                # 摘要失败不终止运行，仅记录错误（下一步还会再尝试）。
                logger.error(f"Summarization failed: {e}")

        # Call LLM
        try:
            # 调用编排模型：传入消息、当前可用工具与超时等参数。
            response: ModelResponse = await generate_response(
                self.model,
                self.messages,
                self._get_tools(),
                self.llm_response_timeout,
                self.extra_args,
                trajectory_id=self.trajectory_id,
            )
        except ContextWindowExceededError:
            # 上下文超限（被动触发）：立即做 ReSum 摘要压缩，本步结束等待下步重试。
            logger.warning("Context exceeded, summarizing")
            self.messages = await self.resum.summarize(self.messages)
            return
        except Timeout:
            # LLM 超时：记录错误并返回，外层循环会在下一步重试。
            logger.error("LLM timeout")
            return
        except Exception as e:
            # 其他 LLM 错误：记录后向上抛出（可能是系统级错误）。
            logger.error(f"LLM error: {e}")
            raise

        # 统计本次调用的 token 用量。
        self._usage_tracker.track(response)
        # 取出回复的候选列表。
        choices = response.choices
        # 空回复或结构异常时：追加一条"继续"的用户消息重新提示模型。
        if not choices or not isinstance(choices[0], Choices):
            logger.bind(message_type="step").warning(
                "LLM returned an empty response with no choices, re-prompting with 'continue'"
            )
            self.messages.append(
                LitellmOutputMessage(
                    role="user", content="Continue. Use final_answer when done."
                )
            )
            return

        # 把模型回复消息规范化为本项目的消息类型。
        response_message = LitellmOutputMessage.model_validate(choices[0].message)
        # 取出模型发起的工具调用列表（可能为 None）。
        tool_calls = getattr(response_message, "tool_calls", None)
        # 取出模型的文本内容（可能为 None）。
        content = getattr(response_message, "content", None)

        # Log reasoning if present (o1/reasoning models)
        # 若模型返回了推理内容（reasoning 模型如 o1），记录到日志。
        if getattr(response_message, "reasoning_content", None):
            logger.bind(message_type="reasoning").info(
                response_message.reasoning_content
            )

        # Log thinking blocks if present (Claude extended thinking)
        # 若模型返回了思考块（Claude 扩展思考），逐块记录到调试日志。
        if getattr(response_message, "thinking_blocks", None):
            if isinstance(response_message.thinking_blocks, list):
                for thinking_block in response_message.thinking_blocks:
                    if thinking_block.get("thinking"):
                        logger.bind(message_type="thinking").debug(
                            thinking_block.get("thinking")
                        )

        # Log response content
        # 模型有文本输出时记录到日志。
        if content:
            logger.bind(message_type="response").info(content)

        # Log tool call summary
        # 有工具调用时记录调用了哪些工具。
        if tool_calls:
            tool_names = [tc.function.name for tc in tool_calls]
            logger.bind(message_type="step").info(
                f"Calling {len(tool_calls)} tool(s): {', '.join(tool_names)}"
            )
        elif not content:
            # 既无文本也无工具调用：记录告警及结束原因（finish_reason）。
            logger.bind(message_type="step").warning("No content and no tool calls")
            try:
                finish_reason = choices[0].finish_reason if choices else None
                logger.bind(message_type="step").warning(
                    f"(finish_reason={finish_reason})"
                )
            except Exception as e:
                logger.error(f"Error getting finish reason: {e}")

        # 把模型回复消息追加进对话历史（含可能的 tool_calls）。
        self.messages.append(response_message)

        if tool_calls:
            # 有工具调用则进入工具执行流程。
            await self._handle_tool_calls(client, tool_calls)
        else:
            # 无工具调用时追加提示，要求模型要么调用工具要么用 final_answer 收尾。
            self.messages.append(
                LitellmOutputMessage(
                    role="user",
                    content="No tools called. Use final_answer to submit your answer. Please continue completing the task.",
                )
            )

    async def _handle_tool_calls(self, client: Any, tool_calls: list[Any]) -> None:
        """Process tool calls."""
        # 收集真正的 MCP 工具调用，批量延后执行。
        mcp_tool_calls: list[Any] = []

        for tool_call in tool_calls:
            # 取出本次调用的工具名。
            name = tool_call.function.name

            # Final answer - validate todos, then handle and return
            # 终止工具：先校验待办清单，再登记最终答案并结束本步。
            if name == "final_answer":
                # Check for incomplete todos
                # 处理器必须已初始化（断言保证内部一致性）。
                assert self.meta_tool_handler
                # 取出所有未完成（未 completed/cancelled）的待办项。
                incomplete = self.meta_tool_handler.get_incomplete_todos()
                if incomplete:
                    # 拼接未完成任务的可读列表。
                    incomplete_list = ", ".join(
                        f"'{t.id}' ({t.status.value})" for t in incomplete
                    )
                    # 组装错误信息：要求先把待办标记为完成或取消。
                    error_msg = (
                        f"ERROR: Cannot submit final_answer with incomplete todos. "
                        f"You have {len(incomplete)} incomplete task(s): {incomplete_list}. "
                        f"Use todo_write to mark each as 'completed' or 'cancelled' first."
                    )
                    # 记录拒绝日志。
                    logger.bind(message_type="tool").warning(
                        f"final_answer rejected: {len(incomplete)} incomplete todos"
                    )
                    # 以 tool 消息形式把错误回传给模型（对应此次 tool_call）。
                    self.messages.append(
                        LitellmOutputMessage(
                            role="tool",
                            tool_call_id=tool_call.id,
                            name="final_answer",
                            content=error_msg,
                        )
                    )
                    # 本步结束，运行继续（模型需先处理待办）。
                    return

                # 校验通过：解析最终答案文本与任务状态。
                answer, status = parse_final_answer(tool_call.function.arguments)
                # 记录最终答案日志。
                logger.bind(message_type="final_answer").info(answer)

                # 置终止标志并保存答案与状态。
                self._finalized = True
                self._final_answer = answer
                self._final_status = status

                # 把答案以 tool 消息形式写回对话（保持消息结构合法）。
                self.messages.append(
                    LitellmOutputMessage(
                        role="tool",
                        tool_call_id=tool_call.id,
                        name="final_answer",
                        content=answer,
                    )
                )
                return

            # Meta-tool - handle locally
            # 元工具（工具带管理/待办规划）：本地直接处理，不走 MCP。
            if name in META_TOOL_NAMES:
                # 记录元工具调用日志（含参数）。
                logger.bind(
                    message_type="tool_call",
                    ref=tool_call.id,
                    name=name,
                    payload=tool_call.function.arguments,
                ).info(f"Meta-tool: {name}")
                assert self.meta_tool_handler
                # 交给元工具处理器执行，返回 JSON 字符串结果。
                result = self.meta_tool_handler.handle(
                    name, tool_call.function.arguments
                )
                # 记录元工具结果日志。
                logger.bind(
                    message_type="tool_result",
                    ref=tool_call.id,
                    name=name,
                    payload=result,
                ).info(f"Meta-tool {name} completed")
                # 结果以 tool 消息追加进对话。
                self.messages.append(
                    LitellmOutputMessage(
                        role="tool",
                        tool_call_id=tool_call.id,
                        name=name,
                        content=result,
                    )
                )
                continue

            # MCP tool - collect for batch execution
            # 普通 MCP 工具：先收集，循环结束后统一执行。
            mcp_tool_calls.append(tool_call)

        # Execute MCP tools (using shared client connection)
        # 延迟注入的图片消息缓冲（图片以独立 user 消息延后追加）。
        deferred_image_messages: list[LitellmInputMessage] = []
        for tool_call in mcp_tool_calls:
            # 逐个执行 MCP 工具调用（复用同一个客户端连接）。
            await self._execute_mcp_tool(client, tool_call, deferred_image_messages)
        # 全部执行完后把延迟的图片消息追加进对话。
        self.messages.extend(deferred_image_messages)

    async def _execute_mcp_tool(
        self,
        client: Any,
        tool_call: Any,
        deferred_image_messages: list[LitellmInputMessage],
    ) -> None:
        """Execute an MCP tool call."""
        # 取出要执行的工具名。
        name = tool_call.function.name

        # 未装载进工具带的工具不允许直接调用。
        if name not in self.toolbelt:
            self.messages.append(
                LitellmOutputMessage(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=name,
                    content=f"Error: '{name}' not in toolbelt. Use toolbelt_add_tool first.",
                )
            )
            return

        # 创建绑定本次调用 ref 与工具名的日志上下文。
        tool_logger = logger.bind(
            ref=tool_call.id,
            name=name,
        )
        # 记录工具调用日志（含参数 payload）。
        tool_logger.bind(
            message_type="tool_call",
            payload=tool_call.function.arguments,
        ).info(f"Calling tool {name}")

        # 派生一个专用于结果日志的 logger。
        tool_result_logger = tool_logger.bind(message_type="tool_result")

        # 把 MCP 工具调用包装成后台任务（ensure_future 立即调度）。
        shielded_task = asyncio.ensure_future(
            call_openai_tool(client.session, tool_call)
        )
        try:
            # 用 shield 保护任务并限时等待：超时后任务本身仍可被安全排空。
            result = await asyncio.wait_for(
                asyncio.shield(shielded_task),
                timeout=self.tool_call_timeout,
            )
        except TimeoutError:
            # 工具调用超时：记录错误、排空被保护的任务并回写超时结果。
            tool_result_logger.error(f"Tool call {name} timed out")
            await drain_shielded_task(shielded_task)
            self.messages.append(
                LitellmOutputMessage(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=name,
                    content="Tool call timed out",
                )
            )
            return
        except Exception as e:
            if is_fatal_mcp_error(e):
                # 致命 MCP 错误（如连接断开）：记录后向上抛出，终止整次运行。
                tool_result_logger.error(f"Fatal MCP error, ending run: {repr(e)}")
                self.messages.append(
                    LitellmOutputMessage(
                        role="tool",
                        tool_call_id=tool_call.id,
                        name=name,
                        content=f"Fatal error: {e}",
                    )
                )
                raise
            # 非致命错误：记录并把错误文本作为工具结果回传给模型。
            tool_result_logger.error(f"Error calling tool {name}: {repr(e)}")
            self.messages.append(
                LitellmOutputMessage(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=name,
                    content=f"Error: {e}",
                )
            )
            return

        # 工具成功但无内容：也回写一条占位结果，保持消息配对完整。
        if not result.content:
            tool_result_logger.error(f"Tool {name} returned no content")
            self.messages.append(
                LitellmOutputMessage(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=name,
                    content="No content returned",
                )
            )
            return

        # 记录成功日志（payload 为各内容块的字典形式）。
        tool_result_logger.bind(
            payload=[block.model_dump() for block in result.content],
        ).info(f"Tool {name} called successfully")

        # 把 MCP 内容块转换为工具结果消息（图片等大块延后注入）。
        messages = content_blocks_to_messages(
            result.content,
            tool_call.id,
            name,
            self.model,
            deferred_image_messages=deferred_image_messages,
        )
        # 对工具结果文本做 head_tail 截断，防止超大输出进入上下文。
        truncate_tool_messages(messages, self.model)
        # 把处理后的消息追加进对话历史。
        self.messages.extend(messages)

    def _build_output(self) -> AgentTrajectoryOutput:
        # 组装轨迹输出：完整历史（含摘要前消息）、最终状态、耗时与用量统计。
        return AgentTrajectoryOutput(
            messages=self.resum.get_full_history(self.messages),
            status=self.status,
            time_elapsed=time.time() - self.start_time if self.start_time else 0,
            usage=self._usage_tracker.to_dict(),
        )

    async def run(self) -> AgentTrajectoryOutput:
        """Run the agent loop with a single MCP connection."""
        try:
            # 用总超时包裹整个运行过程。
            async with asyncio.timeout(self.timeout):
                # Single MCP connection for entire agent lifecycle
                # 整个生命周期复用一条 MCP 连接（async with 管理连接）。
                async with self.mcp_client as client:
                    logger.info(f"Starting ReAct Toolbelt agent with {self.model}")
                    # 初始化工具清单与元工具处理器。
                    await self._initialize_tools(client)

                    # 记录起始时间，状态置为运行中。
                    self.start_time = time.time()
                    self.status = AgentStatus.RUNNING

                    for step in range(self.max_steps):
                        # 已收到 final_answer 则提前退出循环。
                        if self._finalized:
                            logger.info(f"Finalized after {step} steps")
                            break
                        # 记录当前步数（从 1 开始计数）。
                        logger.bind(message_type="step").info(
                            f"Starting step {step + 1}"
                        )
                        # 执行一步 ReAct 循环（LLM 决策 + 工具执行）。
                        await self.step(client)

                    if not self._finalized:
                        # 达到最大步数仍未终止：标记运行失败。
                        logger.error(f"Not finalized after {self.max_steps} steps")
                        self.status = AgentStatus.FAILED
                    else:
                        # 正常终止：标记完成。
                        self.status = AgentStatus.COMPLETED

                    # 返回轨迹输出。
                    return self._build_output()

        except TimeoutError:
            # 总超时：标记错误状态并输出已有轨迹。
            logger.error(f"Timeout after {self.timeout}s")
            self.status = AgentStatus.ERROR
            return self._build_output()

        except asyncio.CancelledError:
            # 被外部取消：标记取消状态并输出已有轨迹。
            logger.error("Cancelled")
            self.status = AgentStatus.CANCELLED
            return self._build_output()

        except Exception as e:
            # 其他异常：按系统错误/任务失败分类标记状态。
            logger.error(f"Error: {e}")
            self.status = (
                AgentStatus.ERROR if is_system_error(e) else AgentStatus.FAILED
            )
            return self._build_output()


async def run(run_input: AgentRunInput) -> AgentTrajectoryOutput:
    """Entry point for the ReAct Toolbelt agent."""
    # 入口函数：创建智能体实例并运行，返回轨迹输出。
    return await ReActAgent(run_input).run()

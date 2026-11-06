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

import asyncio
import time
from typing import Any

from fastmcp import Client as FastMCPClient
from litellm import Choices
from litellm.exceptions import ContextWindowExceededError, Timeout
from litellm.experimental_mcp_client import call_openai_tool, load_mcp_tools
from litellm.files.main import ModelResponse
from loguru import logger
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from runner.agents.models import (
    AgentRunInput,
    AgentStatus,
    AgentTrajectoryOutput,
    LitellmAnyMessage,
    LitellmInputMessage,
    LitellmOutputMessage,
)
from runner.utils.error import is_fatal_mcp_error, is_system_error
from runner.utils.llm import generate_response
from runner.utils.mcp import (
    build_mcp_gateway_schema,
    content_blocks_to_messages,
    drain_shielded_task,
)
from runner.utils.usage import UsageTracker

# ReSum 上下文管理与工具结果截断(可进化的既有安全机制,批评家红线所在)
from .resum import ReSumManager
from .tool_result import truncate_tool_messages
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
        # 运行标识与冻结策略模型
        self.trajectory_id: str = run_input.trajectory_id
        self.model: str = run_input.orchestrator_model
        # 消息历史从初始消息(system + task prompt)开始
        self.messages: list[LitellmAnyMessage] = list(run_input.initial_messages)

        if run_input.mcp_gateway_url is None:
            raise ValueError("MCP gateway URL is required for react toolbelt agent")

        # MCP 网关客户端:代理的所有领域工具经它访问
        self.mcp_client = FastMCPClient(
            build_mcp_gateway_schema(
                run_input.mcp_gateway_url,
                run_input.mcp_gateway_auth_token,
                run_input.mcp_gateway_actor_id,
            )
        )

        # Config
        # 运行配置:总超时/最大步数/单工具超时/单次 LLM 响应超时/工具带上限
        config = run_input.agent_config_values
        self.timeout: int = config.get("timeout", 10800)
        self.max_steps: int = config.get("max_steps", 250)
        self.tool_call_timeout: int = config.get("tool_call_timeout", 3600)
        self.llm_response_timeout: int = config.get("llm_response_timeout", 600)
        self.max_toolbelt_size: int = 80

        self.extra_args: dict[str, Any] = run_input.orchestrator_extra_args or {}

        # Components
        # ReSum 上下文管理器:在上下文爆炸前主动摘要压缩
        self.resum: ReSumManager = ReSumManager(self.model, self.extra_args)

        # Toolbelt state
        # 全量工具表(从网关加载)与当前工具带(代理自选的工具子集)
        self.all_tools: dict[str, ChatCompletionToolParam] = {}
        self.toolbelt: set[str] = set()
        # 元工具处理器(todo_write/toolbelt_* 等本地实现)
        self.meta_tool_handler: MetaToolHandler | None = None

        # Agent state
        # 终态标志与最终答案/状态;运行状态与计时;用量跟踪
        self._finalized: bool = False
        self._final_answer: str | None = None
        self._final_status: str = "completed"
        self.status: AgentStatus = AgentStatus.PENDING
        self.start_time: float | None = None
        self._usage_tracker: UsageTracker = UsageTracker()

    def _get_tools(self) -> list[ChatCompletionToolParam]:
        """Get tools for LLM: meta-tools + toolbelt + final_answer."""
        # 本步暴露给 LLM 的工具 = 元工具 + 工具带内工具 + final_answer
        toolbelt_tools = [self.all_tools[name] for name in self.toolbelt]
        return list(META_TOOLS) + toolbelt_tools + [FINAL_ANSWER_TOOL]

    async def _initialize_tools(self, client: Any) -> None:
        """Load tools from MCP gateway."""
        # 从网关拉取全部工具并转成 OpenAI 格式
        tools: list[ChatCompletionToolParam] = await load_mcp_tools(
            client.session, format="openai"
        )  # pyright: ignore[reportAssignmentType]

        for tool in tools:
            name = tool.get("function", {}).get("name")
            if name:
                self.all_tools[name] = tool

        # 元工具处理器持有全量工具表与工具带(动态装卸)
        self.meta_tool_handler = MetaToolHandler(
            self.all_tools, self.toolbelt, self.max_toolbelt_size
        )

        logger.bind(
            message_type="configure",
            payload=list(self.all_tools.keys()),
        ).info(f"Loaded {len(self.all_tools)} MCP tools (toolbelt starts empty)")

    async def step(self, client: Any) -> None:
        """Execute one step of the ReAct loop."""
        # Proactive ReSum check
        # 每步先做 ReSum 主动检查:上下文接近上限则摘要压缩
        if self.resum.should_summarize(self.messages):
            logger.bind(message_type="resum").info("Summarizing context")
            try:
                self.messages = await self.resum.summarize(self.messages)
            except Exception as e:
                logger.error(f"Summarization failed: {e}")

        # Call LLM
        try:
            # 调用冻结策略:模型 + 消息 + 本步工具集 + 超时
            response: ModelResponse = await generate_response(
                self.model,
                self.messages,
                self._get_tools(),
                self.llm_response_timeout,
                self.extra_args,
                trajectory_id=self.trajectory_id,
            )
        except ContextWindowExceededError:
            # 上下文爆炸:立即摘要压缩后返回,下一步重试
            logger.warning("Context exceeded, summarizing")
            self.messages = await self.resum.summarize(self.messages)
            return
        except Timeout:
            # LLM 超时:本步放弃,循环继续
            logger.error("LLM timeout")
            return
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

        # 记录 token 用量
        self._usage_tracker.track(response)
        choices = response.choices
        if not choices or not isinstance(choices[0], Choices):
            # 空响应:补一条 user 消息要求继续,防止卡死
            logger.bind(message_type="step").warning(
                "LLM returned an empty response with no choices, re-prompting with 'continue'"
            )
            self.messages.append(
                LitellmOutputMessage(
                    role="user", content="Continue. Use final_answer when done."
                )
            )
            return

        # 把响应转成消息并提取工具调用与文本内容
        response_message = LitellmOutputMessage.model_validate(choices[0].message)
        tool_calls = getattr(response_message, "tool_calls", None)
        content = getattr(response_message, "content", None)

        # Log reasoning if present (o1/reasoning models)
        # 有推理内容(o1/reasoning 模型)则记录
        if getattr(response_message, "reasoning_content", None):
            logger.bind(message_type="reasoning").info(
                response_message.reasoning_content
            )

        # Log thinking blocks if present (Claude extended thinking)
        # 有思考块(Claude extended thinking)则逐块记录
        if getattr(response_message, "thinking_blocks", None):
            if isinstance(response_message.thinking_blocks, list):
                for thinking_block in response_message.thinking_blocks:
                    if thinking_block.get("thinking"):
                        logger.bind(message_type="thinking").debug(
                            thinking_block.get("thinking")
                        )

        # Log response content
        if content:
            logger.bind(message_type="response").info(content)

        # Log tool call summary
        if tool_calls:
            tool_names = [tc.function.name for tc in tool_calls]
            logger.bind(message_type="step").info(
                f"Calling {len(tool_calls)} tool(s): {', '.join(tool_names)}"
            )
        elif not content:
            # 既无内容也无工具调用:记录 finish_reason 便于诊断
            logger.bind(message_type="step").warning("No content and no tool calls")
            try:
                finish_reason = choices[0].finish_reason if choices else None
                logger.bind(message_type="step").warning(
                    f"(finish_reason={finish_reason})"
                )
            except Exception as e:
                logger.error(f"Error getting finish reason: {e}")

        # 响应消息入历史
        self.messages.append(response_message)

        if tool_calls:
            # 有工具调用则分发执行
            await self._handle_tool_calls(client, tool_calls)
        else:
            # 没有工具调用:推一条 user 消息催促提交,避免空转
            self.messages.append(
                LitellmOutputMessage(
                    role="user",
                    content="No tools called. Use final_answer to submit your answer. Please continue completing the task.",
                )
            )

    async def _handle_tool_calls(self, client: Any, tool_calls: list[Any]) -> None:
        """Process tool calls."""
        # 收集需要走 MCP 的工具调用(其余在本地处理)
        mcp_tool_calls: list[Any] = []

        for tool_call in tool_calls:
            name = tool_call.function.name

            # Final answer - validate todos, then handle and return
            # final_answer:先校验 todo 全部完成,再解析并落终态
            if name == "final_answer":
                # Check for incomplete todos
                # 校验未完成 todo:有则拒绝提交并回错误消息
                assert self.meta_tool_handler
                incomplete = self.meta_tool_handler.get_incomplete_todos()
                if incomplete:
                    incomplete_list = ", ".join(
                        f"'{t.id}' ({t.status.value})" for t in incomplete
                    )
                    error_msg = (
                        f"ERROR: Cannot submit final_answer with incomplete todos. "
                        f"You have {len(incomplete)} incomplete task(s): {incomplete_list}. "
                        f"Use todo_write to mark each as 'completed' or 'cancelled' first."
                    )
                    logger.bind(message_type="tool").warning(
                        f"final_answer rejected: {len(incomplete)} incomplete todos"
                    )
                    self.messages.append(
                        LitellmOutputMessage(
                            role="tool",
                            tool_call_id=tool_call.id,
                            name="final_answer",
                            content=error_msg,
                        )
                    )
                    return

                # todo 全部完成:解析答案与状态,置终态标志
                answer, status = parse_final_answer(tool_call.function.arguments)
                logger.bind(message_type="final_answer").info(answer)

                self._finalized = True
                self._final_answer = answer
                self._final_status = status

                # 最终答案作为工具结果入历史
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
            # 元工具(todo_write/toolbelt_*):本地处理,结果作为工具消息入历史
            if name in META_TOOL_NAMES:
                logger.bind(
                    message_type="tool_call",
                    ref=tool_call.id,
                    name=name,
                    payload=tool_call.function.arguments,
                ).info(f"Meta-tool: {name}")
                assert self.meta_tool_handler
                result = self.meta_tool_handler.handle(
                    name, tool_call.function.arguments
                )
                logger.bind(
                    message_type="tool_result",
                    ref=tool_call.id,
                    name=name,
                    payload=result,
                ).info(f"Meta-tool {name} completed")
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
            # 普通领域工具:收集起来批量执行
            mcp_tool_calls.append(tool_call)

        # Execute MCP tools (using shared client connection)
        # 复用共享连接逐个执行 MCP 工具;图片消息延迟合并(避免拆散多模态对)
        deferred_image_messages: list[LitellmInputMessage] = []
        for tool_call in mcp_tool_calls:
            await self._execute_mcp_tool(client, tool_call, deferred_image_messages)
        self.messages.extend(deferred_image_messages)

    async def _execute_mcp_tool(
        self,
        client: Any,
        tool_call: Any,
        deferred_image_messages: list[LitellmInputMessage],
    ) -> None:
        """Execute an MCP tool call."""
        name = tool_call.function.name

        if name not in self.toolbelt:
            # 工具不在工具带上:拒绝并提示先 toolbelt_add_tool
            self.messages.append(
                LitellmOutputMessage(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=name,
                    content=f"Error: '{name}' not in toolbelt. Use toolbelt_add_tool first.",
                )
            )
            return

        # 绑定本次调用的日志上下文
        tool_logger = logger.bind(
            ref=tool_call.id,
            name=name,
        )
        tool_logger.bind(
            message_type="tool_call",
            payload=tool_call.function.arguments,
        ).info(f"Calling tool {name}")

        tool_result_logger = tool_logger.bind(message_type="tool_result")

        # 把工具调用包成受保护任务(shield),等待超时也不取消底层执行
        shielded_task = asyncio.ensure_future(
            call_openai_tool(client.session, tool_call)
        )
        try:
            result = await asyncio.wait_for(
                asyncio.shield(shielded_task),
                timeout=self.tool_call_timeout,
            )
        except TimeoutError:
            # 超时:排空底层任务防止悬挂,回超时消息
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
                # 致命 MCP 错误(如连接失效):记录后向上抛,结束整个运行
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
            # 普通错误:作为工具结果回给模型,运行继续
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

        if not result.content:
            # 空结果:回占位错误消息
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

        tool_result_logger.bind(
            payload=[block.model_dump() for block in result.content],
        ).info(f"Tool {name} called successfully")

        # 把内容块转成消息(图片延迟入列),再做工具结果截断,然后入历史
        messages = content_blocks_to_messages(
            result.content,
            tool_call.id,
            name,
            self.model,
            deferred_image_messages=deferred_image_messages,
        )
        truncate_tool_messages(messages, self.model)
        self.messages.extend(messages)

    def _build_output(self) -> AgentTrajectoryOutput:
        # 组装输出:完整历史(含被摘要的原文)、状态、耗时、用量
        return AgentTrajectoryOutput(
            messages=self.resum.get_full_history(self.messages),
            status=self.status,
            time_elapsed=time.time() - self.start_time if self.start_time else 0,
            usage=self._usage_tracker.to_dict(),
        )

    async def run(self) -> AgentTrajectoryOutput:
        """Run the agent loop with a single MCP connection."""
        try:
            # 整个运行限时
            async with asyncio.timeout(self.timeout):
                # Single MCP connection for entire agent lifecycle
                # 整个生命周期复用一条 MCP 连接
                async with self.mcp_client as client:
                    logger.info(f"Starting ReAct Toolbelt agent with {self.model}")
                    # 初始化工具表并开始计时
                    await self._initialize_tools(client)

                    self.start_time = time.time()
                    self.status = AgentStatus.RUNNING

                    # ReAct 主循环:最多 max_steps 步,终态即停
                    for step in range(self.max_steps):
                        if self._finalized:
                            logger.info(f"Finalized after {step} steps")
                            break
                        logger.bind(message_type="step").info(
                            f"Starting step {step + 1}"
                        )
                        await self.step(client)

                    if not self._finalized:
                        # 步数耗尽仍未提交:判为失败
                        logger.error(f"Not finalized after {self.max_steps} steps")
                        self.status = AgentStatus.FAILED
                    else:
                        self.status = AgentStatus.COMPLETED

                    return self._build_output()

        except TimeoutError:
            # 总超时:状态置 ERROR 并产出已有轨迹
            logger.error(f"Timeout after {self.timeout}s")
            self.status = AgentStatus.ERROR
            return self._build_output()

        except asyncio.CancelledError:
            # 被取消
            logger.error("Cancelled")
            self.status = AgentStatus.CANCELLED
            return self._build_output()

        except Exception as e:
            # 系统错误与任务失败的区分:前者通常是基础设施问题
            logger.error(f"Error: {e}")
            self.status = (
                AgentStatus.ERROR if is_system_error(e) else AgentStatus.FAILED
            )
            return self._build_output()


async def run(run_input: AgentRunInput) -> AgentTrajectoryOutput:
    """Entry point for the ReAct Toolbelt agent."""
    # 模块入口:构造代理并运行
    return await ReActAgent(run_input).run()

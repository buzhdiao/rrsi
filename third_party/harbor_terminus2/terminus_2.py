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

# ---- 标准库导入 ----
# 深拷贝：为子代理复制主轨迹步骤时使用
import copy
# shlex.quote：把命令/路径安全引用进 tmux 容器 shell
import shlex
# time.time：统计每次 LLM API 请求耗时
import time
# uuid4：为未显式指定 session_id 的运行生成唯一会话 ID
import uuid
# dataclass：声明 Command / SubagentMetrics 数据类
from dataclasses import dataclass
# datetime/timezone：为轨迹步骤打 UTC 时间戳
from datetime import datetime, timezone
# Path：日志目录、提示词模板文件路径
from pathlib import Path
# Literal：约束 reasoning_effort 取值；override：标记覆写基类方法
from typing import Any, Literal, override

# tenacity：声明式重试（@retry 装饰器 + 重试条件/停止策略）
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

# harbor 代理基类：Terminus2 继承其 setup/run 接口
from harbor.agents.base import BaseAgent
# JSON 纯文本响应解析器（模型以 JSON 格式回指令）
from .terminus_json_plain_parser import (
    TerminusJSONPlainParser,
)
# XML 纯文本响应解析器（模型以 XML 格式回指令）
from .terminus_xml_plain_parser import (
    TerminusXMLPlainParser,
)
# tmux 会话封装：send_keys / capture_pane / 增量输出等终端操作
from .tmux_session import TmuxSession
# harbor 环境基类：在任务容器内执行命令、判断目录存在等
from harbor.environments.base import BaseEnvironment
# LLM 基础设施：基类、上下文超限/输出超限异常、后端枚举、响应对象
from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMBackend,
    LLMResponse,
    OutputLengthExceededError,
)
# Chat：维护消息历史并调用 LLM 的会话对象
from harbor.llms.chat import Chat
# LiteLLM 后端实现（统一接入各家模型 API）
from harbor.llms.lite_llm import LiteLLM
# litellm 调试日志开关配置
from harbor.llms.litellm_config import configure_litellm_debug
# AgentContext：记录 token 数/成本/rollout 明细等运行期汇总
from harbor.models.agent.context import AgentContext
# AgentName：代理名称枚举（含 TERMINUS_2）
from harbor.models.agent.name import AgentName
# RolloutDetail：单次 LLM 调用的 token id / logprobs 明细类型
from harbor.models.agent.rollout_detail import RolloutDetail
# TrajectoryConfig：轨迹导出配置（raw_content / linear_history）
from harbor.models.agent.trajectory_config import TrajectoryConfig
# 轨迹（trajectory）数据模型：ATIF 格式的 Step/Observation/ToolCall 等
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
# 环境内固定路径（agent_dir 等，用于 asciinema 录制路径）
from harbor.models.trial.paths import EnvironmentPaths
# 轨迹 JSON 格式化工具
from harbor.utils.trajectory_utils import format_trajectory_json


# 一条终端命令：keystrokes 为要敲入的按键串（含命令本身与换行），
# duration_sec 为敲入后等待输出稳定的秒数
@dataclass
class Command:
    keystrokes: str
    duration_sec: float


# 观测（observation）与工具调用（tool call）的归属 ID 判定：多命令批次共享同一份
# 终端输出，无法精确归属到某条命令；只有"单命令批次"才能无歧义关联。
def _terminal_observation_source_call_id(
    commands: list[Command], episode: int
) -> str | None:
    # 命令数不是 1（0 条或多条）→ 无法唯一归属，返回 None
    if len(commands) != 1:
        return None
    # 单命令时返回该 episode 第一条命令的工具调用 ID（call_<episode>_1）
    return f"call_{episode}_1"


# 子代理（subagent）的 token/成本累计器：与主对话分开统计，便于开销归因
@dataclass
class SubagentMetrics:
    """Metrics for subagent operations (summarization, etc.)."""

    # 子代理累计消耗的 prompt tokens
    total_prompt_tokens: int = 0
    # 子代理累计产生的 completion tokens
    total_completion_tokens: int = 0
    # 子代理累计命中的缓存 tokens
    total_cached_tokens: int = 0
    # 子代理累计花费（美元）
    total_cost_usd: float = 0.0


# Terminus 2 代理：基于 tmux 终端的极简 agent（论文中的基础 harness/脚手架）。
# 核心循环：LLM 输出结构化文本（JSON/XML）→ 解析成终端命令 → 在 tmux 中执行 →
# 把终端观测回灌给 LLM，直至模型两次确认任务完成或回合数耗尽。
class Terminus2(BaseAgent):
    # 声明支持 ATIF（Agent Trajectory Interchange Format）轨迹导出格式
    SUPPORTS_ATIF: bool = True

    # LLM 后端工厂：按 llm_backend 参数实例化对应后端（def 签名跨多行）
    def _init_llm(
        self,
        llm_backend: LLMBackend | str,
        model_name: str,
        temperature: float | None,
        collect_rollout_details: bool,
        llm_kwargs: dict[str, Any] | None,
        # LiteLLM-specific args
        api_base: str | None,
        session_id: str | None,
        max_thinking_tokens: int | None,
        reasoning_effort: str | None,
        model_info: dict[str, Any] | None,
        use_responses_api: bool,
    ) -> BaseLLM:
        """Initialize the LLM backend based on llm_backend parameter.

        Args:
            llm_backend: The LLM backend to use.
            model_name: Name of the model.
            temperature: Sampling temperature, if explicitly configured.
            collect_rollout_details: Whether to collect token IDs and logprobs.
            llm_kwargs: Additional kwargs passed to the LLM constructor.
            api_base: Base URL for LiteLLM API endpoint.
            session_id: Session ID for LiteLLM.
            max_thinking_tokens: Max thinking tokens for LiteLLM extended thinking.
            reasoning_effort: Reasoning effort level for LiteLLM.
            model_info: Model info dict for LiteLLM custom models.
            use_responses_api: Whether to use the Responses API.

        Returns:
            An initialized LLM instance.

        Raises:
            ValueError: If llm_backend is not a recognized backend.
        """
        # Normalize enum to string value for matching
        # 归一化后端标识：枚举转字符串，便于下方 match 匹配（表达式跨三行）
        backend_value = (
            llm_backend.value if isinstance(llm_backend, LLMBackend) else llm_backend
        )
        # 复制一份构造参数字典（None 视为空），避免污染调用方的 llm_kwargs
        constructor_kwargs = dict(llm_kwargs or {})
        if temperature is not None:
            # 显式配置了采样温度才注入（未配置则交给后端默认值）
            constructor_kwargs["temperature"] = temperature

        # 按后端标识分派实例化
        match backend_value:
            case LLMBackend.LITELLM.value:
                # 取出并移除 litellm 专用开关（它不是 LiteLLM 构造参数）
                litellm_debug = constructor_kwargs.pop("litellm_debug", False)
                # 配置 litellm 的调试日志开关
                configure_litellm_debug(debug=litellm_debug)

                # 实例化 LiteLLM 后端（return 语句跨多行）
                return LiteLLM(
                    model_name=model_name,
                    api_base=api_base,
                    collect_rollout_details=collect_rollout_details,
                    session_id=session_id,
                    max_thinking_tokens=max_thinking_tokens,
                    reasoning_effort=reasoning_effort,
                    model_info=model_info,
                    use_responses_api=use_responses_api,
                    **constructor_kwargs,
                )
            case LLMBackend.TINKER.value:
                # Tinker 后端延迟导入：仅在选用时加载
                from harbor.llms.tinker import TinkerLLM

                # 实例化 Tinker 后端（return 语句跨多行）
                return TinkerLLM(
                    model_name=model_name,
                    collect_rollout_details=collect_rollout_details,
                    **constructor_kwargs,
                )
            case _:
                # 未知后端 → 报错并列出全部受支持的后端
                raise ValueError(
                    f"Unknown llm_backend: {llm_backend!r}. "
                    f"Supported backends: {[b.value for b in LLMBackend]}"
                )

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_turns: int | None = None,
        parser_name: str = "json",
        api_base: str | None = None,
        temperature: float | None = None,
        reasoning_effort: Literal[
            "none", "minimal", "low", "medium", "high", "xhigh", "max", "default"
        ]
        | None = None,
        collect_rollout_details: bool = False,
        session_id: str | None = None,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
        max_thinking_tokens: int | None = None,
        model_info: dict[str, Any] | None = None,
        trajectory_config: TrajectoryConfig | None = None,
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        store_all_messages: bool = False,
        record_terminal_session: bool = True,
        interleaved_thinking: bool = False,
        suppress_max_turns_warning: bool = False,
        use_responses_api: bool = False,
        llm_backend: LLMBackend | str = LLMBackend.LITELLM,
        llm_kwargs: dict[str, Any] | None = None,
        llm_call_kwargs: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
        *args,
        **kwargs,
    ):
        # Terminus 2 构造函数：装配 LLM/解析器/模板与全部运行期状态（def 签名跨多行）
        """Initialize Terminus 2 agent.

        Args:
            logs_dir: Directory to store logs
            model_name: Name of the model to use
            max_episodes: Maximum number of episodes (default: 1000000)
            parser_name: Parser to use - "json" or "xml" (default: "json")
            api_base: Base URL for the API endpoint
            temperature: Optional sampling temperature. If unset, no temperature is
                passed to the LLM backend. (default: None)
            reasoning_effort: Qualitative or quantitative measure of effort (default: None)
            collect_rollout_details: Whether to collect detailed rollout data including token IDs.
                NOTE: Rollout details will be incomplete if context summarization occurs.
                See class docstring for details. (default: False)
            session_id: Session ID for the agent (default: None)
            enable_summarize: Whether to enable context summarization (default: True)
            proactive_summarization_threshold: Number of free tokens below which to trigger
                proactive summarization. Set to 0 to disable proactive summarization. (default: 8000)
            max_thinking_tokens: Maximum thinking tokens for Anthropic extended thinking mode.
                Minimum value is 1024. (default: None)
            model_info: Optional dict containing model information for custom models.
                Used to register the model with litellm. Common fields include:
                - max_input_tokens: Maximum input tokens (context length)
                - max_output_tokens: Maximum output tokens
                - input_cost_per_token: Cost per input token (optional)
                - output_cost_per_token: Cost per output token (optional)
                (default: None)
            trajectory_config: Optional TrajectoryConfig containing trajectory-related configurations.
                Available options:
                - raw_content (bool): If True, dump raw LLM responses into trajectory without
                  parsing into tool_calls. Useful for SFT data export. (default: False)
                - linear_history (bool): If True, split trajectory into separate files when context
                  summarization occurs, ensuring each trajectory represents a continuous linear
                  history sent to the LLM. When False, keep all steps from the main agent
                  in a single trajectory file despite chat history resets. (default: False)
                (default: None)
            tmux_pane_width: Starting tmux pane width (maps to `tmux -x`, default: 160)
            tmux_pane_height: Starting tmux pane height (maps to `tmux -y`, default: 40)
            record_terminal_session: Whether to capture terminal recordings via asciinema.
                (default: True)
            interleaved_thinking: Whether to include reasoning content in chat history
                and send to litellm in next round's conversation (default: False)
            suppress_max_turns_warning: Whether to suppress the warning about artificially
                limiting max_turns (default: False)
            llm_backend: LLM backend to use. Use LLMBackend.LITELLM or "litellm".
                (default: LLMBackend.LITELLM)
            llm_kwargs: Additional kwargs to pass to the LLM constructor.
                ``litellm_debug`` (bool): When True, re-enables litellm's verbose
                debug output (e.g. Provider List on unknown models). Suppressed by
                default. (default: False)
                (default: None)
            llm_call_kwargs: Extra kwargs to forward to LLM calls (e.g., extra_body).
            **kwargs: Additional arguments
        """
        # 调用基类构造（透传 extra_env 等参数）
        super().__init__(logs_dir, model_name, *args, extra_env=extra_env, **kwargs)

        if model_name is None:
            # Terminus 2 必须知道用哪个模型，缺省即报错
            raise ValueError("model_name is required for Terminus 2")

        # 记录主模型名（策略 LLM 的标识）
        self._model_name = model_name
        # 最近一次 LLM 响应实际报告的模型名（用于轨迹记录，先置空）
        self._last_response_model_name: str | None = None
        # 响应格式解析器类型："json" 或 "xml"
        self._parser_name = parser_name
        # 是否收集 token id / logprobs 级别的 rollout 明细
        self._collect_rollout_details = collect_rollout_details
        # 推理努力等级（reasoning effort，控制思考预算的定性/定量度量）
        self._reasoning_effort = reasoning_effort
        # 解析模型元信息（自定义模型的上下文长度/价格等），hosted_vllm 会告警
        resolved_model_info = self._resolve_model_info(model_name, model_info)

        # 按后端类型实例化 LLM（语句跨多行）
        self._llm = self._init_llm(
            llm_backend=llm_backend,
            model_name=model_name,
            temperature=temperature,
            collect_rollout_details=collect_rollout_details,
            llm_kwargs=llm_kwargs,
            api_base=api_base,
            session_id=session_id,
            max_thinking_tokens=max_thinking_tokens,
            reasoning_effort=reasoning_effort,
            model_info=resolved_model_info,
            use_responses_api=use_responses_api,
        )
        # 按格式选择响应解析器实例
        self._parser = self._get_parser()
        # 读入系统提示词模板（按格式选择 terminus-json-plain / terminus-xml-plain）
        self._prompt_template = self._get_prompt_template_path().read_text()
        # 读入命令超时提示模板
        self._timeout_template = self._get_timeout_template_path().read_text()
        # 保存温度设置（None 表示未配置，写轨迹时据此省略该字段）
        self._temperature = temperature

        # Handle deprecated 'episodes' kwarg
        # 兼容已弃用的 'episodes' 关键字参数
        episodes_from_kwargs = kwargs.get("episodes")
        if episodes_from_kwargs is not None:
            # 弃用参数仍可用但发出弃用警告
            self.logger.warning(
                "The 'episodes' parameter is deprecated and will be removed in a future version. "
                "Please use 'max_turns' instead."
            )

        # Determine the final max episodes value with proper precedence:
        # 1. max_turns (new parameter)
        # 2. max_episodes (deprecated but still supported)
        # 3. episodes from kwargs (deprecated)
        # 4. Default value of 1000000
        # 按优先级确定最大回合数：max_turns > max_episodes（弃用）> episodes（弃用）> 默认
        final_max_episodes = None
        if max_turns is not None:
            # 新参数 max_turns 优先
            final_max_episodes = max_turns
        elif kwargs.get("max_episodes") is not None:
            # 其次是弃用的 max_episodes
            final_max_episodes = kwargs.get("max_episodes")
        elif episodes_from_kwargs is not None:
            # 最后是弃用的 episodes
            final_max_episodes = episodes_from_kwargs

        if final_max_episodes is not None:
            if not suppress_max_turns_warning:
                # 人为限制回合数通常降低完成率 → 默认给出提醒
                self.logger.warning(
                    f"max_turns (f.k.a. max_episodes) artificially limited to {final_max_episodes}. "
                    "Consider removing this limit for better task completion."
                )
            # 采用配置的回合上限
            self._max_episodes = final_max_episodes
        else:
            # 未配置 → 使用事实上不设限的默认值 1000000
            self._max_episodes = 1000000
        # 主对话 Chat 对象（每次 run() 重建）
        self._chat: Chat | None = None
        # AgentContext 引用（run() 时注入，汇总 token/成本/明细）
        self._context: AgentContext | None = None
        # 是否处于"待二次确认完成"状态（防止单次 task_complete 误判）
        self._pending_completion = False
        # tmux 会话（setup() 时创建）
        self._session: TmuxSession | None = None
        # 每次 LLM API 请求耗时（毫秒）列表，写入 context.metadata
        self._api_request_times: list[float] = []
        # 已执行的 episode（回合）计数
        self._n_episodes: int = 0
        # 保存调用方显式传入的 session_id（跨 run 复用时重置用）
        self._user_provided_session_id: str | None = session_id
        # 会话 ID：显式传入或随机生成
        self._session_id = session_id if session_id else str(uuid.uuid4())
        # 主轨迹步骤列表（ATIF 格式的 Step 序列）
        self._trajectory_steps: list[Step] = []
        # 是否用 asciinema 录制终端会话（recording.cast）
        self._record_terminal_session = record_terminal_session
        # 透传给每次 LLM 调用的额外关键字参数（如 extra_body）
        self._llm_call_kwargs = dict(llm_call_kwargs) if llm_call_kwargs else {}

        # 摘要（summarization）子代理创建次数计数（语句跨多行 + 既有行内注释）
        self._summarization_count: int = (
            0  # Track number of summarization subagents created
        )
        # 待挂到下一步轨迹里的子代理轨迹引用（语句跨多行）
        self._pending_subagent_refs: list[SubagentTrajectoryRef] | None = (
            None  # Track subagent refs to include in next step
        )
        # 待作为 user 步骤加入轨迹的交接提示词（handoff prompt，语句跨多行）
        self._pending_handoff_prompt: str | None = (
            None  # Track handoff prompt to include as user step
        )
        # 子代理 token/成本累计器（与主对话分开统计）
        self._subagent_metrics = SubagentMetrics()  # Track subagent metrics separately
        # 各子代理的 rollout 明细（语句跨多行）
        self._subagent_rollout_details: list[
            RolloutDetail
        ] = []  # Track rollout details for each subagent
        # 上下文摘要总开关（语句跨多行）
        self._enable_summarize = (
            enable_summarize  # Toggle for proactive and context limit summarization
        )
        # 主动摘要阈值：剩余可用 token 低于该值即提前压缩上下文
        self._proactive_summarization_threshold = proactive_summarization_threshold
        # tmux 面板初始宽度（映射 tmux -x）
        self._tmux_pane_width = tmux_pane_width
        # tmux 面板初始高度（映射 tmux -y）
        self._tmux_pane_height = tmux_pane_height

        # Trajectory configuration
        # 轨迹导出配置（None 归一为空字典）
        self._trajectory_config = trajectory_config or {}
        # raw_content：轨迹中保留 LLM 原始响应（不解析成 tool_calls），用于 SFT（监督微调）数据导出
        self._save_raw_content_in_trajectory = self._trajectory_config.get(
            "raw_content", False
        )
        # linear_history：摘要发生时拆分轨迹文件，保证每份轨迹是发给 LLM 的连续线性历史
        self._linear_history = self._trajectory_config.get("linear_history", False)
        # Optional: include full chat messages in TrialResult metadata (can be large)
        # 可选：把完整聊天消息塞进 TrialResult 元数据（可能非常大）
        self._store_all_messages = store_all_messages
        # 交错思考：是否把推理内容保留在历史并随下一轮请求发回
        self._interleaved_thinking = interleaved_thinking
        # 保留原始 llm_kwargs（写轨迹元数据时用）
        self._llm_kwargs = llm_kwargs

    # 解析模型元信息：优先用调用方提供的，否则针对 hosted_vllm 给出警告
    # （def 签名跨多行）
    def _resolve_model_info(
        self, model_name: str | None, provided_model_info: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        # 调用方显式提供了模型元信息 → 原样使用
        if provided_model_info:
            return provided_model_info
        if model_name and "hosted_vllm" in model_name:
            # hosted_vllm 自定义模型缺元信息会导致上下文/计费判断失准 → 警告
            self.logger.warning(
                "Model info is required when using hosted_vllm models. "
                "Please set `model_info` in your Terminus 2 configuration with "
                "`max_input_tokens`, `max_output_tokens`, and cost fields. "
                "Falling back to LiteLLM defaults, which may cause context or pricing issues."
            )
        # 其余情况返回 None（由 LiteLLM 使用其内置默认值）
        return None

    # 代理注册名（类方法，供 harbor 框架按名实例化）
    @staticmethod
    @override
    def name() -> str:
        # 代理注册名（枚举值 "terminus-2"）
        return AgentName.TERMINUS_2.value

    # 代理版本号
    @override
    def version(self) -> str | None:
        # 代理版本号，写入轨迹元数据
        return "2.0.0"

    # 初始化阶段：创建并启动 tmux 终端会话（含可选的 asciinema 录制）
    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        if self._record_terminal_session:
            # 需要录制：规划本地与容器内两端的 asciinema 录像文件路径
            local_recording_path = environment.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            # 不录制：两端路径均为 None
            local_recording_path = None
            remote_recording_path = None

        # 创建 tmux 会话封装对象（面板尺寸、日志与录像路径、额外环境变量、默认用户）
        self._session = TmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        # 启动 tmux 会话，供后续 send_keys / capture_pane 使用
        await self._session.start()

    # 按格式选择响应解析器实例（json 或 xml）
    def _get_parser(self):
        """Return the appropriate parser instance for this format."""
        if self._parser_name == "json":
            # JSON 格式 → JSON 解析器
            return TerminusJSONPlainParser()
        elif self._parser_name == "xml":
            # XML 格式 → XML 解析器
            return TerminusXMLPlainParser()
        else:
            # 未知格式 → 报错
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    # 返回当前格式对应的系统提示词模板路径
    def _get_prompt_template_path(self) -> Path:
        """Return the path to the prompt template for this format."""
        if self._parser_name == "json":
            # JSON 格式的系统提示词模板路径
            return Path(__file__).parent / "templates" / "terminus-json-plain.txt"
        elif self._parser_name == "xml":
            # XML 格式的系统提示词模板路径
            return Path(__file__).parent / "templates" / "terminus-xml-plain.txt"
        else:
            # 未知格式 → 报错
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    # 返回命令超时提示模板路径（两种格式共用）
    def _get_timeout_template_path(self) -> Path:
        """Return the path to the timeout template for this format."""
        # 命令超时提示模板（两种格式共用）
        return Path(__file__).parent / "templates" / "timeout.txt"

    # 解析 SKILL.md 的 YAML frontmatter，返回 name/description（def 签名跨多行）
    @staticmethod
    def _parse_skill_frontmatter(content: str) -> dict[str, str] | None:
        """Parse YAML frontmatter from SKILL.md content, returning name and description."""
        # 正则：提取 --- 包围的 frontmatter 块
        import re

        # yaml：解析 frontmatter 的 name/description 字段
        import yaml

        # 匹配文件头 "---\n...\n---" 的 frontmatter（DOTALL 允许内容跨行）
        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            # 无 frontmatter → 不是合格的 SKILL.md
            return None
        try:
            # 安全解析 YAML（safe_load 防任意对象注入）
            fm = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            # YAML 语法错误 → 视为无效技能
            return None
        if not isinstance(fm, dict) or "name" not in fm or "description" not in fm:
            # 必须是字典且含 name/description 两个字段
            return None
        # 返回技能名与描述（用于 <available_skills> 广播）
        return {"name": fm["name"], "description": fm["description"]}

    # 技能发现：扫描容器内 SKILL.md，产出 <available_skills> 广播区块（签名跨多行）
    async def _build_skills_section(self, environment: BaseEnvironment) -> str | None:
        """Discover Agent Skills in skills_dir and return an <available_skills> XML block.

        Follows the Agent Skills spec: scans for subdirectories containing SKILL.md
        inside the remote environment, parses YAML frontmatter for name/description,
        and provides the absolute path so the model can ``cat`` the file to activate
        a skill.
        """
        # 未设置 skills_dir（本 trial 无技能）→ 返回 None
        if not self.skills_dir:
            return None

        # 容器内技能目录不存在 → 返回 None
        if not await environment.is_dir(self.skills_dir):
            return None

        # List subdirectories containing SKILL.md in the remote environment
        # 在容器内查找形如 <skills_dir>/<name>/SKILL.md 的文件并排序
        # （find 深度限定 2 层；await 语句跨多行）
        result = await environment.exec(
            f"find {shlex.quote(self.skills_dir)} -mindepth 2 -maxdepth 2"
            " -name SKILL.md -type f | sort",
            timeout_sec=10,
        )

        if result.return_code != 0 or not result.stdout or not result.stdout.strip():
            # 查找失败或无结果 → 无可用技能
            return None

        # 按行拆出每个 SKILL.md 的容器内绝对路径
        skill_md_paths = result.stdout.strip().splitlines()
        entries: list[tuple[str, str, str]] = []  # (name, description, location)

        for skill_md_path in skill_md_paths:
            # 逐个读取 SKILL.md 内容（await 语句跨多行）
            cat_result = await environment.exec(
                f"cat {shlex.quote(skill_md_path)}", timeout_sec=10
            )
            if cat_result.return_code != 0 or not cat_result.stdout:
                # 读取失败 → 跳过该技能
                continue
            # 解析 frontmatter 得到名称与描述
            fm = self._parse_skill_frontmatter(cat_result.stdout)
            if fm is None:
                # frontmatter 不合格 → 跳过
                continue
            # 收集 (名称, 描述, 容器内路径) 三元组
            entries.append((fm["name"], fm["description"], skill_md_path))

        if not entries:
            # 没有合格技能 → 返回 None
            return None

        # 延迟导入标准库 XML 构造工具
        from xml.etree.ElementTree import Element, SubElement, indent, tostring

        # 以 <available_skills> 为根构造技能清单 XML
        root = Element("available_skills")
        for name, description, location in entries:
            # 每个技能一个 <skill> 节点
            skill = SubElement(root, "skill")
            # 子节点：<name> 技能名
            SubElement(skill, "name").text = name
            # 子节点：<description> 描述
            SubElement(skill, "description").text = description
            # 子节点：<location> 容器内路径（模型 cat 该文件即可激活技能）
            SubElement(skill, "location").text = location

        # 缩进美化输出
        indent(root, space="  ")
        # 序列化为字符串并加前导换行，供拼接到增强指令末尾
        return "\n" + tostring(root, encoding="unicode")

    # 返回错误消息中使用的响应类型称谓（如 "JSON response"）
    def _get_error_response_type(self) -> str:
        """Return the response type name for error messages.

        Examples: 'JSON response', 'response'
        """
        if self._parser_name == "json":
            # JSON 格式的响应类型称谓
            return "JSON response"
        elif self._parser_name == "xml":
            # XML 格式的响应类型称谓
            return "response"
        else:
            # 未知格式 → 报错
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    # 返回格式相关的任务完成二次确认消息（防一次误判就交卷）
    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Return the format-specific task completion confirmation message."""
        if self._parser_name == "json":
            # JSON 格式的完成二次确认消息（语句跨多行）
            return (
                f"Current terminal state:\n{terminal_output}\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded and you won't be able to "
                'make any further corrections. If so, include "task_complete": true '
                "in your JSON response again."
            )
        elif self._parser_name == "xml":
            # XML 格式的完成二次确认消息（语句跨多行）
            return (
                f"Current terminal state:\n{terminal_output}\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded and you won't be able to "
                "make any further corrections. If so, include "
                "<task_complete>true</task_complete> again."
            )
        else:
            # 未知格式 → 报错
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    # 统计聊天全部消息的总 token 数（用于摘要触发判断）
    def _count_total_tokens(self, chat: Chat) -> int:
        """Count total tokens across all messages in the chat."""
        # 延迟导入 litellm 的 token 计数器
        from litellm.utils import token_counter

        # 按当前模型对全部历史消息计数（用于主动摘要判断）
        return token_counter(model=self._model_name, messages=chat.messages)

    # 终端输出字节级截断：保头保尾、省略中间（控制回灌给 LLM 的体量）
    def _limit_output_length(self, output: str, max_bytes: int = 10000) -> str:
        """
        Limit output to specified byte length, keeping first and last portions.

        Args:
            output: The terminal output to potentially truncate
            max_bytes: Maximum allowed bytes (default 10000)

        Returns:
            str: Original output if under limit, or truncated with middle omitted
        """
        # UTF-8 字节数未超限 → 原样返回
        if len(output.encode("utf-8")) <= max_bytes:
            return output

        # Calculate portions (half each for first and last)
        # 预算对半分：一半保开头，一半保结尾
        portion_size = max_bytes // 2

        # Convert to bytes for accurate splitting
        # 转成字节串切分，保证字节数（而非字符数）准确
        output_bytes = output.encode("utf-8")

        # Get first portion
        # 取开头一半（errors="ignore" 丢弃被切坏的多字节字符尾部）
        first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")

        # Get last portion
        # 取结尾一半
        last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")

        # Calculate omitted bytes
        # 计算被丢弃的中间字节数（表达式跨多行）
        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode("utf-8"))
            - len(last_portion.encode("utf-8"))
        )

        # 拼回"头 + 省略说明 + 尾"，让模型知道中间被截断（语句跨多行）
        return (
            f"{first_portion}\n[... output limited to {max_bytes} bytes; "
            f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
        )

    # 消息回退：从尾部成对删除消息直到腾出目标量的空闲 token（def 签名跨多行）
    def _unwind_messages_to_free_tokens(
        self, chat: Chat, target_free_tokens: int = 4000
    ) -> None:
        """Remove recent messages until we have enough free tokens."""
        # 取模型上下文窗口上限
        context_limit = self._llm.get_model_context_limit()

        while len(chat.messages) > 1:  # Keep at least the first message
            # 统计当前消息总 token 数
            current_tokens = self._count_total_tokens(chat)
            # 计算剩余可用 token
            free_tokens = context_limit - current_tokens

            if free_tokens >= target_free_tokens:
                # 已腾出足够空间 → 停止回退
                break

            # Remove the most recent pair of messages (user + assistant)
            # 从尾部成对删除消息（user + assistant 一起删，保持对话结构完整）
            if len(chat.messages) >= 2:
                chat._messages = chat.messages[:-2]
            else:
                break

        # 重置响应链：删除消息后 Chat 内部缓存的状态需要重建
        chat.reset_response_chain()
        # 重新统计回退后的剩余可用 token
        free_tokens = context_limit - self._count_total_tokens(chat)
        # 记录回退结果日志（语句跨多行）
        self.logger.debug(
            f"Unwound messages. Remaining messages: {len(chat.messages)}, "
            f"Free tokens: approximately {free_tokens}"
        )

    # 收集子代理 LLM 响应的 rollout 明细（token id / logprobs）
    def _collect_subagent_rollout_detail(self, response: LLMResponse) -> None:
        """Collect rollout details from a subagent LLM response.

        Args:
            response: The LLM response containing token IDs and logprobs
        """
        if not self._collect_rollout_details:
            # 未开启明细收集 → 直接返回
            return

        # 组装该次子代理调用的 rollout 明细（各字段包一层列表，与主循环格式对齐）
        rollout_detail: RolloutDetail = {}
        if response.prompt_token_ids is not None:
            # prompt 的 token id 序列
            rollout_detail["prompt_token_ids"] = [response.prompt_token_ids]
        if response.completion_token_ids is not None:
            # 补全的 token id 序列
            rollout_detail["completion_token_ids"] = [response.completion_token_ids]
        if response.logprobs is not None:
            # 逐 token 对数概率
            rollout_detail["logprobs"] = [response.logprobs]
        if response.extra is not None:
            # 其他附加字段同样包一层列表（字典推导跨多行）
            rollout_detail["extra"] = {
                key: [value] for key, value in response.extra.items()
            }

        if rollout_detail:
            # 有内容才追加，避免空明细污染列表
            self._subagent_rollout_details.append(rollout_detail)

    # 用 LLM 响应的用量信息累计子代理指标
    def _update_subagent_metrics(self, usage_info) -> None:
        """Update subagent metrics with usage information from an LLM response.

        Args:
            usage_info: Usage information from LLM response containing token counts and cost
        """
        if not usage_info:
            # 无用量信息（调用失败等）→ 跳过
            return

        # 累计 prompt tokens
        self._subagent_metrics.total_prompt_tokens += usage_info.prompt_tokens
        # 累计 completion tokens
        self._subagent_metrics.total_completion_tokens += usage_info.completion_tokens
        # 累计缓存命中 tokens
        self._subagent_metrics.total_cached_tokens += usage_info.cache_tokens
        # 累计成本（美元）
        self._subagent_metrics.total_cost_usd += usage_info.cost_usd

    # 从用量信息提取并归一化 (prompt, completion, cached, cost) 四元组
    def _extract_usage_metrics(self, usage_info) -> tuple[int, int, int, float]:
        """Extract and normalize metrics from usage info.

        Args:
            usage_info: Usage information from LLM response

        Returns:
            Tuple of (prompt_tokens, completion_tokens, cached_tokens, cost_usd)
        """
        if not usage_info:
            # 无用量信息 → 返回全零占位
            return 0, 0, 0, 0

        # 归一化为四元组；成本非正数按 0 处理（return 语句跨多行）
        return (
            usage_info.prompt_tokens,
            usage_info.completion_tokens,
            usage_info.cache_tokens,
            usage_info.cost_usd if usage_info.cost_usd > 0 else 0,
        )

    # 依据起始时间戳记录一次 API 请求耗时（毫秒）
    def _track_api_request_time(self, start_time: float) -> None:
        """Track API request time from start timestamp.

        Args:
            start_time: Start time from time.time()
        """
        # 记录结束时刻
        end_time = time.time()
        # 换算成毫秒
        request_time_ms = (end_time - start_time) * 1000
        # 追加进请求耗时列表（最终写入 context.metadata）
        self._api_request_times.append(request_time_ms)

    async def _run_subagent(
        self,
        prompt: str,
        message_history: list[dict[str, Any]],
        steps: list[Step],
        session_id: str,
        agent_name: str,
        filename_suffix: str,
        summary_text: str,
        subagent_name_for_logging: str = "subagent",
    ) -> tuple[LLMResponse, SubagentTrajectoryRef]:
        # 子代理执行统一封装：记步骤 → 调 LLM → 记指标/明细 → 存轨迹（签名跨多行）
        """Run a subagent and return its response and trajectory reference.

        This helper encapsulates the common pattern of:
        1. Appending the prompt step to the trajectory
        2. Calling the LLM with the prompt and message history
        3. Tracking API request time
        4. Updating subagent metrics
        5. Appending the response step to the trajectory
        6. Collecting rollout details
        7. Saving the subagent trajectory

        Args:
            prompt: The prompt to send to the LLM
            message_history: The message history to provide context
            steps: The trajectory steps (will be modified to include prompt and response steps)
            session_id: Session ID for the subagent
            agent_name: Name of the subagent (e.g., "terminus-2-summarization-summary")
            filename_suffix: Suffix for trajectory filename (e.g., "summary", "questions", "answers")
            summary_text: Human-readable summary for the trajectory ref
            subagent_name_for_logging: Name used in logging messages

        Returns:
            tuple: (LLM response, SubagentTrajectoryRef)
        """
        # Append the prompt step
        # 先把子代理收到的 prompt 记为一条 user 步骤（步骤号顺延）
        prompt_step_id = len(steps) + 1
        steps.append(
            Step(
                step_id=prompt_step_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=prompt,
            )
        )
        # 响应步骤号紧随其后
        response_step_id = prompt_step_id + 1

        # 记录发起时刻，用于统计 API 耗时
        start_time = time.time()
        # 调用 LLM（带消息历史与额外调用参数；await 语句跨多行）
        response: LLMResponse = await self._llm.call(
            prompt=prompt,
            message_history=message_history,
            **self._llm_call_kwargs,
        )
        # 记录本次请求耗时
        self._track_api_request_time(start_time)

        # 取出 token 用量信息
        usage_info = response.usage
        # 累计进子代理指标
        self._update_subagent_metrics(usage_info)

        # 把响应（含条件性的指标）追加为 agent 步骤（语句跨多行）
        self._append_subagent_response_step(
            steps,
            response_step_id,
            response,
            usage_info,
            subagent_name_for_logging,
        )

        # 收集该子代理调用的 rollout 明细
        self._collect_subagent_rollout_detail(response)

        # 把子代理轨迹落盘并取得引用（语句跨多行）
        trajectory_ref = self._save_subagent_trajectory(
            session_id=session_id,
            agent_name=agent_name,
            steps=steps,
            usage_info=usage_info,
            filename_suffix=filename_suffix,
            summary_text=summary_text,
        )

        # 返回响应与轨迹引用
        return response, trajectory_ref

    async def _summarize(
        self, chat: Chat, original_instruction: str, session: TmuxSession
    ) -> tuple[str, list[SubagentTrajectoryRef] | None]:
        # 三步上下文摘要（context summarization）：摘要生成 → 提问 → 作答，
        # 最后以"问题对话 + 交接答案"替换聊天历史（def 签名跨多行）
        """Create a summary of the agent's work to pass to a new agent instance.

        This method implements a three-step context summarization process using separate
        subagents to compress conversation history while preserving critical information:

        **Step 1: Summary Generation (SUBAGENT 1)**
        - Message history: Unwound chat.messages (limited context after token freeing)
        - Prompt: Generate comprehensive summary of all work completed so far
        - Response: summary_response.content (detailed narrative of progress)

        **Step 2: Question Asking (SUBAGENT 2)**
        - Message history: [] (fresh start, no conversation context)
        - Prompt: Given the original task, summary from Step 1, and current terminal screen,
        generate questions about information missing from the summary.
        - Response: model_questions (list of clarifying questions)

        **Step 3: Answer Providing (SUBAGENT 3)**
        - Message history: chat.messages + [summary_prompt, summary_response] (extended context)
        - Prompt: Given the questions from Step 2, answer the questions based on full conversation history.
        - Response: answers_response.content (detailed answers)

        **Final handoff:**
        - Chat history replaced with: [system, question_prompt (which includes summary from step 1), model_questions]
        - Result: Compressed context that preserves critical task-specific information, allowing
        the main agent to continue working on the task without losing context.

        **Why three steps?** This question-answer approach ensures the summarization doesn't
        lose important details - the questions subagent identifies gaps, and the answers
        subagent (with full context) fills them in.

        Args:
            chat: Chat object containing conversation history (may be unwound)
            original_instruction: The original task instruction from user
            session: TmuxSession for capturing current terminal state

        Returns:
            tuple: (handoff_prompt, subagent_trajectory_refs)
                - handoff_prompt: The prompt to continue with (includes answers + instructions)
                - subagent_trajectory_refs: List of 3 SubagentTrajectoryRef objects, or None if summarization failed
        """
        if len(chat.messages) == 0:
            # 没有任何对话历史就无需摘要：直接以原指令交接
            return original_instruction, None

        # Increment summarization count
        # 摘要轮次计数 +1（同一次 run 可能多次触发摘要）
        self._summarization_count += 1
        # 收集本次三个子代理的轨迹引用
        subagent_trajectory_refs = []

        # ===== SUBAGENT 1: Summary Generation =====
        # 子代理 1：摘要生成（复用回退后的对话历史，产出工作总结）
        # 子代理会话 ID：主会话 ID + 摘要轮次 + 阶段名（表达式跨多行）
        summary_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-summary"
        )

        # Trajectory needs to reflect what is sent to LLM: essentially the previous chat
        # history, minus the messages that were removed to free up tokens.
        # Calculate how many trajectory steps to include based on remaining chat messages:
        # - Chat has: [system, agent1, user1, agent2, user2, ...]
        # - Trajectory has: [step1_system, step2_agent, step3_agent, ...]
        # - Formula: steps_to_include = 1 + (num_messages - 1) // 2
        # 轨迹须反映实际发给 LLM 的内容：由剩余消息数反推应保留的轨迹步数
        steps_to_include = 1 + (len(chat.messages) - 1) // 2
        # 深拷贝主轨迹前 N 步并剥离指标，作为摘要子代理的轨迹（语句跨多行）
        summary_steps, step_id_counter = self._prepare_copied_trajectory_steps(
            steps_to_include
        )

        # 摘要生成提示词：要求全面总结已完成的动作/学到的事实/解决的难题/当前进度
        # （f-string 跨多行，注释置于语句起始处）
        summary_prompt = f"""You are about to hand off your work to another AI agent.
            Please provide a comprehensive summary of what you have
            accomplished so far on this task:

Original Task: {original_instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant command you executed
            and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings, file
            locations, configurations, error messages, or system state discovered.
3. **Challenging Problems Addressed** - Any significant issues you
            encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.


Be comprehensive and detailed. The next agent needs to understand everything
            that has happened so far in order to continue."""

        # 子代理 1 执行：以回退后的完整历史为上下文生成摘要（语句跨多行）
        summary_response, summary_trajectory_ref = await self._run_subagent(
            prompt=summary_prompt,
            message_history=chat.messages,
            steps=summary_steps,
            session_id=summary_session_id,
            agent_name="terminus-2-summarization-summary",
            filename_suffix="summary",
            summary_text=f"Context summarization {self._summarization_count}: Step 1 - Summary generation",
            subagent_name_for_logging="summary generation LLM call",
        )
        # 记录子代理 1 的轨迹引用
        subagent_trajectory_refs.append(summary_trajectory_ref)

        # ===== SUBAGENT 2: Question Asking =====
        # 子代理 2：提问（全新上下文，只看任务+摘要+当前屏幕，找出摘要缺失的信息）
        # 抓取当前终端可见画面（不截全历史缓冲）
        current_screen = await session.capture_pane(capture_entire=False)
        # 子代理 2 的会话 ID（表达式跨多行）
        questions_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-questions"
        )
        # 子代理 2 从零开始，无继承步骤
        questions_steps = []

        # 提问提示词：任务 + 上一个代理的摘要 + 当前屏幕，要求至少提五个问题
        # （f-string 跨多行）
        question_prompt = f"""You are picking up work from a previous AI agent on this task:

**Original Task:** {original_instruction}

**Summary from Previous Agent:**
{summary_response.content}

**Current Terminal Screen:**
{current_screen}

Please begin by asking several questions (at least five, more if necessary)
about the current state of the solution that are not answered in the summary
from the prior agent. After you ask these questions you will be on your own,
so ask everything you need to know."""

        # 子代理 2 执行：空消息历史（全新视角），只依赖 prompt 内容提问（语句跨多行）
        questions_response, questions_trajectory_ref = await self._run_subagent(
            prompt=question_prompt,
            message_history=[],
            steps=questions_steps,
            session_id=questions_session_id,
            agent_name="terminus-2-summarization-questions",
            filename_suffix="questions",
            summary_text=f"Context summarization {self._summarization_count}: Step 2 - Question asking",
            subagent_name_for_logging="questions subagent",
        )
        # 保存子代理 2 提出的问题文本
        model_questions = questions_response.content
        # 记录子代理 2 的轨迹引用
        subagent_trajectory_refs.append(questions_trajectory_ref)

        # ===== SUBAGENT 3: Answer Providing =====
        # 子代理 3：作答（持完整上下文，回答子代理 2 的问题，弥补摘要信息缺口）
        # 子代理 3 的会话 ID（表达式跨多行）
        answers_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-answers"
        )

        # Reuse the actual trajectory steps (same as summary subagent)
        # At this point, chat.messages has the same unwound history as in summary subagent
        # 复用与子代理 1 相同的拷贝轨迹步骤（此时 chat.messages 仍是回退后的历史）
        answers_steps, step_id_counter = self._prepare_copied_trajectory_steps(
            steps_to_include
        )

        # Add the summary prompt and response steps that are part of the message history
        # Mark these as copied context since they were already part of the summary subagent trajectory
        # 追加摘要 prompt 步骤（标记为复制上下文，因子代理 1 轨迹里已存在）
        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=summary_prompt,
                is_copied_context=True,
            )
        )
        # 步骤号递增
        step_id_counter += 1

        # 追加摘要响应步骤（同样标记为复制上下文；语句跨多行）
        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=summary_response.model_name or self._model_name,
                message=summary_response.content,
                reasoning_content=summary_response.reasoning_content,
                is_copied_context=True,
                extra={
                    "note": "Copied from summary subagent - metrics already recorded there"
                },
            )
        )
        # 步骤号递增
        step_id_counter += 1

        # 作答请求提示词：把子代理 2 的问题逐条转给持完整上下文的子代理 3
        answer_request_prompt = (
            "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n"
            + model_questions
        )

        # The answer subagent should see: unwound chat history + summary prompt + summary response
        # 子代理 3 的消息历史 = 回退后的对话 + 摘要 prompt/响应两条消息
        answers_message_history = chat.messages + [
            {"role": "user", "content": summary_prompt},
            {"role": "assistant", "content": summary_response.content},
        ]

        # 子代理 3 执行：回答问题（语句跨多行）
        answers_response, answers_trajectory_ref = await self._run_subagent(
            prompt=answer_request_prompt,
            message_history=answers_message_history,
            steps=answers_steps,
            session_id=answers_session_id,
            agent_name="terminus-2-summarization-answers",
            filename_suffix="answers",
            summary_text=f"Context summarization {self._summarization_count}: Step 3 - Answer providing",
            subagent_name_for_logging="answers subagent",
        )
        # 记录子代理 3 的轨迹引用
        subagent_trajectory_refs.append(answers_trajectory_ref)

        # Update chat history with the handoff context
        # Note: We only include questions, not answers. The answers will be provided
        # via the handoff_prompt to the next agent iteration.
        # 重建聊天历史完成"交接"：只保留系统消息 + 提问对话（答案走 handoff_prompt，
        # 不进入历史，语句跨多行）
        chat._messages = [
            chat.messages[0],
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": model_questions},
        ]
        # 重置响应链（历史被整体替换）
        chat.reset_response_chain()

        # 交接提示词：答案全文 + "继续任务、不能再提问、遵守终端交互规范"
        handoff_prompt = (
            "Here are the answers the other agent provided.\n\n"
            + answers_response.content
            + "\n\n"
            + "Continue working on this task from where the previous agent left off."
            " You can no longer ask questions. Please follow the spec to interact with "
            "the terminal."
        )

        # 返回交接提示词与三个子代理的轨迹引用
        return handoff_prompt, subagent_trajectory_refs

    async def _check_proactive_summarization(
        self, chat: Chat, original_instruction: str, session: TmuxSession
    ) -> tuple[str, list[SubagentTrajectoryRef] | None] | None:
        # 主动摘要检查：空闲 token 低于阈值时提前压缩上下文（def 签名跨多行）
        """Check if we should proactively summarize due to token usage.

        Returns:
            tuple: (summary_prompt, subagent_trajectory_ref) if summarization occurred, None otherwise
        """
        # 取模型上下文窗口上限
        context_limit = self._llm.get_model_context_limit()
        # 统计当前对话总 token 数
        current_tokens = self._count_total_tokens(chat)
        # 计算剩余可用 token
        free_tokens = context_limit - current_tokens

        if free_tokens < self._proactive_summarization_threshold:
            # 剩余空间低于阈值 → 抢在上下文真正爆掉之前主动摘要（proactive summarization）
            self.logger.debug(
                f"Proactively summarizing. Free tokens: approximately {free_tokens}"
            )
            try:
                # 执行三步摘要（语句跨多行）
                summary_prompt, subagent_trajectory_refs = await self._summarize(
                    chat, original_instruction, session
                )
                # 返回交接提示词与子代理轨迹引用
                return (summary_prompt, subagent_trajectory_refs)
            except Exception as e:
                # 摘要失败只记错误日志：主动摘要失败可退回被动兜底路径
                self.logger.error(f"Error in proactively summarizing: {e}")

        # 未触发摘要 → 返回 None
        return None

    # 重试装饰器：最多尝试 3 次；上下文超限（ContextLengthExceededError）不重试
    # （摘要关闭时重试也会再次超限），CancelledError 因继承 BaseException 而被排除
    @retry(
        stop=stop_after_attempt(3),
        # Do not retry since it is only raised when summarization is OFF, and we will
        # likely hit the context length limit again despite retrying.
        retry=(
            retry_if_not_exception_type(ContextLengthExceededError)
            # To avoid asyncio.CancelledError retries which inherits from BaseException
            # rather than Exception
            & retry_if_exception_type(Exception)
        ),
        reraise=True,
    )
    async def _query_llm(
        self,
        chat: Chat,
        prompt: str,
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> LLMResponse:
        # 主 LLM 调用入口：正常调用 + 上下文超限/输出超限两类异常的自愈处理
        try:
            # 记录发起时刻
            start_time = time.time()
            # 经 Chat 发起主 LLM 调用（await 语句跨多行）
            llm_response = await chat.chat(
                prompt,
                **self._llm_call_kwargs,
            )
            # 记录结束时刻并累计耗时
            end_time = time.time()
            request_time_ms = (end_time - start_time) * 1000
            self._api_request_times.append(request_time_ms)

            # 正常返回 LLM 响应
            return llm_response

        except ContextLengthExceededError:
            # First, we check whether we want to summarize or not
            # 上下文超限：先看摘要开关
            if not self._enable_summarize:
                # 摘要关闭 → 无法自救，原样抛出（重试已被装饰器排除）
                self.logger.debug("Context length exceeded and summarization is OFF.")
                raise

            self.logger.debug("Context length exceeded. Using fallback summarization.")

            if session is None:
                # 没有终端会话就无法构造降级上下文 → 报错
                raise RuntimeError("Cannot handle context length error without session")

            # 第一步：回退删除最近的消息，先腾出约 4000 token 空间
            self._unwind_messages_to_free_tokens(chat, target_free_tokens=4000)

            # 摘要提示词占位：逐级兜底，任一级成功即停
            summary_prompt = None
            # Fallback 1: Try full summary
            # 兜底 1：完整三步摘要（效果最好，开销最大）
            try:
                self.logger.debug("SUMMARIZATION: Attempting full summary")
                summary_prompt, subagent_trajectory_refs = await self._summarize(
                    chat, original_instruction, session
                )
                # Store subagent_refs to include in the trajectory
                # 暂存子代理轨迹引用，稍后挂到主轨迹
                self._pending_subagent_refs = subagent_trajectory_refs
                # Store handoff prompt to add as a user step
                # 暂存交接提示词，稍后作为 user 步骤进轨迹
                self._pending_handoff_prompt = summary_prompt
                self.logger.debug("SUMMARIZATION: Full summary succeeded")
            except Exception as e:
                # 完整摘要失败 → 降级到下一级兜底
                self.logger.debug(f"SUMMARIZATION: Full summary failed: {e}")

            # Fallback 2: Try short summary
            # 兜底 2：极简摘要（单次调用，只看原指令 + 屏幕尾部 1000 字符）
            if summary_prompt is None:
                try:
                    self.logger.debug("SUMMARIZATION: Attempting short summary")
                    # 抓当前终端画面
                    current_screen = await session.capture_pane(capture_entire=False)
                    # 只保留尾部 1000 字符，控制 token 开销
                    limited_screen = current_screen[-1000:] if current_screen else ""

                    # 极简摘要提示词：让模型用 2-3 句话给出下一步
                    short_prompt = f"Briefly continue this task: {original_instruction}\n\nCurrent state: {limited_screen}\n\nNext steps (2-3 sentences):"

                    # 发起这次单轮 LLM 调用（await 语句跨多行）
                    short_llm_response: LLMResponse = await self._llm.call(
                        prompt=short_prompt,
                        **self._llm_call_kwargs,
                    )
                    # 把极简摘要与原指令拼成新的继续提示
                    summary_prompt = f"{original_instruction}\n\nSummary: {short_llm_response.content}"
                    self.logger.debug("SUMMARIZATION: Short summary succeeded")
                except Exception as e:
                    # 极简摘要也失败 → 降级到终极兜底
                    self.logger.error(f"SUMMARIZATION: Short summary failed: {e}")

            # Fallback 3: Ultimate fallback (no LLM calls)
            # 兜底 3：终极兜底（不再调 LLM，直接用原指令 + 屏幕尾部）
            if summary_prompt is None:
                self.logger.debug("SUMMARIZATION: Using ultimate fallback")
                # 抓当前终端画面
                current_screen = await session.capture_pane(capture_entire=False)
                # 只保留尾部 1000 字符
                limited_screen = current_screen[-1000:] if current_screen else ""
                # 拼出"原指令 + 当前状态"的续跑提示（语句跨多行）
                summary_prompt = (
                    f"{original_instruction}\n\nCurrent state: {limited_screen}"
                )

            try:
                # 用降级后的上下文重新发起主调用
                start_time = time.time()
                llm_response = await chat.chat(
                    summary_prompt,
                    **self._llm_call_kwargs,
                )
                end_time = time.time()
                request_time_ms = (end_time - start_time) * 1000
                self._api_request_times.append(request_time_ms)

            except Exception as e:
                # 连兜底调用都失败 → 造一个占位响应让循环继续（不崩溃）
                self.logger.error(f"Even fallback chat failed: {e}")
                llm_response = LLMResponse(
                    content="Technical difficulties. Please continue with the task."
                )

            # 返回（降级路径的）LLM 响应
            return llm_response

        except OutputLengthExceededError as e:
            # 输出长度超限：本轮所有动作都不会被执行，需要让模型重新发（拆小块）
            self.logger.debug(f"Output length exceeded: {e}")

            # 从异常上取被截断的原始输出文本（取不到则用占位串；表达式跨多行）
            truncated_response = getattr(
                e, "truncated_response", "[TRUNCATED RESPONSE NOT AVAILABLE]"
            )

            # Try to salvage a valid response from the truncated output
            # Only available for XML parser
            # 尝试从截断输出中抢救出有效响应（仅 XML 解析器实现了 salvage）
            salvaged_response = None
            has_multiple_blocks = False

            if hasattr(self._parser, "salvage_truncated_response"):
                # 解析器支持抢救 → 在截断文本里找结构完整的响应（语句跨多行）
                salvaged_response, has_multiple_blocks = (
                    self._parser.salvage_truncated_response(truncated_response)
                )

            if salvaged_response:
                self.logger.debug(
                    "Output exceeded length but found valid response. "
                    "Using truncated version."
                )

                # 抢救成功：直接返回截断但有效的响应文本
                return salvaged_response

            # If we get here, we couldn't salvage a valid response
            # Try to parse the truncated response to get warnings
            # 抢救失败 → 解析截断文本，收集解析器警告一并反馈给模型
            warnings_text = ""
            try:
                parse_result = self._parser.parse_response(truncated_response)
                if parse_result.warning:
                    # 拼装警告文本（表达式跨多行）
                    warnings_text = (
                        f"\n\nParser warnings from your truncated response:\n"
                        f"{parse_result.warning}"
                    )
            except Exception as parse_error:
                # 截断文本也解析不动 → 只记调试日志
                self.logger.debug(f"Failed to parse truncated response: {parse_error}")

            # Get the actual output limit for the model
            # 取模型真实的输出上限（未知则用通用措辞）
            output_limit = self._llm.get_model_output_limit()
            if output_limit is not None:
                limit_str = f"{output_limit} tokens"
            else:
                limit_str = "the maximum output length"

            # 错误消息：明确告知动作未执行、必须拆块重发（语句跨多行）
            error_msg = (
                "ERROR!! NONE of the actions you just requested were performed "
                f"because you exceeded {limit_str}. "
                f"Your outputs must be less than {limit_str}. Re-issue this request, "
                f"breaking it into chunks each of which is less than {limit_str}."
            )

            if warnings_text:
                # 附加截断响应的解析器警告
                error_msg += warnings_text

            # 把本轮 user prompt 与截断响应写回历史，保证上下文连续
            chat.messages.append({"role": "user", "content": prompt})
            chat.messages.append({"role": "assistant", "content": truncated_response})
            chat.reset_response_chain()

            # 递归重试：以纠错消息作为新一轮 prompt（await 语句跨多行）
            return await self._query_llm(
                chat=chat,
                prompt=error_msg,
                original_instruction=original_instruction,
                session=session,
            )

        except Exception as e:
            # 未知错误 → 记录后原样抛出
            self.logger.error(f"Unknown Error in LLM interaction: {e}")
            raise e

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> tuple[list[Command], bool, str, str, str, LLMResponse]:
        # LLM 交互 + 响应解析的封装：产出命令、完成标记、反馈与推理内容（签名跨多行）
        # 调用 LLM（内含超限/重试兜底；await 语句跨多行）
        llm_response = await self._query_llm(
            chat, prompt, original_instruction, session
        )

        # 用格式解析器把响应文本解析成命令与完成标记
        result = self._parser.parse_response(llm_response.content)

        # 组装反馈文本（错误/警告），供回灌给模型纠错
        feedback = ""
        if result.error:
            # 硬错误优先
            feedback += f"ERROR: {result.error}"
            if result.warning:
                # 有错误时警告附加在后
                feedback += f"\nWARNINGS: {result.warning}"
        elif result.warning:
            # 只有警告时单独列出
            feedback += f"WARNINGS: {result.warning}"

        if result.warning:
            # 警告也写一份调试日志便于排查
            self.logger.debug(f"Parser warnings: {result.warning}")

        # 把解析出的命令转成内部 Command 结构
        commands = []
        for parsed_cmd in result.commands:
            commands.append(
                Command(
                    keystrokes=parsed_cmd.keystrokes,
                    # 单命令等待上限 60 秒（防一条命令卡死整个回合）
                    duration_sec=min(parsed_cmd.duration, 60),
                )
            )

        # 返回六元组：命令、完成标记、反馈、分析、计划、原始响应
        return (
            commands,
            result.is_task_complete,
            feedback,
            result.analysis,
            result.plan,
            llm_response,
        )

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        # 在终端中执行一批命令，返回（是否超时, 终端输出）（def 签名跨多行）
        """Execute a batch of commands in the terminal.

        Args:
            commands: List of commands to execute
            session: TmuxSession instance

        Returns:
            Tuple of (timeout_occurred, terminal_output)
        """
        # 逐条把命令敲进 tmux 终端
        for command in commands:
            try:
                # 非阻塞发送按键，至少等待 duration_sec 让输出稳定（语句跨多行）
                await session.send_keys(
                    command.keystrokes,
                    block=False,
                    min_timeout_sec=command.duration_sec,
                )
            except TimeoutError:
                # 该命令在等待窗口内未稳定 → 判定超时，用模板生成超时反馈
                # （return 语句跨多行，包含增量输出截断）
                return True, self._timeout_template.format(
                    timeout_sec=command.duration_sec,
                    command=command.keystrokes,
                    terminal_state=self._limit_output_length(
                        await session.get_incremental_output()
                    ),
                )

        # 全部命令执行完毕：取本批次的增量输出（截断到字节上限）
        return False, self._limit_output_length(await session.get_incremental_output())

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        original_instruction: str = "",
    ) -> None:
        # 代理主循环：LLM ↔ 终端反复交互，直至确认完成/回合耗尽/会话死亡
        if self._context is None:
            # 没有上下文对象说明 run() 未被正确调用 → 防御性报错
            raise RuntimeError("Agent context is not set. This should never happen.")

        # 当前轮发给 LLM 的提示词（初始为系统模板渲染出的首轮指令）
        prompt = initial_prompt

        # 清零本 run 的 token/成本统计
        self._context.n_input_tokens = 0
        self._context.n_output_tokens = 0
        self._context.n_cache_tokens = 0
        self._context.cost_usd = None

        if self._session is None:
            # 未 setup 就跑主循环 → 防御性报错
            raise RuntimeError("Session is not set. This should never happen.")

        # Step ID offset accounts for initial steps in the trajectory:
        # Step 1: system message (includes task instruction)
        # Steps 2+: agent episodes (starting from episode 0)

        # 主循环：episode 0..max_episodes
        for episode in range(self._max_episodes):
            # 记录已执行回合数（1 基）
            self._n_episodes = episode + 1
            if not await self._session.is_session_alive():
                # tmux 会话已死（如容器退出）→ 直接结束循环
                self.logger.debug("Session has ended, breaking out of agent loop")
                return

            if original_instruction and self._enable_summarize:
                # 主动摘要检查：剩余 token 低于阈值就提前压缩上下文（语句跨多行）
                proactive_summary_result = await self._check_proactive_summarization(
                    chat,
                    original_instruction,
                    self._session,
                )
                if proactive_summary_result:
                    # 触发了摘要：交接提示词成为本轮 prompt
                    prompt, subagent_refs = proactive_summary_result
                    # Store subagent_refs to add a system step later
                    # 暂存子代理引用，稍后以 system 步骤记录
                    self._pending_subagent_refs = subagent_refs
                    # Also store the handoff prompt to add as a user step
                    # 暂存交接提示词，稍后以 user 步骤记录
                    self._pending_handoff_prompt = prompt

            # Track token counts and cost before this step
            # 记录本步开始时的 token/成本基线（结束后做差得本步指标）
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            # LLM 交互：调用 + 解析（await 语句跨多行）
            (
                commands,
                is_task_complete,
                feedback,
                analysis,
                plan,
                llm_response,
            ) = await self._handle_llm_interaction(
                chat, prompt, original_instruction, self._session
            )

            # If we have pending subagent refs, add a system step to record the delegation
            # This must happen before we build the agent step
            # We use len(self._trajectory_steps) + 1 as the step_id to ensure it's sequential
            # 有待挂的子代理引用 → 先以 system 步骤记录这次"上下文摘要 + 交接"委派
            if self._pending_subagent_refs:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Performed context summarization and handoff to continue task.",
                        # 观测里引用三个子代理轨迹（外部化存储，避免主轨迹爆炸）
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    subagent_trajectory_ref=self._pending_subagent_refs
                                )
                            ]
                        ),
                        # 上下文管理元数据：类型 compaction（压缩），边界 replace（整体替换）
                        extra={
                            "context_management": {
                                "type": "compaction",
                                "boundary": "replace",
                            }
                        },
                    )
                )
                # 清空待挂引用（语句跨多行的 Step 构造已完成）
                self._pending_subagent_refs = None

            # Handle handoff prompt based on linear_history mode
            # 交接提示词处理（分 linear_history 与普通两种轨迹策略）
            if self._pending_handoff_prompt:
                # If linear_history mode is enabled, split trajectory immediately WITHOUT adding handoff step
                # The handoff step will be added to the continuation trajectory during the split
                if self._linear_history:
                    # 线性历史模式：立即拆分轨迹文件，交接步骤在续篇里补
                    self._split_trajectory_on_summarization(
                        self._pending_handoff_prompt
                    )
                else:
                    # For non-linear mode, add the handoff prompt as a user step
                    # 普通模式：交接提示词作为 user 步骤追加进同一份轨迹（语句跨多行）
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="user",
                            message=self._pending_handoff_prompt,
                        )
                    )
                # 清空待挂交接提示词
                self._pending_handoff_prompt = None

            # Create message content from analysis and plan, or use raw response if raw_content is enabled
            # 构造轨迹步骤的 message 内容：原始响应或结构化的 analysis/plan
            if self._save_raw_content_in_trajectory:
                # Use the raw LLM response content for SFT data export
                # raw_content 模式：保留原始响应用于 SFT 数据导出
                message_content = llm_response.content
            else:
                # Parse into structured format (analysis + plan)
                # 结构化模式：拼 "Analysis: ... / Plan: ..."
                message_parts = []
                if analysis:
                    message_parts.append(f"Analysis: {analysis}")
                if plan:
                    message_parts.append(f"Plan: {plan}")
                # 两段都没有就存空串（换行拼接）
                message_content = "\n".join(message_parts) if message_parts else ""

            # 把本步之后的累计 token/成本同步进 context
            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            if feedback and "ERROR:" in feedback:
                # 解析出错：不执行任何命令，把错误反馈给模型要求修正
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{self._get_error_response_type()}."
                )
                # For error cases, we still want to record the step
                # Use the raw response as the message since parsing failed
                # 错误轮也要记轨迹（message 存原始响应，因为解析失败了）
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before

                if llm_response.model_name:
                    # 记录响应实际使用的模型名
                    self._last_response_model_name = llm_response.model_name
                # 追加错误轮的 agent 步骤（观测即纠错 prompt；语句跨多行）
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=llm_response.model_name or self._model_name,
                        message=llm_response.content,
                        reasoning_content=llm_response.reasoning_content,
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    content=prompt,
                                )
                            ]
                        ),
                        # 本步 token/成本做差统计（语句跨多行）
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens
                            - tokens_before_output,
                            cached_tokens=cache_tokens_used
                            if cache_tokens_used > 0
                            else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                # 进入下一轮：以纠错 prompt 重新请求
                continue

            # 正常路径：在 tmux 中执行命令并拿到终端输出
            timeout_occurred, terminal_output = await self._execute_commands(
                commands,
                self._session,
            )

            # Capture the pending completion state before potentially modifying it
            # 先快照"待确认完成"状态（下方会修改该字段）
            was_pending_completion = self._pending_completion

            # Construct the observation (what gets sent back to the LLM)
            # 构造观测（回灌给 LLM 的内容）
            if is_task_complete:
                if self._pending_completion:
                    # 第二次收到完成标记（已确认）→ 观测就是普通终端输出
                    observation = terminal_output
                else:
                    # 第一次收到完成标记 → 置待确认态，观测换成二次确认消息
                    self._pending_completion = True
                    observation = self._get_completion_confirmation_message(
                        terminal_output
                    )
            else:
                # 模型未宣告完成 → 清除待确认态
                self._pending_completion = False
                if feedback and "WARNINGS:" in feedback:
                    # 有警告：警告 + 终端输出一起回灌
                    observation = (
                        f"Previous response had warnings:\n{feedback}\n\n"
                        f"{self._limit_output_length(terminal_output)}"
                    )
                else:
                    # 只有终端输出（做长度截断）
                    observation = self._limit_output_length(terminal_output)

            # Record the step in trajectory
            # 计算本步 token/成本增量（写轨迹指标用）
            cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
            step_cost = chat.total_cost - cost_before

            # Create tool_calls array from commands and task completion
            # Note: Although Terminus 2 doesn't offer native tool calling (it uses text-based command parsing),
            # we represent parsed commands as tool calls for better trajectory analysis and compatibility with tooling.
            # However, when raw_content mode is enabled, we skip tool_calls generation to preserve the raw LLM response.
            tool_calls: list[ToolCall] | None = None
            observation_results: list[ObservationResult] = []

            if not self._save_raw_content_in_trajectory:
                # Only create tool_calls when NOT in raw_content mode
                # 非 raw_content 模式：把解析出的命令表示成 tool_calls
                tool_calls_list: list[ToolCall] = []

                if commands:
                    for i, cmd in enumerate(commands):
                        # 每条命令一个 bash_command 工具调用，ID 按 episode+序号
                        tool_call_id = f"call_{episode}_{i + 1}"
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=tool_call_id,
                                function_name="bash_command",
                                arguments={
                                    "keystrokes": cmd.keystrokes,
                                    "duration": cmd.duration_sec,
                                },
                            )
                        )

                    # Multi-command batches share one terminal output, so only
                    # single-command observations can be linked precisely.
                    # 多命令批次共享一份终端输出：只有单命令批次能精确归属
                    # （source_call_id 为 None 时表示无法唯一关联）
                    observation_results.append(
                        ObservationResult(
                            source_call_id=_terminal_observation_source_call_id(
                                commands, episode
                            ),
                            content=observation,
                        )
                    )

                # Add task_complete as a tool call if the agent marked the task complete
                if is_task_complete:
                    # 完成宣告也建模为一个 mark_task_complete 工具调用
                    task_complete_call_id = f"call_{episode}_task_complete"
                    tool_calls_list.append(
                        ToolCall(
                            tool_call_id=task_complete_call_id,
                            function_name="mark_task_complete",
                            arguments={},
                        )
                    )
                    # If there are no commands, we still need to add an observation result
                    if not commands:
                        # 没有命令时也要留一条观测结果（终端输出）
                        observation_results.append(
                            ObservationResult(
                                content=observation,
                            )
                        )
                elif not commands:
                    # No commands and no task completion, just the observation
                    # 既无命令也未完成：只记录观测
                    observation_results.append(
                        ObservationResult(
                            content=observation,
                        )
                    )

                # 空列表归一化为 None（轨迹里省略字段）
                tool_calls = tool_calls_list or None
            else:
                # In raw_content mode, just add observation without tool_calls
                # raw_content 模式：只记观测，不生成 tool_calls
                observation_results.append(
                    ObservationResult(
                        content=observation,
                    )
                )

            # Build the step object using Pydantic models
            if llm_response.model_name:
                # 记录响应实际使用的模型名
                self._last_response_model_name = llm_response.model_name
            # 追加本回合的正常 agent 步骤（消息/工具调用/观测/指标；语句跨多行）
            self._trajectory_steps.append(
                Step(
                    step_id=len(self._trajectory_steps) + 1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=llm_response.model_name or self._model_name,
                    message=message_content,
                    reasoning_content=llm_response.reasoning_content,
                    tool_calls=tool_calls,
                    observation=Observation(results=observation_results),
                    metrics=Metrics(
                        prompt_tokens=chat.total_input_tokens - tokens_before_input,
                        completion_tokens=chat.total_output_tokens
                        - tokens_before_output,
                        cached_tokens=cache_tokens_used
                        if cache_tokens_used > 0
                        else None,
                        cost_usd=step_cost if step_cost > 0 else None,
                        prompt_token_ids=llm_response.prompt_token_ids,
                        completion_token_ids=llm_response.completion_token_ids,
                        logprobs=llm_response.logprobs,
                    ),
                )
            )

            # Dump trajectory after every episode to ensure we persist even if program terminates
            # 每回合落盘一次轨迹：程序中途被杀也能保住已有进度
            self._dump_trajectory()

            if is_task_complete:
                if was_pending_completion:
                    # Task is confirmed complete (this is the second time task_complete was True), return
                    # 第二次确认完成 → 任务真正结束，退出主循环
                    return
                else:
                    # First completion attempt - ask for confirmation and continue
                    # 第一次宣告完成 → 以确认消息为下一轮 prompt，继续等确认
                    prompt = observation
                    continue

            # 未完成：以本轮观测作为下一轮 prompt
            prompt = observation

    def _reset_per_run_state(self) -> None:
        """Reset all per-run state. The same Terminus2 instance is reused
        across multiple `run()` invocations in multi-step trials, so any
        accumulator that should be scoped to a single step must be reset here.
        """
        # 清空主轨迹步骤
        self._trajectory_steps = []
        # 清空 API 请求耗时记录
        self._api_request_times = []
        # 回合计数归零
        self._n_episodes = 0
        # 摘要轮次归零
        self._summarization_count = 0
        # 子代理指标归零（新累计器）
        self._subagent_metrics = SubagentMetrics()
        # 子代理 rollout 明细清空
        self._subagent_rollout_details = []
        # 完成确认状态复位
        self._pending_completion = False
        # 待挂子代理引用清空
        self._pending_subagent_refs = None
        # 待挂交接提示词清空
        self._pending_handoff_prompt = None
        # 会话 ID 复位：用用户显式提供的或重新生成
        self._session_id = self._user_provided_session_id or str(uuid.uuid4())

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # 一次 rollout 的入口：复位状态、渲染首轮提示、进主循环、finally 收尾统计
        # 多步 trial 复用同一实例 → 先复位所有单次运行状态
        self._reset_per_run_state()
        # 为主对话新建 Chat（决定是否回传交错思考内容）
        self._chat = Chat(self._llm, interleaved_thinking=self._interleaved_thinking)
        # 绑定运行上下文（最终统计写入这里）
        self._context = context

        if self._session is None:
            # 未 setup 就 run → 防御性报错
            raise RuntimeError("Session is not set")

        # Get the terminal state for the initial prompt
        # 取当前终端状态（截断）作为首轮提示的 terminal_state
        terminal_state = self._limit_output_length(
            await self._session.get_incremental_output()
        )

        # 增强指令：在原任务指令上追加 MCP/技能等环境信息
        augmented_instruction = instruction
        if self.mcp_servers:
            # 有 MCP 服务器 → 拼出清单说明
            mcp_info = "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n"
            for s in self.mcp_servers:
                if s.transport == "stdio":
                    # stdio 传输：给出启动命令与参数
                    args_str = " ".join(s.args)
                    mcp_info += f"- {s.name}: stdio transport, command: {s.command} {args_str}\n"
                else:
                    # 其他传输：给出 url
                    mcp_info += f"- {s.name}: {s.transport} transport, url: {s.url}\n"
            # 把 MCP 清单附加到指令之后
            augmented_instruction = instruction + mcp_info

        # 发现并广播可用技能（<available_skills> 区块）
        skills_section = await self._build_skills_section(environment)
        if skills_section:
            # 有技能就附加到增强指令
            augmented_instruction += skills_section

        # 用系统模板渲染首轮提示（指令 + 终端状态；语句跨多行）
        initial_prompt = self._prompt_template.format(
            instruction=augmented_instruction,
            terminal_state=terminal_state,
        )

        # 轨迹第 1 步：system/user 的首轮提示（含任务指令与终端状态）
        self._trajectory_steps.append(
            Step(
                step_id=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=initial_prompt,
            )
        )

        try:
            # 进入主循环（语句跨多行）
            await self._run_agent_loop(
                initial_prompt=initial_prompt,
                chat=self._chat,
                original_instruction=instruction,
            )
        finally:
            # Always capture metrics and rollout details even if the agent times out or errors
            # Combine main chat rollout details with subagent rollout details
            # 无论超时/报错都要收尾统计：合并主对话与子代理的 rollout 明细
            context.rollout_details = (
                self._chat.rollout_details + self._subagent_rollout_details
            )

            # Include subagent metrics in context totals
            # 总 token/成本 = 主对话 + 全部子代理
            context.n_input_tokens = (
                self._chat.total_input_tokens
                + self._subagent_metrics.total_prompt_tokens
            )
            context.n_output_tokens = (
                self._chat.total_output_tokens
                + self._subagent_metrics.total_completion_tokens
            )
            context.n_cache_tokens = (
                self._chat.total_cache_tokens
                + self._subagent_metrics.total_cached_tokens
            )
            # 总成本（非正数记 None）
            total_cost = self._chat.total_cost + self._subagent_metrics.total_cost_usd
            context.cost_usd = total_cost if total_cost > 0 else None
            # 元数据：回合数、各次请求耗时、摘要轮次
            context.metadata = {
                "n_episodes": self._n_episodes,
                "api_request_times_msec": self._api_request_times,
                "summarization_count": self._summarization_count,
            }
            if self._store_all_messages:
                # 可选：把完整聊天消息塞进元数据（体积可能很大）
                context.metadata["all_messages"] = self._chat.messages

            # Dump trajectory to JSON
            # 最终落盘轨迹 JSON
            self._dump_trajectory()

    # ===== Trajectory-related helper functions =====
    # =====（轨迹相关辅助函数）=====

    @staticmethod
    def _remove_metrics_from_copied_steps(steps: list[Step]) -> None:
        """Remove metrics from copied trajectory steps and mark as copied context.

        Args:
            steps: List of trajectory steps to modify in-place
        """
        for step in steps:
            # Mark all copied steps with is_copied_context=True
            # 全部标记为"复制上下文"（非本轨迹原生步骤）
            step.is_copied_context = True

            # Remove metrics to avoid duplication
            # 剥离指标避免重复统计（原轨迹已记过一次）
            if step.metrics:
                step.metrics = None
                if step.extra is None:
                    step.extra = {}
                # 留个说明便于阅读轨迹时理解指标为何为空
                step.extra["note"] = (
                    "Metrics omitted to avoid duplication - already recorded in parent trajectory"
                )

    def _prepare_copied_trajectory_steps(
        self, steps_to_include: int
    ) -> tuple[list[Step], int]:
        """Prepare trajectory steps for subagent by copying and removing metrics.

        Args:
            steps_to_include: Number of steps to include from main trajectory

        Returns:
            Tuple of (copied_steps, next_step_id)
        """
        # 深拷贝主轨迹前 N 步（子代理修改不会影响主轨迹）
        copied_steps = copy.deepcopy(self._trajectory_steps[:steps_to_include])
        # 剥离复制步骤的指标并打上复制标记
        self._remove_metrics_from_copied_steps(copied_steps)
        # 计算下一个可用步骤号
        next_step_id = len(copied_steps) + 1
        # 返回拷贝步骤与下一个步骤号
        return copied_steps, next_step_id

    def _append_subagent_response_step(
        self,
        steps: list[Step],
        step_id: int,
        response: LLMResponse,
        usage_info,
        subagent_name: str,
    ) -> None:
        # 追加子代理响应步骤：有用量则带指标，否则降级为无指标步骤（签名跨多行）
        """Append a response step with conditional metrics to trajectory steps.

        Args:
            steps: List of steps to append to
            step_id: ID for the new step
            response: LLM response
            usage_info: Usage info (may be None)
            subagent_name: Name of subagent for warning message
        """
        if usage_info:
            # 有用量信息 → 响应步骤带完整指标（语句跨多行）
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=response.model_name or self._model_name,
                    message=response.content,
                    reasoning_content=response.reasoning_content,
                    metrics=Metrics(
                        prompt_tokens=usage_info.prompt_tokens,
                        completion_tokens=usage_info.completion_tokens,
                        cached_tokens=usage_info.cache_tokens,
                        cost_usd=usage_info.cost_usd
                        if usage_info.cost_usd > 0
                        else None,
                        prompt_token_ids=response.prompt_token_ids,
                        completion_token_ids=response.completion_token_ids,
                        logprobs=response.logprobs,
                    ),
                )
            )
        else:
            # 拿不到用量 → 记调试日志并追加不带指标的响应步骤
            self.logger.debug(f"Failed to get token usage for {subagent_name}")
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=response.model_name or self._model_name,
                    message=response.content,
                    reasoning_content=response.reasoning_content,
                )
            )

    def _save_subagent_trajectory(
        self,
        session_id: str,
        agent_name: str,
        steps: list[Step],
        usage_info,
        filename_suffix: str,
        summary_text: str,
    ) -> SubagentTrajectoryRef:
        # 子代理轨迹落盘并返回引用句柄（def 签名跨多行）
        """Save a subagent trajectory to disk and return its reference.

        Args:
            session_id: Session ID for the subagent
            agent_name: Name of the subagent (e.g., "terminus-2-summarization-summary")
            steps: List of trajectory steps
            usage_info: Usage information from LLM response
            filename_suffix: Suffix for trajectory filename (e.g., "summary", "questions", "answers")
            summary_text: Human-readable summary for the trajectory ref extra field

        Returns:
            SubagentTrajectoryRef for inclusion in parent trajectory
        """
        # 从用量信息提取并归一化四项指标（语句跨多行）
        total_prompt, total_completion, total_cached, total_cost = (
            self._extract_usage_metrics(usage_info)
        )

        # 构造子代理轨迹对象（记录父会话与摘要轮次便于追溯；语句跨多行）
        trajectory = Trajectory(
            session_id=session_id,
            agent=Agent(
                name=agent_name,
                version=self.version() or "unknown",
                model_name=self._model_name,
                extra={
                    "parent_session_id": self._session_id,
                    "summarization_index": self._summarization_count,
                },
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                total_cached_tokens=total_cached,
                total_cost_usd=total_cost if total_cost > 0 else None,
            ),
        )

        # 子代理轨迹文件路径：主轨迹同目录，带摘要轮次与阶段后缀（语句跨多行）
        trajectory_path = (
            self.logs_dir
            / f"trajectory.summarization-{self._summarization_count}-{filename_suffix}.json"
        )
        try:
            # 序列化并写盘
            with open(trajectory_path, "w") as f:
                f.write(format_trajectory_json(trajectory.to_json_dict()))
            self.logger.debug(
                f"{filename_suffix.capitalize()} subagent trajectory saved to {trajectory_path}"
            )
        except Exception as save_error:
            # 落盘失败只记错误：不影响主流程继续
            self.logger.error(
                f"Failed to save {filename_suffix} subagent trajectory: {save_error}"
            )

        # 返回给父轨迹引用的轻量句柄（文件名 + 摘要说明；语句跨多行）
        return SubagentTrajectoryRef(
            session_id=session_id,
            trajectory_path=str(trajectory_path.name),
            extra={"summary": summary_text},
        )

    def _convert_chat_messages_to_steps(
        self,
        chat_messages: list[dict[str, Any]],
        additional_user_message: str | None = None,
        mark_as_copied: bool = False,
    ) -> list[Step]:
        # 聊天消息 → 轨迹步骤的转换（用于线性历史续篇的轨迹重建；签名跨多行）
        """Convert chat messages to trajectory steps.

        Args:
            chat_messages: List of chat messages with 'role' and 'content' fields
            additional_user_message: Optional additional user message to append as final step
            mark_as_copied: If True, mark all steps with is_copied_context=True (for continuation trajectories)

        Returns:
            List of Step objects representing the chat history
        """
        # 结果步骤列表
        steps = []
        # 步骤号从 1 重新编号
        step_id = 1

        for msg in chat_messages:
            # 取消息角色与内容
            role = msg["role"]
            content = msg["content"]

            # Map chat role to trajectory source
            # 角色映射：chat 的 assistant → 轨迹的 agent
            if role == "assistant":
                source = "agent"
                step = Step(
                    step_id=step_id,
                    source=source,
                    message=content,
                    model_name=self._last_response_model_name or self._model_name,
                )
            else:
                # 其他角色（system/user）直接沿用为 source
                source = role
                step = Step(
                    step_id=step_id,
                    source=source,
                    message=content,
                )

            # Mark as copied context if this is for a continuation trajectory
            # 续篇轨迹中的步骤标记为复制上下文
            if mark_as_copied:
                step.is_copied_context = True

            # 加入结果并递增步骤号
            steps.append(step)
            step_id += 1

        # Add the additional user message if provided
        # Note: The additional user message is NOT marked as copied since it's the new handoff prompt
        if additional_user_message:
            # 附加的 user 消息（新交接提示词）不标 copied：它是续篇的新内容
            steps.append(
                Step(
                    step_id=step_id,
                    source="user",
                    message=additional_user_message,
                )
            )

        # 返回转换后的步骤列表
        return steps

    # 线性历史模式：摘要发生时拆分轨迹文件，当前段落落盘后重置为续篇
    def _split_trajectory_on_summarization(self, handoff_prompt: str) -> None:
        """Split trajectory on summarization when linear_history is enabled.

        Saves current trajectory segment and resets for continuation with full linear history.

        Args:
            handoff_prompt: The handoff prompt containing answers
        """
        # Save current trajectory segment before creating a continuation
        # When _summarization_count is 1, dump to trajectory.json (continuation_index = 0)
        # When _summarization_count is 2, dump to trajectory.cont-1.json (continuation_index = 1)
        # 先把当前轨迹段落盘（第 1 次摘要存 trajectory.json，第 2 次存 trajectory.cont-1.json）
        self._dump_trajectory_with_continuation_index(self._summarization_count - 1)

        # Create new session_id for continuation
        # 为续篇生成新会话 ID：去掉旧 -cont- 后缀再接新的续篇编号（表达式跨多行）
        self._session_id = (
            f"{self._session_id.split('-cont-')[0]}-cont-{self._summarization_count}"
        )

        # After dumping the trajectory till the summarization stage, reset trajectory by
        # converting from chat messages (excluding the last response which will be added
        # by the normal agent loop flow). Mark all these steps as copied context since they
        # were already present in the previous trajectory segment.
        # 用聊天消息重建轨迹（去掉最后一条响应——它会由正常循环补上）；
        # 全部标记为复制上下文，因为它们已在上一段轨迹中出现过
        if self._chat:
            self._trajectory_steps = self._convert_chat_messages_to_steps(
                self._chat.messages[:-1], mark_as_copied=True
            )

    # 按续篇编号落盘轨迹 JSON（ATIF 格式；0 号即主轨迹文件）
    def _dump_trajectory_with_continuation_index(self, continuation_index: int) -> None:
        """Dump trajectory data to JSON file with specified continuation index.

        Args:
            continuation_index: The continuation index to use for filename and metadata.
                               For the initial trajectory, use 0.
                               For the first continuation, use 1, etc.
        """
        if not self._context:
            # 无运行上下文（尚未 run）→ 跳过落盘
            self.logger.debug("No context available, skipping trajectory dump")
            return

        # Construct the trajectory using Pydantic models for validation
        # 用 context 汇总构造终局指标（语句跨多行）
        final_metrics = FinalMetrics(
            total_prompt_tokens=self._context.n_input_tokens,
            total_completion_tokens=self._context.n_output_tokens,
            total_cached_tokens=self._context.n_cache_tokens or 0,
            total_cost_usd=self._context.cost_usd if self._context.cost_usd else None,
        )

        # agent 元信息：解析器名必带，温度/llm_kwargs/续篇编号按需附带
        agent_extra: dict[str, Any] = {
            "parser": self._parser_name,
        }
        if self._temperature is not None:
            # 显式配置过温度才记录
            agent_extra["temperature"] = self._temperature
        if self._llm_kwargs:
            # 有额外构造参数才记录
            agent_extra["llm_kwargs"] = self._llm_kwargs
        if self._linear_history and continuation_index > 0:
            # 线性历史模式下的续篇编号
            agent_extra["continuation_index"] = continuation_index

        # Determine if this trajectory will be continued
        # In linear_history mode, when saving during summarization (i.e., continuation_index < _summarization_count),
        # this trajectory will have a continuation
        # 判断本轨迹是否会有续篇（线性历史 + 摘要尚未全部处理完）
        continued_trajectory_ref = None
        if self._linear_history and continuation_index < self._summarization_count:
            # This trajectory segment will be continued in the next file
            # 本段会被下一文件续写 → 记下续篇文件引用
            next_continuation_index = continuation_index + 1
            continued_trajectory_ref = f"trajectory.cont-{next_continuation_index}.json"

        # 组装完整轨迹对象（会话 ID、代理信息、步骤、终局指标、续篇引用；语句跨多行）
        trajectory = Trajectory(
            session_id=self._session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self._model_name,
                extra=agent_extra,
            ),
            steps=self._trajectory_steps,
            final_metrics=final_metrics,
            continued_trajectory_ref=continued_trajectory_ref,
        )

        # Determine trajectory filename based on continuation index
        # 按续篇编号决定文件名
        if self._linear_history and continuation_index > 0:
            # 续篇：trajectory.cont-<n>.json（语句跨多行）
            trajectory_path = (
                self.logs_dir / f"trajectory.cont-{continuation_index}.json"
            )
        else:
            # 主轨迹：trajectory.json
            trajectory_path = self.logs_dir / "trajectory.json"

        try:
            # 序列化写盘
            with open(trajectory_path, "w") as f:
                json_str = format_trajectory_json(trajectory.to_json_dict())
                f.write(json_str)
            self.logger.debug(f"Trajectory dumped to {trajectory_path}")
        except Exception as e:
            # 落盘失败只记错误：不能让轨迹 IO 问题打断 agent 运行
            self.logger.error(f"Failed to dump trajectory: {e}")

    def _dump_trajectory(self) -> None:
        """Dump trajectory data to JSON file following ATIF format."""
        # 默认落盘：续篇编号取当前摘要轮次（未摘要时为 0 → trajectory.json）
        self._dump_trajectory_with_continuation_index(self._summarization_count)

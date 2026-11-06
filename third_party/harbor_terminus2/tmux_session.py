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

# 模块说明：实现 TmuxSession——在 harbor 环境中托管一个 tmux 会话，
# 负责录像工具（tmux/asciinema）安装、会话启动、按键注入（send-keys，
# 超长按键走粘贴缓冲回退）、面板捕获与增量输出（incremental output），
# 以及 asciinema 终端录像（terminal recording）与时间戳标记的合并。

# 导入 asyncio：提供异步等待原语（wait_for/sleep），用于安装预算与非阻塞补眠。
import asyncio
# 导入 base64：把超长按键负载编码为 ASCII 安全的 base64 分块，便于经 shell 写入文件。
import base64
# 导入 re：用正则判断按键文本是否以换行符结尾（即会触发命令执行）。
import re
# 导入 shlex：对会话名、按键等参数做 shell 安全引用（quote），防止注入与参数拆分。
import shlex
# 导入 time：记录发送起始时刻，用于统计阻塞/非阻塞发送的耗时。
import time
# 导入 uuid：生成唯一令牌，避免并发投递时临时文件/粘贴缓冲区命名冲突。
import uuid
# 导入 Path 与 PurePosixPath：本地宿主机脚本路径用 Path，容器内 POSIX 路径用 PurePosixPath 表达。
from pathlib import Path, PurePosixPath
# 导入 Any：类型注解中使用（字典值可为任意类型）。
from typing import Any

# 导入 AsciinemaHandler：负责把时间戳标记（markers）合并进 asciinema 的 .cast 录像。
from .asciinema_handler import AsciinemaHandler
# 导入 harbor 的 BaseEnvironment（容器环境抽象）与 ExecResult（命令执行结果）。
from harbor.environments.base import BaseEnvironment, ExecResult
# 导入 harbor 的统一日志器。
from harbor.utils.logger import logger


# TmuxSession：封装单个 tmux 会话的生命周期与交互，是智能体（agent）与
# 容器内终端之间的主要通道。
class TmuxSession:
    # 视为"回车"的按键名集合：发送其中任一键会执行当前输入行。
    _ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
    # 正则模式：匹配以回车/换行字符结尾的按键文本（同样会触发命令执行）。
    _ENDS_WITH_NEWLINE_PATTERN = r"[\r\n]$"
    # 需要从按键文本尾部剥离的换行字符集合。
    _NEWLINE_CHARS = "\r\n"
    # 追加在命令末尾的 tmux 完成信号：`tmux wait -S done` 立即置位事件，
    # 让阻塞等待方（`tmux wait done`）立刻返回，用于实现阻塞式发送的完成检测。
    _TMUX_COMPLETION_COMMAND = "; tmux wait -S done"
    # tmux rejects commands exceeding its internal message buffer (~16 KB
    # since tmux 1.9, see https://github.com/tmux/tmux/issues/254) with
    # "command too long". Keep a conservative margin below that hard
    # ceiling; keys that cannot fit are delivered via a paste buffer
    # instead (see _paste_key).
    # 该常量：单条 tmux send-keys 命令允许的最大字节数（在 ~16 KB 硬上限之下留保守余量）。
    _TMUX_SEND_KEYS_MAX_COMMAND_LENGTH = 16_000
    # stderr marker tmux prints when a client command exceeds the build's
    # message size limit. Older tmux builds enforce far smaller limits than
    # the ~16 KB above, so this is also handled at runtime as a fallback.
    # 该常量：tmux 命令超出其消息上限时打印到 stderr 的错误标记，
    # 运行时据此识别比上面假设值更小的旧版 tmux 限制并降级处理。
    _TMUX_COMMAND_TOO_LONG_MARKER = "command too long"
    # Base64 characters staged per `environment.exec` call when writing a
    # paste-buffer file. exec does not go through tmux, so the bound is the
    # shell's ARG_MAX; stay far below it while keeping round-trips few.
    # 该常量：每次 environment.exec 写入粘贴缓冲文件时暂存的 base64 字符数；
    # 取值远低于 shell 的 ARG_MAX，同时尽量减少命令往返次数。
    _PASTE_BASE64_CHUNK_LEN = 65_536
    # Per-command ceiling for tool-install execs (apt/pip/source build). A
    # command whose outbound connection is silently dropped (e.g. asciinema via
    # pip on a closed-internet task) produces no output and would otherwise
    # block until the agent-setup timeout; bounding each one lets the install
    # fall through to the next fallback or degrade to a logged failure.
    # 该常量：单条工具安装命令（apt/pip/源码编译）的超时上限（秒）。
    _TOOL_INSTALL_TIMEOUT_SEC = 120
    # Ceiling for the whole tool-install step, kept well under harbor's 360s
    # agent-setup timeout so even a pathological chain of hung commands cannot
    # consume the setup budget. On expiry recording tools are skipped.
    # 该常量：整个工具安装步骤的总预算（秒），远小于 harbor 360 秒的
    # agent-setup（智能体环境初始化）超时，保证初始化预算不被耗尽。
    _TOOL_INSTALL_BUDGET_SEC = 240
    # 容器内"获取 asciinema 时间戳"辅助脚本的安装路径（用于向录像打时间戳标记）。
    GET_ASCIINEMA_TIMESTAMP_SCRIPT_CONTAINER_PATH = Path(
        "/tmp/get-asciinema-timestamp.sh"
    )
    # 宿主机（本仓库源码目录内）同名脚本路径，启动时上传到容器。
    _GET_ASCIINEMA_TIMESTAMP_SCRIPT_HOST_PATH = Path(__file__).parent / (
        "get-asciinema-timestamp.sh"
    )

    # 初始化会话对象：校验并保存面板尺寸、日志/录像路径、会话名与执行用户等配置。
    def __init__(
        self,
        session_name: str,
        environment: BaseEnvironment,
        logging_path: Path | PurePosixPath,
        local_asciinema_recording_path: Path | None,
        remote_asciinema_recording_path: Path | PurePosixPath | None,
        pane_width: int = 160,
        pane_height: int = 40,
        extra_env: dict[str, str] | None = None,
        user: str | int | None = None,
    ):
        try:
            # 把面板宽度转换为整数（单位：字符列数）。
            self._pane_width = int(pane_width)
            # 把面板高度转换为整数（单位：字符行数）。
            self._pane_height = int(pane_height)
        except (ValueError, TypeError):
            # 无法转换为整数时抛出带说明的异常。
            raise ValueError("pane_width and pane_height must be valid integers.")
        # 宽高必须为正，否则 tmux 无法创建合法会话。
        if self._pane_width <= 0 or self._pane_height <= 0:
            raise ValueError("pane_width and pane_height must be positive integers.")
        # 保存 pipe-pane 终端输出日志（会话原始输出记录）的目标路径。
        self._logging_path = logging_path
        # 保存宿主机侧 asciinema 录像（.cast 终端录制文件）的本地存放路径。
        self._local_asciinema_recording_path = local_asciinema_recording_path
        # 保存容器内 asciinema 录像的远程路径。
        self._remote_asciinema_recording_path = remote_asciinema_recording_path
        # 保存 tmux 会话名。
        self._session_name = session_name
        # 使用 harbor 的全局日志器。
        self._logger = logger
        # 上次捕获的完整终端缓冲区内容，作为增量输出（incremental output）的对比基准。
        self._previous_buffer: str | None = None
        # 是否禁用录像（默认不禁用）。
        self._disable_recording = False
        # 保存所依赖的 harbor 环境对象，所有容器内命令都经它执行。
        self.environment = environment
        # 录像时间戳标记列表：每个元素为 (时间戳秒, 标签)，stop 时合并进录像。
        self._markers: list[tuple[float, str]] = []
        # 创建 tmux 会话时要注入的额外环境变量（KEY=value 形式）。
        self._extra_env: dict[str, str] = extra_env or {}
        # 执行 tmux 命令使用的用户（None 表示默认，或指定用户名/uid）。
        self._user = user

    # 以下为预留的 asciinema 日志路径属性（尚未启用，仅作占位保留）。
    # TODO: Add asciinema logging
    # @property
    # def logging_path(self) -> Path:
    #     return (
    #         Path(DockerComposeManager.CONTAINER_SESSION_LOGS_PATH)
    #         / f"{self._session_name}.log"
    #     )

    # @property
    # def _recording_path(self) -> Path | None:
    #     if self._disable_recording:
    #         return None
    #     return (
    #         Path(DockerComposeManager.CONTAINER_SESSION_LOGS_PATH)
    #         / f"{self._session_name}.cast"
    #     )

    # 尝试安装 tmux 与 asciinema：为整个安装步骤设置总预算，避免挂起的安装
    # 耗尽 agent-setup 预算进而使整次 trial（试验）失败。
    async def _attempt_tmux_installation(self) -> None:
        """Install tmux and asciinema, bounded so a hung install cannot consume
        the agent-setup budget.

        Recording tools are best-effort. On a closed-internet task apt/PyPI may
        be unreachable, so an install command can block producing no output;
        without a bound the only ceiling is harbor's agent-setup timeout, which
        fails the whole trial. Cap the entire step at ``_TOOL_INSTALL_BUDGET_SEC``
        and, on expiry, proceed without recording rather than dying.
        """
        try:
            # 在总预算内等待安装流程完成，超预算即放弃。
            await asyncio.wait_for(
                self._install_recording_tools(),
                timeout=self._TOOL_INSTALL_BUDGET_SEC,
            )
        except asyncio.TimeoutError:
            # 超时则降级：只记录警告并继续运行（录像可能不可用），不让任务失败。
            self._logger.warning(
                "Tool installation exceeded its %ss budget; proceeding without "
                "guaranteed tmux/asciinema (session recording may be "
                "unavailable).",
                self._TOOL_INSTALL_BUDGET_SEC,
            )

    # 一次性安装 tmux 与 asciinema 两个工具，减少命令往返开销。
    async def _install_recording_tools(self) -> None:
        """
        Install both tmux and asciinema in a single operation for efficiency.
        """
        # Check what's already installed
        # 先执行 `tmux -V` 检测 tmux 是否已安装。
        tmux_result = await self.environment.exec(command="tmux -V", user="root")
        # 退出码为 0 说明 tmux 已可用。
        tmux_installed = tmux_result.return_code == 0

        # 只有需要远程 asciinema 录像时才需要检测 asciinema。
        needs_asciinema = self._remote_asciinema_recording_path is not None
        if needs_asciinema:
            # 用版本命令检测 asciinema 是否已安装。
            asciinema_result = await self.environment.exec(
                command="asciinema --version", user="root"
            )
            # 退出码为 0 说明 asciinema 已可用。
            asciinema_installed = asciinema_result.return_code == 0
        else:
            # 不需要录像时视为"已安装"，跳过检测与安装。
            asciinema_installed = True

        # 两者都已就绪时直接返回，无需任何安装。
        if tmux_installed and asciinema_installed:
            self._logger.debug("Both tmux and asciinema are already installed")
            return

        # 汇总缺失、需要安装的工具名列表。
        tools_needed = []
        if not tmux_installed:
            # 缺 tmux 则加入安装清单。
            tools_needed.append("tmux")
        if needs_asciinema and not asciinema_installed:
            # 缺 asciinema 则加入安装清单。
            tools_needed.append("asciinema")

        # 记录即将安装的工具。
        self._logger.debug(f"Installing: {', '.join(tools_needed)}")

        # Detect system and package manager
        # 检测操作系统类型与可用的包管理器。
        system_info = await self._detect_system_info()

        # 若检测到包管理器，则尝试用组合命令一次性装齐所有工具。
        if system_info["package_manager"]:
            # 根据系统信息生成对应的组合安装命令。
            install_command = self._get_combined_install_command(
                system_info, tools_needed
            )
            # 命令非空（识别出了包管理器）时才执行。
            if install_command:
                self._logger.debug(
                    f"Installing tools using {system_info['package_manager']}: {install_command}"
                )
                # 以 root 执行安装，并设单命令超时防止挂起。
                result = await self.environment.exec(
                    command=install_command,
                    user="root",
                    timeout_sec=self._TOOL_INSTALL_TIMEOUT_SEC,
                )

                # 安装命令成功后还需逐一验证工具真正可用。
                if result.return_code == 0:
                    # Verify installations
                    # 验证 tmux：包管理器装完仍不可用时回退到源码编译。
                    if not tmux_installed:
                        verify_tmux = await self.environment.exec(
                            command="tmux -V", user="root"
                        )
                        if verify_tmux.return_code != 0:
                            self._logger.warning(
                                "tmux installation verification failed"
                            )
                            await self._build_tmux_from_source()

                    # 验证 asciinema：仍不可用时回退到 pip 安装。
                    if needs_asciinema and not asciinema_installed:
                        verify_asciinema = await self.environment.exec(
                            command="asciinema --version", user="root"
                        )
                        if verify_asciinema.return_code != 0:
                            self._logger.warning(
                                "asciinema installation verification failed"
                            )
                            await self._install_asciinema_with_pip()

                    return

        # Fallback to individual installations
        # 组合安装不可用或失败时，回退为逐个工具单独安装。
        if not tmux_installed:
            self._logger.debug("Installing tmux from source...")
            # 从源码编译安装 tmux。
            await self._build_tmux_from_source()

        if needs_asciinema and not asciinema_installed:
            self._logger.debug("Installing asciinema via pip...")
            # 用 pip 安装 asciinema。
            await self._install_asciinema_with_pip()

    # 检测容器操作系统与可用的包管理器，为选择安装命令提供依据。
    async def _detect_system_info(self) -> dict[str, str | None]:
        """
        Detect the operating system and available package managers.
        """
        # 初始化系统信息字典：OS 类型、包管理器、更新命令均为未知（None）。
        system_info: dict[str, str | None] = {
            "os": None,
            "package_manager": None,
            "update_command": None,
        }

        # Check for OS release files
        # 读取 /etc/os-release 发行版信息文件，读取失败时输出占位文本 'not found'。
        os_release_result = await self.environment.exec(
            command="cat /etc/os-release 2>/dev/null || echo 'not found'",
            user="root",
        )

        # Check uname for system type
        # 用 uname -s 查询内核/系统类型（如 Darwin、FreeBSD）。
        uname_result = await self.environment.exec(command="uname -s", user="root")

        # Detect package managers by checking if they exist
        # 常见包管理器候选列表，按序探测。
        package_managers = [
            "apt-get",
            "dnf",
            "yum",
            "apk",
            "pacman",
            "brew",
            "pkg",
            "zypper",
        ]

        for pm_name in package_managers:
            # 用 which 静默检测该包管理器是否存在于 PATH。
            check_result = await self.environment.exec(
                command=f"which {pm_name} >/dev/null 2>&1", user="root"
            )
            # 找到第一个可用的包管理器即记录并停止探测。
            if check_result.return_code == 0:
                system_info["package_manager"] = pm_name
                break

        # Try to determine OS from available info
        # 优先依据 /etc/os-release 内容判断 OS 家族。
        if (
            os_release_result.return_code == 0
            and os_release_result.stdout
            and "not found" not in os_release_result.stdout
        ):
            # 输出转小写以便做子串匹配。
            stdout_lower = os_release_result.stdout.lower()
            # 含 ubuntu/debian 判定为 Debian 系发行版。
            if "ubuntu" in stdout_lower or "debian" in stdout_lower:
                system_info["os"] = "debian-based"
            # 含 fedora/rhel 判定为 RHEL 系发行版。
            elif "fedora" in stdout_lower or "rhel" in stdout_lower:
                system_info["os"] = "rhel-based"
            # 含 alpine 判定为 Alpine。
            elif "alpine" in stdout_lower:
                system_info["os"] = "alpine"
            # 含 arch 判定为 Arch Linux。
            elif "arch" in stdout_lower:
                system_info["os"] = "arch"
        # 没有 os-release 时退回用 uname 判断类 Unix 系统。
        elif uname_result.return_code == 0 and uname_result.stdout:
            stdout_lower = uname_result.stdout.lower()
            # darwin 对应 macOS。
            if "darwin" in stdout_lower:
                system_info["os"] = "macos"
            # freebsd 对应 FreeBSD。
            elif "freebsd" in stdout_lower:
                system_info["os"] = "freebsd"

        # 记录检测结果并返回。
        self._logger.debug(f"Detected system: {system_info}")
        return system_info

    # 根据检测到的包管理器生成一次性安装多个工具的组合命令字符串。
    def _get_combined_install_command(
        self, system_info: dict[str, Any], tools: list[str]
    ) -> str:
        """
        Get the appropriate installation command for multiple tools based on system info.
        """
        # 取出已识别的包管理器名称。
        package_manager = system_info.get("package_manager")

        # 未识别到包管理器时返回空串，表示无法组合安装。
        if not package_manager or not isinstance(package_manager, str):
            return ""

        # Build the package list
        # 把工具名拼接为空格分隔的包名列表。
        packages = " ".join(tools)

        # Package manager commands with non-interactive flags
        # 各包管理器的安装命令模板，均带非交互式标志避免卡在确认提示。
        install_commands = {
            "apt-get": f"DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y {packages}",
            "dnf": f"dnf install -y {packages}",
            "yum": f"yum install -y {packages}",
            "apk": f"apk add --no-cache {packages}",
            "pacman": f"pacman -S --noconfirm {packages}",
            "brew": f"brew install {packages}",
            "pkg": f"ASSUME_ALWAYS_YES=yes pkg install -y {packages}",
            "zypper": f"zypper install -y -n {packages}",
        }

        # 查表返回对应命令；未知包管理器返回空串。
        return install_commands.get(package_manager, "")

    # 备选方案：从源码编译安装 tmux（包管理器不可用或安装验证失败时）。
    async def _build_tmux_from_source(self) -> None:
        """
        Build tmux from source as a fallback option.
        """
        try:
            # Install build dependencies based on detected system - with non-interactive flags
            # 按不同发行版准备安装编译依赖（gcc/libevent/ncurses/curl）的候选命令。
            dep_commands = [
                "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libevent-dev libncurses5-dev curl",
                "yum groupinstall -y 'Development Tools' && yum install -y libevent-devel ncurses-devel curl",
                "dnf groupinstall -y 'Development Tools' && dnf install -y libevent-devel ncurses-devel curl",
                "apk add --no-cache build-base libevent-dev ncurses-dev curl",
            ]

            # Try to install build dependencies
            # 依次尝试各候选命令安装编译依赖。
            for cmd in dep_commands:
                result = await self.environment.exec(
                    command=cmd,
                    user="root",
                    timeout_sec=self._TOOL_INSTALL_TIMEOUT_SEC,
                )
                # 某条命令成功即停止尝试其余候选。
                if result.return_code == 0:
                    break

            # Download and build tmux
            # 组装"下载 tmux 3.4 源码并 configure/make/make install"的完整编译命令。
            build_cmd = (
                "cd /tmp && "
                "curl -L https://github.com/tmux/tmux/releases/download/3.4/tmux-3.4.tar.gz -o tmux.tar.gz && "
                "tar -xzf tmux.tar.gz && "
                "cd tmux-3.4 && "
                "./configure --prefix=/usr/local && "
                "make && "
                "make install"
            )

            # 执行源码编译安装（受单命令超时约束）。
            result = await self.environment.exec(
                command=build_cmd,
                user="root",
                timeout_sec=self._TOOL_INSTALL_TIMEOUT_SEC,
            )

            # Verify installation
            # 验证安装：PATH 中的 tmux 或 /usr/local/bin/tmux 任一可执行即可。
            verify_result = await self.environment.exec(
                command="tmux -V || /usr/local/bin/tmux -V", user="root"
            )
            # 验证通过则记录成功日志。
            if verify_result.return_code == 0:
                self._logger.debug("tmux successfully built and installed from source")
            else:
                # 验证失败则记录错误日志（录像功能静默降级）。
                self._logger.error("Failed to install tmux from source")

        except Exception as e:
            # 捕获所有异常并记录，避免源码编译失败中断会话启动流程。
            self._logger.error(f"Failed to build tmux from source: {e}")

    # 备选方案：用 pip 安装 asciinema。
    async def _install_asciinema_with_pip(self) -> None:
        """
        Install asciinema using pip as a fallback.
        """
        try:
            # Try to install python3-pip first - with non-interactive flags
            # 准备各发行版安装 python3-pip 的候选命令（均带非交互标志）。
            pip_install_commands = [
                "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip",
                "yum install -y python3-pip",
                "dnf install -y python3-pip",
                "apk add --no-cache python3 py3-pip",
            ]

            # Try to install pip
            # 依次尝试安装 pip，直到某条命令成功。
            for cmd in pip_install_commands:
                result = await self.environment.exec(
                    command=cmd,
                    user="root",
                    timeout_sec=self._TOOL_INSTALL_TIMEOUT_SEC,
                )
                # 成功即停止尝试其余候选。
                if result.return_code == 0:
                    break

            # Install asciinema using pip
            # 用 pip 安装 asciinema 的候选命令（pip3/pip 两种写法）。
            pip_commands = ["pip3 install asciinema", "pip install asciinema"]

            for cmd in pip_commands:
                # 执行当前 pip 安装命令。
                result = await self.environment.exec(
                    command=cmd,
                    user="root",
                    timeout_sec=self._TOOL_INSTALL_TIMEOUT_SEC,
                )
                # 安装命令成功后还需验证 asciinema 真正可用。
                if result.return_code == 0:
                    # Verify installation
                    # 运行版本命令确认安装生效。
                    verify_result = await self.environment.exec(
                        command="asciinema --version", user="root"
                    )
                    # 验证通过则记录成功并结束安装流程。
                    if verify_result.return_code == 0:
                        self._logger.debug("asciinema successfully installed using pip")
                        return

            # 所有候选命令均失败时记录错误日志。
            self._logger.error("Failed to install asciinema using pip")

        except Exception as e:
            # 捕获异常并记录，避免 pip 安装失败中断会话启动流程。
            self._logger.error(f"Failed to install asciinema with pip: {e}")

    # 组装启动 tmux 会话的完整 shell 命令串（设置终端类型并以分离模式新建会话、开启输出管道记录）。
    @property
    def _tmux_start_session(self) -> str:
        # Build environment variable options for tmux new-session -e KEY=value
        # 为每个额外环境变量生成 `-e KEY=value` 选项（shlex.quote 保证 shell 安全）。
        env_options = "".join(
            f"-e {shlex.quote(f'{key}={value}')} "
            for key, value in self._extra_env.items()
        )

        # 先导出 TERM（xterm-256color 终端类型）与 SHELL（bash），再以分离模式（-d）
        # 创建指定尺寸（-x/-y）与名称（-s）的 tmux 会话（运行登录 bash），
        # 最后用 pipe-pane 把面板输出持续重定向到日志文件。
        return (
            f"export TERM=xterm-256color && "
            f"export SHELL=/bin/bash && "
            f"tmux new-session {env_options}-x {self._pane_width} -y {self._pane_height} -d -s {self._session_name} 'bash --login' \\; "
            f"pipe-pane -t {self._session_name} "
            f"'cat > {self._logging_path}'"
        )

    # 计算字符串的 UTF-8 字节长度。
    @staticmethod
    def _utf8_len(s: str) -> int:
        """Return the UTF-8 byte length of *s*.

        tmux measures command/message sizes in bytes, not Unicode code
        points, so all size checks must use byte length.
        """
        # 编码为 UTF-8 后取字节长度。
        return len(s.encode("utf-8"))

    # 构建 send-keys 命令的公共前缀（目标会话名经 shlex.quote 引用）。
    @property
    def _send_keys_prefix(self) -> str:
        # use `--` to explicitly mark end of options so everything after is
        # treated as keys
        # 返回拼接好的前缀：`--` 显式结束选项解析，其后内容一律按按键名处理。
        return "tmux send-keys -t " + shlex.quote(self._session_name) + " --"

    # 计算单个按键经 shell 引用后仍能放进一条 send-keys 命令的最大字节长度。
    @property
    def _max_escaped_key_length(self) -> int:
        """Largest quoted key that still fits in a single send-keys command."""
        # 最大长度 = 命令总限长 - 前缀字节长 - 分隔空格 1 字节。
        return (
            self._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH
            - self._utf8_len(self._send_keys_prefix)
            - 1
        )

    # 将按键列表拼成一条完整的 send-keys 命令（各按键经 shlex.quote 引用后以空格连接）。
    def _send_keys_command(self, keys: list[str]) -> str:
        return self._send_keys_prefix + " " + " ".join(shlex.quote(key) for key in keys)

    # 判断按键是否过长而无法用单条 send-keys 命令发送（需改走粘贴缓冲通道）。
    def _key_requires_paste(self, key: str) -> bool:
        return self._utf8_len(shlex.quote(key)) > self._max_escaped_key_length

    # 把按键列表分组为若干批次，确保每批拼出的 send-keys 命令不超过 tmux 命令长度限制。
    def _batch_keys_for_send(self, keys: list[str]) -> list[list[str]]:
        """Group *keys* into batches whose send-keys command fits the limit.

        Every key must individually fit in a single command; oversized keys
        (see ``_key_requires_paste``) are delivered via ``_paste_key``.
        """
        # 读取命令总限长与前缀字节长。
        max_len = self._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH
        prefix_len = self._utf8_len(self._send_keys_prefix)

        # batches：最终返回的批次列表；batch：当前累积批次；
        # batch_len：当前批命令的累计字节长（初始为前缀长度）。
        batches: list[list[str]] = []
        batch: list[str] = []
        batch_len = prefix_len

        for key in keys:
            # 计算当前按键经 shell 引用后的字节长。
            escaped_len = self._utf8_len(shlex.quote(key))
            # 单个按键就超限的必须走粘贴缓冲投递，这里直接抛错提示调用方。
            if escaped_len > self._max_escaped_key_length:
                raise ValueError(
                    "key exceeds the tmux send-keys command-length limit; "
                    "it must be sent via a paste buffer instead"
                )
            # 新增开销 = 分隔空格 1 字节 + 引用后的按键字节长。
            addition = 1 + escaped_len  # space + quoted key
            # 放入当前批会超限时，先封存当前批次并另起新批。
            if batch and batch_len + addition > max_len:
                batches.append(batch)
                batch = []
                batch_len = prefix_len
            # 把按键加入当前批并累计长度。
            batch.append(key)
            batch_len += addition

        # 收尾：把最后一批（若有）加入结果。
        if batch:
            batches.append(batch)
        return batches

    # 为按键列表生成一条或多条 send-keys 命令（超出限长时自动分批）。
    def _tmux_send_keys(self, keys: list[str]) -> list[str]:
        """Build one or more ``tmux send-keys`` commands for *keys*.

        If the shell-escaped command would exceed the tmux command-length
        limit, the keys are spread across multiple commands so that each
        individual command stays within the limit.
        """
        # 对每个批次分别生成完整命令字符串。
        return [
            self._send_keys_command(batch) for batch in self._batch_keys_for_send(keys)
        ]

    # 组装 tmux capture-pane 命令：capture_entire 为真时加 -S - 从历史缓冲区
    # 最顶端开始捕获全部内容，否则只捕获当前可见屏幕。
    def _tmux_capture_pane(self, capture_entire: bool = False) -> str:
        if capture_entire:
            # -S - 表示捕获范围从滚动历史的最开头开始。
            extra_args = ["-S", "-"]
        else:
            # 不带 -S 时仅捕获当前可见区域。
            extra_args = []

        # 用空格拼接各部分：-p 以纯文本打印，-t 指定目标会话。
        return " ".join(
            [
                "tmux",
                "capture-pane",
                "-p",
                *extra_args,
                "-t",
                self._session_name,
            ]
        )

    # 启动 tmux 会话：先确保工具已安装，再创建会话并（可选）开启 asciinema 录像。
    async def start(self) -> None:
        # 尝试安装 tmux/asciinema（有总预算限制，失败可降级继续）。
        await self._attempt_tmux_installation()
        # 在容器内执行启动命令，创建 tmux 会话。
        start_session_result = await self.environment.exec(
            command=self._tmux_start_session, user=self._user
        )
        # 创建失败时抛出带 stderr 信息的运行时错误。
        if start_session_result.return_code != 0:
            raise RuntimeError(
                f"Failed to start tmux session. Error: {start_session_result.stderr}"
            )

        # 把 tmux 滚动历史上限调到一千万行，避免长会话丢失历史内容。
        history_limit = 10_000_000
        # 组装全局设置 history-limit 参数的命令。
        command = f"tmux set-option -g history-limit {history_limit}"
        set_history_result = await self.environment.exec(
            command=command, user=self._user
        )
        # 设置失败仅记录调试日志（非致命，使用默认值继续）。
        if set_history_result.return_code != 0:
            self._logger.debug(
                "Failed to increase tmux history-limit: %s",
                (set_history_result.stderr or "").strip(),
            )

        # 需要远程 asciinema 录像时，在会话里启动录像进程。
        if self._remote_asciinema_recording_path:
            self._logger.debug("Starting recording.")
            # 发送 asciinema rec 命令并回车（--stdin 使其捕获键盘输入），
            # 最小等待 1 秒确保录像进程启动完成。
            await self.send_keys(
                keys=[
                    f"asciinema rec --stdin {self._remote_asciinema_recording_path}",
                    "Enter",
                ],
                min_timeout_sec=1.0,
            )
            # 清屏，避免录像开头残留启动时的提示信息。
            await self.send_keys(
                keys=[
                    "clear",
                    "Enter",
                ],
            )

        # 需要录像时，把获取 asciinema 时间戳的辅助脚本从宿主机上传到容器内。
        if self._remote_asciinema_recording_path:
            await self.environment.upload_file(
                source_path=self._GET_ASCIINEMA_TIMESTAMP_SCRIPT_HOST_PATH,
                target_path=str(self.GET_ASCIINEMA_TIMESTAMP_SCRIPT_CONTAINER_PATH),
            )

    # 停止会话：结束 asciinema 录像、把录像下载到本地并合并时间戳标记。
    async def stop(self) -> None:
        if self._remote_asciinema_recording_path:
            self._logger.debug("Stopping recording.")
            # 发送 Ctrl-D 结束 asciinema rec 进程（正常退出并把录像落盘）。
            await self.send_keys(
                keys=["C-d"],
                min_timeout_sec=0.1,
            )

            # Wait a moment for the recording to finish writing
            # 局部导入 asyncio 供下方休眠使用。
            import asyncio

            # 再等 0.5 秒，确保录像文件写完。
            await asyncio.sleep(0.5)

            # 需要本地录像副本时执行下载。
            if self._local_asciinema_recording_path:
                # 递归创建本地目录（已存在则忽略）。
                self._local_asciinema_recording_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                # Ensure recording exists locally before merging markers
                # 从容器下载录像文件到宿主机，供后续合并标记使用。
                await self.environment.download_file(
                    source_path=str(self._remote_asciinema_recording_path),
                    target_path=self._local_asciinema_recording_path,
                )

            # Merge markers into the recording
            # 存在时间戳标记且已有本地录像时执行合并。
            if self._markers and self._local_asciinema_recording_path:
                self._logger.debug(
                    f"Merging {len(self._markers)} markers into recording"
                )
                # 构造 AsciinemaHandler 处理该 .cast 录像与标记列表。
                handler = AsciinemaHandler(
                    self._markers, self._local_asciinema_recording_path
                )
                # 执行标记合并。
                handler.merge_markers()
                self._logger.debug(
                    f"Successfully merged markers into {self._local_asciinema_recording_path}"
                )
            else:
                # 没有标记或没有本地录像时记录调试日志。
                self._logger.debug("No markers to merge")
        ...

    # 判断按键是否为"回车"类按键（会执行当前输入行）。
    def _is_enter_key(self, key: str) -> bool:
        return key in self._ENTER_KEYS

    # 用正则判断按键文本是否以回车/换行字符结尾。
    def _ends_with_newline(self, key: str) -> bool:
        # 在按键文本中搜索"以换行结尾"的模式。
        result = re.search(self._ENDS_WITH_NEWLINE_PATTERN, key)
        # 找到匹配即视为以换行结尾。
        return result is not None

    # 检查 tmux 会话是否仍然存活（has-session 退出码为 0 即存活）。
    async def is_session_alive(self) -> bool:
        """Check if the tmux session is still alive."""
        # 执行 tmux has-session 检查会话是否存在。
        result = await self.environment.exec(
            command="tmux has-session -t {}".format(self._session_name),
            user=self._user,
        )
        # 退出码 0 表示会话存在。
        return result.return_code == 0

    # 判断该按键是否会触发命令执行（是回车键，或以换行字符结尾）。
    def _is_executing_command(self, key: str) -> bool:
        return self._is_enter_key(key) or self._ends_with_newline(key)

    # 去掉按键序列末尾所有会触发执行的按键，用于阻塞模式：
    # 防止用户命令在"完成信号"就位之前被提前执行。
    def _prevent_execution(self, keys: list[str]) -> list[str]:
        # 复制列表，避免修改调用方传入的原始按键序列。
        keys = keys.copy()
        # 持续剥离末尾的执行类按键，直到末尾按键不再触发执行。
        while keys and self._is_executing_command(keys[-1]):
            # 末尾是回车键时直接弹出。
            if self._is_enter_key(keys[-1]):
                keys.pop()
            else:
                # 末尾是带换行的文本按键：先剥掉尾部换行字符。
                stripped_key = keys[-1].rstrip(self._NEWLINE_CHARS)

                # 剥离后仍有内容则原位保留剩余文本。
                if stripped_key:
                    keys[-1] = stripped_key
                else:
                    # 剥离后为空（整个按键就是换行）则弹出。
                    keys.pop()

        # 返回不会立即触发执行的按键序列。
        return keys

    # 准备待发送按键：阻塞模式下剥离末尾执行键后追加 tmux 完成信号，
    # 使"执行用户命令"与"等待命令完成"通过 tmux wait 事件解耦。
    def _prepare_keys(
        self,
        keys: str | list[str],
        block: bool,
    ) -> tuple[list[str], bool]:
        """
        Prepare keys for sending to the terminal.

        Args:
            keys (str | list[str]): The keys to send to the terminal.
            block (bool): Whether to wait for the command to complete.

        Returns:
            tuple[list[str], bool]: The keys to send to the terminal and whether the
                the keys are blocking.
        """
        # 字符串入参统一转为单元素列表。
        if isinstance(keys, str):
            keys = [keys]

        # 非阻塞模式、空按键、或末尾按键不触发执行时，无需处理，按非阻塞返回。
        if not block or not keys or not self._is_executing_command(keys[-1]):
            return keys, False

        # 剥离末尾执行键，防止命令在完成信号就绪前被执行。
        keys = self._prevent_execution(keys)
        # 追加 `; tmux wait -S done` 与回车：命令执行完后立即置位完成事件。
        keys.extend([self._TMUX_COMPLETION_COMMAND, "Enter"])

        # 返回处理后的按键序列，并标记这是阻塞式发送。
        return keys, True

    # 构造按键发送失败的运行时异常，附带会话 ID、动作、命令（截断至 100 字符）、退出码及输出。
    def _send_keys_error(
        self, action: str, command: str, result: ExecResult
    ) -> RuntimeError:
        return RuntimeError(
            f"{self.environment.session_id}: failed to send {action} keys: "
            f"command={command!r:.100}, return_code={result.return_code}, "
            f"stderr={result.stderr!r}, stdout={result.stdout!r}"
        )

    # 判断执行结果是否为 tmux 的"命令过长"错误（旧版 tmux 限制更小时的运行时回退依据）。
    def _is_command_too_long_error(self, result: ExecResult) -> bool:
        return result.return_code != 0 and self._TMUX_COMMAND_TOO_LONG_MARKER in (
            result.stderr or ""
        )

    # 通过 tmux 粘贴缓冲（paste buffer）投递按键，绕过 send-keys 的命令长度限制。
    async def _paste_key(self, key: str, action: str) -> None:
        """Deliver *key* to the pane via a tmux paste buffer.

        ``tmux send-keys`` rejects commands above its internal message size
        limit with "command too long", which previously lost long literal
        payloads such as heredoc answer writes. Instead, the payload is
        staged into a file inside the environment with base64 chunks
        (``environment.exec`` does not go through tmux, so it is not subject
        to that limit) and pasted with ``load-buffer``/``paste-buffer``,
        which can handle arbitrarily large content. ``paste-buffer``'s
        default newline-to-carriage-return conversion makes multi-line
        payloads behave as if typed line by line.
        """
        # 空按键无需投递。
        if not key:
            return

        # 生成唯一令牌，避免并发投递时临时文件/缓冲区名冲突。
        token = uuid.uuid4().hex
        # 容器内临时文件路径（经 shell 引用防注入）。
        quoted_path = shlex.quote(f"/tmp/.harbor-tmux-paste-{token}")
        # tmux 缓冲区名称（与令牌绑定，确保唯一）。
        buffer_name = f"harbor-paste-{token}"
        # 将按键内容按 UTF-8 编码后转为 base64（ASCII 安全，便于经 shell 分块传输）。
        payload = base64.b64encode(key.encode("utf-8")).decode("ascii")

        # 内部辅助：执行一条 shell 命令，失败时抛出发送错误异常。
        async def _run(command: str) -> None:
            result = await self.environment.exec(command=command, user=self._user)
            if result.return_code != 0:
                raise self._send_keys_error(action, command, result)

        try:
            # 分块写入：每块经 base64 -d 解码后写入临时文件（首块覆盖写 >，其余追加写 >>）。
            for offset in range(0, len(payload), self._PASTE_BASE64_CHUNK_LEN):
                # 取出当前 base64 分块（切片步长为块长）。
                chunk = payload[offset : offset + self._PASTE_BASE64_CHUNK_LEN]
                # 首块用覆盖重定向，后续块用追加重定向。
                redirect = ">>" if offset else ">"
                # 用 printf 输出该分块，经 base64 -d 解码后写入/追加到临时文件。
                await _run(f"printf %s {chunk} | base64 -d {redirect} {quoted_path}")

            # 把临时文件加载为 tmux 缓冲区并粘贴到目标会话
            #（-d 表示粘贴后删除缓冲区；多行内容会像逐行键入一样执行）。
            await _run(
                f"tmux load-buffer -b {buffer_name} {quoted_path} && "
                f"tmux paste-buffer -d -b {buffer_name} "
                f"-t {shlex.quote(self._session_name)}"
            )
        finally:
            # 无论成败都清理容器内的临时文件。
            await self.environment.exec(command=f"rm -f {quoted_path}", user=self._user)

    # 用单条 send-keys 命令发送单个按键；若 tmux 报"命令过长"则回退到粘贴缓冲投递。
    async def _send_single_key(self, key: str, action: str) -> None:
        # 生成只含该按键的 send-keys 命令并执行。
        command = self._send_keys_command([key])
        result = await self.environment.exec(command=command, user=self._user)
        # 成功则结束。
        if result.return_code == 0:
            return
        # tmux 报命令过长时改用粘贴缓冲投递。
        if self._is_command_too_long_error(result):
            # A key too long for a send-keys command cannot be a special key
            # name (those are all short), so pasting it literally is safe.
            await self._paste_key(key, action)
            return
        # 其他失败一律抛出发送错误异常。
        raise self._send_keys_error(action, command, result)

    # 按批次发送按键；某批被 tmux 拒绝（命令过长）时逐键重发并启用粘贴回退。
    async def _send_key_batches(self, keys: list[str], action: str) -> None:
        for batch in self._batch_keys_for_send(keys):
            # 生成该批次的 send-keys 命令并执行。
            command = self._send_keys_command(batch)
            result = await self.environment.exec(command=command, user=self._user)
            # 成功则继续处理下一批。
            if result.return_code == 0:
                continue
            # 该 tmux 构建的实际限长小于假设值时，逐键重发，仍被拒者用粘贴缓冲投递。
            if self._is_command_too_long_error(result):
                # This tmux build enforces a smaller command-size limit than
                # the assumed ~16 KB; resend the keys one at a time, pasting
                # any that this tmux still rejects.
                self._logger.debug(
                    "tmux rejected a send-keys command as too long; resending "
                    "keys individually with a paste-buffer fallback"
                )
                for key in batch:
                    # 逐键发送（内部自带粘贴缓冲回退）。
                    await self._send_single_key(key, action)
                continue
            # 其他失败抛出发送错误异常。
            raise self._send_keys_error(action, command, result)

    # 按顺序发送按键：能放进 send-keys 命令的按键批量发送，过长按键单独走粘贴缓冲。
    async def _send_keys_to_session(self, keys: list[str], action: str) -> None:
        """Send *keys* in order, pasting oversized literal keys.

        Keys that fit within the tmux command-length limit are sent with
        ``send-keys`` (batched); keys that cannot fit in a single command
        are delivered via ``_paste_key``.
        """
        # 当前累积的普通按键批次。
        batch: list[str] = []

        # 内部辅助：冲刷当前批次（发送并清空），保证按键顺序不被打乱。
        async def _flush() -> None:
            if batch:
                await self._send_key_batches(batch, action)
                batch.clear()

        for key in keys:
            # 过长按键：先冲刷已累积批次，再用粘贴缓冲单独投递该按键。
            if self._key_requires_paste(key):
                await _flush()
                await self._paste_key(key, action)
            else:
                # 普通按键：累积进当前批次，等待统一发送。
                batch.append(key)

        # 收尾：发送最后累积的批次。
        await _flush()

    # 阻塞式发送：先发按键（含完成信号），再用 tmux wait 等待命令执行完成，超时抛错。
    async def _send_blocking_keys(
        self,
        keys: list[str],
        max_timeout_sec: float,
    ):
        # 记录起始时间用于统计耗时。
        start_time_sec = time.time()

        # 发送按键序列（末尾的完成信号会随命令一起执行）。
        await self._send_keys_to_session(keys, action="blocking")

        # 在超时上限内执行 `timeout Ns tmux wait done`：
        # 该命令阻塞直到 `tmux wait -S done` 置位事件（即命令执行完毕）。
        result = await self.environment.exec(
            f"timeout {max_timeout_sec}s tmux wait done", user=self._user
        )
        # 等待超时（非零退出码）时抛出 TimeoutError。
        if result.return_code != 0:
            raise TimeoutError(f"Command timed out after {max_timeout_sec} seconds")

        # 统计阻塞命令从发送到完成的总耗时并记录。
        elapsed_time_sec = time.time() - start_time_sec
        self._logger.debug(f"Blocking command completed in {elapsed_time_sec:.2f}s.")

    # 非阻塞式发送：发送按键后保证至少等待 min_timeout_sec（不足则补足休眠）。
    async def _send_non_blocking_keys(
        self,
        keys: list[str],
        min_timeout_sec: float,
    ):
        # 记录起始时间。
        start_time_sec = time.time()

        # 发送按键序列（不等待命令完成）。
        await self._send_keys_to_session(keys, action="non-blocking")

        # 计算发送已耗时间。
        elapsed_time_sec = time.time() - start_time_sec

        # 实际耗时不足最小等待时间时补足差值休眠，保证终端有时间渲染输出。
        if elapsed_time_sec < min_timeout_sec:
            await asyncio.sleep(min_timeout_sec - elapsed_time_sec)

    # 对外主入口：向 tmux 会话发送按键（可阻塞等待命令完成）。
    async def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ):
        """
        Execute a command in the tmux session.

        Args:
            keys (str): The keys to send to the tmux session.
            block (bool): Whether to wait for the command to complete.
            min_timeout_sec (float): Minimum time in seconds to wait after executing.
                Defaults to 0.
            max_timeout_sec (float): Maximum time in seconds to wait for blocking
                commands. Defaults to 3 minutes.
        """
        # 阻塞模式下 min_timeout_sec 无意义，记录调试日志提示其会被忽略。
        if block and min_timeout_sec > 0.0:
            self._logger.debug("min_timeout_sec will be ignored because block is True.")

        # 预处理按键：决定是否升级为阻塞式发送（追加 tmux 完成信号）。
        prepared_keys, is_blocking = self._prepare_keys(
            keys=keys,
            block=block,
        )

        # 记录本次发送的按键与超时参数。
        self._logger.debug(
            f"Sending keys: {prepared_keys}"
            f" min_timeout_sec: {min_timeout_sec}"
            f" max_timeout_sec: {max_timeout_sec}"
        )

        # 阻塞式：等待命令执行完成（受 max_timeout_sec 约束）。
        if is_blocking:
            await self._send_blocking_keys(
                keys=prepared_keys,
                max_timeout_sec=max_timeout_sec,
            )
        else:
            # 非阻塞式：发送后至少等待 min_timeout_sec。
            await self._send_non_blocking_keys(
                keys=prepared_keys,
                min_timeout_sec=min_timeout_sec,
            )

    # 捕获 tmux 面板内容（capture_entire 为真时包含完整滚动历史）。
    async def capture_pane(self, capture_entire: bool = False) -> str:
        # 执行 capture-pane 命令并取其标准输出。
        result = await self.environment.exec(
            self._tmux_capture_pane(capture_entire=capture_entire), user=self._user
        )
        # 无输出时返回空字符串。
        return result.stdout or ""

    # 获取当前可见屏幕内容（不含滚动历史）。
    async def _get_visible_screen(self) -> str:
        return await self.capture_pane(capture_entire=False)

    # 在当前完整缓冲区中定位上次缓冲之后的新增内容；无法定位时返回 None。
    async def _find_new_content(self, current_buffer: str) -> str | None:
        # 取上次缓冲（去除首尾空白）；首次调用时为空串。
        pb = "" if self._previous_buffer is None else self._previous_buffer.strip()
        # 上次缓冲内容完整出现在当前缓冲中，才能据此切分出新内容。
        if pb in current_buffer:
            # 记录上次缓冲内容在当前缓冲中的起始下标。
            idx = current_buffer.index(pb)
            # Find the end of the previous buffer content
            # 若上次缓冲为多行，则把切分点改为其最后一个换行符所在下标。
            if "\n" in pb:
                idx = pb.rfind("\n")
            # 返回切分点之后的内容（即上次内容之后的部分）。
            return current_buffer[idx:]
        # 无法可靠定位（如屏幕回滚覆盖了旧内容）时返回 None。
        return None

    # 获取自上次调用以来的新增终端输出；无法可靠判断时回退为当前可见屏幕。
    async def get_incremental_output(self) -> str:
        """
        Get either new terminal output since last call, or current screen if
        unable to determine.

        This method tracks the previous buffer state and attempts to find new content
        by comparing against the current full buffer. This provides better handling for
        commands with large output that would overflow the visible screen.

        Returns:
            str: Formatted output with either "New Terminal Output:" or
                 "Current Terminal Screen:"
        """
        # 捕获完整缓冲区（含滚动历史）作为当前状态。
        current_buffer = await self.capture_pane(capture_entire=True)

        # First capture - no previous state
        # 首次调用：无历史基准可比，保存当前缓冲并直接返回当前可见屏幕。
        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            visible_screen = await self._get_visible_screen()
            return f"Current Terminal Screen:\n{visible_screen}"

        # Try to find new content
        # 尝试在上次缓冲基础上定位新增内容。
        new_content = await self._find_new_content(current_buffer)

        # Update state
        # 无论是否找到新内容，都把当前缓冲保存为下次比较的基准。
        self._previous_buffer = current_buffer

        if new_content is not None:
            # 找到了新增内容且非空白，以"新终端输出"形式返回。
            if new_content.strip():
                return f"New Terminal Output:\n{new_content}"
            else:
                # No new content, show current screen
                # 无实际新内容时回退显示当前可见屏幕。
                return f"Current Terminal Screen:\n{await self._get_visible_screen()}"
        else:
            # Couldn't reliably determine new content, fall back to current screen
            # 无法可靠判定新增内容时同样回退显示当前可见屏幕。
            return f"Current Terminal Screen:\n{await self._get_visible_screen()}"
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

# Google LLC 的 Apache-2.0 许可证头（仓库标准文件头，保留原文）。
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Domain paragraphs for the three search roles on Terminal-Bench 2.1."""
# 【模块说明（中文）】Terminal-Bench 2.1 域的四个搜索角色提示词段落（briefs）：
# ANALYST/DIGESTER/PROPOSER/CRITIC 分别注入 analyst（归因）、digester（消化轨迹）、
# proposer（提编辑方案）、critic（反过拟合审查）的上下文，让它们"说基准的语言"。
# 注意：字符串内容即提示词原文，逐字保留，不做任何改动；{policy} 由 adapter 替换为冻结策略名。

# ANALYST 段：告诉 analyst 它在看什么——终端 agent（冻结策略 LLM + Terminus-2 脚手架）
# 在 Docker 里通过 tmux 按键解题；只有隐藏测试通过才算解决；每任务一条渲染轨迹，
# 末尾 VERIFIER 段是逐测试通过/失败与断言信息；差一点就过的任务杠杆最高；
# 若失败摘要显示被脚手架机制（输出截断、固定等待、上下文摘要、JSON 解析、提前完成）卡住
# 而非判断力问题，应使用 capability_gap（能力缺口）视角。
ANALYST = """The agent is a terminal-use agent: a frozen policy LLM ({policy})
driven by a Terminus-2 scaffold solves command-line / coding tasks
(builds, debugging, data processing, security, system administration) inside
a Docker container by emitting batches of keystrokes into a tmux session and
reading the terminal output back. A task counts as solved only if its hidden
unit tests pass after the agent stops. There is one rendered trace per task
(a failing trial where one exists); the VERIFIER section at the end of each
trace shows the per-test pass/fail with assertion messages. Near-miss tasks
that failed a single verifier test are high-leverage. Where a failure digest
hints the agent was blocked by scaffold mechanics (output truncation, fixed
wait durations on long commands, context summarization, JSON parse errors,
completion signalled before running the deliverable) rather than by
judgment, use the capability_gap lens."""

# DIGESTER 段：告诉 digester 轨迹的文件结构——每个 <task_id>.txt 含任务提示、每一步
# （AGENT 分析/计划、KEYSTROKES 按键、TERMINAL 输出）、运行元数据（奖励/episode/token/
# 摘要次数/异常），末尾 VERIFIER 段是逐测试结果与测试日志尾部；先读 VERIFIER（到底什么
# 失败了），再 grep 锚点（步号、命令、错误串）。
DIGESTER = """The trajectory comes from a terminal-use agent: a frozen policy LLM
solving a command-line / coding task inside a Docker container by emitting
batches of keystrokes into a tmux session and reading the terminal output
back. Each <task_id>.txt holds the task prompt, every step (AGENT
analysis/plan, KEYSTROKES sent, TERMINAL output), run metadata (reward,
episodes, tokens, summarizations, exceptions) and at the end the VERIFIER
ground truth: per-test pass/fail with assertion messages and the test log
tail. Read the VERIFIER section first (what actually failed), then grep for
anchors (step numbers, commands, error strings)."""

# PROPOSER 段：给 proposer 的基准与规则背景——TB2.1 共 89 个容器化终端任务；Terminus-2
# fork 的交互方式（JSON 里 analysis/plan/commands，按键串进 tmux + 等待时长，需两次
# task_complete 才计分）；冻结策略是 {policy}，别假设它有你自己的习惯；脚手架跑在宿主机，
# 只能经 tmux 按键接触任务环境。红线（测量前强制执行）：运行时读/探测 /tests、测试文件、
# 奖励路径、pytest 输出；玩完成信号；在代码/提示/状态里写基准任务名或任务专属文件名/值/触发器。
# 真正的成绩在搜索看不见的 SWE-bench Verified 上判定，记忆任务知识毫无价值；
# 每个编辑的试金石：能否帮一个称职的人类操作员处理大量陌生终端任务？
PROPOSER = """The benchmark is Terminal-Bench 2.1: 89 containerized terminal tasks
(builds, debugging, data processing, security, sysadmin). The scaffold is a
fork of Terminus-2: the policy sees the terminal state and replies with JSON
containing analysis/plan/commands; each command is a raw keystroke string
sent into a tmux session with a wait duration; the loop feeds the resulting
terminal output back; the run only scores if the agent signals task_complete
(twice, by design) before the wall clock runs out and the task's hidden unit
tests then pass. The frozen policy is {policy}; do not assume it shares your habits
or judgment, read the trajectories for how it actually behaves. The scaffold runs on the HOST, not in the container: it may compute and
transform text freely, but every interaction with the task environment goes
through tmux keystrokes.

Out of bounds, enforced before measurement: reading, detecting or referencing
/tests, test files, reward paths or pytest output at runtime; gaming the
completion signal; benchmark task names or task-specific file names, values
or triggers anywhere in code, prompts or state. The run's real outcome is
judged on SWE-bench Verified, a repository-level bug-fixing benchmark the
search never sees, so memorized task knowledge is worthless. Litmus test for
every edit: would this help a competent human operator working on MANY
unfamiliar terminal tasks?"""

# CRITIC 段：给 critic 的域判据——通用终端手艺可以（如"长构建用轮询而非猜固定等待"）；
# 任务知识不行（如"解 ELF 时段偏移是 32"必须 REJECT）。既有的安全机制（上下文摘要、
# 终端输出截断、解析错误恢复、完成双重确认）在没有可用替代品时禁用即拒绝。
CRITIC = """The harness drives a frozen policy LLM through terminal/coding tasks
(Terminal-Bench 2.1) by emitting keystrokes into a tmux session inside a
Docker container; a task is solved when its hidden unit tests pass. General
terminal craft is fine ("poll long-running builds instead of guessing a fixed
wait" = OK); task knowledge is not ("when extracting an ELF, segments are at
offset 32" = REJECT). Existing safety mechanisms that must not be disabled
without a working replacement: context summarization, terminal-output
truncation, parse-error recovery and the completion double-confirmation."""

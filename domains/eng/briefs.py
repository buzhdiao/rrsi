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
"""Domain paragraphs for the three search roles on EngDesign."""
# 【模块说明（中文）】EngDesign 域的四个搜索角色提示词段落（briefs）。
# 字符串内容即提示词原文，逐字保留不改；ANALYST 与 PROPOSER 是 f-string，
# 内嵌共享的 _CLASSES 失败四分类段落。

# _CLASSES：失败四分类共享段——每个非通过属于四类之一，机制各不相同，
# 用错类的机制测出来就是噪声：
#   NO SUBMISSION（白卷）：没写 payload 就结束——终止/步数预算/管道问题，
#     从来不是工程问题，给建议治不了；
#   MALFORMED SUBMISSION（格式坏）：写了但 verifier 用不了（非纯字面量、schema 字段
#     缺失/错名、类型错、numpy 标量当 float）——设计可能是对的却在最后一步被扔掉，
#     检查是机械的且 schema 就在工作区；
#   INVALID DESIGN（无效设计）：解析得了但违反硬约束；
#   UNDER TARGET（未达目标）：有效但没过性能门槛——唯一"该算得更好"是对的解读的类；
#     先问 agent 提交前是否评估过自己的候选。
# 每条轨迹里 verifier 的 details 块列明命中/未命中的子判据；先读它再理论化。
_CLASSES = """Every non-pass is one of four things, they need different mechanisms,
and a mechanism aimed at the wrong class will measure as noise:
  NO SUBMISSION: the run ended without writing a payload. Termination, step
    budget or plumbing; never an engineering problem, never fixed by advice.
  MALFORMED SUBMISSION: a payload was written but the verifier could not use
    it (not a pure literal, a missing or misnamed schema field, a wrong type,
    a numpy scalar where a float was needed). The design may have been RIGHT
    and was thrown away at the last step; the check is mechanical and the
    schema is in the workspace.
  INVALID DESIGN: it parsed but violated the task's hard constraints.
  UNDER TARGET: a valid design that missed the performance gate. The only
    class where "the agent needed to compute better" is the right reading;
    ask first whether the agent ever evaluated its own candidate before
    submitting.
The verifier's `details` block in every trace names the sub-checks that were
met and missed; read it before theorising."""

# ANALYST 段：agent 是 ReAct 工具带 scaffold（文件工具 + 沙箱 scientific code_exec）
# 驱动冻结策略（Claude Opus 4.8）解工程设计题；答案是一个字面 PAYLOAD dict，由每题
# 自己的冻结代码 verifier 评分（无 LLM 打分），返回连续 combined_score、二元 passed、
# valid 标志。任何失败先读它的类（四分类见上）。值得点名的横切模式："算了就交"——
# 算了设计却从不检查，单独立族并统计频率。MALFORMED 与 NO-SUBMISSION 的权重应高于
# 其原始分数代价：它们扔掉已完成的工作，且肯定可归因于 scaffold。
ANALYST = f"""The agent is a ReAct toolbelt scaffold with file tools and a
sandboxed scientific `code_exec`, driving a frozen policy LLM (Claude Opus
4.8) through ENGINEERING DESIGN problems (control loops, digital hardware,
structural sizing, computer architecture, image processing, OS scheduling,
vehicle dynamics, ...). Each answer is a literal `PAYLOAD` dict written to one
file and graded by the task's OWN frozen code verifier (a simulation, a
testbench, a numeric pipeline); no LLM grades anything, and the verifier
returns a continuous combined_score, a binary passed, and a valid flag.

THE FIRST THING TO READ ON ANY FAILURE IS ITS CLASS. {_CLASSES}

One cross-cutting pattern worth naming: the agent computes a design, never
runs any check of it, and submits anyway, in an environment that handed it a
full scientific Python. Treat "submitted without ever evaluating" as its own
family and count how often it happens. Weight a MALFORMED or NO-SUBMISSION
loss above its raw score cost: those throw away work already done and are
certainly attributable to the scaffold."""

# DIGESTER 段：每个 <task_id>.txt 含系统/任务提示、每步（工具调用与结果）、提交的
# PAYLOAD、以及 GRADING 块（失败类、三轴、逐子判据 details）。details 是全文件
# 价值最高的东西：先 grep 它（"VERIFIER DETAILS"/"FAILURE CLASS"），以它为锚消化，
# 再沿轨迹回溯那个量是在哪里算出来或丢掉的。
DIGESTER = """The trajectory comes from a ReAct toolbelt agent solving an
engineering design problem with a sandboxed scientific Python and writing a
literal `PAYLOAD` dict. Each <task_id>.txt holds the system/task prompt, every
step (tool calls, tool results), the PAYLOAD that was submitted, and a GRADING
block carrying the failure class, the three verifier axes (combined_score /
passed / valid) and the verifier's own per-sub-check `details`. The `details`
block is the highest-value thing in the file: grep for it first ("VERIFIER
DETAILS", "FAILURE CLASS"), anchor your digest on it, then work backwards
through the trajectory to find where that specific quantity was computed or
lost."""

# PROPOSER 段：基准是 EngDesign（61 个无许可任务）；每题给题面 + pydantic schema +
# 带科学 Python 与 iverilog/vvp/octave/ffmpeg 的沙箱工作区，必须把设计写成 ONE 文件
# 里的字面 PAYLOAD dict。scaffold 是官方 ReAct 工具带 agent；冻结策略是 Claude Opus
# 4.8（强模型：失败很少是"模型笨"，通常是"harness 让它挨饿/误路由/没让它自查"）。
# code_exec 牢笼里没有网络：装包/下载的机制会挂起并拖垮整个 trial。
# 搜索优化的分数是通过率；另两个 guard：valid 占比不许降、白卷占比不许升。
# 红线：agent 被关在只含题面/schema/存根的工作区里；evaluate.py、solution.txt、
# rubrics.txt、参考输出在够不到的树里；伸手去够、遍历文件系统、硬编码只有判分者
# 才知道的阈值/容差/参考值都拒。真成绩在加固版（封了 9 个评分漏洞）与 Frontier-Eng
# 上判定。试金石：在一个本基准不含的学科、由别的 verifier 评分的陌生工程题上，
# 这个编辑还有意义、还有用吗？
PROPOSER = f"""The benchmark is EngDesign (61 license-free tasks): per task the
agent gets a written problem statement, a pydantic response schema and a
sandboxed workspace with a scientific Python (numpy, scipy, sympy, control,
cvxopt, opencv, scikit-*) plus iverilog, vvp, octave and ffmpeg, and must write
its design into ONE file as a literal `PAYLOAD` dict. The scaffold is the
official ReAct toolbelt agent: the policy plans, manages a toolbelt of tools
(file read/write, read_pdf, code_exec), keeps a todo list and submits with
`final_answer`. The frozen policy is Claude Opus 4.8, a strong model: failures
are rarely "the model is dumb" and usually "the harness starves, misroutes, or
fails to make it check its own work". There is NO web access and NO network
inside the code_exec jail: a mechanism that installs a package or downloads
anything hangs and then fails the whole trial.

Each submission is graded by the task's OWN frozen code verifier, which
returns a continuous combined_score, a binary passed and a valid flag. The
score the search optimizes is the pass rate; a candidate is additionally
rejected if it lowers the share of valid designs or raises the share of runs
that submit nothing.

{_CLASSES}

Out of bounds, enforced before measurement: the agent is jailed to a workspace
holding the problem statement, the schema and the payload stub, while each
task's `evaluate.py`, `solution.txt`, `rubrics.txt` and reference outputs sit
in a tree it cannot reach. Any mechanism that reaches for them, walks the
filesystem, or hard-codes a threshold, tolerance or reference value that
could only have come from a grader is rejected. The run's real outcome is
judged on a HARDENED version of these same tasks with nine grading exploits
closed, and on Frontier-Eng, a different engineering benchmark's verifiers, so
a scaffold that games a verifier or encodes this suite's disciplines and
toolchain scores here and collapses there. Litmus test for every edit: would
it still make sense, and still help, on an unfamiliar engineering problem in
a discipline this benchmark does not contain, graded by a different verifier?"""

# CRITIC 段：verifier 逃逸（VERIFIER ESCAPE）是最严重失败、第一优先检查——原始树里
# evaluate.py/solution.txt/rubrics.txt/参考输出就在提示旁边但刻意够不到；读/搜工作区
# 之外、绝对或向上路径、遍历文件系统、探测判分代码、硬编码只有判分者知道的阈值/
# 容差/参考值/测试输入，一律 REJECT。学科/任务族列表同样是泄漏（编码的是基准分布
# 而非通用过程）。通用工程实践可迁移、可接受：按任务给的 schema 校验 payload、提交
# 前评估候选、处理重复工具错误、保护计算值不被上下文压缩吃掉、预算耗尽前保证有
# 尽力而为的答案。既有安全机制（ReSum 上下文压缩 resum.py、工具结果截断
# tool_result.py、todo 门控的 final_answer 守卫、错误处理）无替代不得禁用。
# 白卷计 0 分，所以同样拒绝没有放弃路径的开放式"继续改进"循环。
CRITIC = """The harness drives a frozen policy LLM (Claude Opus 4.8) through
ENGINEERING DESIGN tasks: it reads a problem statement, computes a design with
a sandboxed scientific Python, and writes the answer as a literal `PAYLOAD`
dict graded by the task's OWN frozen code verifier. VERIFIER ESCAPE is the
most serious failure and the one to check first: the pristine benchmark tree
holds each task's `evaluate.py`, `solution.txt`, `rubrics.txt` and reference
outputs right next to the prompt and is deliberately out of reach. REJECT
anything that reads or searches outside the workspace, uses absolute or
upward paths, walks the filesystem, probes for grading code, or hard-codes a
pass threshold, tolerance, reference value or test input that could only have
come from a grader. A list of disciplines or task families is ALSO leakage: it
encodes the benchmark's distribution rather than a general procedure. General
engineering practice transfers and is fine: validating the payload against
the schema the task itself provides, evaluating a candidate design before
committing to it, handling a repeated tool error, protecting computed values
from context compression, ensuring a best-effort answer exists before the
budget runs out. Existing safety mechanisms that must not be disabled without
a working replacement: ReSum context compression (resum.py), tool-result
truncation (tool_result.py), the todo-gated finalize guard, error handling.
A trial that never writes a payload scores ZERO, so also reject open-ended
"keep improving" loops with no give-up path."""

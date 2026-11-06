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
"""Central config for the EngDesign instance.

Layout:

  third_party/archipelago/runner       FROZEN official archipelago runner. Never edited.
  third_party/archipelago/harness_eng  THE EVOLVABLE SCAFFOLD (main/tools/resum/tool_result/
                 prompts/mechanisms), loaded through RRSI_HARNESS_MODULE.
  mcp_gateway/   the tool ENVIRONMENT. Frozen.
  engdesign_bench/  the benchmark tree, built from the OFFICIAL EngDesign
                 checkout. Holds each task's frozen verifier. NEVER reachable
                 from the agent's sandbox.

WHY THIS BENCHMARK. Each task is graded by its own frozen code verifier that
returns a CONTINUOUS score (raw_score/score_max) for resolution, a BINARY
`passed` for the correctness headline, and a `valid` flag for the damage
guard, with no LLM anywhere in the loop. All per-task variance is therefore
policy variance.

WHAT THE SCAFFOLD IS EVOLVED ON, AND WHERE IT IS JUDGED. Evolution runs on all
61 tasks with no held-out split (the set is too small to spend tasks on one).
The verdict comes from two external surfaces that never enter selection:
EngDesign v1 (the same tasks with grading exploits closed, so a scaffold that
learned to game a verifier shows up as a drop) and Frontier-Eng minus its
EngDesign domain (a different benchmark's verifiers).
"""

# 【模块说明（中文）】EngDesign 实例的集中配置。
# 布局：third_party/archipelago/runner 是冻结的官方 runner（永不编辑）；
# harness_eng 是可演化 scaffold（经 RRSI_HARNESS_MODULE 加载）；mcp_gateway/ 是
# 工具环境（冻结）；engdesign_bench/ 是由官方 EngDesign 检出构建的基准树，
# 装着每题的冻结 verifier，agent 沙箱永远够不到。
# 为什么选这个基准：每题由自己的冻结代码 verifier 评分——连续分（分辨率）、二元
# passed（正确性头条）、valid 标志（损伤 guard），全程无 LLM；因此每任务方差
# 全是策略方差。
# 在上演化、在哪儿裁决：全部 61 题上演化、无 held-out（集合太小，舍不得花题）；
# 裁决来自两个从不参与选择的外部面：EngDesign v1（同题但封了评分漏洞——靠
# 钻 verifier 空子的 scaffold 在那里会掉分）与 Frontier-Eng 去掉 EngDesign 域
# （另一基准的 verifier）。

# os：环境变量读取与回写
import os

# ---- Vertex / models ---------------------------------------------------------
# —— Vertex / 模型 ——
# VERTEX_PROJECT：GCP 项目（也可经环境变量注入子进程）
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")
# VERTEX_LOCATION：Vertex 区域（默认 global）
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
# 同步进 Vertex SDK 认的老变量名（已设置时不覆盖）
os.environ.setdefault("VERTEXAI_PROJECT", VERTEX_PROJECT)
os.environ.setdefault("VERTEXAI_LOCATION", VERTEX_LOCATION)

# Frozen policy under evolution. With opus-4-8 the suite sits in the band where
# a scaffold change can actually separate: not saturated, not on the floor.
# 演化对象：冻结策略模型。选 opus-4-8 是因为本套件在它的分数带里恰好能分离出
# scaffold 变化的效应——既没饱和也没躺在地板上
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "vertex_ai/claude-opus-4-8")
# 采样参数留空：不设 temperature / reasoning_effort（保持默认，冻结）
ORCHESTRATOR_EXTRA_ARGS: dict = {}      # no temperature / reasoning_effort

# Anthropic-on-Vertex lives in specific projects; spreading trials over both
# doubles the usable quota. Route choice is deterministic in (task, trial), and
# this list is FROZEN after the baseline: champion and candidate must draw the
# same route for the same task or the comparison includes a quota difference.
# Anthropic-on-Vertex 住在特定项目里；trial 摊到两个项目上等于配额翻倍。
# 路由对 (task, trial) 确定性；此列表在 baseline 之后冻结——冠军与候选同任务必须
# 同路由，否则比较里混进了配额差异
ANTHROPIC_VERTEX_PROJECTS = [
    p.strip() for p in os.environ.get(
        "ANTHROPIC_VERTEX_PROJECTS", VERTEX_PROJECT).split(",")
    if p.strip()
]


# 定义 policy_endpoints：构造端点列表——每个条目都是同一个冻结策略模型，只是
# vertex_project 不同
def policy_endpoints() -> list[dict]:
    """[{model, extra_args}] -- every entry is the same frozen policy model."""
    # 从 ORCHESTRATOR_MODEL 取裸模型 id
    model_id = ORCHESTRATOR_MODEL.split("/")[-1]
    # 每个项目一个端点条目
    return [{"model": f"vertex_ai/{model_id}",
             "extra_args": {"vertex_project": p, "vertex_location": VERTEX_LOCATION}}
            for p in ANTHROPIC_VERTEX_PROJECTS]


# POLICY_ENDPOINTS：模块加载时构造一次的端点列表（run_tasks 按 (task,trial) 轮转取用）
POLICY_ENDPOINTS = policy_endpoints()

# ---- Paths -------------------------------------------------------------------
# —— 路径 ——
# REPO：本目录（domains/eng/）
REPO = os.path.dirname(os.path.abspath(__file__))
# BENCH_ROOT：基准树根（可用环境变量指到别处）
BENCH_ROOT = os.environ.get("BENCH_ROOT", os.path.join(REPO, "engdesign_bench"))
# BENCH_TASKS：任务目录（每题一个子目录）
BENCH_TASKS = os.path.join(BENCH_ROOT, "benchmarks")
# GRADING_PYTHON：跑冻结 verifier 的专用 venv python
GRADING_PYTHON = os.environ.get(
    "GRADING_PYTHON", os.path.join(REPO, ".venvs", "engdesign", "bin", "python"))
# WORKSPACE_BASE：agent 工作区根（每 trial 一份，按 job 隔离）
WORKSPACE_BASE = os.environ.get("WORKSPACE_BASE", os.path.join(REPO, "workspaces"))
# RUNS：产物根目录
RUNS = os.environ.get("RRSI_RUNS_DIR", os.path.join(REPO, "runs"))
# SPLIT_PATH：任务划分文件
SPLIT_PATH = os.environ.get("RRSI_SPLIT_PATH", os.path.join(REPO, "data", "split_engd.json"))

# The graded deliverable. Every task's frontier_eval/candidate_destination.txt
# names this same path, and it is the ONLY thing the verifier reads.
# 被打分的交付物相对路径：每题的 candidate_destination.txt 都指向它，且 verifier
# 只读这一个东西
CANDIDATE_REL = "submission/payload.py"

# ---- MCP gateway -------------------------------------------------------------
# —— MCP gateway ——
# GATEWAY_HOST/PORT：工具环境网关的监听地址
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8994"))
# GATEWAY_URL：streamable-http 端点
GATEWAY_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}/mcp/"

# ---- Agent config ------------------------------------------------------------
# —— Agent 配置 ——
# 最大步数 / agent 总超时 / 单次 LLM 响应超时（都可被环境变量覆盖）
AGENT_CONFIG_VALUES = {
    "max_steps": int(os.environ.get("MAX_STEPS", "250")),
    "timeout": int(os.environ.get("AGENT_TIMEOUT", "5400")),
    "llm_response_timeout": int(os.environ.get("LLM_RESPONSE_TIMEOUT", "600")),
}
# A trial that exceeds this comes back with no payload and SCORES ZERO. That is
# deliberate and must stay that way for the baseline and every candidate:
# refilling only the trials that failed would launder the score upward, and
# asymmetrically, since a candidate that issues more model calls hits more rate
# limits and more timeouts -- exactly the cost the token guard measures.
# 单 trial 的外层硬超时：超时的 trial 没有产出、按 0 分计。这是刻意设计，baseline
# 与每个候选都必须保持：只补跑失败的 trial 会把分数洗干净地抬上去，且不对称——
# 发更多模型调用的候选会撞更多限流与超时，而这正是 token guard 要度量的成本
TIMEOUT_PER_TASK = int(os.environ.get("TIMEOUT_PER_TASK", "5400"))

# ---- Verifier ----------------------------------------------------------------
# —— Verifier ——
# Per-task wall clock for the frozen verifier. The verifiers run in seconds,
# so this is a crash backstop, not a budget.
# 冻结 verifier 的每题 wall clock：verifier 本身秒级跑完，这只是防挂死的后备，不是预算
VERIFY_TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", "420"))
# 并发评分线程数
VERIFY_JOBS = int(os.environ.get("VERIFY_JOBS", "12"))

# ---- Concurrency -------------------------------------------------------------
# —— 并发 ——
# Must match the concurrency the BASELINE was measured at: a champion and a
# candidate measured at different concurrency are not comparable.
# 必须与 baseline 的测量并发一致：不同并发下测的冠军与候选不可比
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "12"))

# ---- Eval protocol -----------------------------------------------------------
# —— 评估协议 ——
# k=4 is the floor imposed by calibration: bench/calibrate.py measures the null
# by splitting the baseline's own trials 2-vs-2 on the same tasks, which needs
# at least four. It must match the n data/metric.json was calibrated at.
# k=4 是校准强加的下限：calibrate.py 把 baseline 自己的 trial 2 对 2 劈分来测 null，
# 至少要 4 个；且必须与 data/metric.json 校准时用的 n 一致
N_ATTEMPTS = int(os.environ.get("N_ATTEMPTS", "4"))

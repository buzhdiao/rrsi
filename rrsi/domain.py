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
"""The Domain interface: everything RRSI needs from a benchmark environment.

RRSI itself never runs an agent, grades a deliverable or reads a trajectory
format. A domain adapter (domains/<name>/adapter.py) supplies those, and the
texts that make the three search roles speak the benchmark's language.

    run / score        Evaluate(H', D, k)              rrsi.evaluate
    load_trial         trajectories for Analyze        rrsi.loop.build_traces
    render_trace       what the analyst/digester/proposer read
    task_row           one-line summary per trace in the task tables
    smoke              liveness check before evaluation (not a selection rule)
    critic_patterns    deterministic leakage denylist   rrsi.critic
    component_signals  diff regexes -> component tag    rrsi.components
    guards             non-compensatory domain checks   rrsi.selection (Sec. 3.3)
    briefs             domain paragraphs for analyst / digester / proposer / critic
"""
# 【模块说明（中文）】Domain 接口：RRSI 需要基准环境提供的全部能力。
# RRSI 核心自己从不运行 agent、从不给交付物打分、从不读轨迹格式——这些都被
# 推到 domain 适配器（domains/<name>/adapter.py）里，连同让三个搜索角色
# （proposer/analyst/critic）"说基准语言"的提示词文本。
# 接口与核心模块的对应：run/score -> Evaluate(H',D,k)（rrsi.evaluate）；
# load_trial -> Analyze 的轨迹来源（rrsi.loop.build_traces）；render_trace 是
# analyst/digester/proposer 读到的渲染轨迹；task_row 是任务表的每行摘要；
# smoke 是评估前的活性检查（不是选择规则）；critic_patterns 是确定性泄漏黑名单
# （rrsi.critic）；component_signals 是 diff 正则 -> 组件标签（rrsi.components）；
# guards 是非补偿性的 domain 验收判据（rrsi.selection, 论文 3.3 节）；
# briefs 是给四个角色的 domain 段落。

# 延迟类型注解求值
from __future__ import annotations

# importlib.util：按文件路径动态加载 domains/<name>/adapter.py（domain 不是常规包成员）
import importlib.util
# sys：把动态加载的模块注册进 sys.modules，保证 isinstance/导入一致性
import sys
# Path：路径对象
from pathlib import Path

# ROOT：仓库根目录（rrsi/domain.py 的上一级）
ROOT = Path(__file__).resolve().parent.parent
# DOMAINS：domains/ 目录，各基准实例都在其下
DOMAINS = ROOT / "domains"


class Domain:  # Domain 基类：基准适配器的抽象接口（默认实现极少，多数方法要求子类必须实现）
    name: str = "base"  # domain 名（coding/workspace/eng），也是分支名 evolve/<name> 的后缀
    harness_path: str = "harness"        # relative to domains/<name>/
    # 可演化的 harness 目录，相对 domains/<name>/ 的路径
    root: Path                            # domains/<name> in the MAIN checkout
    # 该 domain 在主检出中的目录（由 load_domain 注入，适配器文件不必自己算）

    # ---- task sets ---------------------------------------------------------
    # —— 任务集合划分 ——
    def evolve_ids(self) -> list[str]: raise NotImplementedError  # evolve 集：搜索用的任务 id 列表（选择只看它）；子类必须实现
    def heldout_ids(self) -> list[str]: return []  # 同分布 held-out 集：搜索从不接触，用于最终评估；默认为空
    def smoke_ids(self, incumbent_per_task: dict | None = None) -> list[str]:  # smoke 检查用的少量任务 id
        raise NotImplementedError  # 子类必须实现（可参考现任 per-task 分数挑选代表性任务）

    # ---- Evaluate ----------------------------------------------------------
    # —— Evaluate(H', D, k) 的执行与打分 ——
    def run(self, root: Path, runs_dir: Path, job: str, ids: list[str], k: int,  # 在 root（一个 worktree）下运行 harness
            log_prefix: str = "") -> None:
        """Run the harness checked out under `root` (a worktree) on `ids`
        with k trials, writing to runs_dir/jobs/<job>. Must be resume-safe."""
        # docstring：对 ids 中的每个任务跑 k 个 trial，产物写到 runs_dir/jobs/<job>；
        # 必须可续跑（resume-safe）：重跑时只补缺失的 trial
        raise NotImplementedError  # 子类必须实现（harbor / MCP gateway 等各不相同）

    def score(self, runs_dir: Path, job: str, ids: list[str], k: int  # 把 trial 产物折算成每任务结果
              ) -> tuple[dict, dict]:
        """-> ({task_id: TaskResult}, extra aggregates)."""
        # docstring：返回 ({task_id: TaskResult}, domain 级聚合指标 extra)
        raise NotImplementedError  # 子类必须实现（奖励定义因基准而异）

    def guards(self, incumbent, candidate) -> list[str]:  # 非补偿性判据：违反的 domain 准则列表
        """Violated non-compensatory domain criteria (empty = none)."""
        # docstring：返回被违反的非补偿性 domain 准则（空列表 = 全部通过）；
        # 论文 3.3 节：某些底线（如有效产出率）不参与加权，一票否决
        return []  # 默认无 guard

    def regression_threshold(self, k: int) -> float:  # 归因（attribution）里"单任务回退"的阈值
        """Per-task mean drop that counts as a regression in attribution."""
        # docstring：单任务均值下降多少算一次回退；默认 1/k（一个 trial 的满跌幅）
        return 1.0 / max(1, k)  # k=2 时即 0.5：均值跌超过半个 trial 视为回退

    # ---- evidence ----------------------------------------------------------
    # —— 证据读取（供 analyst/digester/proposer）——
    def load_trial(self, runs_dir: Path, job: str, task_id: str, trial: int):  # 加载一条原始 trial 记录
        raise NotImplementedError  # 子类必须实现（轨迹格式由基准决定）

    def render_trace(self, rec, detail: bool = False) -> str:  # 把轨迹渲染成可读文本
        raise NotImplementedError  # detail=True 时展开每条消息的内容上限（digester 用）

    def task_row(self, task_id: str, rec, tr) -> str:  # 任务表中的一行摘要（分数、状态、步骤数、命中的失败模式）
        raise NotImplementedError  # 子类必须实现

    # ---- gates and texts -----------------------------------------------------
    # —— 闸门与文本 ——
    def smoke(self, root: Path, runs_dir: Path, job: str, ids: list[str]  # 评估前的活性检查
              ) -> tuple[bool, dict]:
        raise NotImplementedError  # 子类必须实现：编译/构造/跑几个任务，返回 (是否存活, 详情)

    critic_patterns: list = []  # domain 专属的确定性泄漏黑名单（正则, 理由）列表，进入 critic.precheck
    component_signals: list = []  # diff 正则 -> 组件标签的 domain 专属信号（优先于通用信号）
    briefs: dict = {}                    # analyst / digester / proposer / critic
    # 四个搜索角色各自的 domain 段落（键：analyst/digester/proposer/critic）
    source_exts: set = {".py", ".txt", ".md", ".json"}  # 提案者可新建文件的扩展名白名单（Workspace.write 的闸门）

    def harness_dir(self, root: Path) -> Path:  # 定义 harness_dir：worktree 内 harness 包的绝对路径
        return (root / "domains" / self.name / self.harness_path).resolve()  # root 是 worktree 根，拼出 domains/<name>/<harness_path> 并 resolve

    def constitution(self, root: Path) -> tuple[str, str]:  # 定义 constitution：读取提案者的"宪法"两件套
        d = root / "domains" / self.name  # 主检出中的 domain 目录
        return (d / "SKILL.md").read_text(), (d / "PATTERNS.md").read_text()  # 返回 (SKILL.md, PATTERNS.md)：行为宪法与模式库，永远注入提案者上下文


def load_domain(name: str) -> Domain:  # 定义 load_domain：按名字动态加载 domain 适配器
    p = DOMAINS / name / "adapter.py"  # 适配器文件路径
    if not p.is_file():  # 不存在
        raise SystemExit(f"unknown domain {name!r} (no {p})")  # 友好报错并列出期望路径
    spec = importlib.util.spec_from_file_location(f"domains.{name}.adapter", p)  # 以固定模块名构造导入 spec（文件级导入，不依赖包结构）
    mod = importlib.util.module_from_spec(spec)  # 由 spec 创建空模块对象
    sys.modules[spec.name] = mod  # 先注册进 sys.modules（dataclass/pickle 等需要模块可寻址）
    spec.loader.exec_module(mod)  # 执行适配器代码，填充模块命名空间
    dom = mod.DOMAIN  # 适配器约定导出单例 DOMAIN
    dom.root = DOMAINS / name  # 注入 domain 根目录（核心不要求适配器自己感知位置）
    return dom  # 返回就绪的 domain 实例

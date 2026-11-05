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
"""Evaluate(H', D_evolve, k): the empirical score and cost of Eq. (estimate).

    S_hat(H) = 1/(k|D|) sum_x sum_j r(x, tau_x^(j))
    C_hat(H) = 1/(k|D|) sum_x sum_j c(tau_x^(j))

with r in [0,1] and c the policy tokens of a trajectory. A missing trial (crash,
timeout, infrastructure) contributes r = 0 with the full denominator, never an
absent slot, so a candidate cannot look better by destroying the trials it
finds hard.

Rewards may carry weights. Every coding / engineering trial has weight 1; a
Harvey LAB trial has reward = criteria passed / criteria total and weight =
criteria total, so S_hat is the fraction of criteria passed over all tasks, the
benchmark's own metric (Appendix, Harvey LAB).
"""
# 【模块说明（中文）】Evaluate(H', D_evolve, k)：论文 Eq. (estimate) 的经验分数与成本。
#   Ŝ(H) = 1/(k|D|) * Σ_x Σ_j r(x, τ_x^(j))（加权奖励的平均）
#   Ĉ(H) = 1/(k|D|) * Σ_x Σ_j c(τ_x^(j))（每 trial 平均策略 token 成本）
# r∈[0,1]，c 是一条轨迹的策略 token 数。缺失 trial（崩溃/超时/基础设施故障）按 r=0
# 计入且分母不减——绝不留空位，使候选无法靠"弄坏自己搞不定的任务"来显得更好。
# 奖励可带权重：coding/eng 每个 trial 权重 1；Harvey LAB 的 trial 奖励=通过准则数/总准则数、
# 权重=总准则数，于是 Ŝ 恰为"全部任务通过的准则占比"，即该基准自己的指标。

# 延迟类型注解求值
from __future__ import annotations

# json：EvalResult 的落盘/回读
import json
# dataclass 工具：asdict 序列化、dataclass/field 声明
from dataclasses import asdict, dataclass, field
# Path：评估产物路径
from pathlib import Path


@dataclass  # dataclass：一个任务的全部 trial 结果
class TaskResult:
    rewards: list[float]                  # one per trial, missing trials -> 0.0
    # 每个 trial 一个奖励值（缺失 trial 记 0.0，分母不减）
    weights: list[float] = field(default_factory=list)   # default 1.0 each
    # 每个 trial 的权重；为空时 __post_init__ 补默认 1.0
    tokens: list = field(default_factory=list)           # int or None per trial
    # 每个 trial 的策略 token 数（int）或 None（该次没记到成本）
    missing: int = 0  # 缺失 trial 计数（崩溃/超时/基础设施故障）
    extra: dict = field(default_factory=dict)  # 该任务的附加信息（domain 自由使用）

    def __post_init__(self):  # dataclass 构造后的规范化钩子
        if not self.weights:  # 没提供权重
            self.weights = [1.0] * len(self.rewards)  # 补成等长全 1（coding/eng 的默认语义）
        if not self.tokens:  # 没提供 token 计数
            self.tokens = [None] * len(self.rewards)  # 补成等长全 None（成本列不可用）

    @property  # 计算属性
    def mean(self) -> float:  # 定义 mean：该任务的加权平均奖励
        w = sum(self.weights)  # 权重和（分母）
        return (sum(r * x for r, x in zip(self.rewards, self.weights)) / w) if w else 0.0  # 加权平均；分母为 0 时返回 0（避免除零）


@dataclass  # dataclass：一次完整评估（job）的结果
class EvalResult:
    job: str  # 评估作业名（base / r3A / heldout_champ 等），也是产物目录名
    k: int  # 每 task 的 trial 数（记录在案，用于校准折算）
    per_task: dict                        # task_id -> TaskResult
    # 每任务结果字典
    S: float                              # S_hat
    # 经验总分 Ŝ（加权奖励占比）
    C: float | None                       # C_hat (None if no token counts)
    # 经验成本 Ĉ（每 trial 平均策略 token）；拿不到 token 计数时为 None
    n_expected: int  # 期望 trial 总数（|D|*k），判定"缺了多少"的分母
    missing: int  # 实际缺失的 trial 数
    extra: dict = field(default_factory=dict)   # domain aggregates (pass counts, ...)
    # domain 级聚合（通过数、valid 率等），guards 会用到

    def to_json(self) -> dict:  # 定义 to_json：序列化为可落盘的 dict
        d = {"job": self.job, "k": self.k, "S": self.S, "C": self.C,  # 顶层标量字段
             "n_expected": self.n_expected, "missing": self.missing,  # 期望/缺失计数（有效性判定依据）
             "extra": self.extra,  # domain 聚合
             "per_task": {t: asdict(r) for t, r in self.per_task.items()}}  # 每任务结果逐个 asdict
        return d  # 返回完整可 JSON 化结构

    @classmethod  # 类方法：反序列化构造器
    def from_json(cls, d: dict) -> "EvalResult":  # 定义 from_json：从 dict 重建 EvalResult
        per = {t: TaskResult(**r) for t, r in d["per_task"].items()}  # 每任务用 ** 解包重建 TaskResult（__post_init__ 会补权重/tokens）
        return cls(job=d["job"], k=d["k"], per_task=per, S=d["S"], C=d["C"],  # 其余字段按名回填
                   n_expected=d["n_expected"], missing=d["missing"],
                   extra=d.get("extra") or {})  # extra 缺失时兜底空 dict（兼容旧文件）

    def save(self, path: Path | str) -> None:  # 定义 save：落盘到 eval.json
        Path(path).write_text(json.dumps(self.to_json(), indent=1))  # JSON（indent=1）覆写文件——断点续跑靠它

    @classmethod  # 类方法
    def load(cls, path: Path | str) -> "EvalResult":  # 定义 load：从 eval.json 回读
        return cls.from_json(json.loads(Path(path).read_text()))  # 读文件 -> dict -> from_json


def aggregate(job: str, k: int, per_task: dict, extra: dict | None = None) -> EvalResult:  # 定义 aggregate：把每任务 trial 记录折算成 Ŝ 与 Ĉ
    """Fold per-task trial records into S_hat and C_hat."""
    # docstring：把每任务的 trial 记录折叠成总分数 Ŝ 与总成本 Ĉ
    num = den = 0.0  # num=加权奖励总和；den=权重总和（缺 trial 也不减分母）
    toks = []  # 收集所有有效 token 计数
    missing = 0  # 累计缺失 trial
    for tr in per_task.values():  # 遍历每个任务
        for r, w in zip(tr.rewards, tr.weights):  # 遍历该任务每个 trial 的（奖励, 权重）
            num += r * w  # 累计加权奖励
            den += w  # 累计权重——缺失 trial 已按 r=0 进入 rewards，因此分母不变
        toks += [x for x in tr.tokens if isinstance(x, (int, float)) and x > 0]  # 只收正的数值 token 数（None/0 不代表真实成本）
        missing += tr.missing  # 累加缺失计数
    return EvalResult(job=job, k=k, per_task=per_task,  # 组装 EvalResult
                      S=(num / den) if den else 0.0,  # Ŝ = 加权奖励 / 权重和
                      C=(sum(toks) / len(toks)) if toks else None,  # Ĉ = 平均每 trial 策略 token；无数据则 None
                      n_expected=len(per_task) * k, missing=missing,  # 期望 trial 数 = 任务数 * k
                      extra=dict(extra or {}))  # 透传 domain 聚合


def evaluate(domain, root: Path, runs_dir: Path, job: str, ids: list[str],  # 定义 evaluate：一次完整评估的入口
             k: int, log_prefix: str = "") -> EvalResult:
    """Run H' (the harness checked out under `root`) on `ids` with k trials and
    score it. Resume-safe: the domain runner fills only missing trials."""
    # docstring：让 H'（检出在 root 的 harness，通常是一个 worktree）在 ids 上跑 k trials 并打分。
    # 可续跑：domain runner 只补缺失的 trial
    domain.run(root, runs_dir, job, ids, k, log_prefix=log_prefix)  # 第一步：真正跑 benchmark（由 domain 适配器决定怎么跑）
    per_task, extra = domain.score(runs_dir, job, ids, k)  # 第二步：读取产物，折成每任务 TaskResult 与 domain 聚合
    return aggregate(job, k, per_task, extra)  # 第三步：折叠出 (Ŝ, Ĉ) 的 EvalResult


def relative_cost_change(C_cand: float | None, C_inc: float | None) -> float:  # 定义 relative_cost_change：相对成本变化 ΔC
    """Delta C = (C' - C_t) / C_t; 0 when either side has no token count."""
    # docstring：ΔC = (C' - C_t) / C_t（相对量）；任一侧没有 token 计数时按 0 处理——
    # 成本规则对"测不到成本"的候选保持中性，不奖不罚
    if not C_cand or not C_inc:  # 任意一侧为 None（或 0，无法作分母）
        return 0.0  # 返回 0
    return (C_cand - C_inc) / C_inc  # 相对变化率：成本规则 Eq. (tokenbudget) 的输入

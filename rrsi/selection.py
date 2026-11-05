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
"""Algorithm 2 of the paper, the selection side, as pure functions.

For every screened candidate H' with measurement (S', C'):

    Delta S = S' - S_t,   Delta C = (C' - C_t) / C_t
    c = [Delta C <= beta0 + beta1 Delta S]             if Delta S > delta      Eq. (tokenbudget)
        [w_s Delta S - w_c Delta C + w_n nu(l') > 0]   otherwise
    admissible  iff  S' >= S* - delta  and  c  (and every domain guard holds)  Eq. (floor)

    H_{t+1} = argmax_{H' admissible} S',  or H_t if none is admissible
    S*      = max(S*, S_{t+1})
"""
# 【模块说明（中文）】论文 Algorithm 2（选择侧），全部实现为纯函数。
# 对每个通过筛选、带测量 (S', C') 的候选 H'：
#   ΔS = S' - S_t（分数增益）；ΔC = (C'-C_t)/C_t（相对成本）
#   成本规则 c（Eq. tokenbudget）：
#     若 ΔS > delta（真实增益）：要求 ΔC <= beta0 + beta1*ΔS（L1 预算：加的 token 要由增益买单）
#     否则（噪声带内）：w_s*ΔS - w_c*ΔC + w_n*nu(l') > 0（shaped 规则：带内增益几乎不值钱，
#     只认省 token 或新结构组件 novelty）
#   可接纳 admissible（Eq. floor）：S' >= S* - delta（noise-adjusted floor 噪声调整地板）
#   且 c 成立，且每条 domain guard 都通过（非补偿性判据）
#   H_{t+1} = 可接纳候选中 argmax S'；没有则保持 H_t（单调不回退）
#   S* = max(S*, S_{t+1})（历史最好，只升不降）

# 延迟类型注解求值
from __future__ import annotations

# dataclass：Candidate / Decision 两个纯数据结构
from dataclasses import dataclass, field

# novelty：结构组件新颖项 nu(l')（components.py）
from .components import novelty
# EvalResult / relative_cost_change：评估结果与相对成本计算
from .evaluate import EvalResult, relative_cost_change


@dataclass  # dataclass：一个候选 harness 的全部已知情报
class Candidate:
    variant: str  # 变体标签（A/B/...，同轮各候选独立起草）
    edits: list[dict]                     # each tagged (component, hypothesis, ...)
    # 编辑清单：每条带 (component 标签 l', hypothesis 假设 h', ...) 元数据
    ev: EvalResult | None = None  # 评估结果；None 表示未完成测量（被闸门拦下或评估无效）
    diff_path: str | None = None  # diff 补丁文件路径（可审计）
    branch: str | None = None  # 该候选所在的 git 分支 <domain>/r<t><variant>
    commit: str | None = None  # 该候选的提交（worktree 中 commit_path 产物）
    gate_failure: str | None = None       # critic_reject / smoke_fail / eval_invalid
    # 闸门失败原因：critic 拒绝 / smoke 未过 / 评估无效（三者都发生在测量之前或之外）
    detail: str = ""  # 失败详情（写入历史，供审计）

    @property  # 计算属性
    def components(self) -> list[str]:  # 定义 components：本候选触及的组件标签列表
        return [e.get("component") for e in self.edits if e.get("component")]  # 收集每条编辑的 component（novelty 的输入）


@dataclass  # dataclass：一个候选的裁决记录
class Decision:
    variant: str  # 对应的候选标签
    admissible: bool  # 是否可接纳（Eq. floor 全链路判定结果）
    reason: str  # 裁决理由（人类可读，写入 decisions.json 与历史）
    S: float | None = None  # 候选分数 S'
    C: float | None = None  # 候选成本 C'
    delta_S: float | None = None  # 分数增益 ΔS
    delta_C: float | None = None  # 相对成本 ΔC
    novelty: int = 0  # 结构新颖项 nu(l')
    guards: list = field(default_factory=list)  # domain guard 的违反列表

    def to_json(self) -> dict:  # 定义 to_json：直接导出实例字典
        return self.__dict__  # dataclass 无嵌套对象，__dict__ 即可 JSON 化（落盘 decisions.json）


def cost_rule(delta_S: float, delta_C: float, nov: int, delta: float,  # 定义 cost_rule：Algorithm 2 第 5 行的成本规则 c
              cfg) -> tuple[bool, str]:
    """c of Algorithm 2, line 5."""
    # docstring：Algorithm 2 第 5 行的判定 c，返回 (是否通过, 解释文字)
    if delta_S > delta:  # 分支一：增益超出噪声带（真实增益）——用 L1 成本预算（Eq. tokenbudget）
        budget = cfg.beta0 + cfg.beta1 * delta_S  # 允许的成本预算 = beta0（白送部分）+ beta1*ΔS（增益买单部分）
        ok = delta_C <= budget  # 相对成本不得超过预算——加推理 token 必须由实测增益支付（cost rule）
        return ok, (f"gain {delta_S:+.4f} > delta {delta:.4f}; cost change "  # 拼装裁决理由（记录进 decisions.json）
                    f"{delta_C:+.3f} {'<=' if ok else '>'} budget {budget:.3f} "
                    f"(beta0 {cfg.beta0} + beta1 {cfg.beta1} * dS)")
    shaped = cfg.w_s * delta_S - cfg.w_c * delta_C + cfg.w_n * nov  # 分支二：噪声带内——shaped 规则：带内增益权重低，省 token 加分，新结构组件（novelty）加分
    ok = shaped > 0  # 严格大于 0 才通过
    return ok, (f"gain {delta_S:+.4f} within delta {delta:.4f}; shaped "  # 解释文字：写出三个分量的线性组合值
                f"{cfg.w_s}*dS - {cfg.w_c}*dC + {cfg.w_n}*nu = {shaped:+.4f} "
                f"{'>' if ok else '<='} 0 (nu={nov})")


def judge(cand: Candidate, incumbent: EvalResult, S_star: float, delta: float,  # 定义 judge：对单个候选做完整可接纳性裁决
          cfg, incumbent_counts: dict, guards: list[str] | None = None) -> Decision:
    ev = cand.ev  # 取候选的评估结果
    if ev is None:  # 没有测量（闸门失败或未评估）
        return Decision(cand.variant, False, cand.gate_failure or "not evaluated")  # 不可接纳，理由即闸门失败名
    dS = ev.S - incumbent.S  # ΔS = S' - S_t：相对现任的分数增益
    dC = relative_cost_change(ev.C, incumbent.C)  # ΔC = (C'-C_t)/C_t：相对成本变化
    nov = novelty(cand.components, incumbent_counts)  # nu(l')：触及的结构组件中现任从未成功采用的数量
    d = Decision(cand.variant, False, "", S=ev.S, C=ev.C, delta_S=dS, delta_C=dC,  # 建立裁决对象（先默认不可接纳，逐条检查）
                 novelty=nov, guards=list(guards or []))
    floor = S_star - delta  # noise-adjusted floor（Eq. floor）：历史最好分 S* 减去噪声带 delta
    if ev.S < floor:  # 低于地板
        d.reason = (f"below noise-adjusted floor: S' {ev.S:.4f} < S* {S_star:.4f} "  # 理由：分数落在评估噪声内，视为没进步
                    f"- delta {delta:.4f}")
        return d  # 一票否决
    ok, why = cost_rule(dS, dC, nov, delta, cfg)  # 通过地板：再查成本规则（L1 预算或 shaped 规则）
    if not ok:  # 成本规则不通过
        d.reason = f"cost rule failed: {why}"  # 理由：加的 token 没有被增益支付（或带内组合分 <= 0）
        return d  # 否决
    if guards:  # domain 非补偿性判据有违反（如 eng 的 valid 率大跌）
        d.reason = "domain guard violated: " + "; ".join(guards)  # 理由：违反了不可补偿的底线准则
        return d  # 否决
    d.admissible = True  # 三关全过：可接纳
    d.reason = f"admissible: {why}"  # 理由记录成本规则的通过分支
    return d  # 返回裁决


def select_round(cands: list[Candidate], incumbent: EvalResult, S_star: float,  # 定义 select_round：Algorithm 2 的整轮入口
                 delta: float, cfg, incumbent_counts: dict,
                 guard_fn=None) -> tuple[Candidate | None, list[Decision]]:
    """Judge every candidate, return (winner or None, decisions)."""
    # docstring：裁决每个候选，返回（获胜者或 None, 全部裁决记录）
    decisions = []  # 收集每个候选的 Decision
    for c in cands:  # 遍历本轮全部候选
        g = guard_fn(incumbent, c.ev) if (guard_fn and c.ev is not None) else []  # domain guards 只对有测量的候选计算（没有测量无从谈起）
        decisions.append(judge(c, incumbent, S_star, delta, cfg,  # 逐个裁决
                               incumbent_counts, g))
    adm = [(c, d) for c, d in zip(cands, decisions) if d.admissible]  # 过滤出可接纳的（候选, 裁决）对
    if not adm:  # 一个都不满足
        return None, decisions  # H_{t+1} = H_t（本轮不动，搜索被正则化刹住）
    winner, _ = max(adm, key=lambda cd: cd[1].S)  # argmax：在可接纳集中取 S' 最大的（max 遍历保持并列时的先后稳定性）
    return winner, decisions  # 返回获胜者与全部裁决

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
"""The edit history L_t and the three summaries the paper computes from it.

    L_t = {(t_i, l_i, h_i, d_i, Delta S_i, Delta C_i, a_i) : i <= n_t}    Eq. (history)
    T_t = {l_i : i <= n_t}                                                 (tried components)
    g_t(l) = max{Delta S_i : l_i = l, t - t_i <= n_prune},  max(empty) = -inf   Eq. (yield)
    U_t = K \\ T_t,  sigma_t = 1[S_t - S_{t-w} <= delta]                    Eq. (explore)
    B_t = {l in T_t : g_t(l) <= 0}                                          Eq. (prune)

One JSONL record per EDIT. A candidate harness that bundles n edits (n <= b_t)
receives one measurement (Delta S, Delta C, a) and every edit in the bundle
carries it; as b_t anneals to 1 the record becomes evidence about a single
component. Candidates dropped before measurement (critic reject, smoke fail,
invalid evaluation) are recorded with delta_S = None and do not enter T_t or g_t.
"""
# 【模块说明（中文）】编辑历史 L_t，以及论文从它计算出的三个汇总量。
#   L_t = {(t_i, l_i, h_i, d_i, ΔS_i, ΔC_i, a_i)}：Eq. (history)——每条记录含轮次、组件、假设、diff、增益、成本、是否被接受
#   T_t = {l_i}：已尝试组件集合
#   g_t(l) = max{ΔS_i : l_i=l, t-t_i <= n_prune}：Eq. (yield)——组件近期的最大增益，空集为 -inf
#   U_t = K \ T_t（未尝试组件），sigma_t = 1[S_t - S_{t-w} <= delta]：停滞标志，Eq. (explore)
#   B_t = {l ∈ T_t : g_t(l) <= 0}：修剪集，Eq. (prune)
# 每个编辑一条 JSONL 记录。打包 n 个编辑（n<=b_t）的候选只得到一次测量 (ΔS, ΔC, a)，
# 包内每个编辑都携带它；随着 b_t 退火到 1，记录逐渐变成关于单一组件的证据。
# 在测量之前就被丢弃的候选（critic 拒绝、smoke 失败、评估无效）记为 delta_S=None，
# 不进入 T_t 或 g_t——没有测量的东西不算证据。

# 延迟类型注解求值
from __future__ import annotations

# json：JSONL 读写
import json
# math：-inf 表示"近期无增益"
import math
# time：每条记录打时间戳（可审计）
import time
# Path：历史文件路径
from pathlib import Path

# K：组件词表（过滤有效记录用）
from .components import K

# MEASURED_OUTCOMES：三种"已被测量"的结局——进入统计口径的记录
MEASURED_OUTCOMES = ("ACCEPTED", "REJECTED", "LOST")


class History:  # History 类：编辑历史 JSONL 的读写与汇总
    def __init__(self, path: Path | str):  # 构造：只存路径，惰性读写（文件不存在即空历史）
        self.path = Path(path)  # runs/<domain>/history.jsonl

    # ------------------------------------------------------------ storage --
    # —— 存储层 ——
    def records(self) -> list[dict]:  # 定义 records：读入全部历史记录
        if not self.path.exists():  # 文件不存在（baseline 前或新 run）
            return []  # 空历史
        out = []  # 记录列表
        for line in self.path.read_text().splitlines():  # 逐行解析
            line = line.strip()  # 去空白
            if line:  # 跳过空行
                out.append(json.loads(line))  # 每行一个 JSON 对象
        return out  # 返回全部记录

    def append(self, rec: dict) -> None:  # 定义 append：追加一条记录（JSONL 行）
        rec = dict(rec)  # 拷贝，避免修改调用方的字典
        rec.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))  # 缺时间戳就补（审计需要）
        self.path.parent.mkdir(parents=True, exist_ok=True)  # 父目录就绪
        with open(self.path, "a") as f:  # 追加模式打开
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # 一行一条 JSON，立即落盘——崩溃也不丢已写记录

    def append_candidate(self, t: int, variant: str, edits: list[dict],  # 定义 append_candidate：把一个候选的测量结果摊到每条编辑
                         outcome: str, delta_S: float | None,
                         delta_C: float | None, accepted: bool,
                         S: float | None, C: float | None,
                         diff: str | None, detail: str = "") -> None:
        """Write the per-edit records of one candidate harness H'."""
        # docstring：写出一个候选 harness H' 的逐编辑记录
        for e in edits or [{"id": "C1", "component": None, "hypothesis": None}]:  # 无编辑清单时写一条占位（候选本身也是历史事实）
            self.append({  # 每个编辑一条记录
                "t": t, "variant": variant, "edit_id": e.get("id"),  # 轮次、变体、编辑 id
                "component": e.get("component"),  # 组件标签 l'
                "hypothesis": e.get("hypothesis") or e.get("mechanism"),  # 假设 h'（兜底用整体机制说明）
                "targets_mode": e.get("targets_mode"),  # 针对的失败模式
                "predicted_affected": e.get("predicted_affected"),  # 预测受影响的任务（归因用）
                "diff": diff,  # diff 文件路径（可审计）
                "delta_S": None if delta_S is None else round(delta_S, 6),  # 分数增益（未测量为 None）
                "delta_C": None if delta_C is None else round(delta_C, 6),  # 相对成本变化
                "accepted": bool(accepted),  # 是否被接受（a_i）
                "outcome": outcome,  # 结局：ACCEPTED/LOST/REJECTED/闸门失败名
                "S": None if S is None else round(S, 6),  # 候选总分
                "C": None if C is None else round(C, 1),  # 候选成本
                "bundle": len(edits or []),  # 打包大小 n（b_t 退火后趋近 1）
                "detail": detail[:600] if detail else "",  # 详情截 600 字符
            })

    def replace_round(self, t: int, keep=lambda r: False) -> None:  # 定义 replace_round：删除第 t 轮的逐编辑记录（重新裁决前调用）
        """Drop the per-edit records of round t (a re-adjudication rewrites
        them); records satisfying `keep` survive."""
        # docstring：丢弃第 t 轮的逐编辑记录（重新裁决会重写它们）；满足 keep 谓词的幸免
        recs = [r for r in self.records()  # 读全部记录
                if not (r.get("t") == t and r.get("edit_id")) or keep(r)]  # 过滤掉 (t 轮且有 edit_id) 的记录——baseline 记录（edit_id=None）保留
        self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))  # 整文件重写（唯一一处非追加写）

    def has(self, t: int, variant: str) -> bool:  # 定义 has：某轮某变体是否已有记录（断点续跑判据）
        return any(r.get("t") == t and r.get("variant") == variant  # 存在匹配记录
                   for r in self.records())  # 则视为已写过，避免重复落账

    # ---------------------------------------------------------- summaries --
    # —— 汇总层 ——
    def measured(self) -> list[dict]:  # 定义 measured：有测量的有效记录
        return [r for r in self.records() if r.get("delta_S") is not None  # 必须有 ΔS（测量过）
                and r.get("component") in K]  # 且组件标签合法

    def tried(self) -> set[str]:  # 定义 tried：T_t——已尝试组件集合
        """T_t: components with at least one measured edit."""
        # docstring：T_t：至少有一次被测量编辑的组件集合
        return {r["component"] for r in self.measured()}  # 集合推导

    def attempted(self) -> dict[str, int]:  # 定义 attempted：每组件"发出过"的编辑数（无论是否被测量）
        """Edits per component that were shipped at all, measured or not."""
        # docstring：每组件实际提交过的编辑计数（含未测量的——诊断用）
        counts = {c: 0 for c in K}  # 初始化全部组件
        for r in self.records():  # 遍历全部记录
            if r.get("component") in counts and r.get("edit_id"):  # 合法组件且有编辑 id
                counts[r["component"]] += 1  # 计数
        return counts  # 返回计数表

    def accepted_edits(self) -> dict[str, list[dict]]:  # 定义 accepted_edits：现任 harness 中的机制清单（按组件）
        """Machinery currently in the incumbent, per component (accepted edits)."""
        # docstring：现任 harness 当前持有的机制，按组件分组（即被接受的编辑）
        out: dict[str, list[dict]] = {c: [] for c in K}  # 初始化
        for r in self.records():  # 遍历全部记录
            if r.get("accepted") and r.get("component") in out:  # 只看被接受的
                out[r["component"]].append(  # 记下轮次/编辑 id/假设/增益
                    {"t": r["t"], "edit_id": r.get("edit_id"),
                     "hypothesis": r.get("hypothesis"),
                     "delta_S": r.get("delta_S")})
        return out  # 修剪集 B_t 会把这些"要拆除的机器"连同组件一起交给提案者

    def incumbent_component_counts(self) -> dict[str, int]:  # 定义 incumbent_component_counts：现任在各组件上的被接受编辑数
        return {c: len(v) for c, v in self.accepted_edits().items()}  # 直接取长度——novelty 的判据：计数为 0 = 从未成功采用

    def yield_g(self, t: int, n_prune: int) -> dict[str, float]:  # 定义 yield_g：g_t(l) 近期收益
        """g_t(l) for every tried component; -inf when nothing recent."""
        # docstring：每个已尝试组件的 g_t(l)；近期没有测量记录则为 -inf
        g = {c: -math.inf for c in self.tried()}  # 初始化为 -inf（Eq. yield：max(空)=-inf）
        for r in self.measured():  # 遍历被测量的记录
            if t - int(r["t"]) <= n_prune:  # 只看最近 n_prune 轮窗口内的
                c = r["component"]  # 该组件
                g[c] = max(g.get(c, -math.inf), float(r["delta_S"]))  # 取最大增益
        return g  # 返回 {组件: g_t(l)}

    def prune_set(self, t: int, n_prune: int) -> list[dict]:  # 定义 prune_set：修剪集 B_t
        """B_t with, for each component, the accepted machinery to remove."""
        # docstring：B_t，并附上每个组件下需要拆除的"现任机制"清单
        g = self.yield_g(t, n_prune)  # 先算近期收益
        acc = self.accepted_edits()  # 现任机制清单
        out = []  # 结果列表
        for c in sorted(g):  # 按组件名排序（输出顺序稳定）
            if g[c] <= 0:  # 近期最大增益 <= 0（包括 -inf：近期压根没测过它）——组件已不再"挣得自己的位置"
                out.append({"component": c,  # 记录组件
                            "recent_best_gain": (None if g[c] == -math.inf  # -inf 显示为 None（更可读）
                                                 else round(g[c], 5)),
                            "accepted_edits_in_incumbent": acc.get(c, [])})  # 附上要拆除的机制明细
        return out  # 交给提案者：删除这些机制本身就是合法编辑（prune 编辑）

    # ------------------------------------------------------- for prompts --
    # —— 供提示词使用 ——
    def render(self, n: int = 40) -> list[dict]:  # 定义 render：给提案者上下文的紧凑历史视图
        """Compact view of the most recent records for the proposer's context.
        Measured outcomes dominate: gate failures without a measurement are
        kept only for their most recent few, because a wall of aborts is a
        feedback loop, not evidence."""
        # docstring：给提案者上下文的最近记录紧凑视图。有测量的记录优先：
        # 无测量的闸门失败只保留最近几条——一墙的失败记录是反馈回路，不是证据
        recs = self.records()  # 全部记录
        kept, unmeasured = [], 0  # 保留列表与未测量计数
        for r in reversed(recs):  # 从最新往旧扫
            if r.get("delta_S") is None:  # 无测量的记录
                unmeasured += 1  # 计数
                if unmeasured > 4:  # 超过 4 条
                    continue  # 丢弃（防闸门失败刷屏挤占有用证据）
            kept.append(r)  # 保留
            if len(kept) >= n:  # 凑够 n 条
                break  # 停止
        keys = ("t", "variant", "edit_id", "component", "hypothesis",  # 只投影这些字段（省上下文）
                "targets_mode", "delta_S", "delta_C", "accepted", "outcome",
                "bundle", "detail")
        return [{k: r.get(k) for k in keys if r.get(k) is not None}  # None 字段整个去掉（JSON 更紧凑）
                for r in reversed(kept)]  # 再反转回时间正序


# --------------------------------------------------------------- explore --
# —— 探索指令（模块级函数）——
def stall_flag(trajectory: list[float], t: int, w: int, delta: float) -> int:  # 定义 stall_flag：停滞标志 sigma_t
    """sigma_t = 1[S_t - S_{t-w} <= delta]; 0 while fewer than w rounds exist."""
    # docstring：sigma_t = 1[S_t - S_{t-w} <= delta]；历史不足 w 轮时为 0（Eq. explore）
    if t < w or t >= len(trajectory) or t - w < 0:  # 数据不足（早期轮次或轨迹缺失）
        return 0  # 不判定停滞（保守：不乱触发探索）
    return int(trajectory[t] - trajectory[t - w] <= delta)  # w 轮内提升没超过噪声带 => 停滞 = 1（布尔转 int）


def exploration(t: int, stall: int, tried: set[str], m_draft: int) -> dict:  # 定义 exploration：组装探索指令 E_t
    """E_t = (sigma_t, U_t, m_draft) plus the text handed to the proposer."""
    # docstring：E_t = (sigma_t, U_t, m_draft) 加上交给提案者的指令文本
    untried = [c for c in K if c not in tried]  # U_t = K \ T_t：从未被测量过的组件
    if stall and untried:  # 已停滞且有未尝试组件：触发保留探索槽位
        text = (f"STALL: the incumbent has not moved by more than the noise band "  # 指令文本一：停滞模式
                f"over the last rounds (sigma_t = 1). {m_draft} candidate slot(s) "
                f"this round are RESERVED for exploratory edits on components the "
                f"run has never exercised: {untried}. A variant holding a reserved "
                f"slot must put at least one edit on one of those components.")
    elif untried:  # 未停滞但仍有未尝试组件
        text = (f"Components not yet exercised in this run: {untried}. Not "  # 指令文本二：软提示（非强制）
                f"mandatory this round (sigma_t = 0), but evidence about them is "
                f"still missing.")
    else:  # 全部组件都试过了
        text = "Every component in K has been exercised at least once."  # 指令文本三：陈述事实
    return {"sigma": stall, "untried": untried, "m_draft": m_draft, "text": text}  # 结构化 E_t（loop.py 里 reserved 槽位据此强制执行）

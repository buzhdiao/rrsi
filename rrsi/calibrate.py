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
"""Estimate the noise band delta by repeatedly evaluating the unchanged base
harness (Sec. stability-aware acceptance).

delta bounds |S_hat(H) - S_hat(H)| between two independent evaluations of the
SAME harness. With R >= 2 independent base evaluations the null difference is
observed directly; with a single k-trial evaluation the standard error of S_hat
is bootstrapped over trials within each task, and the null difference of two
independent evaluations has standard deviation sqrt(2) * se. Either way

    delta = z * sd(null Delta S),   z = cfg.delta_z (2.0 by default),

so an unchanged harness clears the floor S* - delta about 97.5% of the time
and a gain must exceed the band before the L1 cost rule treats it as real.
"""
# 【模块说明（中文）】噪声带（noise band）delta 的校准：反复评估"未改动的基础 harness"，
# 用同一 harness 两次独立评估之间的分数差（零假设差）估计评估噪声。
# delta 界定的是 |Ŝ(H) - Ŝ(H)|——同一个 harness 两次独立评估的分数差。
# 有 R>=2 次独立基础评估时，零假设差被直接观测；只有单次 k-trial 评估时，
# 用 bootstrap 在每个任务的 trials 内重采样估计 Ŝ 的标准误 se，两次独立评估之差的标准差
# 为 sqrt(2)*se。最终 delta = z * sd(null ΔS)，z 默认 2.0（约 97.5% 分位），
# 使得未改动的 harness 也能以约 97.5% 的概率越过地板 S* - delta（noise-adjusted floor），
# 而候选的增益必须超出噪声带，成本规则（cost rule）才把它当作真实增益。

# 延迟类型注解求值，支持 "float | None" 等写法
from __future__ import annotations

# json：把校准结果序列化成 calibration.json
import json
# math：sqrt 用于 sqrt(2)*se 的零假设差标准差换算
import math
# random：bootstrap 重采样的伪随机数发生器
import random
# statistics：stdev/pstdev 计算样本/总体标准差
import statistics as st
# Path：路径对象，读写校准文件
from pathlib import Path

# 复用评估结果的数据结构 EvalResult（每个任务的 trials 奖励与权重）
from .evaluate import EvalResult


def bootstrap_se(ev: EvalResult, reps: int = 2000, seed: int = 7) -> float:  # 定义 bootstrap_se：用 bootstrap 重采样估计 Ŝ 的标准误
    """se(S_hat) by resampling trials within each task (weights respected)."""
    # docstring：在每个任务的 trials 内重采样（尊重权重），估计 Ŝ 的标准误 se
    rng = random.Random(seed)  # 固定种子的 RNG：结果可复现，同一输入永远得到同一 delta（可审计性）
    tasks = [tr for tr in ev.per_task.values() if tr.rewards]  # 收集所有"至少有一个 trial 记录"的任务；没有任何奖励的任务无法重采样，跳过
    vals = []  # vals 收集 reps 次"重采样出的 Ŝ"，其离散程度即标准误
    for _ in range(reps):  # 默认重复 2000 次 bootstrap
        num = den = 0.0  # num=加权奖励和，den=权重和；每轮重采样都从零开始累计
        for tr in tasks:  # 遍历每个任务
            n = len(tr.rewards)  # 该任务有 n 个 trial
            for _j in range(n):  # 有放回地抽 n 次（bootstrap 的核心：与原样本等量的重采样）
                i = rng.randrange(n)  # 随机抽一个 trial 下标（有放回）
                num += tr.rewards[i] * tr.weights[i]  # 累计该 trial 的加权奖励
                den += tr.weights[i]  # 累计权重（保持 Harvey LAB 的"按准则数加权"语义）
        vals.append(num / den if den else 0.0)  # 这次重采样的 Ŝ 值；分母为 0 时记 0
    return st.pstdev(vals) if len(vals) > 1 else 0.0  # vals 的总体标准差即 se(S_hat)；只有一个样本时无法估计，返回 0


def pooled(evals: list[EvalResult]) -> EvalResult:  # 定义 pooled：把同一 harness 的多次评估拼成一次大评估
    """Concatenate the trials of several evaluations of the same harness."""
    # docstring：把同一 harness 的多次评估的 trials 首尾相接
    from .evaluate import TaskResult, aggregate  # 函数内导入：避免模块级循环依赖（evaluate 不依赖 calibrate，但保持局部性更稳）
    per: dict = {}  # task_id -> 拼接后的 TaskResult
    for ev in evals:  # 遍历每次基础评估
        for t, tr in ev.per_task.items():  # 遍历该评估的每个任务
            p = per.setdefault(t, TaskResult(rewards=[], weights=[], tokens=[]))  # 取出（或新建）该任务的累积容器
            p.rewards += tr.rewards  # 追加这次评估的奖励列表（trial 级拼接）
            p.weights += tr.weights  # 追加对应权重（与 rewards 等长对齐）
            p.tokens += tr.tokens  # 追加 token 计数（用于成本 C）
            p.missing += tr.missing  # 累计缺失 trial 数（基础设施失败也要记账）
    return aggregate("pooled", sum(e.k for e in evals), per)  # 用 aggregate 折算成总 k 等于各次 k 之和的合并 EvalResult


def calibrate(evals: list[EvalResult], z: float = 2.0, reps: int = 2000) -> dict:  # 定义 calibrate：主入口，从基础评估估计 delta
    """delta from one or more evaluations of the base harness."""
    # docstring：从一次或多次基础 harness 评估中估计噪声带 delta
    if not evals:  # 防御：没有评估结果就无法校准
        raise ValueError("no base evaluations")  # 直接报错，让调用方先跑 baseline
    if len(evals) >= 2:  # 情形 A：有多次独立评估，可以直接观测零假设差
        scores = [e.S for e in evals]  # 各次评估的总分 Ŝ
        diffs = [abs(a - b) for i, a in enumerate(scores) for b in scores[i + 1:]]  # 两两组合的 |Ŝi - Ŝj|，作为观测记录（诊断用）
        sd_null = st.stdev(scores) * math.sqrt(2) if len(scores) > 1 else 0.0  # 一次评估分数的标准差 * sqrt(2) = 两次独立评估之差的标准差（方差相加）
        method = "repeated base evaluations"  # 记录所用方法：重复基础评估
        observed = {"S_per_eval": scores, "max_abs_diff": max(diffs)}  # 观测量：每次分数与最大两两差，写进校准 JSON 供审计
    else:  # 情形 B：只有单次评估，零假设差无法直接观测
        sd_null = 0.0  # 直接观测项不可用
        method = "bootstrap over trials of one base evaluation"  # 方法名：对单次评估的 trials 做 bootstrap
        observed = {}  # 无额外观测量
    pooled_ev = pooled(evals)  # 把所有评估拼接成一个大数据集，供 bootstrap 使用（样本越多 se 估计越稳）
    se = bootstrap_se(pooled_ev, reps=reps)  # 在拼接数据上重采样出 se(S_hat)
    sd_boot = math.sqrt(2) * se * math.sqrt(pooled_ev.k / evals[0].k)  # 把"合并数据（k 更大）的 se"折算回"单次评估（k0）"的尺度：se 与 sqrt(k) 成反比，再乘 sqrt(2) 得到差的标准差
    # With repeated evals prefer the direct observation unless it is degenerate.
    # 优先用直接观测的 sd_null，除非它退化（例如所有评估分数完全相同，说明噪声太小或样本太少）
    sd_use = sd_null if (len(evals) >= 2 and sd_null > 0) else sd_boot  # 方法选择：重复评估且非退化 -> 直接观测；否则回退到 bootstrap
    return {"delta": round(z * sd_use, 6), "z": z, "sd_null": round(sd_use, 6),  # delta = z * sd(null ΔS)，保留 6 位小数；同时返回所用 sd
            "sd_null_bootstrap": round(sd_boot, 6), "se_bootstrap": round(se, 6),  # 同时给出 bootstrap 侧的估计量，便于两种方法对照
            "method": method, "n_evals": len(evals), "k": evals[0].k,  # 方法名、评估次数、每次的 k，写明校准语境
            "n_tasks": len(pooled_ev.per_task), "S_base": round(evals[0].S, 6),  # 任务数与基础 harness 分数，供 status/审计展示
            "C_base": evals[0].C, **observed}  # 基础成本与（若有）重复评估的观测量一并落盘


def write(path: Path | str, cal: dict) -> None:  # 定义 write：把校准结果写到 runs/<domain>/calibration.json
    Path(path).write_text(json.dumps(cal, indent=1))  # JSON 序列化（indent=1 与仓库其他产物风格一致）并原子覆写文件


def read_delta(path: Path | str) -> float | None:  # 定义 read_delta：从校准文件读回 delta
    p = Path(path)  # 转成 Path 对象
    if not p.exists():  # 文件不存在（还没校准过）
        return None  # 返回 None，让调用方（Run.delta）决定报错还是要求先跑 calibrate
    return float(json.loads(p.read_text())["delta"])  # 解析 JSON 并取 delta 字段（保证是 float）

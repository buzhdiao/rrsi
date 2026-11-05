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
"""Hyperparameters of RRSI (paper Appendix "Hyperparameters") plus the
engineering knobs a run needs that are not part of the method.

Symbols follow the paper:

  T          number of rounds, t = 0..T-1
  k          trials per task inside Evaluate (Eq. estimate)
  m          candidate harnesses drawn per round, |C_t| before screening
  b_min/b_max  bounds of the annealed L0 edit budget b_t (Eq. anneal)
  w          stall window of the exploration flag sigma_t (Eq. explore)
  m_draft    candidate slots reserved for exploratory edits when stalled
  delta      empirical noise band (Sec. stability-aware acceptance)
  beta0/beta1  L1 cost rule for gaining candidates (Eq. tokenbudget)
  w_s/w_c/w_n  shaped rule for candidates inside the noise band (Alg. 2 l.5)
  n_prune    window of the recent-yield summary g_t (Eq. yield / prune)

Units: scores S are fractions in [0,1] (Eq. estimate), cost C is mean policy
tokens per trial, Delta C is RELATIVE (Eq. tokenbudget), so beta0 = 0.10 means
"10% more tokens for free" and beta1 = 40 means "each +1pp of S buys +40%".
"""
# 【模块说明（中文）】RRSI 的超参数（论文附录 Hyperparameters），外加运行所需的工程旋钮
# （不属于方法本身）。符号与论文一致：
#   T=轮数；k=每次评估每任务 trial 数（Eq. estimate）；m=每轮候选数（筛选前的 |C_t|）；
#   b_min/b_max=退火编辑预算 b_t 的上下界（Eq. anneal）；w=停滞标志 sigma_t 的窗口（Eq. explore）；
#   m_draft=停滞后保留给探索性编辑的候选槽位数；delta=经验噪声带（stability-aware acceptance）；
#   beta0/beta1=增益候选的 L1 成本规则（cost rule, Eq. tokenbudget）；
#   w_s/w_c/w_n=噪声带内候选的 shaped 规则（Alg. 2 第 5 行）；n_prune=近期收益 g_t 的窗口（Eq. yield/prune）。
# 单位：分数 S 是 [0,1] 内的比例；成本 C 是每 trial 平均策略 token；ΔC 是相对量——
# 所以 beta0=0.10 表示"白送 10% token"，beta1=40 表示"每 +1pp 分数要花 +40% token 来换"。

# 延迟类型注解求值
from __future__ import annotations

# json：读取 domains/<name>/rrsi.json 配置文件
import json
# dataclass 工具：asdict 导出、dataclass 声明、field 默认值工厂、fields 反射字段名
from dataclasses import asdict, dataclass, field, fields
# Path：配置文件路径
from pathlib import Path


@dataclass  # dataclass 装饰器：自动生成 __init__/__repr__，字段即超参数清单
class RRSIConfig:
    # ---- horizon and estimator -------------------------------------------
    # —— 视界与估计量 ——
    T: int = 20  # 总轮数（t = 0..T-1），退火调度的终点
    k: int = 2  # Evaluate 中每任务的 trial 次数（决定噪声带大小）
    m: int = 2  # 每轮独立起草的候选 harness 数（筛选前）
    # ---- proposal side (Algorithm 1) -------------------------------------
    # —— 提案侧（Algorithm 1）——
    b_min: int = 1  # 退火编辑预算下界：最后几轮每候选只允许 1 个独立编辑
    b_max: int = 4  # 退火编辑预算上界：首轮最多 4 个独立编辑打包
    w: int = 3  # 停滞窗口 w：sigma_t 比较 S_t 与 S_{t-w}
    m_draft: int = 1  # 停滞后（sigma_t=1）保留给"未尝试组件"探索的候选槽位数
    # ---- selection side (Algorithm 2) ------------------------------------
    # —— 选择侧（Algorithm 2）——
    delta: float | None = None      # None -> runs/<domain>/calibration.json
    # 噪声带 delta：None 表示不在配置里写死，改由 calibrate.py 的校准结果（calibration.json）提供
    delta_z: float = 2.0            # calibration: delta = z * sd(null Delta S)
    # 校准分位系数 z：delta = z * sd(null ΔS)，z=2 对应约 97.5% 分位
    beta0: float = 0.10  # L1 成本规则的截距：白送的相对成本预算（10% token）
    beta1: float = 40.0  # L1 成本规则斜率：每 +1 单位 ΔS（即 +100pp）可多花 40 倍相对成本（每 +1pp 约 +0.4）
    w_s: float = 100.0  # shaped 规则中 ΔS 的权重（噪声带内）
    w_c: float = 15.0  # shaped 规则中相对成本 ΔC 的权重（惩罚）
    w_n: float = 0.5  # shaped 规则中 novelty nu 的权重（仅决胜用）
    n_prune: int = 4  # 近期收益 g_t 的回看窗口（轮）；也是修剪集 B_t 的窗口
    # ---- engineering knobs (not part of the method) ----------------------
    # —— 工程旋钮（不属于方法本身）——
    repair_rounds: int = 5          # critic -> proposer repair attempts
    # critic 打回后允许的"修复轮"上限（有界修复 bounded repair）
    invalid_missing_frac: float = 0.15  # 缺失 trial 超过该比例则判定评估本身无效（基础设施故障）
    n_fail_traces: int = 22  # 每轮送入分析师的最低分轨迹条数（失败透镜）
    n_success_traces: int = 6  # 送入分析师的最高分轨迹条数（成功透镜，防止改坏成功习惯）
    eval_parallel: int = 1          # candidates evaluated concurrently
    # 并行评估的候选数（ThreadPoolExecutor 的 max_workers）
    proposer_model: str = "claude-opus-4-8"  # 提案者角色用的搜索模型
    analyst_model: str = "claude-opus-4-8"  # 分析师角色用的搜索模型
    critic_model: str = "claude-opus-4-8"  # critic 角色用的搜索模型
    notes: dict = field(default_factory=dict)  # 未知字段的兜底容器：rrsi.json 里不属于方法的自定义键收进 notes，不丢也不报错

    @classmethod  # 类方法：替代构造器
    def load(cls, path: Path | str, **overrides) -> "RRSIConfig":  # 定义 load：从 rrsi.json 加载并合并命令行覆盖项
        raw = json.loads(Path(path).read_text())  # 读入 domain 的 JSON 配置原文
        known = {f.name for f in fields(cls)}  # 反射取出 dataclass 的全部合法字段名
        kw = {k: v for k, v in raw.items() if k in known}  # 只保留认识的字段（类型校验交给 dataclass 构造）
        kw["notes"] = {k: v for k, v in raw.items() if k not in known}  # 不认识的键（如 policy_model、dataset 等给 domain 用的）收进 notes 保留
        kw.update({k: v for k, v in overrides.items() if v is not None})  # 命令行覆盖：None 表示未提供，跳过；非 None 才覆盖 JSON 值
        return cls(**kw)  # 构造配置对象

    def dump(self) -> dict:  # 定义 dump：导出为可 JSON 化的 dict
        return asdict(self)  # asdict 递归转换 dataclass（notes 也一并保留），写进 frontier.json 便于复现实验

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

# 下方是 Google LLC 的 Apache-2.0 许可证头（本仓库所有源文件的标准开头）。
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
"""Annealed L0 update budget, Eq. (anneal) of the paper:

    b_t = ceil( b_min + (b_max - b_min) * 1/2 (1 + cos(pi t / T)) ),  t = 0..T-1

Early rounds may bundle several coordinated edits in one candidate; late rounds
become sparse and attributable. The constraint bounds ||z_t||_0, the number of
independent edits active in one proposal, and nothing else: the set of
mechanisms the harness may eventually contain is not restricted.
"""
# 【模块说明（中文）】本文件实现论文 Eq. (anneal) 的"退火编辑预算"（annealed edit budget）b_t：
#   b_t = ceil( b_min + (b_max - b_min) * 1/2 (1 + cos(pi * t / T)) )，t = 0..T-1
# 早期轮次允许把多个相互配合的编辑打包进一个候选；后期轮次变得稀疏、每个编辑可归因。
# 该约束只限制 ||z_t||_0（一个提案中"独立编辑"的数量，即 L0 范数），不限制 harness 最终
# 可能包含哪些机制——编辑空间保持开放，这正是 RRSI 的核心立场。

# from __future__ import annotations：让类型注解支持 "int | None" 等延迟求值写法（Python 3.10+ 语法兼容）
from __future__ import annotations

# 导入 math：用到 cos/pi/ceil 三个函数，实现余弦退火曲线
import math


def edit_budget(t: int, T: int, b_min: int, b_max: int) -> int:  # 定义 edit_budget：给定轮次 t、总轮数 T、预算上下界，返回本轮的 b_t
    """b_t for round t (0-indexed) of a T-round run."""
    # docstring：返回 T 轮运行中第 t 轮（0 起始）的退火编辑预算 b_t
    if T <= 0:  # 防御：总轮数非法（0 或负数）时无法定义退火曲线
        return int(b_max)  # 直接退回上界 b_max，避免除零，行为保守（宽松预算）
    t = max(0, min(int(t), int(T)))  # 把 t 夹紧到 [0, T] 区间，防止调用方传入越界轮次导致 cos 参数越界
    v = b_min + (b_max - b_min) * 0.5 * (1.0 + math.cos(math.pi * t / T))  # 余弦退火公式：t=0 时 v=b_max（起步宽松），t=T 时 v=b_min（收尾严格）
    # Guard against 1.0000000002 -> 2 from floating error at t = T.
    # 守卫：在 t=T 处浮点误差可能让 v 变成 1.0000000002，ceil 后错误地变成 2；先四舍五入到 9 位小数再向上取整
    return int(math.ceil(round(v, 9)))  # ceil 得到整数预算（L0 预算必须是整数个独立编辑），再转 int 返回


def budget_table(T: int, b_min: int, b_max: int) -> list[int]:  # 定义 budget_table：一次性算出整条退火调度表
    return [edit_budget(t, T, b_min, b_max) for t in range(T)]  # 列表推导：对 t=0..T-1 逐轮调用 edit_budget，得到 [b_0, b_1, ..., b_{T-1}]，供 `status` 命令打印调度全貌

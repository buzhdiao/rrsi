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

#!/usr/bin/env python3
# Google LLC 的 Apache-2.0 许可证头（仓库标准文件头，保留原文；shebang 行保留在最前）。
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
"""Summarise SWE-bench Verified arms run through harbor: resolve rate over the
full 500-instance denominator (a missing or errored trial counts as unresolved),
infrastructure failures, token cost, and the paired per-instance comparison.

  python3 domains/coding/scripts/swe_summary.py runs/coding/jobs/swe_base runs/coding/jobs/swe_best
"""
# 【模块说明（中文）】汇总经 harbor 跑出的 SWE-bench Verified 各 arm（基线/最优候选等）：
# 以完整 500 题为分母的解决率（缺失或报错的 trial 一律按未解决计）、异常分布、
# token 成本，以及两 arm 之间按实例配对的净胜负比较。纯汇报脚本，不参与搜索。

# 一行导入：json（解析 result.json）、sys（命令行参数）、glob（找 trial 目录）、
# collections（异常类型计数）
import json, sys, glob, collections
# Path：路径对象
from pathlib import Path

# N_TOTAL：SWE-bench Verified 的完整分母 500 题（缺失/报错都算未解决，不缩分母）
N_TOTAL = 500


# 定义 load：读一个 arm 作业目录下所有 trial 的 result.json，按实例名汇总
def load(job_dir):
    # 结果容器：实例短名 -> {resolved, exc, tokens}
    out = {}
    # 通配每个 trial 目录（形如 <task>__<hash>）下的 result.json
    for f in glob.glob(str(Path(job_dir) / "*__*" / "result.json")):
        # 读入该 trial 的结果
        r = json.load(open(f))
        # 实例短名：task_name 取 "/" 后最后一段
        name = str(r.get("task_name", "")).split("/")[-1]
        # verifier 给出的奖励（1.0 = 解决）
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        # 异常信息（dict 或字符串；无则空 dict）
        e = r.get("exception_info") or {}
        # agent 结果段（token 统计）
        ar = r.get("agent_result") or {}
        # 记录：是否解决（严格 == 1.0）、异常类型、输入+输出 token 数
        out[name] = {"resolved": rew == 1.0,
                     "exc": (e.get("exception_type") if isinstance(e, dict) else str(e)) if e else None,
                     "tokens": (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0)}
    # 返回该 arm 的实例级汇总
    return out


# 定义 report：打印一个 arm 的汇总行，返回解决数
def report(label, d):
    # n：跑出的 trial 数；res：解决的实例数
    n = len(d); res = sum(1 for v in d.values() if v["resolved"])
    # 异常类型计数（只统计有异常的 trial）
    exc = collections.Counter(v["exc"] for v in d.values() if v["exc"])
    # 有 token 数据的 trial 的 token 数列表
    toks = [v["tokens"] for v in d.values() if v["tokens"]]
    # 打印：trial 数/500、解决数、以 500 为分母的比率（括号里附"以实际跑数为分母"的比率）、
    # 异常分布、平均每 trial token（千位）
    print(f"{label:9s} trials={n}/{N_TOTAL} resolved={res}  rate={res / N_TOTAL:.4f} "
          f"(over {N_TOTAL}; {res / max(1, n):.4f} over ran)  exceptions={dict(exc)}  "
          f"tokens/trial={sum(toks) / max(1, len(toks)) / 1000:.0f}k")
    # 返回解决数（供外部比较用）
    return res


# 脚本入口：python3 swe_summary.py <job_dir1> [<job_dir2> ...]
if __name__ == "__main__":
    # 读入全部 arm：目录名 -> 实例级汇总
    arms = {Path(p).name: load(p) for p in sys.argv[1:]}
    # 逐 arm 打印汇总
    for k, d in arms.items():
        report(k, d)
    # 恰好两个 arm 时做按实例配对比较（同一实例在两 arm 下的解决与否）
    if len(arms) == 2:
        # 解包两个 arm（保序：命令行先后即先后）
        (a, da), (b, db) = arms.items()
        # 两 arm 都跑过的实例交集
        common = set(da) & set(db)
        # up：b 解决而 a 未解决的实例数（b 的净收益）
        up = sum(1 for t in common if db[t]["resolved"] and not da[t]["resolved"])
        # down：a 解决而 b 未解决的实例数（b 的净损失）
        down = sum(1 for t in common if da[t]["resolved"] and not db[t]["resolved"])
        # 打印配对结果：共同实例数、b 赢/输多少、净值（带符号）
        print(f"paired on {len(common)} common instances: {b} gains {up}, loses {down} (net {up - down:+d})")

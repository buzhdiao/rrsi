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
"""What the score cannot see.

The accept rule reads three numbers. This module reads the behaviour behind
them, so an edit-history entry can distinguish a mechanism that never fired from one
that fired hard and still did nothing. The first invites a better implementation
of the same idea; the second rules out the whole family. Without this
distinction the proposer re-ships the same hypothesis in a new costume every
few rounds and burns the budget.

The axes here are the ones a scaffold can actually move on this benchmark:
how much the agent COMPUTES (code_exec volume and error rate), whether it ever
checks its own submission before finalising, how often it ends with nothing
submitted, and the shape of its losses (malformed / invalid / under target).

Nothing here participates in grading.
"""

# 【模块说明（中文）】分数看不见的东西：验收规则只读三个数，本模块读数字背后的行为，
# 让编辑史条目能区分"机制从未触发"与"触发了但没用"——前者值得换个实现再来，
# 后者要整个想法家族一起排除；没有这个区分，proposer 每隔几轮就换个马甲重投同一
# 假设，烧光预算。这里的轴是 scaffold 在本基准上真正能动的量：agent 的计算量
# （code_exec 次数与错误率）、定稿前是否检查过自己的提交、白卷率，以及失败的形状
# （malformed / invalid / under target）。本模块的一切都不参与打分。

# argparse：命令行参数
import argparse
# json：读 meta/verdict/traj
import json
# re：malformed 错误特征正则
import re
# sys：sys.path 调整
import sys
# Path：路径对象
from pathlib import Path

# ROOT：bench/ 的上一级（domains/eng/）
ROOT = Path(__file__).resolve().parent.parent
# 把域根加入 sys.path 以导入 config
sys.path.insert(0, str(ROOT))
# config：本域集中配置（RUNS、CANDIDATE_REL 等）
import config  # noqa: E402

# JOBS：作业根目录 runs/jobs/
JOBS = Path(config.RUNS) / "jobs"

# _MALFORMED：verifier 报错里"payload 格式坏"的特征词（validation/literal/PAYLOAD/schema/pydantic/field）
_MALFORMED = re.compile(r"validation|literal|PAYLOAD|schema|pydantic|field",
                        re.I)


# 定义 _tool_calls：从轨迹里抽出 (工具名, 参数) 序列（只看 assistant 消息的工具调用）
def _tool_calls(traj: dict):
    # 遍历消息列表
    for m in (traj or {}).get("messages") or []:
        # 非 assistant 消息不含工具调用
        if m.get("role") != "assistant":
            continue
        # 遍历该消息的每个工具调用
        for tc in m.get("tool_calls") or []:
            # function 段（名字 + JSON 字符串参数）
            fn = tc.get("function") or {}
            # 逐个产出 (工具名, 参数)
            yield fn.get("name") or "", (fn.get("arguments") or "")


# 定义 trial_stats：单条 trial 的行为统计——agent 计算了多少、有没有回读检查、白卷与否
def trial_stats(traj: dict, meta: dict | None = None) -> dict:
    # meta 缺省空 dict
    meta = meta or {}
    # 消息列表
    msgs = (traj or {}).get("messages") or []
    # 工具名序列与两个计数器
    names = []
    n_payload_writes = 0
    n_reads_payload = 0
    # 扫描全部工具调用
    for name, args in _tool_calls(traj):
        # 记录工具名
        names.append(name)
        # 写 payload：write_file 或 code_exec 的参数里出现交付路径（submission/payload.py）
        if name in ("write_file", "code_exec") and config.CANDIDATE_REL in args:
            n_payload_writes += 1
        # 回读 payload：read_text_file 或 code_exec 的参数里出现 payload.py
        # （"提交前是否检查过自己"的行为信号）
        if name in ("read_text_file", "code_exec") and "payload.py" in args:
            n_reads_payload += 1
    # A failed code_exec is visible only in the tool RESULT, not the call.
    # code_exec 的失败只体现在工具 RESULT 里，调用侧看不出来——要另扫 tool 消息
    n_failed = 0
    for m in msgs:
        # 只看 code_exec 的工具结果消息
        if m.get("role") == "tool" and m.get("name") == "code_exec":
            # 结果文本
            c = m.get("content")
            # 三种失败特征：success=False（单双引号两种序列化）或出现 Traceback
            if isinstance(c, str) and ("'success': False" in c
                                       or '"success": false' in c.lower()
                                       or "Traceback" in c):
                n_failed += 1
    # 工具调用计数表
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    # 汇总返回：步数、工具调用数、code_exec 次数与失败数、写文件/读文件次数、
    # payload 写入与回读次数、todo 写入次数、完整计数表，以及 meta 里的白卷/token/耗时/状态
    return {
        "n_steps": sum(1 for m in msgs if m.get("role") == "assistant"),
        "n_tool_calls": len(names),
        "n_code_exec": counts.get("code_exec", 0),
        "n_code_exec_failed": n_failed,
        "n_write_file": counts.get("write_file", 0),
        "n_read": counts.get("read_text_file", 0) + counts.get("read_pdf", 0),
        "n_payload_writes": n_payload_writes,
        "n_payload_readbacks": n_reads_payload,
        "n_todo_write": counts.get("todo_write", 0),
        "tool_calls": counts,
        "no_payload": bool(meta.get("no_payload")),
        "total_tokens": meta.get("total_tokens"),
        "wall_s": meta.get("wall_s"),
        "status": meta.get("status"),
    }


# 定义 analyse：把整个作业的行为聚合成一份汇总（分数背后的行为画像）
def analyse(job: str, ids: list[str] | None = None) -> dict:
    """Aggregate behaviour over a whole job."""
    # docstring：聚合一个作业的全部行为
    # 作业目录
    jdir = JOBS / job
    # 每条 trial 的统计行
    rows = []
    # 遍历作业下全部 trial 目录（任务/t 排序）
    for tdir in sorted(jdir.glob("*/t*")):
        # 指定了任务子集时过滤
        if ids is not None and tdir.parent.name not in ids:
            continue
        # 三个源文件路径
        mp, vp, tp = tdir / "meta.json", tdir / "verdict.json", tdir / "traj.json"
        # 没有 meta 的 trial（没跑完）跳过
        if not mp.is_file():
            continue
        # meta 必有；verdict/traj 缺则按空处理
        meta = json.loads(mp.read_text())
        verdict = json.loads(vp.read_text()) if vp.is_file() else {}
        traj = json.loads(tp.read_text()) if tp.is_file() else {}
        # 行为统计
        s = trial_stats(traj, meta)
        # verifier 错误文本
        err = str(verdict.get("error") or "")
        # 附上三个分数轴与 malformed 标记
        s["passed"] = bool(verdict.get("passed"))
        s["valid"] = float(verdict.get("valid") or 0.0)
        s["score"] = float(verdict.get("combined_score") or 0.0)
        # malformed：有报错且命中格式坏特征（与 no_payload 互斥统计）
        s["malformed"] = bool(err) and bool(_MALFORMED.search(err))
        # 收集该行
        rows.append(s)
    # 一条 trial 都没有：返回空汇总
    if not rows:
        return {"job": job, "n_trials": 0}

    # 局部函数 mean：数值字段的均值（保留 4 位）
    def mean(key, default=0.0):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else default

    # 局部函数 rate：满足谓词的 trial 占比（保留 4 位）
    def rate(pred):
        return round(sum(1 for r in rows if pred(r)) / len(rows), 4)

    # 全作业工具调用总量表
    tool_totals: dict[str, float] = {}
    for r in rows:
        for k, v in (r.get("tool_calls") or {}).items():
            tool_totals[k] = tool_totals.get(k, 0) + v
    # trial 总数
    n = len(rows)
    # 汇总返回：作业名、trial 数、平均步数、平均 code_exec 次数、code_exec 错误率、
    # 平均 payload 写入数、payload 回读率、白卷率、失败三分解、通过率、均分、
    # 平均 token/耗时、超时率、每轮平均各工具调用次数（降序）
    return {
        "job": job,
        "n_trials": n,
        "mean_steps": mean("n_steps"),
        "mean_code_exec": mean("n_code_exec"),
        "code_exec_error_rate": (round(sum(r["n_code_exec_failed"] for r in rows)
                                       / max(1, sum(r["n_code_exec"] for r in rows)), 4)),
        "mean_payload_writes": mean("n_payload_writes"),
        "payload_readback_rate": rate(lambda r: r["n_payload_readbacks"] > 0),
        "no_payload_rate": rate(lambda r: r["no_payload"]),
        # The loss decomposition: these three partition every non-pass and want
        # opposite mechanisms (see evolve/trace_render.render_grading).
        # 失败分解：这三率切分所有非通过 trial，各自需要相反的机制（malformed 要
        # "守住最后一步"，under-target 要"算得更好"），分开看才不会互相污染
        "malformed_rate": rate(lambda r: r["malformed"] and not r["no_payload"]),
        "invalid_rate": rate(lambda r: (not r["malformed"]) and (not r["no_payload"])
                             and r["valid"] < 1.0),
        "under_target_rate": rate(lambda r: r["valid"] >= 1.0 and not r["passed"]),
        "pass_rate": rate(lambda r: r["passed"]),
        "mean_score": mean("score"),
        "mean_tokens": mean("total_tokens", None),
        "mean_wall_s": mean("wall_s", None),
        "timeout_rate": rate(lambda r: r.get("status") == "timeout"),
        "tool_calls_per_run": {k: round(v / n, 3) for k, v in
                               sorted(tool_totals.items(), key=lambda kv: -kv[1])},
    }


# 脚本入口：python3 diagnostics.py --job <job>
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # --job：要分析的作业名
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    # 打印整份行为汇总
    print(json.dumps(analyse(a.job), indent=1))

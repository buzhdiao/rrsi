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
"""Scoring / aggregation for a job dir, and the statistics the accept rule uses.

THREE AXES, all from the frozen verifier, each doing a different job:

  combined_score  continuous in [0,1] (raw_score / score_max). The PRIMARY
                  statistic. It is what the paired test runs on, because a
                  design that goes from 0.42 to 0.71 has visibly improved even
                  though `passed` never flipped -- and on a 61-task suite that
                  extra resolution is the difference between seeing an effect
                  and not.
  passed          binary, at the task's own gate. The HEADLINE and the hard
                  correctness floor: a candidate is
                  never accepted if it costs passes, no matter what the
                  continuous mean says.
  valid           the candidate met the task's hard constraints. The DAMAGE
                  axis. A scaffold can raise the average by pushing designs into
                  aggressive territory that more often violates a constraint
                  outright; the continuous mean alone will not show that, and it
                  is exactly what fails on the hardened test surface.

MISSING TRIALS COUNT AS ZERO, never as absent. A crashed, timed-out or
empty-payload trial is a failure of the scaffold, and dropping it from the
denominator would pay a candidate for the trials it destroyed.

Acceptance is a PAIRED comparison: per-task differences d_i on the SAME tasks,
summarised as mean(d) and se = sd(d)/sqrt(n). Pairing removes between-task
difficulty variance, which here runs from tasks solved every time to tasks no
model has ever solved.
"""
# 【模块说明（中文）】作业目录的打分/聚合，以及验收规则用的统计量。
# 三条轴（都来自冻结 verifier，各司其职）：
#   combined_score：[0,1] 连续分（raw_score/score_max），主统计量，配对检验跑在它上面——
#     设计从 0.42 提到 0.71 是肉眼可见的改进，哪怕 passed 从未翻转；61 题套件上这点
#     额外分辨率就是"看见效应"与"看不见"的差别。
#   passed：任务自家门槛上的二元值。头条指标 + 硬正确性底线（hard correctness floor）：
#     只要掉一个通过就不许接受，无论连续均值多好看。
#   valid：设计满足硬约束与否。损伤（damage）轴：scaffold 可以把设计推向更激进、更常
#     违反约束的区域来抬高均值——单看连续均值看不出来，而这正是加固测试面上会崩的东西。
# 缺失 trial 一律记 0，绝不按"不存在"处理：崩溃/超时/白卷是 scaffold 的失败，
# 把它从分母里丢掉等于奖励候选摧毁 trial。
# 验收是配对比较：同任务上的差 d_i，汇总为 mean(d) 与 se = sd(d)/sqrt(n)。
# 配对消除了任务间难度方差（这里从"每次都解出"到"无模型解出"横跨极大）。

# json：读 verdict/meta/metric.json
import json
# math：开方算标准误
import math
# statistics：标准差
import statistics as st
# sys：sys.path 调整
import sys
# Path：路径对象
from pathlib import Path

# 把域根加入 sys.path 以导入 config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# config：本域集中配置
import config  # noqa: E402

# JOBS：作业根目录
JOBS = Path(config.RUNS) / "jobs"
# BENCH：基准任务树根
BENCH = Path(config.BENCH_TASKS)

# METRIC_PATH：校准产物 data/metric.json
METRIC_PATH = Path(config.REPO) / "data" / "metric.json"
# METRIC：读入校准常数（没有文件则空 dict，用缺省值兜底）
METRIC = json.loads(METRIC_PATH.read_text()) if METRIC_PATH.is_file() else {}
# HARD_DROP_DELTA：单任务"重伤/大涨"阈值（calibrate.py 用 null p99 实测得出；缺省 0.50）
HARD_DROP_DELTA = METRIC.get("hard_drop_delta", 0.50)


# 定义 all_task_ids：基准树里的全部任务 id（有 frontier_eval/ 才算），排序
def all_task_ids() -> list[str]:
    return sorted(td.name for td in BENCH.glob("*/*")
                  if (td / "frontier_eval").is_dir())


# 定义 split：读任务划分文件；没有就把全部任务都归入 evolve 集
def split() -> dict:
    # 划分文件路径
    p = Path(config.SPLIT_PATH)
    # 有文件直接读
    if p.is_file():
        return json.loads(p.read_text())
    # 兜底：evolve = 全部任务
    return {"evolve": [{"id": i} for i in all_task_ids()]}


# 定义 task_ids：指定任务集的 id 列表
def task_ids(set_name: str = "evolve") -> list[str]:
    return [t["id"] for t in split()[set_name]]


# 定义 trial_records：某任务某作业下，每个存在的 trial 目录一条记录。
# 有 meta 没 verdict 的 trial 尚未评分，以 graded=False 单独上报（score_job 分开统计）
def trial_records(job: str, task_id: str) -> list[dict]:
    """One record per trial DIRECTORY that exists. A trial with a meta.json but
    no verdict has not been graded yet and is reported separately by score_job."""
    # 结果列表
    out = []
    # 任务目录
    tdir = JOBS / job / task_id
    # 任务没跑过
    if not tdir.is_dir():
        return out
    # 遍历排序后的 trial 目录
    for trial in sorted(tdir.glob("t*")):
        # 没 meta = 从未跑完，直接跳过（score_job 会按槽位缺计）
        if not (trial / "meta.json").is_file():
            continue
        # verdict 文件
        v = trial / "verdict.json"
        # 没 verdict = 跑了但还没评分
        if not v.is_file():
            out.append({"graded": False})
            continue
        # 读 verdict（损坏按空 dict——verifier 保证总留 verdict，这里只防御）
        try:
            d = json.loads(v.read_text())
        except Exception:  # noqa: BLE001
            d = {}
        # 读 meta（损坏按空 dict）
        meta = {}
        try:
            meta = json.loads((trial / "meta.json").read_text())
        except Exception:  # noqa: BLE001
            pass
        # 组装记录：三轴 + verifier 状态 + 白卷/token/耗时/运行状态
        out.append({"graded": True,
                    "score": float(d.get("combined_score") or 0.0),
                    "passed": bool(d.get("passed")),
                    "valid": float(d.get("valid") or 0.0),
                    "vstatus": d.get("status"),
                    "no_payload": bool(meta.get("no_payload")),
                    "tokens": meta.get("total_tokens"),
                    "wall_s": meta.get("wall_s"),
                    "status": meta.get("status")})
    # 返回记录列表
    return out


# 定义 score_job：按 n trials/任务聚合一个作业。ids x n 的每个（任务, trial）槽位
# 都被计入：没有 graded 记录的槽位在三轴上各贡献一个 0（no_payload 贡献 1）
def score_job(job: str, ids: list[str], n: int) -> dict:
    """Aggregate one job over a task list at n trials/task.

    Every (task, trial) slot in ids x n is accounted for: a slot with no graded
    record contributes a zero on all three axes.
    """
    # 每任务统计
    per_task: dict[str, dict] = {}
    # 缺失/未评分的 trial 计数
    missing = ungraded = 0
    # 全局 token / 耗时样本
    toks, walls = [], []
    # 逐任务聚合
    for tid in ids:
        # 该任务的 trial 记录
        recs = trial_records(job, tid)
        # 其中已评分的
        graded = [r for r in recs if r.get("graded")]
        # 未评分的（跑了但 verify 没出 verdict）单独累加
        ungraded += len(recs) - len(graded)
        # 三轴 + 白卷的取值（只取前 n 个）
        scores = [r["score"] for r in graded][:n]
        passes = [1.0 if r["passed"] else 0.0 for r in graded][:n]
        valids = [r["valid"] for r in graded][:n]
        nopay = [1.0 if r["no_payload"] else 0.0 for r in graded][:n]
        # 该任务缺失的槽位数
        hole = n - len(scores)
        # 累加缺失
        missing += hole
        # 缺失槽位补 0（missing-as-zero：不缩分母）
        scores += [0.0] * hole
        passes += [0.0] * hole
        valids += [0.0] * hole
        # no_payload 缺失补 1（连跑都没跑成 = 什么都没交）
        nopay += [1.0] * hole
        # 收集 token/耗时样本（只统计有数据的）
        for r in graded:
            if isinstance(r.get("tokens"), int):
                toks.append(r["tokens"])
            if isinstance(r.get("wall_s"), (int, float)):
                walls.append(r["wall_s"])
        # 记录每任务统计
        per_task[tid] = {
            "mean": sum(scores) / n,
            "scores": scores,
            "pass_rate": sum(passes) / n,
            "passes": int(sum(passes)),
            "valid_rate": sum(valids) / n,
            "no_payload_rate": sum(nopay) / n,
        }
    # 期望 trial 总数
    n_exp = len(ids) * n
    # 返回聚合：作业名、任务数、n、期望 trial 数、缺失/未评分数、四率、均值、
    # 平均 token/耗时、每任务明细
    return {
        "job": job,
        "n_tasks": len(ids),
        "n_attempts": n,
        "n_expected_trials": n_exp,
        "missing_trials": missing,
        "ungraded_trials": ungraded,
        "mean_score": (sum(t["mean"] for t in per_task.values()) / len(ids)) if ids else 0.0,
        "total_passes": sum(t["passes"] for t in per_task.values()),
        "pass_rate": (sum(t["pass_rate"] for t in per_task.values()) / len(ids)) if ids else 0.0,
        "valid_rate": (sum(t["valid_rate"] for t in per_task.values()) / len(ids)) if ids else 0.0,
        "no_payload_rate": (sum(t["no_payload_rate"] for t in per_task.values()) / len(ids)) if ids else 0.0,
        "mean_tokens_per_trial": (sum(toks) / len(toks)) if toks else None,
        "mean_wall_s": (sum(walls) / len(walls)) if walls else None,
        "per_task": per_task,
    }


# 定义 paired：同任务配对差统计——验收规则的统计核心（t 检验 + 伤/益名单）
def paired(champ: dict, cand: dict, ids: list[str], key: str = "mean") -> dict:
    """Paired per-task difference statistics on the same tasks."""
    # 每任务差值（两任都有数据的任务）
    d = [cand[i][key] - champ[i][key] for i in ids if i in champ and i in cand]
    # 样本数
    n = len(d)
    # 样本 <2 算不出统计量：返回中性缺省
    if n < 2:
        return {"n": n, "delta": 0.0, "se": 0.0, "t": 0.0,
                "hard_drops": [], "big_gains": [], "improved": [], "worsened": []}
    # 差值均值
    delta = sum(d) / n
    # 标准误 se = sd(d)/sqrt(n)（配对消除了任务间难度方差）
    se = st.stdev(d) / math.sqrt(n)
    # 实际参与比较的任务
    used = [i for i in ids if i in champ and i in cand]
    # 返回：样本数、delta、se、t 值、重伤/大涨名单（按 HARD_DROP_DELTA）、
    # 严格变好/变坏名单
    return {
        "n": n,
        "delta": delta,
        "se": se,
        "t": (delta / se) if se > 1e-12 else 0.0,
        "hard_drops": [i for i in used
                       if cand[i][key] - champ[i][key] <= -HARD_DROP_DELTA],
        "big_gains": [i for i in used
                      if cand[i][key] - champ[i][key] >= HARD_DROP_DELTA],
        "improved": [i for i in used if cand[i][key] > champ[i][key]],
        "worsened": [i for i in used if cand[i][key] < champ[i][key]],
    }


# 定义 screen_subset：廉价筛选（screen）用的轮转分层子采样。
# 无偏抽样会把预算浪费在动不了的任务上。分层按冠军自己的每任务连续均值：
# 一半给可动中段（候选的效应所在），四分之一给冠军几乎必过的任务（回归绊线），
# 其余给地板任务（真突破最先在那里显形）。窗口随迭代轮转，避免筛选集每轮都是
# 同 30 题——那会悄悄变成第二个训练集
def screen_subset(ids: list[str], seed: int, k: int, champ_per_task: dict) -> list[str]:
    """Rotating STRATIFIED subsample for the cheap screen.

    An unbiased draw would spend most of its trials on tasks that cannot move.
    The strata are taken from the champion's own per-task continuous mean:
    half the budget on the movable middle (where a candidate's effect lives), a
    quarter on tasks the champion nearly always passes (a regression tripwire),
    and the rest on the floor (where a real breakthrough would show first).
    The window rotates with the iteration so the screen is not the same 30 tasks
    every round, which would quietly turn into a second training set.
    """
    # 局部函数 band：按冠军均值给任务分层（<=0.05 地板；>=0.95 巩固；中间可动）
    def band(i):
        m = (champ_per_task.get(i) or {}).get("mean", 0.0)
        return "floor" if m <= 0.05 else ("solid" if m >= 0.95 else "movable")

    # 三个层组
    groups = {"movable": [], "solid": [], "floor": []}
    for i in ids:
        groups[band(i)].append(i)
    # 预算分配：一半可动、四分之一巩固、其余地板
    want = {"movable": k // 2, "solid": k // 4, "floor": k - k // 2 - k // 4}
    # 选中的任务列表
    out: list[str] = []
    for g, need in want.items():
        # 该层池子空则跳过
        pool = groups[g]
        if not pool:
            continue
        # 轮转偏移随 seed（迭代轮次）变化——每轮窗口不同
        off = (seed * 7) % len(pool)
        # 从偏移处环形取 need 个
        out += [pool[(off + j) % len(pool)] for j in range(min(need, len(pool)))]
    # Backfill from whatever is left if a stratum was short.
    # 某层不够数时，从剩余任务里补齐
    if len(out) < min(k, len(ids)):
        rest = [i for i in ids if i not in out]
        off = (seed * 13) % max(1, len(rest))
        out += [rest[(off + j) % len(rest)] for j in range(min(k - len(out), len(rest)))]
    # 去重排序返回
    return sorted(set(out))


# 脚本入口：python3 score.py --job <job> [--n 4] [--set evolve]
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    # --job：作业名
    ap.add_argument("--job", required=True)
    # --n：每任务 trial 数
    ap.add_argument("--n", type=int, default=config.N_ATTEMPTS)
    # --set：任务集名
    ap.add_argument("--set", default="evolve")
    a = ap.parse_args()
    # 打印聚合（不含庞大的 per_task 明细）
    res = score_job(a.job, task_ids(a.set), a.n)
    print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=1))

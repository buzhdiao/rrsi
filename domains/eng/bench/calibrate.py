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
"""Calibrate the metric constants from a completed baseline job.

Nothing about the accept rule is guessed. The noise scale of this benchmark is
an empirical fact, and it decides whether an iteration's move is signal or a
coin flip. Getting it wrong fails in one of two directions: accepting on a
statistic the null itself clears ratchets the training score upward on noise,
while a hard-drop limit borrowed from a different metric makes acceptance
structurally impossible (most candidates identical to the champion would be
vetoed).

Four questions:

1. HEADROOM. What fraction of tasks is pinned at 0.0 or 1.0 across all trials?
   Those cannot move and contribute nothing to a paired comparison. Measured on
   BOTH axes, because they disagree here: a task can be immovable on `passed`
   while its continuous score wanders freely, and that wandering is exactly the
   resolution this benchmark was chosen for.

2. CONTINUOUS NOISE. What does a paired comparison between two runs of the SAME
   harness look like? Measured by HALF-SPLITTING the baseline's own trials, so
   the true difference is zero by construction and everything observed is noise.

3. PASS NOISE. The same half-split on the binary axis, which sets the pass-count
   noise band the hard correctness floor uses.

4. DAMAGE. The null distribution of hard drops, big gains, and their NET, by
   resampling each task from its own baseline rate.

Requires a baseline with n >= 4 (to split 2-vs-2).

Writes data/metric.json (consumed by bench/score.py and loop.py).

Usage: python3 bench/calibrate.py --job baseline --n 4 [--write]
"""

# 【模块说明（中文）】从一个已完成的 baseline 作业校准指标常数（写 data/metric.json）。
# 验收规则的一切阈值都不拍脑袋：本基准的噪声尺度是经验事实，它决定一次迭代的成绩
# 是信号还是掷硬币。搞错有两种死法：用 null 自身就能越过的统计量去验收（在噪声上
# 把训练分越推越高），或从别的指标借来一个 hard-drop 限制（让验收在结构上不可能，
# 连与冠军完全相同的候选都会被否）。四个问题：
# 1) HEADROOM（余量）：多少任务在所有 trial 里钉死在 0.0/1.0（两条轴都测，因为这里
#    两条轴会不一致——passed 不动而连续分自由漂移，漂移正是选这个基准的原因）；
# 2) 连续噪声：同一 harness 两跑之间的配对差长什么样——把 baseline 自己的 trial 对半
#    劈开（half-split），真差按构造为 0，观测到的一切都是噪声（null 零假设）；
# 3) 通过数噪声：二元轴上的同样对半劈，定出硬正确性底线（noise-adjusted floor）的噪声带；
# 4) 损伤（damage）：按每任务自身基线率重采样，得到 hard drops、big gains 及其净值（NET）
#    的 null 分布。
# 要求 baseline n>=4（才能 2 对 2 劈分）。产物由 bench/score.py 与 loop.py 消费。

# argparse：命令行参数
import argparse
# json：读写 metric.json
import json
# math：开方算标准误
import math
# random：null 重采样的可复现随机源
import random
# statistics：均值/标准差
import statistics as st
# sys：sys.path 调整
import sys
# Path：路径对象
from pathlib import Path

# ROOT：bench/ 的上一级，即 domains/eng/
ROOT = Path(__file__).resolve().parent.parent
# 把域根加入 sys.path，以便导入 config
sys.path.insert(0, str(ROOT))
# 把 bench/ 加入 sys.path，以便导入 score
sys.path.insert(0, str(ROOT / "bench"))

# config：本域集中配置（N_ATTEMPTS 等）
import config  # noqa: E402
# score：同目录的打分/统计模块（trial_records / score_job / task_ids）
import score as S  # noqa: E402


# 定义 _axis_values：某任务在指定轴上的 n 个 trial 值（graded 才算，缺失补 0）
def _axis_values(job: str, tid: str, n: int, axis: str) -> list[float]:
    # 取该任务全部已评分 trial 记录
    recs = [r for r in S.trial_records(job, tid) if r.get("graded")]
    # score 轴取连续分；passed 轴取 1.0/0.0；只取前 n 个
    vals = [(r["score"] if axis == "score" else (1.0 if r["passed"] else 0.0))
            for r in recs][:n]
    # 不足 n 的补 0（missing-as-zero，与打分口径一致）
    return vals + [0.0] * max(0, n - len(vals))


# 定义 half_split_null：把同一作业的 trial 对半劈开做配对差——null（零假设）校准的核心。
# 返回配对差的均值/标准误/t 值，以及"每任务绝对变动"的分位数分布（hard_drop_delta 必须
# 越过它：一次下跌只有比"原封不动的 harness 自己都能产生的波动"更大，才算"毁了这题"）
def half_split_null(job: str, ids: list[str], n: int, axis: str) -> dict:
    """Paired delta between two disjoint halves of the same job's trials.

    Also returns the distribution of the ABSOLUTE per-task movement, which is
    what hard_drop_delta must clear: a fall only counts as "this task was
    destroyed" if it is larger than what an unchanged harness produces on its
    own.
    """
    # docstring：同一作业 trial 的两个不相交半区的配对差；另返回每任务绝对变动的分布
    # 每半区的 trial 数（n//2，n>=4 时至少 2）
    half = n // 2
    # 每任务的配对差（a 半区均值 - b 半区均值）
    d = []
    # 逐任务计算
    for tid in ids:
        # 该任务在该轴上的 n 个 trial 值
        v = _axis_values(job, tid, n, axis)
        # 劈成两半：前 half 与其后 half
        a, b = v[:half], v[half:half * 2]
        # 缺任一半区（trial 太少）跳过
        if not a or not b:
            continue
        # 配对差：真效应按构造为 0，观测值全是噪声
        d.append(st.mean(a) - st.mean(b))
    # 样本 <2 算不出标准差：只回报样本数
    if len(d) < 2:
        return {"n": len(d)}
    # 配对差的标准误 se = sd(d)/sqrt(n)——noise-adjusted floor 的噪声尺度
    se = st.stdev(d) / math.sqrt(len(d))
    # 每任务绝对变动的升序样本（算分位数用）
    absd = sorted(abs(x) for x in d)

    # 局部函数 q：取排序列的 f 分位数（越界时取最大）
    def q(f):
        return absd[min(len(absd) - 1, int(f * len(absd)))]

    # 返回：轴名、样本数、配对差均值、标准误、t 值、每半区 trial 数、绝对变动 p90/p95/p99/max
    return {"axis": axis, "n": len(d), "delta": round(st.mean(d), 5),
            "se": round(se, 5), "t": round(st.mean(d) / se, 3) if se else 0.0,
            "half_trials_each": half,
            "abs_move_p90": round(q(0.90), 4), "abs_move_p95": round(q(0.95), 4),
            "abs_move_p99": round(q(0.99), 4), "abs_move_max": round(absd[-1], 4)}


# 定义 pass_count_null：原封不动的 harness 的总通过数会漂多远（重采样 null）。
# 正确性底线用"通过数"表述，所以需要的是通过数本身的散布，而不是每任务均值的散布；
# 每个任务按自己的基线通过率重采样，真效应按构造为 0
def pass_count_null(job: str, ids: list[str], n: int, reps: int = 2000,
                    seed: int = 7) -> dict:
    """How far the TOTAL pass count of an unchanged harness wanders.

    The correctness floor is stated in passes, so it needs the spread of the
    pass count itself, not of a per-task mean. Each task is resampled from its
    own baseline pass rate, which makes the true effect zero by construction.
    """
    # 固定种子的随机源（可复现）
    rng = random.Random(seed)
    # 每任务的基线通过率
    rates = []
    for tid in ids:
        # 该任务 passed 轴的 n 个 0/1 值，取均值即通过率
        v = _axis_values(job, tid, n, "passed")
        rates.append(sum(v) / n)
    # 期望总通过数 = 各任务 通过率 x n 之和
    base = sum(r * n for r in rates)
    # 每次重模拟的总通过数与期望之差
    diffs = []
    for _ in range(reps):
        # 按每任务通过率独立伯努利重抽 n 次，累计全局通过数
        tot = sum(1 for r in rates for _ in range(n) if rng.random() < r)
        # 记录与期望的偏差
        diffs.append(tot - base)
    # 排序以便取分位数
    diffs.sort()

    # 局部函数 q：取排序列的 f 分位数
    def q(f):
        return diffs[min(len(diffs) - 1, int(f * len(diffs)))]

    # 返回：标准差、p05（比它更差只有 5%）、p01（比它更差只有 1%）、重复次数
    return {"sd": round(st.stdev(diffs), 2), "p05": round(q(0.05), 1),
            "p01": round(q(0.01), 1), "reps": reps}


# 定义 null_hard_drops："什么都没变"时 hard drops（重伤）、big gains（大涨）及其净值 NET
# 的分布（重采样 null）。为什么看 NET 而不是只数 drops：trial 少时每任务分数很粗，
# 真有效应的候选往往两头都动；只数下跌的阈值罚的是效应量本身而非"不对称的损伤"。
# NET 才孤立出"毁掉的比修好的多"——这正是 damage guard 要抓的东西
def null_hard_drops(per: dict, n: int, hard_drop: float,
                    reps: int = 400, seed: int = 11) -> dict:
    """Distribution of hard DROPS, big GAINS, and their NET, when nothing changed.

    Why the NET matters and counting drops alone does not: with few trials the
    per-task score is coarse, so any candidate with a real effect moves tasks in
    BOTH directions, while a threshold derived from a no-op null counts only the
    downside. Such a limit penalises EFFECT SIZE rather than asymmetric damage.
    The net isolates "destroyed more than it fixed", which is what the guard was
    always meant to catch.
    """
    # 固定种子的随机源
    rng = random.Random(seed)
    # 每次模拟的 drops 数 / gains 数 / 净值
    drops, gains, nets = [], [], []
    for _ in range(reps):
        # 本轮的重伤任务数与大涨任务数
        hd = bg = 0
        # 逐任务：从其观测分数里重抽一个"伪候选"均值
        for v in per.values():
            # 该任务的观测分数池
            scores = v["scores"]
            # 伪候选均值：有放回抽 n 个取平均
            cand = st.mean([rng.choice(scores) for _ in range(n)])
            # 与基线均值之差
            d = cand - v["mean"]
            # 跌破 -hard_drop：记一次重伤
            if d <= -hard_drop:
                hd += 1
            # 涨过 +hard_drop：记一次大涨
            elif d >= hard_drop:
                bg += 1
        # 收集本轮三个量
        drops.append(hd)
        gains.append(bg)
        nets.append(hd - bg)
    # 各自排序取分位数
    drops.sort()
    gains.sort()
    nets.sort()

    # 局部函数 q：取排序列的 f 分位数
    def q(arr, f):
        return arr[min(len(arr) - 1, int(f * len(arr)))]

    # 返回：drops 均值/p95/p99、gains 均值、NET 均值/p95/p99、重复次数、所用 hard_drop 阈值
    return {"drops_mean": round(st.mean(drops), 1),
            "drops_p95": q(drops, 0.95), "drops_p99": q(drops, 0.99),
            "gains_mean": round(st.mean(gains), 1),
            "net_mean": round(st.mean(nets), 1),
            "net_p95": q(nets, 0.95), "net_p99": q(nets, 0.99),
            "reps": reps, "hard_drop_delta": hard_drop}


# 定义 main：校准主流程——算余量、三条 null、派生全部验收常数，打印并（可选）写 metric.json
def main():
    # 命令行参数
    ap = argparse.ArgumentParser()
    # --job：baseline 作业名（必填）
    ap.add_argument("--job", required=True)
    # --set：任务集名（默认 evolve）
    ap.add_argument("--set", default="evolve")
    # --n：每任务 trial 数（默认取配置 N_ATTEMPTS）
    ap.add_argument("--n", type=int, default=config.N_ATTEMPTS)
    # --write：真写 data/metric.json（否则只报告）
    ap.add_argument("--write", action="store_true",
                    help="write data/metric.json (otherwise report only)")
    args = ap.parse_args()

    # 该任务集的全部任务 id
    ids = S.task_ids(args.set)
    # 汇总 baseline 作业
    res = S.score_job(args.job, ids, args.n)
    # 每任务统计
    per = res["per_task"]
    # 任务数
    n_tasks = len(per)

    # 余量统计（连续轴）：均值≈1 的"永远过"任务数 / 均值≈0 的"从不过"任务数
    always_s = sum(1 for v in per.values() if v["mean"] >= 0.999)
    never_s = sum(1 for v in per.values() if v["mean"] <= 0.001)
    # 余量统计（二元轴）：同样的两个极端
    always_p = sum(1 for v in per.values() if v["pass_rate"] >= 0.999)
    never_p = sum(1 for v in per.values() if v["pass_rate"] <= 0.001)

    # 任务内 trial 间标准差（样本>1 的任务），取中位数——描述单任务波动的一般水平
    within = [st.stdev(v["scores"]) for v in per.values() if len(v["scores"]) > 1]
    sd_trial = st.median(within) if within else 0.0

    # 连续轴 half-split null：配对差的噪声尺度（noise-adjusted floor 的基础）
    null_score = half_split_null(args.job, ids, args.n, "score")
    # 二元轴 half-split null
    null_pass = half_split_null(args.job, ids, args.n, "passed")
    # 通过数重采样 null
    pcn = pass_count_null(args.job, ids, args.n)

    # A per-task fall only counts as "broke this task" if it exceeds what the
    # null ACTUALLY produces. Anchored on the measured p99 of the half-split's
    # absolute per-task movement rather than on a guessed constant or on the
    # within-task sd (which is 0 for the median task here, and would collapse
    # the threshold to the trial step -- far coarser than the metric warrants).
    # hard_drop_delta：单任务跌幅只有超过 null 实际产生的 p99（再加 0.03 余量）才算
    # "毁了这题"。锚在实测分位数上而非拍脑袋常数，也不用任务内 sd（中位任务 sd=0，
    # 会把阈值塌缩到一个 trial 的步长——比指标本身的分辨率粗得多）；下限 0.25
    hard_drop = max(0.25, round(null_score.get("abs_move_p99", 0.25) + 0.03, 2))
    # 在该阈值下重采样 null 的 drops/gains/NET 分布
    nhd = null_hard_drops(per, args.n, hard_drop)

    # FLOOR THE DAMAGE GUARD AT 1, deliberately, and record why. The resampling
    # null draws from each task's four OBSERVED scores, so it cannot produce an
    # outcome the baseline never showed and therefore understates the true
    # trial-to-trial spread -- it returns net p99 = 0 here, i.e. "a candidate may
    # never break even one task more than it fixes". That is the exact shape of
    # the guard that made acceptance structurally impossible on the previous
    # project. One net destroyed task is slack for a single unlucky draw; two is
    # the asymmetric damage this guard exists to catch.
    # damage guard 的净值上限：刻意下限为 1 并记录原因。重采样 null 只从每任务观测到的
    # 4 个分数里抽，产生不出 baseline 没展示过的结果，因此低估真实 trial 间散布
    # （这里 net p99=0，等于"候选永远不许比修好的多毁一题"——正是上个项目里让验收
    # 在结构上不可能的那种 guard）。净值毁 1 题是给单次坏运气留的余量；2 才是此 guard
    # 要抓的不对称损伤
    max_net = max(1, nhd["net_p99"])

    # 汇总全部校准产物
    out = {
        "source_job": args.job, "set": args.set, "n_attempts": args.n,
        "n_tasks": n_tasks,
        # baseline 的缺失 trial 数（供参考）
        "missing_trials": res["missing_trials"],
        "mean_score": round(res["mean_score"], 4),
        "total_passes": res["total_passes"],
        "pass_rate": round(res["pass_rate"], 4),
        "valid_rate": round(res["valid_rate"], 4),
        "no_payload_rate": round(res["no_payload_rate"], 4),
        # headroom, on both axes
        # 余量：两条轴各自的"永远过/从不过/可动"占比
        "score_always": round(always_s / n_tasks, 4) if n_tasks else 0,
        "score_never": round(never_s / n_tasks, 4) if n_tasks else 0,
        "score_movable_fraction": round((n_tasks - always_s - never_s) / n_tasks, 4)
                                  if n_tasks else 0,
        "pass_always": round(always_p / n_tasks, 4) if n_tasks else 0,
        "pass_never": round(never_p / n_tasks, 4) if n_tasks else 0,
        "pass_movable_fraction": round((n_tasks - always_p - never_p) / n_tasks, 4)
                                 if n_tasks else 0,
        # 任务内 trial 间标准差的中位数
        "median_within_task_sd": round(sd_trial, 4),
        # noise
        # 噪声：连续轴 null 全文、其标准误、t=2 可检出的最小差值、二元轴 null、通过数 null
        "null_paired_score": null_score,
        "null_paired_se": null_score.get("se"),
        "detectable_delta_at_t2": (round(2 * null_score["se"], 5)
                                   if null_score.get("se") else None),
        "null_paired_pass": null_pass,
        "null_pass_count": pcn,
        # The pass-count floor: how far below the best-ever pass count a
        # candidate may sit and still be considered. Set at the null's 1st
        # percentile so an unchanged harness clears it ~99% of the time.
        # 通过数的 noise-adjusted floor：候选可以比历史最好通过数低多少仍入围——
        # 取 null 的 1 百分位，使原封不动的 harness 约 99% 的时候都能过线；下限 2
        "noise_band_passes": max(2, int(round(abs(pcn["p01"])))),
        # damage
        # 损伤：hard_drop 阈值、null 分布、净值上限（含说明）、drops p99（下限 2）
        "hard_drop_delta": hard_drop,
        "null_hard_drops": nhd,
        "max_net_hard_drops": max_net,
        "_max_net_hard_drops_note": (
            f"measured null net p99 = {nhd['net_p99']}, floored to {max_net}; "
            "the resampling null draws only from observed trial values and so "
            "understates the true spread"),
        "max_hard_drops": max(2, nhd["drops_p99"]),
    }
    # 打印完整校准结果
    print(json.dumps(out, indent=1))

    # 提示：t=2.0 的验收需要配对差 >= 2*se（可检出最小效应）
    if null_score.get("se"):
        print(f"\n[calibrate] a t=2.0 accept needs delta >= "
              f"{2 * null_score['se']:.4f} combined_score "
              f"({2 * null_score['se'] * 100:.2f}pp)", flush=True)
    # 提示：通过数 null 的 sd/p01 推出的 noise_band_passes
    print(f"[calibrate] pass-count null: sd {pcn['sd']}, p01 {pcn['p01']} "
          f"-> noise_band_passes = {out['noise_band_passes']}", flush=True)
    # 提示：drops/gains/NET 的 null 分位数推出的 max_net_hard_drops
    print(f"[calibrate] null drops mean {nhd['drops_mean']} p99 {nhd['drops_p99']} | "
          f"gains mean {nhd['gains_mean']} | NET p99 {nhd['net_p99']} "
          f"-> max_net_hard_drops = {max_net} (floored)", flush=True)
    # 警告：连续轴可动任务不足一半——scaffold 效应的施展空间有限
    if out["score_movable_fraction"] < 0.5:
        print(f"[calibrate] WARNING only {out['score_movable_fraction']:.0%} of "
              f"tasks can move on the continuous axis "
              f"({out['score_never']:.0%} never, {out['score_always']:.0%} "
              f"always) -- limited room for a scaffold effect", flush=True)

    # --write 时把结果落盘（score.py / loop.py 消费）
    if args.write:
        (ROOT / "data" / "metric.json").write_text(json.dumps(out, indent=1))
        print("wrote data/metric.json")


# 脚本入口
if __name__ == "__main__":
    main()

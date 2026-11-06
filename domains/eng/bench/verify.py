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
"""Grade a job's trials with each task's FROZEN verifier.

For every runs/jobs/<job>/<task>/t<k>/payload.py this copies the pristine task
dir to a throwaway location, drops the payload in at the task's own
`candidate_destination`, runs `frontier_eval/run_eval.sh` under the grading venv,
and writes the resulting metrics as verdict.json next to the payload.

Three properties this file exists to guarantee:

  * The benchmark tree is NEVER written to. Verifiers write metrics.json,
    artifacts.json and scratch files into their working directory; running them
    in place would mutate the pristine tree and make later trials incomparable.
  * A trial that produced no payload is still GRADED, and scores zero. Dropping
    it would pay a candidate for the trials it destroyed.
  * The grader always leaves a verdict behind. A verifier that crashes, hangs or
    writes no metrics yields an explicit zero with a machine-readable reason
    rather than an absent file, because an absent file is indistinguishable from
    "not graded yet" and would silently shrink the denominator.

Idempotent: a trial that already has a verdict.json is skipped.
"""
# 【模块说明（中文）】用每题自己的冻结 verifier 给作业的 trial 评分。
# 对每个 runs/jobs/<job>/<task>/t<k>/payload.py：把原始任务目录拷到一次性位置、
# 把 payload 放进该题自己的 candidate_destination、在 grading venv 下跑
# frontier_eval/run_eval.sh、把产出的 metrics 写成 payload 旁的 verdict.json。
# 本文件要保证的三个性质：
# * 基准树绝不落笔：verifier 会往工作目录写 metrics.json/artifacts.json/临时文件，
#   原地跑会污染原始树，使后续 trial 失去可比性；
# * 白卷 trial 照样评分并得 0：丢掉它等于奖励候选摧毁 trial（missing-as-zero）；
# * 评分器总留 verdict：verifier 崩溃/挂起/不写 metrics 都产生一个带机器可读原因的
#   显式 0，而不是缺文件——缺文件与"还没评"无法区分，会悄悄缩小分母。
# 幂等：已有 verdict.json 的 trial 跳过。

# argparse：命令行参数
import argparse
# json：读写 verdict/metrics
import json
# os：环境变量传递
import os
# shutil：拷贝任务目录与 payload
import shutil
# subprocess：运行 verifier 脚本
import subprocess
# sys：sys.path 调整
import sys
# tempfile：一次性评分工作目录
import tempfile
# time：计时
import time
# ThreadPoolExecutor：线程池并发评分（verifier 是子进程，线程足够）
from concurrent.futures import ThreadPoolExecutor
# Path：路径对象
from pathlib import Path

# ROOT：bench/ 的上一级（domains/eng/）
ROOT = Path(__file__).resolve().parent.parent
# 把域根加入 sys.path 以导入 config
sys.path.insert(0, str(ROOT))
# config：本域集中配置（RUNS/BENCH_TASKS/GRADING_PYTHON/VERIFY_*）
import config  # noqa: E402

# JOBS：作业根目录
JOBS = Path(config.RUNS) / "jobs"
# BENCH：基准任务树根
BENCH = Path(config.BENCH_TASKS)


# 定义 task_dirs：全部任务的 {id -> 目录}（有 frontier_eval/ 才算任务）
def task_dirs() -> dict[str, Path]:
    return {td.name: td for td in BENCH.glob("*/*") if (td / "frontier_eval").is_dir()}


# 定义 grade_one：给一条 trial 评分——拷树、放 payload、跑 verifier、解析 metrics。
# 任何失败路径都返回带显式状态/原因的 verdict（绝不返回"没有结果"）
def grade_one(task_dir: Path, payload: Path, timeout: int) -> dict:
    # 开始计时
    t0 = time.time()
    # verdict 缺省：全 0、状态 ok（后续按实际覆盖）
    verdict = {"combined_score": 0.0, "valid": 0.0, "passed": False,
               "status": "ok", "error": None, "details": None}
    # 一次性临时目录：保证基准树不被写（性质 1）
    with tempfile.TemporaryDirectory(prefix="engd_grade_") as tmp:
        # 工作副本路径（与任务同名，便于任务内相对路径工作）
        work = Path(tmp) / task_dir.name
        try:
            # 整棵拷贝任务目录（保留符号链接）
            shutil.copytree(task_dir, work, symlinks=True)
        except Exception as e:  # noqa: BLE001
            # 拷贝失败：显式 ERR_COPY
            verdict.update(status="ERR_COPY", error=repr(e)[:300])
            return verdict
        # 交付目标路径：该题 frontier_eval/candidate_destination.txt 指定的相对路径
        dest = work / (work / "frontier_eval" / "candidate_destination.txt").read_text().strip()
        # 建父目录
        dest.parent.mkdir(parents=True, exist_ok=True)
        # 有 payload 就放进去；没有就放空 PAYLOAD（白卷照样评 0——性质 2）
        if payload.is_file():
            shutil.copy(payload, dest)
        else:
            dest.write_text("PAYLOAD = {}\n")
        # verifier 命令：run_eval.sh <grading python> <工作目录> <交付路径>
        cmd = ["bash", str(work / "frontier_eval" / "run_eval.sh"),
               config.GRADING_PYTHON, str(work), str(dest)]
        # 环境变量：给任务内部子进程的超时留 30 秒余量（外层先到）
        env = {**os.environ, "ENGDESIGN_TASK_TIMEOUT_S": str(max(30, timeout - 30))}
        try:
            # 运行 verifier（输出捕获，不外泄）
            subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            # verifier 超时：显式 ERR_TIMEOUT（性质 3：总有 verdict）
            verdict.update(status="ERR_TIMEOUT",
                           error=f"verifier exceeded {timeout}s")
            verdict["wall_s"] = round(time.time() - t0, 1)
            return verdict
        except Exception as e:  # noqa: BLE001
            # 启动失败等：显式 ERR_EXEC
            verdict.update(status="ERR_EXEC", error=repr(e)[:300])
            verdict["wall_s"] = round(time.time() - t0, 1)
            return verdict
        # artifacts.json：verifier 的逐子判据明细
        arts = work / "artifacts.json"
        details = None
        if arts.is_file():
            try:
                details = (json.loads(arts.read_text()) or {}).get("details")
            except Exception:  # noqa: BLE001
                details = None
        # metrics.json：verifier 的正式产出
        m = work / "metrics.json"
        if not m.is_file():
            # 没写 metrics：显式 ERR_NO_METRICS，附 eval.stderr.txt 尾部（截 400 字符）
            err = work / "eval.stderr.txt"
            verdict.update(status="ERR_NO_METRICS",
                           error=(err.read_text()[-400:] if err.is_file() else ""))
            verdict["wall_s"] = round(time.time() - t0, 1)
            return verdict
        try:
            # 解析 metrics
            metrics = json.loads(m.read_text())
        except Exception as e:  # noqa: BLE001
            # 解析失败：显式 ERR_PARSE
            verdict.update(status="ERR_PARSE", error=repr(e)[:300])
            verdict["wall_s"] = round(time.time() - t0, 1)
            return verdict

    # 提取连续分
    score = metrics.get("combined_score")
    verdict.update(
        combined_score=float(score) if isinstance(score, (int, float)) else 0.0,
        valid=float(metrics.get("valid") or 0.0),
        passed=bool(metrics.get("passed")),
        raw_score=metrics.get("raw_score"),
        score_max=metrics.get("score_max"),
        error=metrics.get("error"),
        # The verifier's own per-sub-criterion breakdown ("iou_score 0.83 vs
        # pass_threshold 0.8", "phase margin correct, settling time wrong").
        # This is the single most informative artefact for the analyst: it says
        # WHICH part of the design was wrong, which a scalar score cannot.
        # verifier 自己的逐子判据明细：对 analyst 是信息量最大的工件——它说明设计
        # 的哪个部分错了，这是标量分数做不到的
        details=details,
    )
    # Clamp: a verifier that reports outside [0,1] (a sentinel, a ratio that
    # beat the reference) must not distort the mean across tasks.
    # 截断到 [0,1]：verifier 报出的哨兵值/超过参考的比值不许扭曲跨任务均值
    verdict["combined_score"] = max(0.0, min(1.0, verdict["combined_score"]))
    # 记录评分耗时
    verdict["wall_s"] = round(time.time() - t0, 1)
    # 返回 verdict
    return verdict


# 定义 main：扫描作业下全部待评 trial，线程池并发评分并落盘 verdict.json
def main():
    # 命令行参数
    ap = argparse.ArgumentParser()
    # --job：作业名（必填）
    ap.add_argument("--job", required=True)
    # --jobs：并发 worker 数（默认 VERIFY_JOBS）
    ap.add_argument("--jobs", type=int, default=config.VERIFY_JOBS)
    # --timeout：单个 verifier 的超时（默认 VERIFY_TIMEOUT）
    ap.add_argument("--timeout", type=int, default=config.VERIFY_TIMEOUT)
    # --force：强制重评已有 verdict 的 trial
    ap.add_argument("--force", action="store_true", help="re-grade existing verdicts")
    a = ap.parse_args()

    # 任务表
    tds = task_dirs()
    # 待评列表
    todo = []
    # 作业目录
    jdir = JOBS / a.job
    # 遍历全部 trial 目录
    for tdir in sorted(jdir.glob("*/t*")):
        # 任务 id = 父目录名
        tid = tdir.parent.name
        # 不认识的任务（不在基准树）跳过
        if tid not in tds:
            continue
        if not (tdir / "meta.json").is_file():
            continue                      # trial never ran; score_job counts it 0
            # 从未跑完的 trial 不评——score_job 会按槽位缺计 0
        # 幂等：已有 verdict 且未 force 就跳过
        if (tdir / "verdict.json").is_file() and not a.force:
            continue
        # 进入待评队列
        todo.append((tds[tid], tdir))
    # 打印待评数量
    print(f"[verify] job={a.job} to grade: {len(todo)}", flush=True)
    # 没有待评的直接返回
    if not todo:
        return

    # 局部函数 work：评一条 trial 并写 verdict.json
    def work(item):
        td, tdir = item
        v = grade_one(td, tdir / "payload.py", a.timeout)
        (tdir / "verdict.json").write_text(json.dumps(v, indent=1))
        return tdir.parent.name, v

    # 非 ok 的 verifier 运行计数
    bad = 0
    # 线程池并发评分
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for tid, v in ex.map(work, todo):
            # 非 ok（超时/无 metrics/解析失败等）计数并打印原因
            if v["status"] != "ok":
                bad += 1
                print(f"  [{v['status']}] {tid}: {str(v['error'])[:120]}", flush=True)
    # 打印总结
    print(f"[verify] done; non-ok verifier runs: {bad}/{len(todo)}", flush=True)


# 脚本入口
if __name__ == "__main__":
    main()

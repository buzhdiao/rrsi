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
"""Grade a job's trials with a Frontier-Engineering task's own frozen verifier.

Kept SEPARATE from bench/verify.py on purpose: that file is called by the
running evolution loop, and a bug introduced there would silently corrupt an
in-flight candidate's measurement. This one only ever runs for the final
out-of-distribution eval.

The contract is the same shape as EngDesign's, because EngDesign was built into
the Frontier layout in the first place: every task carries a `frontier_eval/`
folder naming its candidate destination, its eval command, its working
directory, and the venv the command must run under. The differences are that the
command is per-task rather than a shared wrapper, and that the venv is per
FAMILY (`frontier-v1-main`, `frontier-eval-driver`, ...) rather than one shared
grading environment.

  metrics.json -> {combined_score, valid, ...}

TWO THINGS THE CALLER MUST KNOW.

`combined_score` here is in each task's NATIVE UNITS (m/s, Mbps, /100, ...) and
upstream is explicit that it must never be averaged across tasks. So this file
reports per-task scores and leaves aggregation to the caller, which must compare
WITHIN a task (win / tie / loss against a fixed reference) rather than across
them. A mean over these numbers is meaningless.

A task whose family venv was never built fails with a clear marker rather than a
zero, because "the environment is missing" and "the agent's program was bad" are
different facts and averaging them together would understate the scaffold.
"""

# 【模块说明（中文）】用 Frontier-Engineering 任务自带的冻结 verifier 给 trial 评分。
# 刻意与 bench/verify.py 分开：那个文件被运行中的进化循环调用，那里引入的 bug 会悄悄
# 污染在途候选的测量；本文件只在最终 OOD 评估时跑。
# 契约形状与 EngDesign 相同（EngDesign 本来就是按 Frontier 布局造的）：每题有
# frontier_eval/ 目录写明候选目的地、评估命令、工作目录、命令要用的 venv。区别在于：
# 命令是每题自己的而非共享包装；venv 按家族划分（frontier-v1-main、frontier-eval-driver
# ...）而非共享评分环境。
# 调用方必须知道的两件事：
# 1) 这里的 combined_score 是每题的原生单位（m/s、Mbps、/100...），上游明言绝不跨题
#    平均。本文件只报每题分数，聚合交给调用方——必须在题内与固定参考比（胜/平/负），
#    对这些数求均值没有意义；
# 2) 家族 venv 没建的任务以清晰标记失败而非计 0："环境缺失"与"agent 程序差"是两件
#    事实，混在一起平均会低估 scaffold。

# argparse：命令行参数
import argparse
# json：读 manifest/metrics、写结果
import json
# os：环境变量
import os
# shutil：拷贝任务目录与候选
import shutil
# subprocess：运行每题自己的评估命令
import subprocess
# sys：sys.path 调整
import sys
# tempfile：一次性评分目录
import tempfile
# time：计时
import time
# ThreadPoolExecutor：并发评分
from concurrent.futures import ThreadPoolExecutor
# Path：路径对象
from pathlib import Path

# ROOT：bench/ 的上一级（domains/eng/）
ROOT = Path(__file__).resolve().parent.parent
# 把域根加入 sys.path 以导入 config
sys.path.insert(0, str(ROOT))
# config：本域集中配置（RUNS）
import config  # noqa: E402

# JOBS：作业根目录
JOBS = Path(config.RUNS) / "jobs"

# The seven tasks of Frontier's own EngDesign domain are all in our evolve set,
# so they are not out-of-distribution and must never enter this eval.
# Frontier 自带 EngDesign 域的 7 题全在我们的 evolve 集里，不是分布外数据，绝不进入本次评估
EXCLUDED_DOMAINS = {"EngDesign"}


# 定义 _read：读文本文件并去空白；不存在返回默认值
def _read(p: Path, default: str = "") -> str:
    return p.read_text(encoding="utf-8").strip() if p.is_file() else default


# 定义 load_manifest：读官方转换器的 manifest，产出 {任务id -> frontier 元数据}。
def load_manifest(manifest: Path) -> dict[str, dict]:
    """Per-task {task_rel, env_name, eval_command, eval_cwd, candidate_destination}.

    Read from the OFFICIAL converter's output rather than re-derived here. Two
    of the four fields cannot be recovered from the task directory at all:
    `env_name` lives in `frontier_eval/conf/batch/v1.yaml` (most tasks have no
    `env_name.txt`), and `eval_command` carries a `{repo_root}` placeholder that
    points outside the task. Re-implementing that resolution would be a second
    source of truth for which venv grades which task -- and a wrong venv does
    not error, it silently scores the task badly.
    """
    # docstring：必须读官方转换器的输出而非自行推导——env_name 在 conf/batch/v1.yaml
    # 里、eval_command 带指向任务外的 {repo_root} 占位；重新实现这套解析等于给"哪个
    # venv 评哪题"立第二个真相源，而选错 venv 不会报错，只会悄悄把分打坏
    # 结果表
    out = {}
    # 逐任务
    for t in json.loads(manifest.read_text()):
        # frontier 元数据
        fr = t.get("frontier") or {}
        # 任务相对路径
        rel = fr.get("task_rel") or ""
        # 没路径或命中排除域跳过
        if not rel or rel.split("/")[0] in EXCLUDED_DOMAINS:
            continue
        # 以扁平 id 为键记录
        out[rel.replace("/", "__")] = fr
    # 返回任务表
    return out


# 定义 venv_python：某家族 venv 的 python 路径
def venv_python(froot: Path, env_name: str) -> Path:
    return froot / ".venvs" / env_name / "bin" / "python"


# 定义 grade_one：给一条 Frontier trial 评分（命令是每题自己的，占位符按 manifest 解析）
def grade_one(froot: Path, fr: dict, candidate: Path, timeout: int) -> dict:
    # 开始计时
    t0 = time.time()
    # 任务相对路径
    rel = fr["task_rel"]
    # 题目源目录
    task_dir = froot / "benchmarks" / rel
    # 家族 venv 名（缺省 frontier-eval-driver）
    env_name = fr.get("env_name") or "frontier-eval-driver"
    # 该家族的 python
    py = venv_python(froot, env_name)
    # verdict 缺省：分数/valid 均为 None（与 0 区分），状态 ok
    v = {"task": rel, "env_name": env_name,
         "combined_score": None, "valid": None, "status": "ok", "error": None}
    if not py.is_file():
        # 家族 venv 没建：显式 ERR_NO_VENV（不是 0——环境缺失与程序差是两回事）
        v.update(status="ERR_NO_VENV",
                 error=f"family venv not built: {py} (run setup_v1_task_envs.sh)")
        return v

    # 一次性临时目录：保证 Frontier 树不被写
    with tempfile.TemporaryDirectory(prefix="fr_grade_") as tmp:
        # 工作副本
        work = Path(tmp) / task_dir.name
        try:
            # 整棵拷贝任务目录
            shutil.copytree(task_dir, work, symlinks=True)
        except Exception as e:  # noqa: BLE001
            # 拷贝失败：显式 ERR_COPY
            v.update(status="ERR_COPY", error=repr(e)[:300])
            return v
        # 交付目标相对路径
        dest_rel = fr.get("candidate_destination") or ""
        if not dest_rel:
            # manifest 没写交付目的地：显式 ERR_NO_DEST
            v.update(status="ERR_NO_DEST", error="no candidate_destination in manifest")
            return v
        # 目标绝对路径（工作副本内）
        dest = work / dest_rel
        if candidate is not None and candidate.is_file():
            # 有候选：放进交付位置
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(candidate, dest)
        # else: the shipped initial_program stays in place, which is exactly the
        # reference-replay case (grade the baseline the benchmark ships).
        # 没有候选：随基准发行的初始程序原地不动——这正是"参考重放"用例（给基准
        # 自带的 baseline 打分）

        # 评估命令模板
        cmd_tmpl = fr.get("eval_command") or ""
        if not cmd_tmpl:
            # manifest 没写命令：显式 ERR_NO_CMD
            v.update(status="ERR_NO_CMD", error="no eval_command in manifest")
            return v
        # {repo_root} points at SHARED evaluator code outside the task dir (a
        # whole domain's tasks often share one evaluate_unified.py), so it must
        # resolve to the real checkout, not to our throwaway copy.
        # 占位符替换：{python}=家族 venv、{repo_root}=真实检出根（共享评估代码在
        # 任务目录之外，必须指向真检出而非一次性拷贝）、{benchmark}=工作副本、
        # {candidate}=交付路径
        cmd = (cmd_tmpl.replace("{python}", str(py))
                       .replace("{repo_root}", str(froot))
                       .replace("{benchmark}", str(work))
                       .replace("{candidate}", str(dest)))
        # 工作目录（缺省任务根）
        cwd_rel = fr.get("eval_cwd") or "."
        cwd = (work / cwd_rel).resolve()
        try:
            # 运行评估命令（shell=True，因为命令是完整命令行模板）
            r = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True,
                               text=True, timeout=timeout,
                               env={**os.environ, "PYTHONNOUSERSITE": "1"})
        except subprocess.TimeoutExpired:
            # 超时：显式 ERR_TIMEOUT
            v.update(status="ERR_TIMEOUT", error=f"verifier exceeded {timeout}s")
            v["wall_s"] = round(time.time() - t0, 1)
            return v
        except Exception as e:  # noqa: BLE001
            # 其他异常：显式 ERR_EXEC
            v.update(status="ERR_EXEC", error=repr(e)[:300])
            v["wall_s"] = round(time.time() - t0, 1)
            return v

        # 找 metrics.json：先看 eval_cwd 下，再看任务根下
        m = None
        for cand in (cwd / "metrics.json", work / "metrics.json"):
            if cand.is_file():
                m = cand
                break
        if m is None:
            # 没产出 metrics：显式 ERR_NO_METRICS，附 stderr/stdout 尾部与返回码
            v.update(status="ERR_NO_METRICS",
                     error=(r.stderr or r.stdout)[-500:], returncode=r.returncode)
            v["wall_s"] = round(time.time() - t0, 1)
            return v
        try:
            # 解析 metrics
            metrics = json.loads(m.read_text())
        except Exception as e:  # noqa: BLE001
            # 解析失败：显式 ERR_PARSE
            v.update(status="ERR_PARSE", error=repr(e)[:300])
            v["wall_s"] = round(time.time() - t0, 1)
            return v

    # 填入分数与 valid（原生单位，不做聚合）；另附前 20 个 metrics 字段与返回码
    v.update(combined_score=metrics.get("combined_score"),
             valid=metrics.get("valid"),
             metrics={k: metrics[k] for k in list(metrics)[:20]},
             returncode=r.returncode)
    # 记录耗时
    v["wall_s"] = round(time.time() - t0, 1)
    # 返回 verdict
    return v


# 定义 main：OOD 终审评分入口——评作业的 payload 或重放每题自带 baseline
def main():
    # 命令行参数
    ap = argparse.ArgumentParser()
    # --frontier-root：Frontier 检出根（必填）
    ap.add_argument("--frontier-root", required=True)
    # --manifest：官方转换器的输出
    ap.add_argument("--manifest", default=os.environ.get("FRONTIER_MANIFEST", "data/frontier_manifest.json"),
                    help="output of scripts/frontier_eng/frontier_to_apex.py")
    # --job：给了就评该作业的 payload；不给则重放每题自带的 baseline
    ap.add_argument("--job", default="", help="grade a job's payloads; omit to "
                                              "replay each task's shipped baseline")
    # --jobs：并发数
    ap.add_argument("--jobs", type=int, default=4)
    # --timeout：单个 verifier 超时（Frontier verifier 更慢，默认 1800 秒）
    ap.add_argument("--timeout", type=int, default=1800)
    # --tasks：逗号分隔子集
    ap.add_argument("--tasks", default="", help="comma-separated subset")
    # --out：结果 JSON 输出路径
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # Frontier 根
    froot = Path(a.frontier_root).resolve()
    # 读任务表
    tasks = load_manifest(Path(a.manifest))
    # 给了子集就过滤（id 或末段名都认）
    if a.tasks:
        want = {t.strip() for t in a.tasks.split(",") if t.strip()}
        tasks = {k: v for k, v in tasks.items()
                 if k in want or k.split("__")[-1] in want}
    # 打印任务数与根
    print(f"[fverify] {len(tasks)} tasks (EngDesign domain excluded), "
          f"root={froot}", flush=True)

    # 局部函数 work：确定候选文件并评分
    def work(item):
        key, td = item
        cand = None
        if a.job:
            # 评作业：取该任务 t0 的 payload.py（没有就按参考重放处理）
            p = JOBS / a.job / key / "t0" / "payload.py"
            cand = p if p.is_file() else None
        return grade_one(froot, td, cand, a.timeout)

    # 结果行
    rows = []
    # 线程池并发评分，逐条打印
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for v in ex.map(work, sorted(tasks.items())):
            rows.append(v)
            print(f"  {v['task']:52s} score={str(v['combined_score']):>12s} "
                  f"valid={str(v['valid']):>5s} {v['status']}"
                  + (f" :: {str(v['error'])[:70]}" if v["error"] else ""),
                  flush=True)

    # 产出了分数的任务数
    scored = [r for r in rows if r["combined_score"] is not None]
    # 缺家族 venv 的任务数
    novenv = [r for r in rows if r["status"] == "ERR_NO_VENV"]
    # 打印总结：出分数/缺 venv/其他失败三类分列
    print(f"\n[fverify] produced a score: {len(scored)}/{len(rows)} | "
          f"missing family venv: {len(novenv)} | "
          f"other failures: {len(rows) - len(scored) - len(novenv)}")
    # 重申：分数是每题原生单位，不可平均；题内与固定参考比
    print("[fverify] scores are per-task NATIVE UNITS -- do not average them; "
          "compare within a task against a fixed reference.")
    # --out 给了就落盘完整结果
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=str))
        print(f"[fverify] wrote {a.out}")


# 脚本入口
if __name__ == "__main__":
    main()

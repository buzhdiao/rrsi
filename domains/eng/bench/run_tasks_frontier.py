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
"""Run the scaffold over Frontier-Engineering tasks (the OOD verdict surface).

Separate from bench/run_tasks.py because the DELIVERABLE CONTRACT is different,
and that difference is the point of the surface: on EngDesign the agent writes a
literal `PAYLOAD` dict describing a design, while here it edits a PROGRAM that
the verifier then executes. A scaffold that only learned to fill in a dict has
learned nothing that transfers; one that learned how to work an unfamiliar
engineering problem end to end should carry over.

Everything else is deliberately identical to the evolve-set runner -- same
engine, same jailed gateway, same frozen policy, same k -- so the only thing
that differs between the two surfaces is the benchmark.

TWO DEVIATIONS FROM UPSTREAM, both recorded because they affect what the numbers
mean:

  * Upstream's benchmark is iterate-to-improve-within-budget. This is SINGLE
    SHOT: the agent gets one pass and the result is graded. Absolute numbers are
    therefore not comparable to Frontier's leaderboard; they are a consistent
    A/B between two scaffolds.
  * Frontier lists `verification/` as read-only context the solver may see, and
    this runner honours that (it copies what the task's own manifest exposes).
    That is upstream's design, not ours, and both arms face it identically.
"""

# 【模块说明（中文）】在 Frontier-Engineering 任务上跑 scaffold——这是 OOD（分布外）
# 终审面。与 bench/run_tasks.py 分开，因为交付契约不同，而差异正是这个面的意义：
# EngDesign 上 agent 写描述设计的字面 PAYLOAD dict；这里它编辑一段"程序"，verifier
# 随后执行它。只学会填 dict 的 scaffold 什么也迁移不过去；学会端到端处理陌生工程
# 问题的 scaffold 才能带过来。其余一切刻意与 evolve 跑法一致——同一引擎、同一牢笼
# gateway、同一冻结策略、同一 k——两个面之间唯一的差异就是基准本身。
# 与上游的两处偏差（都影响数字的含义）：
# * 上游是"预算内迭代改进"，这里是 SINGLE SHOT：agent 只有一轮，结果即评分。绝对数
#   不可与 Frontier 榜单比；它是两个 scaffold 之间一致的 A/B。
# * Frontier 把 verification/ 列为解题者可读的上下文，本 runner 遵守（按任务自己的
#   manifest 暴露什么就复制什么）。那是上游的设计，且两个 arm 面对它的方式相同。

# argparse：命令行参数
import argparse
# asyncio：并发跑 trial
import asyncio
# hashlib：确定性路由哈希
import hashlib
# json：读 manifest、写 traj/meta
import json
# os：环境变量与目录创建
import os
# shutil：工作区复制/清理
import shutil
# sys：sys.path 调整
import sys
# time：计时
import time
# traceback：异常栈
import traceback
# Path：路径对象
from pathlib import Path

# ROOT：bench/ 的上一级（domains/eng/）
ROOT = Path(__file__).resolve().parent.parent
# 把官方 archipelago 引擎加入 sys.path
sys.path.insert(0, str(ROOT.parent.parent / "third_party" / "archipelago"))
# 默认 harness 模块为 harness_eng.main（与 evolve 跑法一致）
os.environ.setdefault("RRSI_HARNESS_MODULE", "harness_eng.main")
# 把域根加入 sys.path 以导入 config
sys.path.insert(0, str(ROOT))

# config：本域集中配置
import config  # noqa: E402
# AgentRunInput：引擎的 agent 运行输入模型
from runner.agents.models import AgentRunInput  # noqa: E402
# get_agent_impl：从注册表取 agent 实现
from runner.agents.registry import get_agent_impl  # noqa: E402
# SYSTEM_PROMPT：可演化 scaffold 的系统提示词
from harness_eng.prompts import SYSTEM_PROMPT  # noqa: E402

# JOBS：作业根目录
JOBS = Path(config.RUNS) / "jobs"
# EXCLUDED_DOMAINS：排除的域——Frontier 自带的 EngDesign 7 题全在我们的 evolve 集里，
# 不是分布外数据，绝不进入这次评估
EXCLUDED_DOMAINS = {"EngDesign"}

# The task wrapper. Kept structurally parallel to the evolve set's, because a
# scaffold that only works with one phrasing of the contract has learned the
# phrasing rather than the job. What changes is the deliverable: a program the
# verifier will run, not a literal dict.
#
# IT MUST STAY AS NEUTRAL AS THE EVOLVE SET'S WRAPPER. A single helpful line
# here (for example telling the agent to pass `workdir` on every code_exec
# call) can pre-empt exactly the failure an accepted scaffold mechanism exists
# to recover from; the OOD measurement would then read a handicap written into
# the wrapper, not a property of the scaffold. Anything the wrapper teaches,
# the scaffold no longer gets credit for.
# 任务包装模板：结构与 evolve 集的保持平行——只认一种契约措辞的 scaffold 学到的是
# 措辞而非工作。变化的是交付物：要被 verifier 执行的程序，而非字面 dict。
# 它必须与 evolve 集的包装一样中性：这里任何一句"帮忙"的话（比如提醒每次 code_exec
# 传 workdir）都可能提前化解某个被接受的 scaffold 机制本来要恢复的失败；届时 OOD
# 测量读到的是包装里写死的补偿，而不是 scaffold 的性质。包装教过的，scaffold 不再记功。
TASK_TEMPLATE = """=== ENGINEERING TASK ===
The full problem statement is in the workspace. Read it first:

  {task_md}

=== WORKSPACE ===
{workspace}

Everything the task provides is inside this workspace. There is no web access.
Do not read or write anything outside it.

=== DELIVERABLE (this file is the ONLY thing that counts) ===
{candidate_file}

Edit that file. It is a Python program which will be EXECUTED by an automated
evaluator after you finish, and its behaviour is what is scored. It already
contains a working baseline solution, so improving it is the job; leaving it
unchanged scores whatever the baseline scores.

Work is only credited through this file. Anything you print, explain in chat, or
leave in a scratch file earns nothing.
"""


# 定义 load_tasks：读 Frontier manifest，产出任务记录列表（排除 EngDesign 域）
def load_tasks(manifest: Path) -> list[dict]:
    # 结果列表
    out = []
    # manifest 每项一个任务
    for t in json.loads(manifest.read_text()):
        # frontier 字段
        fr = t.get("frontier") or {}
        # 任务相对路径
        rel = fr.get("task_rel") or ""
        # 没路径或命中排除域（EngDesign）则跳过
        if not rel or rel.split("/")[0] in EXCLUDED_DOMAINS:
            continue
        # 记录：id 用 "__" 拼接的扁平名、rel、域、frontier 元数据
        out.append({"id": rel.replace("/", "__"), "rel": rel,
                    "domain": rel.split("/")[0], "frontier": fr})
    # 返回任务列表
    return out


# 定义 gradeable：该任务能否被打分——家族 venv 没建的话跑了也读不出数
def gradeable(froot: Path, t: dict) -> bool:
    """A task whose family venv was never built cannot be graded, so running the
    agent on it would burn a rollout for a number nobody can read. Skipped in
    BOTH arms, which keeps the comparison paired."""
    # 家族 venv 名（缺省 frontier-eval-driver）
    env = (t["frontier"].get("env_name") or "frontier-eval-driver")
    # venv 的 python 存在即可评
    return (froot / ".venvs" / env / "bin" / "python").is_file()


# 定义 trial_dir：某 job 某任务第 trial 次的目录
def trial_dir(job: str, tid: str, trial: int) -> Path:
    return JOBS / job / tid / f"t{trial}"


# 定义 is_done：trial 是否已完成（meta.json 存在且非空）——续跑跳过的依据
def is_done(d: Path) -> bool:
    m = d / "meta.json"
    return m.is_file() and m.stat().st_size > 0


# 定义 build_workspace：物化一题 agent 可见的工作区；返回 (工作区, 交付文件路径)
def build_workspace(froot: Path, t: dict, job: str, trial: int) -> tuple[Path, Path]:
    # 题目源目录
    src = froot / "benchmarks" / t["rel"]
    # 工作区路径（按 job 与 trial 隔离）
    ws = Path(config.WORKSPACE_BASE) / job / f"{t['id']}_t{trial}"
    # 重跑先删旧
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    # 建全新工作区
    ws.mkdir(parents=True)
    # 复制清单：manifest 的 copy_files（缺省 "." 整目录）；这是任务自己的 manifest
    # 暴露什么就复制什么（包括上游设计的 verification/ 只读上下文）
    listed = t["frontier"].get("copy_files") or ["."]
    for rel in listed:
        # "." 表示整目录；否则拼子路径
        s = src if rel == "." else src / rel
        d = ws if rel == "." else ws / rel
        # 目录整棵复制，但跳过 frontier_eval 与 __pycache__
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("frontier_eval", "__pycache__"))
        elif s.is_file():
            # 单文件先建父目录再拷
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    # frontier_eval/ holds the harness wiring (which venv, which command), not
    # task content. It is never exposed: it would tell the agent exactly how it
    # is about to be invoked.
    # frontier_eval/ 装的是 harness 接线（哪个 venv、哪条命令），不是任务内容；
    # 绝不暴露——它会告诉 agent 自己即将被怎样调用
    shutil.rmtree(ws / "frontier_eval", ignore_errors=True)
    # 交付文件路径（manifest 的 candidate_destination，缺省 baseline/init.py）
    cand = ws / (t["frontier"].get("candidate_destination") or "baseline/init.py")
    # 返回工作区与交付路径
    return ws, cand


# 定义 run_trial：跑单个 trial（与 evolve 跑法同构，差别只在交付物是程序）
async def run_trial(froot: Path, t: dict, trial: int, job: str,
                    sem: asyncio.Semaphore) -> dict:
    # 任务 id
    tid = t["id"]
    # 输出目录
    out_dir = trial_dir(job, tid, trial)
    # 信号量限流
    async with sem:
        # 已完成：跳过（续跑）
        if is_done(out_dir):
            return {"id": tid, "trial": trial, "status": "skipped"}
        # 建输出目录
        out_dir.mkdir(parents=True, exist_ok=True)
        # 建工作区，拿交付路径
        ws, cand = build_workspace(froot, t, job, trial)
        # 找题面文件：优先 Task.md，其次 README.md
        task_md = next((ws / n for n in ("Task.md", "README.md")
                        if (ws / n).is_file()), ws / "Task.md")

        # 用户消息：任务模板填入工作区/题面/交付文件
        user_msg = TASK_TEMPLATE.format(workspace=str(ws), task_md=str(task_md),
                                        candidate_file=str(cand))
        # 端点列表（与 evolve 跑法相同的多项目摊配额）
        eps = config.POLICY_ENDPOINTS
        # 确定性路由哈希
        h = int(hashlib.md5(tid.encode()).hexdigest()[:8], 16)
        # 选端点；无列表则退回单模型
        ep = eps[(h + trial) % len(eps)] if eps else {
            "model": config.ORCHESTRATOR_MODEL, "extra_args": {}}
        # 组装 agent 输入（与 evolve 完全同构）
        run_input = AgentRunInput(
            trajectory_id=f"fr-{job}-{tid}-t{trial}",
            initial_messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user_msg}],
            mcp_gateway_url=config.GATEWAY_URL,
            mcp_gateway_auth_token=None,
            orchestrator_model=ep["model"],
            orchestrator_extra_args={**config.ORCHESTRATOR_EXTRA_ARGS,
                                     **ep["extra_args"]},
            agent_config_values=dict(config.AGENT_CONFIG_VALUES),
        )
        # 取 agent 实现
        impl = get_agent_impl("react_toolbelt_agent")
        # 计时
        t0 = time.time()
        # 状态与输出缺省
        status, output = "error", None
        try:
            # 限时运行
            output = await asyncio.wait_for(impl(run_input),
                                            timeout=config.TIMEOUT_PER_TASK)
            status = str(output.status)
        except asyncio.TimeoutError:
            # 超时
            status = "timeout"
        except Exception as e:  # noqa: BLE001
            # 其他异常：打印栈，不中断其他 trial
            print(f"[error] {tid} t{trial}: {e!r}", flush=True)
            traceback.print_exc()
        # wall 时间
        wall = time.time() - t0

        # meta：任务/trial/状态/域/耗时/模型
        meta = {"id": tid, "trial": trial, "status": status, "domain": t["domain"],
                "wall_s": round(wall, 1), "model": ep["model"]}
        # 有输出就落盘轨迹
        if output is not None:
            try:
                # 序列化轨迹
                dump = output.model_dump(mode="json")
                (out_dir / "traj.json").write_text(json.dumps(dump, indent=1, default=str))
                meta["n_messages"] = len(dump.get("messages") or [])
                meta["usage"] = dump.get("usage") or {}
            except Exception as e:  # noqa: BLE001
                # 序列化失败记进 meta
                meta["traj_error"] = repr(e)
        # The graded artefact is named payload.py for symmetry with the evolve
        # runner, so bench/verify_frontier.py finds it at the same path.
        # 被评的工件也叫 payload.py（与 evolve 跑法对称），verify_frontier.py 按同路径取
        text = cand.read_text(errors="replace") if cand.is_file() else ""
        (out_dir / "payload.py").write_text(text)
        # payload 字符数
        meta["payload_chars"] = len(text)
        # 白卷标记（文本全空）
        meta["no_payload"] = not text.strip()
        # 写 meta.json（完成标记）
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        # 清理工作区
        shutil.rmtree(ws, ignore_errors=True)
        # 打印完成行
        print(f"[done] {tid} t{trial} status={status} {wall:.0f}s", flush=True)
        # 返回 meta
        return meta


# 定义 main：OOD 评估的调度入口
async def main():
    # 命令行参数
    ap = argparse.ArgumentParser()
    # --job：作业名（必填）
    ap.add_argument("--job", required=True)
    # --frontier-root：Frontier 检出根（必填）
    ap.add_argument("--frontier-root", required=True)
    # --manifest：任务 manifest（默认环境变量或 data/frontier_manifest.json）
    ap.add_argument("--manifest", default=os.environ.get("FRONTIER_MANIFEST", "data/frontier_manifest.json"))
    # --n：每任务 trial 数（默认 1——OOD 终审是 single shot）
    ap.add_argument("--n", type=int, default=1)
    # --concurrency：并发
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENT)
    # --tasks：逗号分隔子集
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()

    # Frontier 根
    froot = Path(a.frontier_root).resolve()
    # 读任务列表
    tasks = load_tasks(Path(a.manifest))
    # 给了子集就过滤
    if a.tasks:
        want = {x.strip() for x in a.tasks.split(",") if x.strip()}
        tasks = [t for t in tasks if t["id"] in want or t["rel"] in want]
    # 可评任务（venv 齐备）
    runnable = [t for t in tasks if gradeable(froot, t)]
    # 被跳过的任务（缺家族 venv）
    skipped = [t["rel"] for t in tasks if t not in runnable]
    # 打印计划摘要
    print(f"[frun] job={a.job} {len(runnable)} runnable of {len(tasks)} "
          f"(EngDesign domain excluded; {len(skipped)} skipped for a missing "
          f"family venv)", flush=True)
    # 列出跳过明细
    if skipped:
        print(f"[frun] skipped: {', '.join(sorted(skipped))}", flush=True)
    # 建工作区根与作业目录
    os.makedirs(config.WORKSPACE_BASE, exist_ok=True)
    (JOBS / a.job).mkdir(parents=True, exist_ok=True)

    # 并发信号量
    sem = asyncio.Semaphore(a.concurrency)
    # 全部 (任务, trial) 槽位
    todo = [(t, k) for t in runnable for k in range(a.n)]
    # 并发执行
    results = await asyncio.gather(*(run_trial(froot, t, k, a.job, sem)
                                     for t, k in todo))
    # 按状态聚合
    by = {}
    for r in results:
        by[r.get("status", "?")] = by.get(r.get("status", "?"), 0) + 1
    # 打印总结
    print(f"[frun] SUMMARY {json.dumps(by)}", flush=True)


# 脚本入口
if __name__ == "__main__":
    asyncio.run(main())

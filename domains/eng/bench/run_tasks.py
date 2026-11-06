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
"""Run the react_toolbelt agent over EngDesign tasks, k trials each.

One job = one (task list x n_attempts) sweep into runs/jobs/<job>/:

    runs/jobs/<job>/<task_id>/t<trial>/payload.py   the graded deliverable
    runs/jobs/<job>/<task_id>/t<trial>/traj.json    full trajectory (analyst)
    runs/jobs/<job>/<task_id>/t<trial>/meta.json    status/steps/tokens/wall

Resume-safe: a trial with a non-empty meta.json is skipped, so a killed job can
be re-run and only fills the holes. This is what lets the CONFIRM stage reuse
the SCREEN stage's trials instead of throwing them away.

THE WORKSPACE IS BUILT FROM `copy_files.txt`, NOTHING ELSE. Each task's own
frontier_eval/copy_files.txt lists the files the benchmark intends the solver to
see: the problem statement, the response schema, the payload stub, and images
where a task has them. `evaluate.py`, `solution.txt`, `rubrics.txt` and
`reference.txt` sit next to those in the pristine tree and are never copied. The
gateway additionally jails code_exec inside the workspace, because a shell can
reach anything the path checks do not cover.

The agent scaffold (third_party/archipelago/harness_eng) is imported through the
engine registry via RRSI_HARNESS_MODULE, so the code that runs here is exactly
the code the proposer edits and git diffs.

Usage:
  python3 bench/run_tasks.py --job baseline --n 4
  python3 bench/run_tasks.py --job r3A_screen --ids AB_01,XY_05 --n 1
"""

# 【模块说明（中文）】用 react_toolbelt agent 跑 EngDesign 任务，每题 k 个 trial。
# 一个 job = 一次（任务列表 x n_attempts）扫荡，产物进 runs/jobs/<job>/：
# payload.py 是被打分的交付物，traj.json 是完整轨迹（analyst 用），meta.json 是
# 状态/步数/token/耗时。可续跑：meta.json 非空的 trial 直接跳过，被杀的 job 重跑只补洞——
# 这让 CONFIRM 阶段能复用 SCREEN 阶段的 trial 而不是扔掉重跑。
# 工作区只按 copy_files.txt 构建：每题自己的 frontier_eval/copy_files.txt 列出基准
# 打算让解题者看到的文件（题面、响应 schema、payload 存根、图片）；evaluate.py、
# solution.txt、rubrics.txt、reference.txt 与它们同在原始树里但绝不复制。gateway
# 还把 code_exec 关进工作区牢笼——shell 能到达路径检查覆盖不到的任何地方。
# agent scaffold 经引擎注册表按 RRSI_HARNESS_MODULE 导入，所以这里跑的代码就是
# proposer 编辑、git diff 里的那份代码。

# argparse：命令行参数
import argparse
# asyncio：并发跑 trial（信号量限流）
import asyncio
# hashlib：md5 做确定性路由哈希
import hashlib
# json：写 traj/meta
import json
# os：环境变量与目录创建
import os
# shutil：构建/清理工作区
import shutil
# sys：sys.path 调整与退出
import sys
# time：计 wall 时间
import time
# traceback：异常时打印栈
import traceback
# Path：路径对象
from pathlib import Path

# ROOT：bench/ 的上一级（domains/eng/）
ROOT = Path(__file__).resolve().parent.parent
# 把官方 archipelago 引擎加入 sys.path（runner 注册表从这里导入）
sys.path.insert(0, str(ROOT.parent.parent / "third_party" / "archipelago"))
# 默认 harness 模块为 harness_eng.main（可演化 scaffold 的入口）
os.environ.setdefault("RRSI_HARNESS_MODULE", "harness_eng.main")
# 把域根加入 sys.path 以导入 config
sys.path.insert(0, str(ROOT))

# config：本域集中配置
import config  # noqa: E402
# AgentRunInput：引擎的 agent 运行输入模型
from runner.agents.models import AgentRunInput  # noqa: E402
# get_agent_impl：从注册表取 agent 实现（跑的就是被演化的 scaffold）
from runner.agents.registry import get_agent_impl  # noqa: E402
# SYSTEM_PROMPT / TASK_TEMPLATE：可演化 scaffold 的系统提示词与任务模板（proposer 可改）
from harness_eng.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    TASK_TEMPLATE,
)

# JOBS：作业根目录 runs/jobs/
JOBS = Path(config.RUNS) / "jobs"
# BENCH：基准任务树根（每题一个目录，含冻结 verifier）
BENCH = Path(config.BENCH_TASKS)


# ------------------------------------------------------------------- tasks ---
# —— 任务枚举与选择 ——
# 定义 all_tasks：枚举基准树里的全部任务，每题一条 {id, domain, dir}
def all_tasks() -> list[dict]:
    """One record per benchmark task: {id, domain, dir}."""
    # 结果列表
    out = []
    # 遍历两级目录（学科/任务），排序保证确定性
    for td in sorted(BENCH.glob("*/*")):
        # 有 frontier_eval/ 目录的才算任务（那是 harness 接线所在）
        if (td / "frontier_eval").is_dir():
            out.append({"id": td.name,
                        "domain": td.parent.name.replace("EngDesignOpen_", ""),
                        "dir": str(td)})
    # 返回任务记录列表
    return out


# 定义 select_tasks：按划分文件的任务集 + 显式 id 列表 + 数量上限筛任务
def select_tasks(set_name: str, ids: str, limit: int) -> list[dict]:
    # 划分文件路径
    split_p = Path(config.SPLIT_PATH)
    # 有划分文件：只取指定集合里的任务
    if split_p.is_file():
        wanted = {t["id"] for t in json.loads(split_p.read_text())[set_name]}
        tasks = [t for t in all_tasks() if t["id"] in wanted]
    else:
        # 没有划分文件：取全部任务
        tasks = all_tasks()
    # 显式给了 id 列表：再按交集过滤
    if ids:
        want = {i.strip() for i in ids.split(",") if i.strip()}
        tasks = [t for t in tasks if t["id"] in want]
    # limit>0 时截断；否则全量
    return tasks[:limit] if limit else tasks


# 定义 trial_dir：某 job 某任务第 trial 次的目录路径
def trial_dir(job: str, task_id: str, trial: int) -> Path:
    return JOBS / job / task_id / f"t{trial}"


# 定义 is_done：该 trial 是否已完成（meta.json 存在且非空）——续跑跳过的依据
def is_done(d: Path) -> bool:
    m = d / "meta.json"
    return m.is_file() and m.stat().st_size > 0


# --------------------------------------------------------------- workspace ---
# —— 工作区构建与交付物收割 ——
# 定义 build_workspace：把一题"agent 可见的切片"物化成工作区；每个 trial 全新一份
def build_workspace(task: dict, job: str, trial: int) -> Path:
    """Materialise the agent-visible slice of a task. Fresh every trial."""
    # 题目源目录（原始树）
    src = Path(task["dir"])
    # 工作区路径：WORKSPACE_BASE/<job>/<task>_t<trial>（按 job 与 trial 隔离）
    ws = Path(config.WORKSPACE_BASE) / job / f"{task['id']}_t{trial}"
    # 已存在（重跑）则先删干净
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    # 创建全新工作区
    ws.mkdir(parents=True)
    # 读该题自己的 copy_files.txt：基准允许解题者看到的文件清单（题面/schema/存根/图片）
    listed = (src / "frontier_eval" / "copy_files.txt").read_text().split()
    # 逐项复制进工作区
    for rel in listed:
        # 源与目标路径
        s = src / rel
        d = ws / rel
        # 先建父目录
        d.parent.mkdir(parents=True, exist_ok=True)
        # 目录整棵复制；单文件直接拷
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        elif s.is_file():
            shutil.copy2(s, d)
    # The stub must exist even if a task's copy_files did not list it: it is the
    # graded path, and an agent that has nowhere to write scores zero for a
    # reason that has nothing to do with its engineering.
    # 存根必须存在（哪怕 copy_files 没列它）：这是被打分的路径；agent 没地方写而得
    # 零分，与工程能力无关，会污染测量
    cand = ws / config.CANDIDATE_REL
    if not cand.exists():
        cand.parent.mkdir(parents=True, exist_ok=True)
        cand.write_text("PAYLOAD = {}\n")
    # 返回工作区路径
    return ws


# 定义 harvest：把被打分的交付物从工作区拷出来；返回 (文本, 是否真的写了东西)
def harvest(ws: Path, out_dir: Path) -> tuple[str, bool]:
    """Copy the graded deliverable out of the workspace. Returns (text, wrote)."""
    # 交付路径
    cand = ws / config.CANDIDATE_REL
    # 文本缺省为空
    text = ""
    # 交付文件存在才读（容忍非 UTF-8 字节）
    if cand.is_file():
        text = cand.read_text(encoding="utf-8", errors="replace")
    # 落盘为 trial 的 payload.py（评分对象）
    (out_dir / "payload.py").write_text(text)
    # "wrote" means the agent left something other than the untouched stub.
    # "wrote"意味着 agent 留下的不是没动过的存根（存根特征串还在=没写）
    stub_marker = "TODO: solve the task described in LLM_prompt.txt"
    return text, bool(text.strip()) and stub_marker not in text


# ------------------------------------------------------------------ trials ---
# —— 单 trial 执行 ——
# 定义 _total_tokens：从引擎的 usage 结构里尽量抠出总 token 数；抠不出返回 None
def _total_tokens(usage: dict) -> int | None:
    # 先试两个平铺字段名
    for key in ("total_tokens", "total_token_count"):
        if isinstance((usage or {}).get(key), int):
            return usage[key]
    # 再逐段求和（prompt/completion 等分项）
    tot, found = 0, False
    for v in (usage or {}).values():
        if isinstance(v, dict):
            for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
                if isinstance(v.get(key), int):
                    tot += v[key]
                    found = True
    # 找到过任何分项才返回和，否则 None（没有可信用量）
    return tot if found else None


# 定义 run_trial：跑单个 trial（协程）——建工作区、组输入、限时运行、落盘轨迹/交付物/meta
async def run_trial(task: dict, trial: int, job: str, sem: asyncio.Semaphore) -> dict:
    # 任务 id
    tid = task["id"]
    # 输出目录
    out_dir = trial_dir(job, tid, trial)
    # 信号量限流：最多 concurrent 个 trial 同时跑
    async with sem:
        # 已完成（续跑）：直接跳过
        if is_done(out_dir):
            return {"id": tid, "trial": trial, "status": "skipped"}
        # 建输出目录
        out_dir.mkdir(parents=True, exist_ok=True)
        # 建全新工作区
        ws = build_workspace(task, job, trial)

        # 用户消息：任务模板填入工作区/题面/schema/交付物四个路径
        user_msg = TASK_TEMPLATE.format(
            workspace=str(ws),
            prompt_file=str(ws / "LLM_prompt.txt"),
            schema_file=str(ws / "output_structure.py"),
            candidate_file=str(ws / config.CANDIDATE_REL),
        )
        # Same frozen model over several Vertex projects: spread trials so
        # throughput is not capped by one project's quota. Deterministic in
        # (task, trial) so resuming a job does not reshuffle routes -- champion
        # and candidate must draw the same route for the same task.
        # 同一冻结模型分布在多个 Vertex 项目上跑：摊开吞吐，不被单项目配额卡死。
        # 路由对 (task, trial) 确定性——续跑不重洗路由，冠军与候选同任务必须走同一路由
        eps = config.POLICY_ENDPOINTS
        # 任务 id 的 md5 哈希（确定性伪随机）
        h = int(hashlib.md5(tid.encode()).hexdigest()[:8], 16)
        # 选端点；没配端点列表就退回单模型
        ep = eps[(h + trial) % len(eps)] if eps else {
            "model": config.ORCHESTRATOR_MODEL, "extra_args": {}}
        # 组装 agent 运行输入：系统提示词 + 任务消息、gateway 地址、模型与参数、agent 配置
        run_input = AgentRunInput(
            trajectory_id=f"engd-{job}-{tid}-t{trial}",
            initial_messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            mcp_gateway_url=config.GATEWAY_URL,
            mcp_gateway_auth_token=None,
            orchestrator_model=ep["model"],
            orchestrator_extra_args={**config.ORCHESTRATOR_EXTRA_ARGS,
                                     **ep["extra_args"]},
            agent_config_values=dict(config.AGENT_CONFIG_VALUES),
        )

        # 从注册表取 react_toolbelt_agent 实现（即被演化的 scaffold）
        impl = get_agent_impl("react_toolbelt_agent")
        # 开始计时
        t0 = time.time()
        # 状态与输出缺省
        status, output = "error", None
        try:
            # 限时运行（超 TIMEOUT_PER_TASK 即杀——超时 trial 按白卷计 0，这是刻意设计）
            output = await asyncio.wait_for(impl(run_input),
                                            timeout=config.TIMEOUT_PER_TASK)
            status = str(output.status)
        except asyncio.TimeoutError:
            # 超时：状态记 timeout
            status = "timeout"
        except Exception as e:  # noqa: BLE001
            # 其他异常：状态记 error，打印栈（不中断其他 trial）
            status = "error"
            print(f"[error] {tid} t{trial}: {e!r}", flush=True)
            traceback.print_exc()
        # 该 trial 的 wall 时间
        wall = time.time() - t0

        # meta：任务/trial/状态/学科/耗时/模型/Vertex 项目
        meta = {"id": tid, "trial": trial, "status": status,
                "domain": task["domain"], "wall_s": round(wall, 1),
                "model": ep["model"],
                "vertex_project": ep["extra_args"].get("vertex_project")}
        # 有输出就落盘轨迹
        if output is not None:
            try:
                # 序列化输出对象
                dump = output.model_dump(mode="json")
                # 写 traj.json（analyst 的原料）
                (out_dir / "traj.json").write_text(
                    json.dumps(dump, indent=1, default=str))
                # 消息数与 token 用量记入 meta
                meta["n_messages"] = len(dump.get("messages") or [])
                meta["usage"] = dump.get("usage") or {}
                meta["total_tokens"] = _total_tokens(meta["usage"])
            except Exception as e:  # noqa: BLE001
                # 序列化失败记进 meta，不影响评分
                meta["traj_error"] = repr(e)

        # 收割交付物：文本与"是否真写了"
        text, wrote = harvest(ws, out_dir)
        # payload 字符数
        meta["payload_chars"] = len(text)
        # 白卷标记（评分时白卷按 0 计）
        meta["no_payload"] = not wrote
        # 写 meta.json——写完这一刻起该 trial 视为"已完成"（续跑跳过的标记）
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        # 删除工作区（产物已收割，不留残余）
        shutil.rmtree(ws, ignore_errors=True)
        # 打印完成行
        print(f"[done] {tid} t{trial} status={status} "
              f"payload={len(text)}c{' NONE' if not wrote else ''} "
              f"{wall:.0f}s", flush=True)
        # 返回 meta
        return meta


# 定义 main：整个扫荡的调度入口
async def main():
    # 命令行参数
    ap = argparse.ArgumentParser()
    # --job：作业名（必填）
    ap.add_argument("--job", required=True)
    # --set：任务集名（默认 evolve）
    ap.add_argument("--set", default="evolve")
    # --ids：逗号分隔的任务子集（默认空=全集）
    ap.add_argument("--ids", default="")
    # --n：每任务 trial 数（默认 N_ATTEMPTS=4）
    ap.add_argument("--n", type=int, default=config.N_ATTEMPTS)
    # --limit：任务数上限（0=不限）
    ap.add_argument("--limit", type=int, default=0)
    # --concurrency：并发（默认 MAX_CONCURRENT）
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENT)
    args = ap.parse_args()

    # 选任务
    tasks = select_tasks(args.set, args.ids, args.limit)
    # 一个都没有：退出码 2
    if not tasks:
        print("[run] no tasks selected")
        sys.exit(2)
    # 建工作区根与作业目录
    os.makedirs(config.WORKSPACE_BASE, exist_ok=True)
    (JOBS / args.job).mkdir(parents=True, exist_ok=True)

    # 全部 (任务, trial 槽位) 组合
    todo = [(t, k) for t in tasks for k in range(args.n)]
    # 其中尚未完成的（续跑只补洞）
    pending = [x for x in todo if not is_done(trial_dir(args.job, x[0]["id"], x[1]))]
    # 打印计划摘要
    print(f"[run] job={args.job} tasks={len(tasks)} x{args.n} "
          f"({len(pending)} pending) model={config.ORCHESTRATOR_MODEL} "
          f"gateway={config.GATEWAY_URL} conc={args.concurrency}", flush=True)

    # 并发信号量
    sem = asyncio.Semaphore(args.concurrency)
    # 并发执行全部槽位（已完成的在 run_trial 内部秒回 skipped）
    results = await asyncio.gather(*(run_trial(t, k, args.job, sem) for t, k in todo))

    # 按状态聚合结果
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
    # 白卷计数
    nopay = sum(1 for r in results if r.get("no_payload"))
    # 打印总结
    print(f"[run] SUMMARY {json.dumps(by_status)} no_payload={nopay}", flush=True)


# 脚本入口
if __name__ == "__main__":
    asyncio.run(main())

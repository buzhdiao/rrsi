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
# 模块说明:Harvey LAB 运行器,驱动内置(vendored)的官方 react_toolbelt 引擎。
# LAB 任务契约:任务 = tasks/<领域>/<工作流>[/<场景>]/{task.json, documents/};
# 把 documents/ 复制进临时工作区,代理把交付物写到工作区 output/;运行结束后把
# output/* 复制到 runs/<LABEL>/<任务>/output/,轨迹存为 runs/<LABEL>/<任务>/
# trajectory.json;已有非空产出的任务跳过(断点续跑);封闭工作区:网关不带
# web_search(LAB 契约)。harness 模块经 RRSI_HARNESS_MODULE(默认
# harness_workspace.main)从本检出的 third_party/archipelago 树导入,因此从候选
# worktree 运行本驱动即评测该候选的 harness。
"""Harvey LAB runner driving the vendored official react_toolbelt engine.

Harvey LAB task contract:

  - task = tasks/<area>/<workflow>[/<scenario>]/{task.json, documents/}
  - copy documents/ into a scratch workspace; agent writes deliverables to
    workspace output/
  - after the run, copy output/* -> runs/<LABEL>/<task_name>/output/ and the
    trajectory to runs/<LABEL>/<task_name>/trajectory.json
  - resume-skip tasks that already have non-empty output
  - closed workspace: gateway runs WITHOUT web_search (LAB contract)

The harness module is imported from this checkout's third_party/archipelago
tree via RRSI_HARNESS_MODULE (default harness_workspace.main), so running the
driver from a candidate worktree evaluates that candidate's harness.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import traceback

# 定位本目录;把 archipelago 树加进 sys.path 以导入 runner 包
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "third_party", "archipelago"))
os.environ.setdefault("RRSI_HARNESS_MODULE", "harness_workspace.main")

from runner.agents.models import AgentRunInput  # noqa: E402
from runner.agents.registry import get_agent_impl  # noqa: E402

# 各类运行参数(均可环境变量覆盖):LAB 根、模型标签、编排模型、网关地址、并发
HARVEY_LAB_ROOT = os.environ.get("HARVEY_LAB_ROOT", "harvey-labs")
MODEL_LABEL = os.environ.get("MODEL_LABEL", "opus48_base")
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "vertex_ai/claude-opus-4-8")
GATEWAY_URL = f"http://127.0.0.1:{os.environ.get('GATEWAY_PORT', '8991')}/mcp/"
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "6"))
TIMEOUT_PER_TASK = int(os.environ.get("TIMEOUT_PER_TASK", "3600"))
# 任务产出目录与临时工作区目录
JOBS_DIR = os.environ.get("RRSI_JOBS_DIR", os.path.join(HERE, "runs"))
RUNS = os.path.join(JOBS_DIR, MODEL_LABEL)
WORKSPACES = os.path.join(os.environ.get("RRSI_WORKSPACES", os.path.join(HERE, "workspaces")), MODEL_LABEL.replace("/", "__"))

# 代理运行配置:最大步数、总超时、单次 LLM 响应超时
AGENT_CONFIG_VALUES = {
    "max_steps": int(os.environ.get("MAX_STEPS", "250")),
    "timeout": int(os.environ.get("AGENT_TIMEOUT", "10800")),
    "llm_response_timeout": int(os.environ.get("LLM_RESPONSE_TIMEOUT", "600")),
}

# Official ReAct system prompt (react_toolbelt README), web_search mention
# removed: LAB is a closed-workspace benchmark (documents provided; no web).
# 官方 ReAct 系统提示(react_toolbelt README),删去 web_search 提法:LAB 是
# 封闭工作区基准(文档已提供,无网络)。
SYSTEM_PROMPT = """You are an AI assistant that completes tasks by reasoning and using tools.

## Think Before Acting

Before making tool calls, briefly explain your reasoning in 1-3 sentences:
- What you learned from the previous step
- What you're doing next and why

Don't over-explain. Be concise but show your thinking.

## Tools

**Always Available (Meta-Tools):**
- `todo_write` - Task planning: create/update todos. Takes `todos` array [{id, content, status}] and `merge` boolean.
- `toolbelt_list_tools` / `toolbelt_inspect_tool` / `toolbelt_add_tool` / `toolbelt_remove_tool` - Tool management
- `final_answer` - Submit your answer (status: completed/blocked/failed)

**Domain Tools:** Use `toolbelt_list_tools` to discover, then `toolbelt_add_tool` to add them. They include file reading/writing, `code_exec` (shell + a rich scientific Python for producing xlsx/docx/pptx/pdf/csv deliverables), and `read_pdf`.

## Workflow

1. Plan: Use `todo_write` to create todos for complex tasks
2. Discover: Use `toolbelt_list_tools` to find relevant tools
3. Execute: Work through todos, use `todo_write` with `merge=true` to update status
4. Complete: Call `final_answer` (all todos must be completed/cancelled first)

## Rules

- Update todo status with `todo_write`: set `in_progress` when starting, `completed` when done
- Show your work for calculations; use exact values without rounding unless told otherwise
- All file paths are absolute
- `final_answer` is rejected if todos are incomplete"""

# 任务包装提示模板:标题、指令、文档目录、输出目录、期望交付物(精确文件名)
# 及关键规则(封闭工作区、只写最终交付物、不许越出工作区、二进制格式用代码解析)
WRAPPER_TEMPLATE = """=== TASK ===
{title}

=== INSTRUCTIONS ===
{instructions}

=== DOCUMENTS FOLDER (read-only source materials) ===
{documents_dir}

=== OUTPUT DIRECTORY ===
{output_dir}

=== EXPECTED DELIVERABLES (write with these EXACT filenames) ===
{deliverables}

IMPORTANT:
- All source materials are files under {documents_dir}. There is no web access;
  everything you need is in the documents folder.
- Save ONLY the final deliverables to {output_dir}, using the exact filenames
  listed above. Do not save intermediate or temporary files there.
- You MUST only access files within {workspace}. Do NOT access any files or
  directories outside of this path.
- If a file cannot be read directly (e.g., .xlsx, .docx, .pptx), use code to
  extract and process its contents.
- If you encounter ambiguous or conflicting information, analyze the conflict,
  explain your reasoning, and justify the approach you choose."""


def find_tasks(subset_file: str | None) -> list[dict]:
    """Enumerate LAB tasks (both nesting depths). task name = path with '/'->'__'."""
    # 枚举 LAB 任务(两种嵌套深度都支持);任务名 = 相对路径中 '/' 换成 '__'
    tasks = []
    troot = os.path.join(HARVEY_LAB_ROOT, "tasks")
    for dirpath, dirnames, filenames in os.walk(troot):
        # 任务目录判据:含 task.json 且有 documents/ 子目录
        if "task.json" in filenames and os.path.isdir(os.path.join(dirpath, "documents")):
            rel = os.path.relpath(dirpath, troot)
            tasks.append({"id": rel, "name": rel.replace("/", "__"), "dir": dirpath})
    # 按 id 排序保证枚举顺序确定
    tasks.sort(key=lambda t: t["id"])
    if subset_file:
        # 指定子集文件(JSON 数组)时过滤
        keep = set(json.load(open(subset_file)))
        tasks = [t for t in tasks if t["id"] in keep]
    return tasks


def is_done(name: str) -> bool:
    # 断点续跑判断:output/ 目录存在且非空即视为已完成
    out = os.path.join(RUNS, name, "output")
    return os.path.isdir(out) and bool(os.listdir(out))


async def run_one(task: dict, sem: asyncio.Semaphore) -> dict:
    # 运行单个任务(受信号量限流):建工作区、构造输入、跑代理、回收产出
    name = task["name"]
    async with sem:
        if is_done(name):
            # 已有产出则跳过
            print(f"[skip] {name}", flush=True)
            return {"name": name, "status": "skipped"}

        # 读取任务定义
        tj = json.load(open(os.path.join(task["dir"], "task.json")))
        # 临时工作区:documents/(只读源材料)+ output/(交付物)
        workspace = os.path.join(WORKSPACES, name)
        ws_docs = os.path.join(workspace, "documents")
        ws_output = os.path.join(workspace, "output")
        if os.path.exists(workspace):
            shutil.rmtree(workspace)
        os.makedirs(workspace, exist_ok=True)
        shutil.copytree(os.path.join(task["dir"], "documents"), ws_docs)
        os.makedirs(ws_output, exist_ok=True)

        # 期望交付物 = task.json 中 deliverables 的键(缺省 response.md)
        deliverables = sorted((tj.get("deliverables") or {}).keys()) or ["response.md"]
        # 用模板构造任务提示(含精确文件名契约)
        user_msg = WRAPPER_TEMPLATE.format(
            title=tj.get("title", task["id"]),
            instructions=tj.get("instructions", ""),
            documents_dir=ws_docs,
            output_dir=ws_output,
            workspace=workspace,
            deliverables="\n".join(f"- {d}" for d in deliverables),
        )
        # 组装代理运行输入:系统提示 + 任务提示、网关地址、模型与配置
        run_input = AgentRunInput(
            trajectory_id=f"workspace-{name}"[:120],
            initial_messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            mcp_gateway_url=GATEWAY_URL,
            mcp_gateway_auth_token=None,
            orchestrator_model=ORCHESTRATOR_MODEL,
            orchestrator_extra_args={},
            agent_config_values=dict(AGENT_CONFIG_VALUES),
        )

        # 取官方 react_toolbelt 代理实现并运行
        impl = get_agent_impl("react_toolbelt_agent")
        status = "error"
        run_dir = os.path.join(RUNS, name)
        os.makedirs(run_dir, exist_ok=True)
        try:
            # 单任务限时;成功则落盘完整轨迹
            output = await asyncio.wait_for(impl(run_input), timeout=TIMEOUT_PER_TASK)
            status = str(output.status)
            with open(os.path.join(run_dir, "trajectory.json"), "w") as f:
                json.dump(output.model_dump(mode="json"), f, indent=2, default=str)
        except asyncio.TimeoutError:
            # 超时:标记并继续
            status = "timeout"
            print(f"[timeout] {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            # 其他异常:打印并继续下一个任务
            print(f"[error] {name}: {e!r}", flush=True)
            traceback.print_exc()

        # 回收产出:把工作区 output/* 复制进 runs/<任务>/output/
        n_files = 0
        if os.path.isdir(ws_output) and os.listdir(ws_output):
            final_out = os.path.join(run_dir, "output")
            os.makedirs(final_out, exist_ok=True)
            for item in os.listdir(ws_output):
                src = os.path.join(ws_output, item)
                dst = os.path.join(final_out, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
                n_files += 1

        # 清理临时工作区并汇报
        shutil.rmtree(workspace, ignore_errors=True)
        print(f"[done] {name} status={status} files={n_files}", flush=True)
        return {"name": name, "status": status, "n_output_files": n_files}


async def main():
    # 入口:解析过滤参数并并发运行全部任务
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None, help="JSON file: list of task ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    # 确保工作区与产出目录存在
    os.makedirs(WORKSPACES, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)
    # 枚举任务并可按子集/名称子串/数量上限过滤
    tasks = find_tasks(args.subset)
    if args.only:
        subs = [s for s in args.only.split(",") if s]
        tasks = [t for t in tasks if any(s in t["name"] for s in subs)]
    if args.limit:
        tasks = tasks[: args.limit]

    # 打印运行概要
    print(f"[driver] {len(tasks)} tasks | model={ORCHESTRATOR_MODEL} "
          f"label={MODEL_LABEL} | gateway={GATEWAY_URL} "
          f"| concurrency={MAX_CONCURRENT}", flush=True)
    # 信号量限流并发跑全部任务,然后按状态汇总
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    results = await asyncio.gather(*(run_one(t, sem) for t in tasks))
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"[driver] SUMMARY {json.dumps(by_status)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

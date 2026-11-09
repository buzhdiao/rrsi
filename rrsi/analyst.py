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

# Google LLC 的 Apache-2.0 许可证头（仓库标准文件头，保留原文）。
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
"""Analyze(H_t, D_evolve) -> F_t  (Algorithm 1, line 1).

The batch analyst never reads raw traces. It dispatches read-only digesters
over the incumbent's own evaluation and aggregates their digests into a
three-lens report:

  failure_modes    what blockers lost points, clustered across tasks
  capability_gaps  what the agent tried to do but could not
  success_habits   what let passing tasks finish cleanly (kept across rounds
                   as a regression guard for the proposer)
"""
# 【模块说明（中文）】Analyze(H_t, D_evolve) -> F_t（Algorithm 1 第 1 行）。
# 批量分析师（batch analyst）自己从不读原始轨迹：它派发只读的 digester 子代理
# 去调查现任 harness 自己评估出的轨迹，再把各摘要聚合成三透镜报告 F_t：
#   failure_modes    什么阻塞机制丢了分（跨任务聚类）
#   capability_gaps  agent 想做什么却做不到
#   success_habits   什么让通过的任务干净收尾（跨轮保留，作为提案者的"回归护栏"）

# 延迟类型注解求值
from __future__ import annotations

# json：报告/摘要的序列化
import json
# ThreadPoolExecutor：并行跑 digester 子任务
from concurrent.futures import ThreadPoolExecutor
# Path：摘要文件落盘路径
from pathlib import Path

# digest_task：单个轨迹的只读调查子代理（digester.py）
from .digester import digest_task
# generate：统一 LLM 客户端
from .llm import generate

# MAX_TURNS：分析师代理的最大交互轮数（防失控循环）
MAX_TURNS = 30
# DIGEST_PARALLELISM：digest_many 并行度（6 个摘要子代理同时跑）
DIGEST_PARALLELISM = 6

# SYSTEM_TMPL：分析师的系统提示词（注入 domain_brief）
SYSTEM_TMPL = """You are the batch analyst in a harness-evolution loop. A frozen
policy LLM, driven by an evolvable scaffold, ran the evolve set of a benchmark;
some trials scored well, some did not. Your job: produce a three-lens analysis
report that a harness engineer will act on.

{domain_brief}

You do NOT read traces yourself. You dispatch read-only digester subagents
that investigate one trace each and return structured digests. Budget their
use: digest FAILED / low-scoring tasks first (failure lens), prioritising
coverage of every suspected failure cluster over digesting every failure;
digest 3-5 representative SUCCESSFUL tasks (success lens, prefer ones that
resemble a big failure cluster); use the capability_gap lens or follow-up
questions where a failure digest hints the agent was blocked by the
environment or the scaffold's plumbing rather than by its own judgment.

Actions (STRICT JSON, one per turn):
  {{"action": "digest_many", "requests": [
      {{"task_id": "<id exactly as it appears in the task table>",
       "lens": "failure|capability_gap|success",
       "questions": ["optional targeted questions"]}}, ...]}}
      (up to 8 per call, they run in parallel; each returns a digest)
  {{"action": "report",
   "failure_modes": [
     {{"mode": "snake_case_label", "n_tasks": 0, "affected_tasks": [...],
      "description": "generalized mechanism, entity-free",
      "needed_instead": "...",
      "representative_evidence": [{{"task_id": "...", "where": "...", "quote": "..."}}]}}],
   "capability_gaps": [
     {{"gap": "snake_case_label", "n_tasks": 0, "affected_tasks": [...],
      "description": "what the agent could not do and why, entity-free",
      "representative_evidence": [...]}}],
   "success_habits": [
     {{"habit": "snake_case_label", "n_tasks": 0,
      "description": "the reusable behavior, entity-free",
      "risk_if_broken": "what regresses if a harness change disrupts it"}}]}}

Aggregation rules:
1. MERGE digests describing the same underlying mechanism even if worded
   differently; SPLIT a label that covers two distinct mechanisms.
2. RANK failure_modes by total impact (number of tasks weighted by how much
   score the mode costs on each).
3. Descriptions must be entity-free and task-agnostic (no task names, no
   subject-matter facts, no task-specific values); evidence quotes may
   contain them.
4. Keep prior mode names when the same mechanism recurs (stable naming);
   prior names are provided in the context.
5. Do NOT prescribe code changes and do NOT attribute blame to model vs
   scaffold.
6. If digests contradict or a cluster is unclear, dispatch follow-up digests
   with targeted questions before reporting. Report once, at the end."""
# 【提示词要点（中文）】分析师在 harness 进化循环中做批量分析：冻结策略模型在可演化
# 脚手架驱动下跑完了 evolve 集。它自己不读轨迹，而是派发只读 digester 子代理。
# 摘要预算策略：失败透镜优先（覆盖每个疑似失败簇，而非每个失败）；成功透镜选 3-5 个
# 代表性成功任务；怀疑是环境/脚手架管道问题时用 capability_gap 透镜。
# 两个动作：digest_many（一次最多 8 个并行请求）与 report（三透镜结构化报告）。
# 聚合规则：同机制合并、一标签两机制要拆开；按总影响排序；描述必须去实体化、任务无关；
# 同机制复现时保持命名稳定；不开药方、不归责模型或脚手架；有矛盾先追加调查再报告。


def prepare_rendered(domain, traces: dict, rendered_dir: Path) -> None:  # 定义 prepare_rendered：把轨迹渲染成文本文件
    """把本轮选中的轨迹逐条渲染成明文文件，供 digester 子代理只读调查；
    已渲染过的任务直接跳过（断点续跑不重复劳动）。"""
    rendered_dir.mkdir(parents=True, exist_ok=True)  # 建渲染目录 runs/<domain>/r<t>/analysis/rendered/
    for tid, rec in traces.items():  # 遍历每个选中的轨迹
        out = rendered_dir / f"{tid}.txt"  # 目标文件 <task_id>.txt
        if not out.exists():  # 已存在则跳过（断点续跑不重复渲染）
            out.write_text(domain.render_trace(rec, detail=True))  # detail=True：展开每条消息的内容上限，digester 要看工具实际返回


def task_table(domain, traces: dict, per_task: dict) -> str:  # 定义 task_table：构造任务表（分析师的主上下文）
    """构造任务表：每任务一行（行文本由 domain 适配器生成），按分数升序
    拼接成整段文本——最差的任务排在最上面。"""
    rows = []  # (分数, 行文本) 对
    for tid, rec in traces.items():  # 遍历每条轨迹
        tr = per_task.get(tid)  # 该任务的 TaskResult
        rows.append((tr.mean if tr else 0.0, domain.task_row(tid, rec, tr)))  # 行文本由 domain 生成（分数/状态/步骤数）
    # 按分数升序排——最差任务在最上面，吸引分析师优先处理
    return "\n".join(r for _, r in sorted(rows, key=lambda x: x[0]))  # 返回排好序的任务表文本


def analyze(domain, traces: dict, per_task: dict, round_dir: Path,  # 定义 analyze：Analyze(H_t, D) 的主入口
            prior_modes: list | None = None, prior_habits: list | None = None,
            model: str | None = None) -> dict:
    """Analyze(H_t, D) -> F_t 的主入口：渲染轨迹后进入至多 MAX_TURNS 轮的
    JSON 动作交互——digest_many 派发摘要子代理、report 交出三透镜报告；
    prior_modes/prior_habits 只提供历史命名，让跨轮的模式名保持稳定。"""
    rendered_dir = round_dir / "analysis" / "rendered"  # 渲染轨迹目录
    prepare_rendered(domain, traces, rendered_dir)  # 先渲染全部轨迹到磁盘
    digests_dir = round_dir / "analysis" / "digests"  # 摘要落盘目录（供 load_digests 复用）
    digests_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    system = SYSTEM_TMPL.format(domain_brief=domain.briefs["analyst"])  # 系统提示词注入 domain 段落
    context = "\n\n".join([  # 拼装首轮上下文
        "=== TASK TABLE (one rendered trace per task; lowest scores first) ===",  # 任务表（分数升序）
        task_table(domain, traces, per_task),
        "=== PRIOR FAILURE-MODE NAMES (for stable naming) ===",  # 历史失败模式名：让跨轮命名稳定、可对照
        json.dumps([{"mode": m.get("mode"), "description": m.get("description")}  # 只给名字与描述（省上下文）
                    for m in (prior_modes or [])], ensure_ascii=False, indent=1),
        "=== PRIOR SUCCESS-HABIT NAMES (for stable naming) ===",  # 历史成功习惯名
        json.dumps([{"habit": h.get("habit"), "description": h.get("description")}  # 同上
                    for h in (prior_habits or [])], ensure_ascii=False, indent=1),
        "=== TASK ===",  # 任务说明
        "Dispatch digesters, then produce the report. First action:",  # 指令：先派摘要、后报告
    ])
    transcript = ""  # 交互记录（滚动追加）
    n_digests = 0  # 已完成摘要计数（写进报告做统计）
    for _ in range(MAX_TURNS):  # 最多 30 轮交互
        prompt = (context + "\n\n=== INTERACTION LOG ===\n" + transcript  # 每轮重发完整上下文 + 交互日志
                  + "\nReply with exactly one JSON action object.")  # 硬约束：每轮恰一个 JSON 动作
        raw = generate(prompt, system=system, json_only=True, model=model)  # 调用搜索模型
        try:  # 解析动作
            act = json.loads(raw)  # 直接解析
        except json.JSONDecodeError:  # 坏 JSON
            transcript += f"\n[you] {raw[:300]}\n[result] ERROR: invalid JSON"  # 反馈错误让它重试
            continue  # 下一轮
        if isinstance(act, list):  # 输出成了数组
            act = next((x for x in act if isinstance(x, dict)), None)  # 取第一个对象元素
        if not isinstance(act, dict):  # 仍不是对象
            transcript += ("\n[you] (non-object)\n[result] ERROR: reply with "  # 纠正格式
                           "EXACTLY ONE JSON action object")
            continue  # 下一轮
        a = act.get("action")  # 取动作名
        if a == "report":  # 动作一：交报告（终止动作）
            report = {k: act.get(k) or [] for k in  # 三个透镜缺失时补空列表
                      ("failure_modes", "capability_gaps", "success_habits")}
            report["failure_modes"].sort(key=lambda m: -(m.get("n_tasks") or 0))  # 失败模式按受影响任务数降序——高影响排前
            report["n_digests"] = n_digests  # 记录调查了多少条轨迹
            return report  # 返回 F_t
        if a == "digest_many":  # 动作二：批量派发摘要
            reqs = (act.get("requests") or [])[:8]  # 截到 8 个（硬上限）
            for r in reqs:  # 规范化 task_id
                if isinstance(r, dict) and r.get("task_id") is not None:  # 防御 None
                    r["task_id"] = str(r["task_id"])  # 统一转字符串（LLM 可能给数字 id）
            valid = [r for r in reqs if isinstance(r, dict) and r.get("task_id") in traces]  # 过滤掉不存在的任务 id
            if reqs and not valid:  # 全部无效
                print(f"[analyst] WARNING all {len(reqs)} requests had unknown "  # 打警告并提示合法 id 样例（帮助诊断）
                      f"task_ids; known ids look like {next(iter(traces))!r}",
                      flush=True)

            def run(r):  # 单个摘要任务的闭包
                d = digest_task(rendered_dir, r["task_id"], r.get("lens", "failure"),  # 起 digester 子代理（默认失败透镜）
                                domain.briefs["digester"], r.get("questions"),
                                model=model)
                (digests_dir / f"{r['task_id']}_{r.get('lens', 'failure')}.json"  # 摘要按任务+透镜命名落盘
                 ).write_text(json.dumps(d, ensure_ascii=False, indent=1))
                return d  # 返回摘要内容
            with ThreadPoolExecutor(max_workers=DIGEST_PARALLELISM) as ex:  # 并行度 6：子代理之间互相独立
                digests = list(ex.map(run, valid))  # 阻塞收集全部摘要
            n_digests += len(digests)  # 累加计数
            result = json.dumps(digests, ensure_ascii=False)  # 摘要以 JSON 文本回填交互日志
            if len(reqs) - len(valid):  # 有请求被跳过
                result += f"\n[{len(reqs) - len(valid)} requests skipped: unknown task_id]"  # 告知跳过数与原因
            print(f"[analyst] {len(digests)} digests done (total {n_digests})",  # 控制台进度
                  flush=True)
            transcript += (f"\n[you] digest_many ({len(valid)} requests)\n"  # 交互日志记录本次派发
                           f"[result] {result}")
        else:  # 未知动作
            transcript += (f"\n[you] {json.dumps(act)[:300]}\n"  # 反馈错误
                           f"[result] ERROR: unknown action {a}")
    return {"failure_modes": [], "capability_gaps": [], "success_habits": [],  # 轮数耗尽：返回空报告（不编造）
            "error": "analyst hit max turns", "n_digests": n_digests}


def load_digests(round_dir: Path) -> list:  # 定义 load_digests：从磁盘读回本轮全部摘要
    """读回本轮落盘的全部摘要 JSON（按文件名排序，保证顺序稳定可复现），
    供下游把摘要喂给提案者；目录不存在则返回空列表。"""
    dg = round_dir / "analysis" / "digests"  # 摘要目录
    if dg.exists():  # 存在
        return [json.loads(p.read_text()) for p in sorted(dg.glob("*.json"))]  # 按文件名排序读入全部摘要（排序保证顺序稳定可复现）
    return []  # 没有目录（如断点续跑绕过了分析）返回空列表

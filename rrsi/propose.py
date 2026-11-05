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
"""The regularized proposer P_reg(. | H_t, F_t, L_t, b_t, E_t) of Algorithm 1.

A Claude Opus 4.8 coding agent that edits the harness IN PLACE inside a
candidate's worktree through a strict-JSON action protocol and finishes with
done(edits=[...]). Every edit is tagged with (component l', hypothesis h');
the diff d' is taken from git afterwards. The agent is conditioned on:

  F_t   the analyst's three-lens report, the per-task digests and read-only
        access to the rendered trajectories
  L_t   the edit history (component, hypothesis, Delta S, Delta C, accepted)
  b_t   the annealed edit budget: at most b_t independent edits (||z_t||_0 <= b_t)
  E_t   exploration directives (stall flag, untried components, reserved slots)
  B_t   components to prune, with the accepted machinery to remove

Budget, reserved exploration slots and tagging are enforced in done(): an
over-budget or untagged submission is bounced back, never accepted.
"""
# 【模块说明（中文）】Algorithm 1 的正则化提案者 P_reg(. | H_t, F_t, L_t, b_t, E_t)。
# 一个 Claude Opus 4.8 编码代理：在候选 worktree 内"就地"编辑 harness，走严格 JSON
# 动作协议，以 done(edits=[...]) 收尾。每个编辑打上 (组件 l', 假设 h') 标签；
# diff d' 事后从 git 取。代理被条件化于：
#   F_t  分析师三透镜报告 + 每任务摘要 + 渲染轨迹的只读访问
#   L_t  编辑历史（组件、假设、ΔS、ΔC、是否被接受）
#   b_t  退火编辑预算：至多 b_t 个独立编辑（||z_t||_0 <= b_t）
#   E_t  探索指令（停滞标志、未尝试组件、保留槽位）
#   B_t  待修剪组件及其要拆除的现任机制
# 预算、保留探索槽位与标签都在 done() 里强制执行：超预算或未打标签的提交被打回，
# 绝不直接接受。

# 延迟类型注解求值
from __future__ import annotations

# json：动作协议与上下文 JSON 化
import json
# os：路径分隔符（Workspace 的囚笼判断）
import os
# re：轨迹行号解析
import re
# Path：harness 目录路径
from pathlib import Path

# K：组件词表（done() 校验编辑标签）
from .components import K
# generate：统一 LLM 客户端
from .llm import generate

# MAX_TURNS：提案者代理最大交互轮数
MAX_TURNS = 40
# MAX_EDITS：单候选的文件操作硬上限（防失控写爆 worktree）
MAX_EDITS = 80
# TRACE_READ_CAP：单次轨迹读取的字符上限（上下文保护）
TRACE_READ_CAP = 60_000

# SYSTEM_TMPL：提案者的系统提示词（角色设定 + 严格 JSON 动作协议 + done() 契约 + 宪法）
SYSTEM_TMPL = """You are a harness engineer agent. You directly modify the source
code of an LLM-agent scaffold (the "harness") to fix recurring failure modes
observed on an evolve set of tasks. The policy LLM is frozen and is a DIFFERENT
model from you: do not assume it shares your capabilities, habits or judgment.
Improve the harness from ITS perspective, using the trajectories as evidence of
how it actually behaves. The ONLY thing you can change is the scaffold code in
your working directory; the tool environment, the grader and the task set are
frozen.

{domain_brief}

You interact through a STRICT JSON protocol: reply with EXACTLY ONE action
object per turn, nothing else. Actions:
  {{"action": "list_files"}}
  {{"action": "read_file", "path": "relative/path"}}
  {{"action": "list_traces"}}
      (this round's traces: task_id, score, status, steps, modes hit)
  {{"action": "read_trace", "task_id": "<id from the task list>", "from_step": 30,
   "to_step": 60, "detail": true}}
      (READ-ONLY rendered trajectory segment: the raw evidence behind the
      failure modes. Step range optional. "detail": true expands per-message
      caps so you can see what tool results actually contained. The grading
      verdict is always included.)
  {{"action": "edit_file", "path": "p", "old": "exact substring", "new": "replacement"}}
      (old must occur EXACTLY ONCE in the file)
  {{"action": "write_file", "path": "p", "content": "full file content"}}
      (for NEW files only; never overwrite an existing file this way. Paths are
      relative to the harness package. A new module must be imported from the
      entry module to do anything, and it IS included in the reviewed diff.)
  {{"action": "done", "summary": "one-line summary of this candidate",
   "edits": [
     {{"id": "C1",
      "component": "one of: {components}",
      "hypothesis": "one sentence: the mechanism and WHY it should move the score",
      "targets_mode": "failure mode / capability gap / habit it targets",
      "why_not_lower_lever": "why a plain instruction edit would NOT fix this
          (or, for a prompt edit, why prose IS the right lever here)",
      "trigger_condition": "the exact, checkable condition under which the
          mechanism activates ('always' is almost never right)",
      "predicted_affected": ["<task id>", "..."],
      "retroactive_check": "three-part counterfactual: (corrective) which cited
          failing tasks would have moved had this existed, walking the actual
          trajectory; (preservative) which success habits could this disrupt
          and why it won't; (transfer) why it generalizes beyond this evolve set",
      "regression_risk": "what could break outside predicted_affected"}},
     ...]}}

THERE IS NO ABORT ACTION. You must ship a candidate. A round that ships
nothing tests nothing: the history records what was MEASURED, and a mechanism
you declined to build has no measurement behind it. If your best idea violates
a hard rule, it is not your best idea; construct a different one, preferably on
a component the history shows was never exercised.

done() contract:
- An edit = ONE independent, attributable change (it works on its own and can
  answer "which tasks will it move" by itself). Dependent parts are ONE edit.
  Ship at most THIS ROUND'S EDIT BUDGET b_t given in the context, never more.
  Ship fewer if the evidence supports fewer.
- Every edit names its `component` from the fixed vocabulary above. The tag
  is validated against the diff; a mislabelled edit is re-tagged from the diff.
- If the context says a RESERVED EXPLORATION SLOT applies to you, at least one
  edit must be on one of the listed never-exercised components.
- If the context lists COMPONENTS TO PRUNE that hold accepted machinery, an
  edit that removes that machinery is a legitimate edit (component = the
  pruned component, hypothesis = "prune: ..."). Prefer it when the evidence
  says the machinery stopped earning its place.
- predicted_affected lists CONCRETE task ids from this round's traces.
  Predictions are checked against the evaluation and your hit/miss record
  (scoreboard) is shown back to you; over-claiming counts against you.

You must follow the constitution (SKILL.md) in the context: no task-specific
entities, names or values anywhere in code, prompts or state; every edit
targets a reported failure mode, capability gap or habit; keep each edit's
diff scoped to its mechanism; the code runs unattended on every task in the
set, so an unhandled exception kills the whole candidate. The interface
contract (module entry points, class names, trajectory output schema) stays
unchanged; model name, step budget and timeouts are injected externally so
editing them has no effect. All code and comments in English."""
# 【提示词要点（中文）】提案者是"harness 工程师"：直接修改脚手架源码去修复 evolve 集上
# 反复出现的失败模式。策略模型是冻结的且与你不同——从它的视角出发改 harness；
# 唯一可改的是工作目录里的脚手架代码，工具环境/判分器/任务集全部冻结。
# 动作协议：list_files / read_file / list_traces / read_trace（只读轨迹切片）/
# edit_file（old 必须恰好出现一次）/ write_file（仅限新文件）/ done。
# 没有 abort 动作——必须交付候选：什么都不交付就什么都测不到。done 契约：一个编辑 =
# 一个独立可归因的改动；至多 b_t 个；每个编辑必须声明组件标签（会按 diff 重校）；
# 保留探索槽位必须落在未尝试组件上；拆除 B_t 机制是合法编辑；预测会与评估对账并回显
# 命中率——过度声称要扣分。必须遵守宪法（无任务专属实体、每编辑对准失败模式、
# 接口契约不变、模型名/步数预算/超时外部注入改了也无效、代码与注释全英文）。


class Workspace:  # Workspace 类：提案者的文件操作层——路径囚禁在候选 harness 目录内
    """Path-jailed file operations inside one candidate's harness directory."""
    # docstring：一个候选 harness 目录内的"路径囚禁"文件操作

    def __init__(self, harness_dir: Path, source_exts: set):  # 构造：harness 目录 + 允许新建的扩展名
        self.dir = Path(harness_dir).resolve()  # resolve 消解符号链接，囚笼判断基于真实路径
        self.exts = source_exts  # domain 给出的源码扩展名白名单

    def _safe(self, rel: str) -> Path:  # 定义 _safe：相对路径 -> 囚笼内的绝对路径
        rel = str(rel).lstrip("/")  # 去掉开头的 /（防绝对路径注入）
        name = self.dir.name  # harness 目录名
        if rel == name or rel.startswith(name + "/"):  # 提案者可能带上了目录名前缀
            rel = rel[len(name):].lstrip("/")  # 容错地剥掉它
        p = (self.dir / rel).resolve()  # 拼接并 resolve（消解 ..）
        if not str(p).startswith(str(self.dir) + os.sep) and p != self.dir:  # 解析后越出囚笼
            raise ValueError(f"path escapes harness dir: {rel}")  # 拒绝（防 ../ 逃逸写任意文件）
        return p  # 返回安全路径

    def list_files(self) -> str:  # 定义 list_files：列出全部文件（含大小）
        return "\n".join(f"{p.relative_to(self.dir)} ({p.stat().st_size}B)"  # 相对路径 + 字节数
                         for p in sorted(self.dir.rglob("*"))  # 递归遍历并排序
                         if p.is_file() and "__pycache__" not in p.parts)  # 跳过缓存目录

    def read(self, rel: str) -> str:  # 定义 read：读文件全文
        p = self._safe(rel)  # 过囚笼
        return p.read_text() if p.exists() else f"ERROR: no such file {rel}"  # 不存在返回错误占位（让代理自己纠正）

    def edit(self, rel: str, old: str, new: str) -> str:  # 定义 edit：精确子串替换（edit_file 动作）
        p = self._safe(rel)  # 过囚笼
        if not p.exists():  # 文件不存在
            return f"ERROR: no such file {rel}"  # 错误
        src = p.read_text()  # 读原文
        n = src.count(old)  # old 出现次数
        if n == 0:  # 找不到
            return "ERROR: old string not found (must match exactly, including whitespace)"  # 错误：要求精确匹配
        if n > 1:  # 出现多次
            return f"ERROR: old string occurs {n} times; provide a longer unique context"  # 错误：要求更长唯一上下文（防改错位置）
        p.write_text(src.replace(old, new, 1))  # 恰好替换第一处
        return f"OK: edited {rel}"  # 成功

    def write(self, rel: str, content: str) -> str:  # 定义 write：新建文件（write_file 动作）
        p = self._safe(rel)  # 过囚笼
        if p.suffix not in self.exts:  # 扩展名不在白名单
            return f"ERROR: extension {p.suffix} not allowed"  # 拒绝（防写二进制/脚本）
        if p.exists():  # 文件已存在
            return f"ERROR: {rel} exists; use edit_file"  # 拒绝覆盖——已有文件必须走 edit
        p.parent.mkdir(parents=True, exist_ok=True)  # 父目录就绪
        p.write_text(content)  # 写入新文件
        return f"OK: created {rel}"  # 成功

    def dump(self) -> str:  # 定义 dump：整包源码倾倒（进入提案者上下文的 H_t）
        parts = []  # 文件片段列表
        for p in sorted(self.dir.rglob("*")):  # 递归遍历
            if p.is_file() and p.suffix in self.exts and "__pycache__" not in p.parts:  # 只倒白名单扩展名的源码
                parts.append(f"===== FILE: {p.relative_to(self.dir)} =====\n{p.read_text()}")  # 带文件名分隔头的全文
        return "\n\n".join(parts)  # 拼成一个大文本


def _list_traces(domain, traces: dict, per_task: dict, findings: list) -> str:  # 定义 _list_traces：任务表（list_traces 动作的输出）
    modes_by_task: dict = {}  # task_id -> 命中的失败模式标签列表
    for f in findings or []:  # 遍历分析师摘要
        label = f.get("blocker") or f.get("wanted") or f.get("lens") or "?"  # 取最能代表该摘要的标签
        modes_by_task.setdefault(str(f.get("task_id")), []).append(str(label)[:80])  # 按任务聚合（截 80 字符）
    rows = []  # (分数, 行) 列表
    for tid, rec in traces.items():  # 遍历本轮轨迹
        tr = per_task.get(tid)  # 该任务的结果
        rows.append((tr.mean if tr else 0.0,  # 排序键：任务均分
                     domain.task_row(tid, rec, tr)  # domain 生成的行摘要
                     + f" | modes={modes_by_task.get(tid, [])}"))  # 追加命中模式
    # 分数升序——最差任务在最上面，吸引提案者优先处理；无轨迹时返回占位提示
    return "\n".join(r for _, r in sorted(rows, key=lambda x: x[0])) or \
        "ERROR: no traces available"  # 空占位


def _read_trace(domain, traces: dict, task_id, from_step, to_step, cache: dict,  # 定义 _read_trace：只读轨迹切片（read_trace 动作）
                detail: bool) -> str:
    task_id = str(task_id)  # 规范化 id
    rec = traces.get(task_id)  # 查轨迹记录
    if rec is None:  # 不存在
        return f"ERROR: no trace for {task_id} (use list_traces for valid ids)"  # 提示先 list_traces
    key = (task_id, bool(detail))  # 缓存键：任务 + 明细级别
    if key not in cache:  # 未缓存
        cache[key] = domain.render_trace(rec, detail=bool(detail))  # 渲染一次并缓存（同一轨迹可能被多次切片）
    rendered = cache[key]  # 取渲染文本
    if from_step is None and to_step is None:  # 未给步数范围
        return rendered[:TRACE_READ_CAP] + (  # 直接返回并截断
            f"\n...[capped at {TRACE_READ_CAP} chars; use from_step/to_step]"  # 提示可用范围参数
            if len(rendered) > TRACE_READ_CAP else "")
    lo, hi = int(from_step or 0), int(to_step or 10**9)  # 解析范围（缺省 0..无穷）
    keep, in_grades = [], False  # 保留行；是否处于判分区
    for line in rendered.splitlines():  # 逐行过滤
        if line.startswith("=== ") and ("GRAD" in line or "VERIFIER" in line):  # 判分/verifier 段开始
            in_grades = True  # 该段永远保留（判分结论是关键证据）
        m = re.match(r"\[step (\d+)\]", line)  # 解析行首步号
        if in_grades or m is None or lo <= int(m.group(1)) <= hi:  # 判分区行、无步号行、或步号在范围内则保留
            keep.append(line)
    out = "\n".join(keep)  # 拼回
    return out[:TRACE_READ_CAP] + (  # 截断保护
        f"\n...[capped at {TRACE_READ_CAP} chars]" if len(out) > TRACE_READ_CAP else "")


def propose(domain, harness_dir: Path, report: dict, history_rows: list,  # 定义 propose：提案者主入口（一个候选变体的起草）
            skill_md: str, patterns_md: str, budget: int, explore: dict,
            reserved_slot: bool, prune_set: list, traces: dict | None = None,
            per_task: dict | None = None, findings: list | None = None,
            scoreboard: list | None = None, variant_brief: str = "",
            repair_brief: dict | None = None, model: str | None = None) -> dict:
    """Run the proposer agent for one candidate variant. Edits files in place
    under harness_dir. Returns {status, edits, summary, n_edits, log}."""
    # docstring：为一个候选变体运行提案者代理。在 harness_dir 下就地编辑文件。
    # 返回 {status, edits, summary, n_edits, log}
    ws = Workspace(harness_dir, domain.source_exts)  # 建立囚禁式文件操作层
    traces = traces or {}  # 空值兜底
    per_task = per_task or {}  # 空值兜底
    render_cache: dict = {}  # 轨迹渲染缓存（本变体内复用）
    system = SYSTEM_TMPL.format(domain_brief=domain.briefs["proposer"],  # 系统提示词注入 domain 段落与组件词表
                                components=" | ".join(K))
    stable = "\n\n".join([  # 稳定前缀块：宪法 + 模式库——送 LLM 时打 ephemeral 缓存标记省钱
        "=== CONSTITUTION (SKILL.md) ===", skill_md,
        "=== PATTERN LIBRARY (PATTERNS.md) ===", patterns_md,
    ])
    explore_text = explore.get("text", "")  # 探索指令文本（E_t）
    if reserved_slot:  # 本变体持有保留探索槽位
        explore_text += ("\n\nRESERVED EXPLORATION SLOT: this variant holds one. At "  # 追加硬性提醒
                         "least one of your edits MUST be on a never-exercised "
                         f"component from: {explore.get('untried')}.")
    prune_text = (json.dumps(prune_set, ensure_ascii=False, indent=1)  # B_t 序列化（含要拆除的现任机制明细）
                  if prune_set else "(none)")  # 空则显式写 (none)
    context = "\n\n".join([  # 拼装大上下文（每轮重发）
        *(["=== THIS VARIANT'S BRIEF ===", variant_brief] if variant_brief else []),  # 变体简报（多候选独立性说明）
        "=== EDIT HISTORY L_t (every measured edit: component, hypothesis, "  # 编辑历史：被否证的假设不要重画
        "Delta S, Delta C, accepted). A rejected mechanism is negative evidence; "
        "do not redraw it unchanged. An accepted one carries the gain it "
        "produced; refine what has known credit, not what merely preceded a "
        "rise. ===",
        json.dumps(history_rows, ensure_ascii=False, indent=1),
        "=== ATTRIBUTION SCOREBOARD (how past edits' predictions fared; "  # 归因记分板：预测命中率与未预测的回退
        "unpredicted_regressions are tasks an edit likely broke) ===",
        json.dumps(scoreboard or [], ensure_ascii=False, indent=1),
        "=== EXPLORATION DIRECTIVES E_t ===", explore_text,  # 探索指令
        "=== COMPONENTS TO PRUNE B_t (exercised, no strictly improving edit in "  # 修剪集 B_t
        "the recent window; remove the accepted machinery listed, it has "
        "stopped earning its place) ===", prune_text,
        "=== THREE-LENS ANALYSIS REPORT F_t (failure modes ranked; capability "  # 分析师报告 F_t
        "gaps often need tool/plumbing fixes; success_habits are behaviors "
        "your change MUST NOT break) ===",
        json.dumps(report, ensure_ascii=False, indent=1),
        "=== PER-TASK DIGESTS (evidence anchors; use read_trace for raw evidence) ===",  # 每任务摘要（证据锚点）
        json.dumps(findings or [], ensure_ascii=False, indent=1),
        "=== CURRENT HARNESS SOURCE H_t ===", ws.dump(),  # 现任 harness 全文
        "=== THIS ROUND'S EDIT BUDGET b_t ===",  # 本轮编辑预算 b_t（退火说明）
        f"You may ship AT MOST {budget} independent edit(s) in this candidate "
        f"(the budget anneals over the run: early rounds explore, late rounds "
        f"make single attributable changes). Ship fewer if the evidence "
        f"supports fewer.",
        "=== TASK ===",  # 任务说明：普通轮 vs 修复轮二选一
        ("REPAIR ROUND: your previous edits for this candidate are already in "  # 修复轮：critic 的反对意见 + 最小修复要求
         "the working tree (reflected in CURRENT HARNESS SOURCE above). The "
         "reviewer raised the objections below. Fix ONLY what the objections "
         "require (remove leaked content, split or re-declare edits, wire up "
         "dead code, or delete the offending part) with minimal additional "
         "edits, then call done again with the corrected edits array.\n\n"
         "=== REVIEWER OBJECTIONS ===\n"
         + json.dumps(repair_brief, ensure_ascii=False, indent=1) + "\nFirst action:")
        if repair_brief else
        "Address the highest-impact failure modes / capability gaps within your "  # 普通轮：在预算内对准高影响失败模式
        "edit budget. Implement via the JSON actions, then call done with the "
        "edits array. First action:",
    ])

    log, transcript, n_edits, aborts_left = [], "", 0, 3  # 动作日志、交互文本、实际文件编辑数、abort 容忍余量
    for _turn in range(MAX_TURNS):  # 最多 40 轮
        prompt = ("=== ROUND CONTEXT ===\n" + context  # 每轮：完整上下文 + 交互日志
                  + "\n\n=== INTERACTION LOG ===\n" + transcript
                  + "\nReply with exactly one JSON action object.")
        raw = generate(prompt, system=system, json_only=True, model=model,  # 调用模型：stable 前缀走缓存
                       cache_prefix=stable)
        try:  # 解析动作
            act = json.loads(raw)  # 直接解析
        except json.JSONDecodeError:  # 坏 JSON
            transcript += f"\n[you] {raw[:500]}\n[result] ERROR: not valid JSON"  # 反馈重试
            continue
        if isinstance(act, list):  # 输出成数组
            act = next((x for x in act if isinstance(x, dict)), None)  # 取第一个对象
        if not isinstance(act, dict):  # 不是对象
            transcript += ("\n[you] (non-object)\n[result] ERROR: reply with "  # 纠正格式
                           "EXACTLY ONE JSON action object")
            continue
        a = act.get("action")  # 动作名
        log.append(act if a in ("done", "abort") else  # done/abort 全量入日志
                   {k: (v if len(str(v)) < 200 else str(v)[:200] + "...")  # 其余动作截断记录（防日志爆炸）
                    for k, v in act.items()})
        if a == "done":  # 终止动作：提交编辑清单（预算/槽位/标签在此强制执行）
            edits = act.get("edits") or act.get("candidates") or []  # 取编辑清单（容忍旧字段名）
            problems = []  # 违约问题列表
            if n_edits == 0 and edits:  # 声明了编辑却一次文件都没改
                transcript += ("\n[you] done\n[result] ERROR: you declared edits but "  # 打回：先实现再 done
                               "made ZERO file changes. Implement them with "
                               "edit_file/write_file, then call done.")
                continue
            if n_edits > 0 and not edits:  # 改了文件却没声明编辑
                problems.append("edits array is empty")  # 问题：清单为空
            if len(edits) > budget:  # 超出退火预算 b_t
                problems.append(f"{len(edits)} edits exceed the budget b_t = {budget}")  # 问题：超预算
            for e in edits:  # 逐编辑校验必填字段
                miss = [f for f in ("id", "component", "hypothesis", "targets_mode",  # 缺失的必填字段
                                    "predicted_affected", "retroactive_check")
                        if not e.get(f)]
                if miss:  # 有缺失
                    problems.append(f"edit {e.get('id', '?')} missing {miss}")  # 记录问题
                if e.get("component") and str(e["component"]).lower() not in K:  # 组件标签不在词表
                    problems.append(f"edit {e.get('id')} component "  # 记录问题
                                    f"{e['component']!r} not in {K}")
            if reserved_slot and explore.get("untried") and edits and not any(  # 保留槽位校验（按声明标签）
                    str(e.get("component", "")).lower() in explore["untried"]
                    for e in edits):
                problems.append("this variant holds a RESERVED EXPLORATION SLOT: at "  # 问题：没落在未尝试组件上
                                "least one edit must be on a never-exercised "
                                f"component from {explore['untried']}")
            if problems and n_edits > 0:  # 有问题且已动手（纯清零退回会死循环）
                transcript += (f"\n[you] done\n[result] ERROR: {problems}. Call done "  # 打回并列出全部问题
                               f"again fixed (drop or merge edits if over budget; "
                               f"add the required edit if a slot is reserved).")
                continue  # 下一轮自行修复
            for e in edits:  # 全部通过：规范化编辑
                e["component"] = str(e.get("component", "")).lower()  # 标签转小写
                e.setdefault("mechanism", e.get("hypothesis"))  # mechanism 字段兜底为 hypothesis
            return {"status": "done", "summary": act.get("summary"),  # 返回成功结果
                    "edits": edits,
                    "mechanism": act.get("summary") or  # 整体机制说明兜底
                    "; ".join(str(e.get("hypothesis")) for e in edits)[:200],
                    "targets_mode": ", ".join(str(e.get("targets_mode"))  # 目标失败模式串接
                                              for e in edits)[:200],
                    "n_edits": n_edits, "log": log}
        if a == "abort":  # 代理想放弃
            if aborts_left > 0:  # 还有容忍余量
                aborts_left -= 1  # 扣一次
                transcript += ("\n[you] abort\n[result] ERROR: there is no abort action. "  # 打回：协议里没有 abort
                               "Pick the most defensible mechanism you can build "
                               "within the hard rules, implement it, and call done. "
                               "A rejection is data; an abort is not.")
                continue  # 再给机会
            return {"status": "abort", "reason": act.get("reason"),  # 三次 abort 后接受现实：本变体失败
                    "edits": [], "n_edits": n_edits, "log": log}
        try:  # 工具动作分发
            if a == "list_files":  # 列文件
                result = ws.list_files()
            elif a == "read_file":  # 读文件
                result = ws.read(act["path"])
            elif a == "list_traces":  # 任务表
                result = _list_traces(domain, traces, per_task, findings or [])
            elif a == "read_trace":  # 轨迹切片
                result = _read_trace(domain, traces, act.get("task_id", ""),
                                     act.get("from_step"), act.get("to_step"),
                                     render_cache, act.get("detail", False))
            elif a == "edit_file":  # 编辑已有文件
                if n_edits >= MAX_EDITS:  # 文件操作额度耗尽
                    result = "ERROR: edit budget exhausted; call done"  # 拒绝并催收尾
                else:
                    result = ws.edit(act["path"], act["old"], act["new"])  # 执行替换
                    n_edits += result.startswith("OK")  # 成功才计数
            elif a == "write_file":  # 新建文件
                if n_edits >= MAX_EDITS:  # 额度耗尽
                    result = "ERROR: edit budget exhausted; call done"  # 拒绝
                else:
                    result = ws.write(act["path"], act["content"])  # 执行新建
                    n_edits += result.startswith("OK")  # 成功才计数
            else:  # 未知动作
                result = f"ERROR: unknown action {a}"
        except Exception as e:  # noqa: BLE001
            result = f"ERROR: {e}"  # 工具异常转文本反馈（不杀死整个起草）
        cap = 150_000 if a in ("read_file", "list_files", "read_trace",  # 读取类动作的输出上限放宽
                               "list_traces") else 4000  # 其余动作 4000 字符
        transcript += f"\n[you] {json.dumps(act)[:1500]}\n[result] {str(result)[:cap]}"  # 记入交互日志（动作截 1500）
    return {"status": "max_turns", "edits": [], "n_edits": n_edits, "log": log}  # 轮数耗尽：本变体失败

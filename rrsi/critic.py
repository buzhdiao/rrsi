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
"""Critic(H_t, H') -> {0,1}: the leakage screen (Algorithm 1, last line).

Reads the candidate diff BEFORE any evaluation is spent and rejects edits
that encode task names, entity names, task-specific values, answers or other
suite-specific logic, as well as edits that add inert machinery. Two layers:

1. Deterministic pre-checks: a generic denylist (credentials, infra names)
   plus the domain's own patterns (grader paths, task ids, judge references).
2. LLM review (Claude Opus 4.8) of intent and content, not style.

A rejection sends the objections back to the proposer for a bounded number of
repair rounds; a candidate that cannot be repaired is dropped and recorded in
the history without a measurement.
"""
# 【模块说明（中文）】Critic(H_t, H') -> {0,1}：泄漏筛查（Algorithm 1 的最后一行）。
# 在花掉任何评估开销之前读候选 diff，拒绝编码了任务名、实体名、任务专属值、答案
# 或其他套件专属逻辑的编辑，也拒绝"死机器"（看似有机制实则无效）的编辑。两层结构：
#   1. 确定性预检：通用黑名单（凭据、基础设施名）+ domain 专属模式（判分器路径、任务 id、裁判引用）
#   2. LLM 审查（Claude Opus 4.8）：审查意图与内容，而非风格
# 拒绝会把反对意见发回提案者做有界轮数的修复（bounded repair）；修不好的候选被丢弃，
# 只在历史里留档（delta_S=None），不进入 T_t / g_t（未测量的编辑不算证据）。

# 延迟类型注解求值
from __future__ import annotations

# json：声明编辑清单与 critic 输出的序列化
import json
# re：黑名单正则与从输出中抢救 JSON 对象
import re

# generate：统一的 Vertex AI Claude 客户端（llm.py）
from .llm import generate

# GENERIC_PATTERNS：跨 domain 通用的确定性黑名单——(正则, 理由) 对
GENERIC_PATTERNS = [
    # 匹配 Google API key（AIza+35 位）、OpenAI 风格 key（sk-+20 位）、或 api_key="..." 赋值——凭据硬编码
    (r"AIza[0-9A-Za-z_-]{35}|sk-[A-Za-z0-9]{20,}|api_key\s*=\s*[\"\'][^\"\']{8,}", "credential in diff"),
]

# SYSTEM_TMPL：critic 的系统提示词（占位 domain_brief 注入基准专属判据）
SYSTEM_TMPL = """You are a strict reviewer of harness (agent scaffold) code changes
in an evolution loop. The harness is evolved against the very tasks it is
measured on, so anti-overfitting review is critical. The change was made by
another model in response to aggregated failure modes. Your job is to catch
changes that would be cheating, degenerate, or self-destructive.

{domain_brief}

REJECT if ANY of:

1. LEAKAGE / TASK-SPECIALIZATION: the diff hard-codes knowledge of specific
   evolve-set tasks: task names or ids, entity/company/case names, task-specific
   file names, expected outputs or answers, magic constants that only make
   sense for one task, per-task templates, a list of the suite's domains, or
   branching on task-identifying features ("if the task mentions X"). General
   professional practice and general procedure are fine. Litmus test: would
   this change still make sense, and still help, on an unfamiliar task from a
   different suite in the same kind of work?
2. DEGENERATE: the diff is effectively a no-op (dead code, unused flag,
   comment-only) while claiming a mechanism, or it deletes/disables an
   existing safety mechanism (context compression, output truncation, error
   handling, termination guards) without a working replacement.
3. GRADER GAMING: the change tries to read, detect, reconstruct or influence
   the verifier / judge / rubric at runtime, hard-codes a threshold or reference
   value that could only have come from a grader, or games the completion or
   output format for the grader rather than improving actual task performance.
4. UNDECLARED BUNDLING: multiple DECLARED edits per candidate are allowed and
   normal (the declared edit list is in the payload). Reject only if the diff
   contains independent changes not covered by ANY declared edit, or one
   declared edit secretly bundles several independent mechanisms.
5. RUNTIME MEMORY / SKILL LEAKAGE: if an edit adds a memory/state or skill
   mechanism, inspect what it PERSISTS or INJECTS at runtime. Entity-free
   general procedures are fine. REJECT if the code path can persist or inject
   task-specific runtime data across trials (captured file contents, tool
   outputs, computed answers, task/entity/file names, per-task solutions keyed
   by task identity): the harness is evaluated on the tasks it evolves on, so
   that is memorization, not a general improvement.
6. UNBOUNDED WORK: an added check, retry or "keep improving" loop with no
   give-up path, or anything that implies finishing should wait for a
   confirmation that may never come. A bounded pass with an explicit exit is
   fine.

Otherwise ACCEPT. You are reviewing intent and content, not style.
Runtime correctness (undefined names, crashes, syntax) is NOT your job:
deterministic compile/constructor/smoke checks handle that after you, and you
only see the diff (not the full files), so never speculate that something is
undefined or would crash.
Return STRICT JSON:
{{"verdict": "accept" | "reject", "reasons": ["..."], "risk_notes": ["..."]}}"""
# 【提示词要点（中文）】critic 在进化循环中审查 harness 代码改动；harness 是在"它被测量的
# 同一批任务"上演化出来的，所以反过拟合审查至关重要。六类必拒：
# 1) 泄漏/任务特化（硬编码任务名/实体/答案/魔法常数/按任务特征分支）；试金石：换一个
#    套件的陌生任务，这条改动是否仍合理仍有用？2) 退化（无操作的死代码、删安全机制）；
# 3) 钻判分器空子（运行时探测/影响 verifier、硬编码只能来自判分器的阈值）；
# 4) 未声明的捆绑（diff 含任何声明都没覆盖的独立改动）；5) 运行时记忆/技能泄漏
#    （跨 trial 持久化任务专属运行时数据 = 记忆化，不是通用改进）；6) 无界工作
#    （没有放弃路径的重试/"持续改进"循环）。其余接受；只审意图与内容，运行期正确性
#    不是它的职责（由编译/构造/smoke 检查兜底）。输出严格 JSON verdict/reasons/risk_notes。


def precheck(diff: str, domain_patterns: list) -> list[str]:  # 定义 precheck：第一层确定性黑名单筛查
    """第一层筛查：对 diff 跑通用黑名单 + domain 专属模式的正则匹配，
    返回命中的违规理由列表（空列表即通过，不花任何 LLM 调用）。"""
    hits = []  # 命中的理由列表
    for pat, why in GENERIC_PATTERNS + list(domain_patterns or []):  # 通用黑名单在前、domain 专属模式在后
        if re.search(pat, diff):  # 任一正则在 diff 中命中
            hits.append(why)  # 记录该违规的理由
    return hits  # 返回全部命中（空列表 = 通过预检）


def review(domain, diff: str, summary: str, targets_mode: str,  # 定义 review：第二层 LLM 审查入口
           edits: list | None = None, state_files: str = "",
           model: str | None = None) -> dict:
    """第二层审查：预检通过后，把候选摘要、声明的编辑清单、diff 与（若机制
    持久化状态）状态文件一并交给 LLM 裁决，返回 {verdict, reasons, risk_notes}；
    连续三轮解析不出合法 JSON 则按拒绝处理（宁错杀不漏放）。"""
    hard = precheck(diff, domain.critic_patterns)  # 先跑确定性预检（domain 模式来自 adapter 的 critic_patterns）
    if hard:  # 预检命中
        return {"verdict": "reject", "reasons": [f"precheck: {h}" for h in hard],  # 直接拒绝，不花 LLM 调用（省时且可复现）
                "risk_notes": []}
    if not diff.strip():  # 空 diff（什么都没改）
        return {"verdict": "reject", "reasons": ["empty diff"], "risk_notes": []}  # 拒绝：没有机制可审
    system = SYSTEM_TMPL.format(domain_brief=domain.briefs["critic"])  # 组装系统提示词（注入 domain 判据段落）
    payload = (  # 用户消息载荷：候选摘要 + 声明编辑 + diff + 状态文件
        f"CANDIDATE SUMMARY: {summary}\nTARGETS: {targets_mode}\n\n"  # 摘要与目标失败模式
        f"=== DECLARED EDITS (independent changes in this candidate) ===\n"  # 声明编辑清单：核对"未声明捆绑"的基准
        f"{json.dumps(edits or [], ensure_ascii=False, indent=1)[:20_000]}\n\n"  # JSON 化并截断 2 万字符（上下文保护）
        f"=== DIFF ===\n{diff[:120_000]}\n\n"  # diff 正文截断 12 万字符
        f"=== STATE FILES (if a mechanism persists state) ===\n{state_files[:20_000]}"  # 若机制持久化状态，附状态文件内容（查记忆泄漏）
    )
    last = ""  # 记录最后一次原始输出（解析失败时写进拒绝理由）
    for _ in range(3):  # 最多重试 3 次（LLM 偶发输出不合法 JSON）
        out = generate(payload, system=system, json_only=True, model=model)  # 调用搜索模型，json_only 模式
        last = out  # 保存原始输出
        try:  # 尝试解析
            v = json.loads(out)  # 直接解析整个输出
            if isinstance(v, dict) and v.get("verdict") in ("accept", "reject"):  # 校验结构合法且 verdict 合法
                return v  # 合法裁决直接返回
        except json.JSONDecodeError:  # 整体不是 JSON
            m = re.search(r"\{.*\}", out, re.S)  # 退路：非贪婪搜出第一个 {...} 块（DOTALL 跨行）
            if m:  # 找到了疑似 JSON 片段
                try:  # 再试一次
                    return json.loads(m.group(0))  # 解析片段成功即返回
                except json.JSONDecodeError:  # 片段也坏了
                    pass  # 进入下一轮重试
    return {"verdict": "reject",  # 三次都失败：按拒绝处理（宁可错杀不可漏放——被拒只损失一个候选）
            "reasons": [f"critic output unparseable after 3 attempts: {last[:200]}"],  # 理由写明是解析失败
            "risk_notes": []}

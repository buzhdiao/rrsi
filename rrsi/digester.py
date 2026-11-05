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
"""Trace digester: a READ-ONLY subagent that inspects one rendered trajectory
on behalf of the analyst, so the analyst's context carries structured
conclusions, never whole traces.

Tools (jailed to the round's rendered-traces directory): read_file, glob,
grep, bash (write operations rejected). Returns structured JSON hard-capped at
DIGEST_MAX_CHARS. Lenses: failure / capability_gap / success.
"""
# 【模块说明（中文）】轨迹摘要子代理（trace digester）：代表分析师深挖"一条"渲染轨迹的
# 只读调查员——让分析师上下文里装的是结构化结论，而不是整条轨迹。
# 工具（被囚禁在本轮的渲染轨迹目录内）：read_file、glob、grep、bash（拒绝一切写操作）。
# 返回结构化 JSON，硬性上限 DIGEST_MAX_CHARS。三种透镜：failure / capability_gap / success。

# 延迟类型注解求值
from __future__ import annotations

# json：动作与摘要的序列化
import json
# re：bash 黑名单正则、grep 的行匹配
import re
# subprocess：只读 bash 工具的执行
import subprocess
# Path：轨迹目录与文件路径
from pathlib import Path

# generate：统一 LLM 客户端
from .llm import generate

# MAX_TURNS：摘要子代理最大轮数（15 轮内必须 return）
MAX_TURNS = 15
# DIGEST_MAX_CHARS：摘要 JSON 的字符硬上限（保证分析师上下文可控）
DIGEST_MAX_CHARS = 6000
# TOOL_OUT_CAP：单次工具输出回填进交互日志的上限
TOOL_OUT_CAP = 25_000
# BASH_TIMEOUT：bash 命令超时（秒），防卡死
BASH_TIMEOUT = 30

# ALLOWED_BASH：只读 bash 白名单——分段（按 |;& 切）后首词必须在此集合
ALLOWED_BASH = {"grep", "egrep", "head", "tail", "wc", "cat", "ls", "find",
                "cut", "sort", "uniq", "awk", "jq", "sed", "tr", "paste"}
# DENY_BASH：bash 黑名单正则——重定向(>>)、反引号、命令替换 $(、sed -i、rm/mv/cp/tee/touch/mkdir/chmod/python
DENY_BASH = re.compile(r"(>>?|`|\$\(|sed\s+[^|]*-i|\brm\b|\bmv\b|\bcp\b|"
                       r"\btee\b|\btouch\b|\bmkdir\b|\bchmod\b|\bpython)")

# SCHEMAS：三种透镜各自的返回 JSON 模板（嵌入系统提示词，让子代理知道字段语义）
SCHEMAS = {
    # failure 透镜：什么机制丢了分、失败如何展开、证据引用、判分器报告、需要的替代路径
    "failure": """{"task_id": "...", "lens": "failure",
 "blocker": "one sentence: what mechanism lost the points",
 "narrative": "2-5 sentences: how the failure unfolded, concrete",
 "evidence": [{"where": "step 42", "quote": "short exact quote"}],
 "verifier_evidence": "what the grader/verifier itself reported as missed",
 "capability_note": "optional: anything the agent tried but could not do",
 "needed_instead": "1-2 sentences: what the successful path required"}""",
    # capability_gap 透镜：agent 想做什么、为什么做不到、证据、见到的部分替代方案
    "capability_gap": """{"task_id": "...", "lens": "capability_gap",
 "wanted": "what the agent was trying to accomplish",
 "why_couldnt": "what stopped it (tool limits, missing info, dead ends)",
 "evidence": [{"where": "step 12", "quote": "..."}],
 "workaround_seen": "optional: any partial workaround it attempted"}""",
    # success 透镜：可复用的好习惯及其位置、哪个习惯是承重墙
    "success": """{"task_id": "...", "lens": "success",
 "habits": [{"habit": "reusable behavior that made this run clean",
             "where_shown": "step range"}],
 "risk_if_removed": "which habit is load-bearing and what breaks without it"}""",
}

# SYSTEM_TMPL：摘要子代理的系统提示词
SYSTEM_TMPL = """You are a trajectory digester: a read-only investigator that
inspects ONE agent trajectory in depth and returns a compact structured
digest. Another agent (the batch analyst) will rely on your digest without
reading the trace itself, so be precise and evidence-anchored.

{domain_brief}

The rendered trajectory files live in your working directory as <task_id>.txt.

Your lens for this assignment: {lens}
{focus}

You interact via STRICT JSON, one action per turn:
  {{"action": "read_file", "path": "<task_id>.txt", "offset": 1200, "limit": 300}}
      (line-based; omit offset/limit to read from the start)
  {{"action": "glob", "pattern": "*.txt"}}
  {{"action": "grep", "pattern": "regex", "path": "<task_id>.txt", "max_hits": 40}}
      (returns matching lines with line numbers)
  {{"action": "bash", "cmd": "grep -n 'FAILED' <task_id>.txt | head -30"}}
      (read-only shell: grep/head/tail/awk/jq/sed(no -i)/wc/...; any write
       operation is rejected)
  {{"action": "return", "digest": {{...}}}}

Investigate efficiently: read the grading / verifier section first (what
actually failed), grep for anchors (step numbers, tool names, key values,
error strings), then read the relevant slices. Do not read whole files top
to bottom.

Finish with action "return". The digest MUST follow this schema and MUST be
under {cap} characters total:
{schema}

Rules: quotes must be exact and short; every claim needs a "where"; do not
speculate beyond what the trace shows; no blame attribution to "model vs
harness"."""
# 【提示词要点（中文）】只读调查员，一次深挖一条轨迹并返回紧凑摘要；分析师只依赖摘要
# 不读轨迹，因此必须精确、以证据为锚。高效调查：先读判分/verifier 段（到底什么失败），
# 再 grep 锚点（步骤号/工具名/关键值/错误串），然后只读相关切片，不要从头读到尾。
# 以 return 动作收尾，摘要必须符合 schema 且少于 {cap} 字符。规则：引用必须精确且短；
# 每个论断都要有 where；不越轨迹内容猜测；不做"模型 vs 脚手架"的归责。


def _bash_ok(cmd: str) -> bool:  # 定义 _bash_ok：只读 bash 命令的两层检查
    if DENY_BASH.search(cmd):  # 第一层：黑名单正则命中（重定向/替换/危险命令）直接拒绝
        return False
    for seg in re.split(r"[|;&]+", cmd):  # 第二层：按管道与命令分隔符切段，逐段检查
        seg = seg.strip()  # 去空白
        if seg and seg.split()[0] not in ALLOWED_BASH:  # 每段首词必须在白名单里（防 grep x; rm y 之类）
            return False
    return True  # 两层都过：允许执行


def _safe(root: Path, rel: str) -> Path:  # 定义 _safe：路径囚笼——禁止越出轨迹目录
    p = (root / rel).resolve()  # 拼接并 resolve（消解 .. 与符号链接）
    if not str(p).startswith(str(root.resolve())):  # 解析后不在根目录内
        raise ValueError("path escapes the traces dir")  # 拒绝（防 ../../etc/passwd 式逃逸）
    return p  # 返回安全路径


def _read_file(root: Path, rel: str, offset, limit) -> str:  # 定义 _read_file：按行读文件片段
    p = _safe(root, rel)  # 先过路径囚笼
    if not p.exists():  # 文件不存在
        return f"ERROR: no such file {rel}"  # 错误占位（回给子代理自行纠正）
    lines = p.read_text().splitlines()  # 按行读入
    lo = max(0, (offset or 1) - 1)  # offset 是 1-based 行号，转 0-based 下标；缺省从第 1 行
    hi = lo + (limit or 400)  # 缺省读 400 行
    body = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[lo:hi], start=lo))  # 每行带原始行号前缀（便于引用证据位置）
    return body + (f"\n...[file has {len(lines)} lines]" if hi < len(lines) else "")  # 未读到底时提示总行数（引导用 offset 继续）


def _grep(root: Path, pattern: str, rel: str, max_hits: int) -> str:  # 定义 _grep：带行号的正则搜索
    p = _safe(root, rel)  # 路径囚笼
    if not p.exists():  # 文件不存在
        return f"ERROR: no such file {rel}"  # 错误占位
    try:  # 编译子代理给的正则
        rx = re.compile(pattern)  # 可能非法
    except re.error as e:  # 非法正则
        return f"ERROR: bad regex: {e}"  # 回传错误信息
    out = []  # 命中行收集
    for i, line in enumerate(p.read_text().splitlines(), 1):  # 逐行扫描（1-based 行号）
        if rx.search(line):  # 命中
            out.append(f"{i}: {line[:400]}")  # 记录行号与该行（截 400 字符防超长行刷屏）
            if len(out) >= max_hits:  # 达到命中上限
                out.append("...[max hits reached]")  # 提示截断
                break  # 提前结束
    return "\n".join(out) if out else "(no matches)"  # 空结果给占位串


def digest_task(traces_dir: Path, task_id: str, lens: str, domain_brief: str,  # 定义 digest_task：单个轨迹的调查主循环
                questions: list | None = None, model: str | None = None) -> dict:
    lens = lens if lens in SCHEMAS else "failure"  # 透镜名非法时回落到 failure（默认透镜）
    focus = ""  # 分析师的定向问题段
    if questions:  # 有定向问题
        # 要求子代理必须回答；最多列 5 条
        focus = "Specific questions from the analyst you MUST answer:\n" + \
            "\n".join(f"- {q}" for q in questions[:5])  # 逐条列出
    system = SYSTEM_TMPL.format(lens=lens, focus=focus, cap=DIGEST_MAX_CHARS,  # 组装系统提示词：透镜、定向问题、字符上限、schema
                                schema=SCHEMAS[lens], domain_brief=domain_brief)
    transcript = f"Assigned trace: {task_id}.txt\nFirst action:"  # 首轮告知分配到的轨迹文件
    for _ in range(MAX_TURNS):  # 最多 15 轮
        raw = generate(transcript, system=system, json_only=True, model=model)  # 调用模型（整段 transcript 即用户消息）
        try:  # 解析动作
            act = json.loads(raw)  # 直接解析
        except json.JSONDecodeError:  # 坏 JSON
            transcript += f"\n[you] {raw[:300]}\n[result] ERROR: invalid JSON"  # 反馈重试
            continue
        if isinstance(act, list):  # 输出成数组
            act = next((x for x in act if isinstance(x, dict)), None)  # 取第一个对象
        if not isinstance(act, dict):  # 不是对象
            transcript += ("\n[you] (non-object)\n[result] ERROR: reply with "  # 纠正格式
                           "EXACTLY ONE JSON action object, not a list or value")
            continue
        a = act.get("action")  # 动作名
        if a == "return":  # 终止动作：交摘要
            digest = act.get("digest") or {}  # 取摘要对象
            blob = json.dumps(digest, ensure_ascii=False)  # 序列化以测长度
            if len(blob) > DIGEST_MAX_CHARS:  # 超过硬上限
                transcript += (f"\n[you] return ({len(blob)} chars)\n[result] "  # 打回要求缩短——上限是硬约束
                               f"ERROR: digest is {len(blob)} chars, cap is "
                               f"{DIGEST_MAX_CHARS}. Shorten and return again.")
                continue  # 不接受，继续轮
            digest.setdefault("task_id", task_id)  # 补齐任务 id（防子代理漏填）
            digest.setdefault("lens", lens)  # 补齐透镜名
            return digest  # 返回摘要
        try:  # 工具动作分发
            if a == "read_file":  # 读文件片段
                result = _read_file(traces_dir, act["path"], act.get("offset"),
                                    act.get("limit"))
            elif a == "glob":  # 列文件名
                result = "\n".join(sorted(  # 排序保证稳定
                    p.name for p in traces_dir.glob(act.get("pattern", "*"))))  # 默认匹配全部
            elif a == "grep":  # 正则搜索
                result = _grep(traces_dir, act.get("pattern", ""),
                               act.get("path", ""), int(act.get("max_hits", 40)))
            elif a == "bash":  # 只读 shell
                cmd = act.get("cmd", "")  # 取命令
                if not _bash_ok(cmd):  # 两层检查未过
                    result = ("ERROR: command rejected (read-only shell; allowed: "  # 拒绝并说明白名单
                              "grep/head/tail/awk/jq/sed/wc/cat/ls/find/cut/sort/"
                              "uniq/tr; no redirection)")
                else:  # 检查通过
                    r = subprocess.run(cmd, shell=True, cwd=traces_dir,  # 在轨迹目录内执行（cwd 囚笼）
                                       capture_output=True, text=True,
                                       timeout=BASH_TIMEOUT)  # 30 秒超时防卡死
                    result = (r.stdout or "") + (r.stderr or "")  # stdout+stderr 拼接为结果
            else:  # 未知动作
                result = f"ERROR: unknown action {a}"
        except Exception as e:  # noqa: BLE001
            result = f"ERROR: {e}"  # 工具异常转成文本回给子代理（不让单次失败杀死整个调查）
        transcript += (f"\n[you] {json.dumps(act)[:600]}\n"  # 动作截 600 字符记入日志
                       f"[result] {str(result)[:TOOL_OUT_CAP]}")  # 工具输出截 25000 字符
    return {"task_id": task_id, "lens": lens,  # 轮数耗尽：返回带错误标记的占位摘要
            "error": "digester hit max turns without returning"}

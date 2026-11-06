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
# 模块说明:把一条 LAB trial 记录(轨迹 + 量规判定)渲染成分析师、消化器与提议者
# 阅读的文本。
"""Render a LAB trial record (trajectory + rubric verdicts) into the text the
analyst, digester and proposer read."""

# 各类渲染截断上限:工具结果/工具参数/助手发言字符数,以及整份文本上限
TOOL_RESULT_HEAD, TOOL_ARGS_HEAD, ASSISTANT_HEAD, MAX_CHARS = 700, 400, 1200, 300_000


def _clip(s, n):
    # 通用截断函数:None 视为空串;超长保留前 n 字符并标注被截掉的长度
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"


def render_messages(messages: list, tool_head=TOOL_RESULT_HEAD, args_head=TOOL_ARGS_HEAD,
                    asst_head=ASSISTANT_HEAD) -> str:
    # 先建立 tool_call_id -> 工具名 的映射:工具结果消息常缺 name,需从调用反查
    call_names = {}
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if tc.get("id"):
                call_names[tc["id"]] = (tc.get("function") or {}).get("name")
    # 输出行列表与步号计数(assistant 回合即一步)
    lines, step = [], 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            # 系统提示固定截 2000 字符
            lines.append(f"[SYSTEM PROMPT] {_clip(m.get('content'), 2000)}")
        elif role == "user":
            # 任务题面固定截 4000 字符
            lines.append(f"[TASK PROMPT] {_clip(m.get('content'), 4000)}")
        elif role == "assistant":
            # 步号加一,渲染助手发言与工具调用
            step += 1
            if m.get("content"):
                lines.append(f"[step {step}] ASSISTANT: {_clip(m['content'], asst_head)}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                lines.append(f"[step {step}] TOOL_CALL {fn.get('name')}"
                             f"({_clip(fn.get('arguments', ''), args_head)})")
        elif role == "tool":
            # 工具结果行:工具名优先取消息自带,其次反查表,兜底 unknown_tool
            nm = m.get("name") or call_names.get(m.get("tool_call_id")) or "unknown_tool"
            lines.append(f"[step {step}] TOOL_RESULT {nm}: {_clip(m.get('content'), tool_head)}")
    # 拼接全部行
    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        # 超全局上限:保留头 60% + 尾 40%,中段占位
        head, tail = int(MAX_CHARS * 0.6), int(MAX_CHARS * 0.4)
        text = text[:head] + f"\n...[TRUNCATED {len(text) - head - tail} chars]...\n" + text[-tail:]
    return text


def render_grades(criteria: list) -> str:
    # 渲染量规判定:逐判据 PASS/FAIL、判据标题与评审理由
    lines = []
    for i, c in enumerate(criteria):
        verdict = "PASS" if c.get("verdict") == "pass" else "FAIL"
        lines.append(f"criterion {i} [{verdict}] {_clip(c.get('title') or c.get('criteria'), 400)}\n"
                     f"  judge: {_clip(c.get('reasoning', ''), 1200)}")
    return "\n".join(lines)


def render_full(rec: dict, detail: bool = False) -> str:
    # 拼装整条 trial 的文本:头部信息、轨迹、最终输出、量规评分(真值)
    tr = rec.get("trajectory") or {}
    usage = tr.get("usage") or {}
    # detail 模式放宽各消息截断上限
    caps = ({"tool_head": 4000, "args_head": 1200, "asst_head": 4000} if detail else {})
    msgs = tr.get("messages") or []
    scores = rec.get("scores") or {}
    return "\n".join([
        f"TASK: {rec.get('task_id')} | practice area: {rec.get('domain')} | "
        f"trial: {rec.get('trial')}",
        f"STATUS: {tr.get('status')} | steps(assistant turns): "
        f"{sum(1 for m in msgs if m.get('role') == 'assistant')} | "
        f"compactions: {usage.get('compaction_count')} | total_tokens: {usage.get('total_tokens')}",
        f"CRITERIA: {scores.get('passed')} / {scores.get('total')} passed "
        f"(all_pass={scores.get('all_pass')}) | deliverables written: "
        f"{rec.get('output_files')}",
        "", "=== TRAJECTORY ===", render_messages(msgs, **caps),
        "", "=== FINAL OUTPUT ===", _clip(tr.get("output"), 6000),
        "", "=== RUBRIC GRADING (ground truth verdicts) ===",
        render_grades(scores.get("criteria") or []),
    ])

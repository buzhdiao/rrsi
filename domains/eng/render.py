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
# 模块说明:把一条已评分(graded)的工程 trial(试验)记录渲染成分析师(analyst)/
# 消化器(digester)阅读的文本。trial 记录捆绑了 ground truth(评分真值)所依赖的
# 全部证据:任务与领域、运行状态/步数/token 数、GRADING 评分块(三个验证轴,加上
# 携带诊断信息的逐子判据 details)、代理实际提交的 PAYLOAD(唯一被评分的产物)、
# 以及工具调用轨迹(按消息截断)。details 块是本基准值得持续进化的原因:标量分数
# 只能说明设计错了,details 能说明错在哪个子项(如 "iou_score 0.71 对 pass_threshold
# 0.8"、"phase_margin 正确但 settling_time 错误"),从而区分需要相反机制的失败族。
"""Render one graded trial into the text form the analyst/digester read.

A trial record bundles everything the ground truth depends on:

  - the task, its engineering domain, the run status / step count / tokens
  - the GRADING block: the three verifier axes plus -- the part that carries the
    diagnosis -- the verifier's own per-sub-criterion `details`
  - the PAYLOAD the agent actually submitted (the only graded artefact)
  - the trajectory (tool calls + results), clipped per message

The `details` block is the reason this benchmark is worth evolving on. A scalar
score says a design was wrong; `details` says WHICH part was wrong -- "iou_score
0.71 against pass_threshold 0.8", "phase_margin correct, settling_time wrong",
"schema validation failed on field `config`". Those distinguish failure families
that need opposite mechanisms, and none of them is visible in the score.

"""

import json

# 工具结果消息渲染时保留的头部字符数上限
TOOL_RESULT_HEAD = 700
# 工具调用参数渲染时保留的头部字符数上限
TOOL_ARGS_HEAD = 400
# 助手发言渲染时保留的头部字符数上限
ASSISTANT_HEAD = 1200
# 整份渲染文本的最大字符数,超出则做"头 60% + 尾 40%"的中段截断
MAX_CHARS = 300_000
# 验证器 details(逐子判据结果)块保留的头部字符数;details 指明具体错在哪个子项
DETAILS_HEAD = 2500
# The problem statement was hardcoded at 3000 characters and, unlike every other
# cap, was NOT widened by detail mode -- so the one thing a reader most needs to
# reason about a failure was the one thing always truncated. 26 of the 61 task
# prompts exceed 3000 characters (the longest is 111k), meaning the proposer was
# diagnosing 43% of tasks against a cut-off list of constraints.
# 普通模式下任务题面(task prompt)保留的头部字符数
TASK_PROMPT_HEAD = 3000
# detail 模式下任务题面放宽到的字符数——题面是诊断失败最关键的依据,必须放宽
TASK_PROMPT_HEAD_DETAIL = 40_000


def _clip(s, n):
    # 通用截断函数:None 视为空串,其余转成字符串
    s = "" if s is None else str(s)
    # 未超长原样返回;超长则保留前 n 字符并追加 "...[+被截掉的字符数]" 提示
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"


def render_messages(messages: list, tool_head: int = TOOL_RESULT_HEAD,
                    args_head: int = TOOL_ARGS_HEAD,
                    asst_head: int = ASSISTANT_HEAD,
                    prompt_head: int = TASK_PROMPT_HEAD) -> str:
    # Tool RESULT messages usually carry no `name`, so the rendered trace said
    # "TOOL_RESULT None: ..." and the reader could not tell which tool produced
    # which output -- on a run whose whole diagnosis is "what did code_exec
    # actually return", that is the single most important label in the file.
    # The name is recoverable from the CALL that the result answers.
    # 先扫描全部消息,建立 tool_call_id -> 工具名 的映射:工具结果消息通常不带
    # name 字段,需要从产生它的 TOOL_CALL 反查,渲染出的轨迹才能标明每条结果
    # 来自哪个工具(诊断"code_exec 到底返回了什么"时这是最关键的标签)
    call_names = {}
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if tc.get("id"):
                call_names[tc["id"]] = (tc.get("function") or {}).get("name")

    # 输出行列表与步号计数(每个 assistant 回合算一步)
    lines = []
    step = 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            # 系统提示固定截到 2500 字符
            lines.append(f"[SYSTEM PROMPT] {_clip(m.get('content'), 2500)}")
        elif role == "user":
            # 任务题面按 prompt_head 截断
            lines.append(f"[TASK PROMPT] {_clip(m.get('content'), prompt_head)}")
        elif role == "assistant":
            # assistant 回合步号加一
            step += 1
            content = m.get("content")
            if content:
                # 有文本内容则先渲染助手发言
                lines.append(f"[step {step}] ASSISTANT: {_clip(content, asst_head)}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                # 渲染工具调用:工具名 + 截断后的参数
                lines.append(f"[step {step}] TOOL_CALL {fn.get('name')}"
                             f"({_clip(fn.get('arguments', ''), args_head)})")
        elif role == "tool":
            # 工具结果:优先用消息自带 name,否则用前面反查表,再兜底 unknown_tool
            nm = (m.get("name")
                  or call_names.get(m.get("tool_call_id"))
                  or "unknown_tool")
            # 渲染工具结果行:工具名 + 截断后的返回内容
            lines.append(f"[step {step}] TOOL_RESULT {nm}: "
                         f"{_clip(m.get('content'), tool_head)}")
    # 拼接全部行
    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        # 超过全局上限时保留头 60% 和尾 40%,中段用占位符标注被截掉的字符数
        head, tail = int(MAX_CHARS * 0.6), int(MAX_CHARS * 0.4)
        text = (text[:head] + f"\n...[TRUNCATED {len(text) - head - tail} chars]...\n"
                + text[-tail:])
    return text


def render_grading(task: dict, verdict: dict, diag: dict | None = None,
                   details_head: int = DETAILS_HEAD) -> str:
    """The ground truth for this trial, most diagnostic line first.

    Read the FAILURE CLASS line before anything else. It splits every loss into
    kinds that need different mechanisms:

      NO SUBMISSION  - the run ended without writing a payload at all. Nothing
                       about engineering explains this; it is termination,
                       budget, or plumbing.
      MALFORMED      - a payload was written but the verifier could not use it:
                       not a literal, missing a schema field, wrong type. The
                       design may have been right and was thrown away at the
                       last step. This is a contract-compliance bug.
      INVALID DESIGN - the submission parsed but violated the task's hard
                       constraints. The agent proposed something the physics or
                       the spec forbids.
      UNDER TARGET   - a valid design that simply did not perform well enough.
                       This is the genuinely engineering-limited case, and the
                       only one where "the agent needed to compute better" is
                       the right reading.
    """
    # verdict/diag 允许传 None,兜底成空 dict
    verdict = verdict or {}
    diag = diag or {}
    # 提取各评分字段:status 异常表示验证器自身出错;valid 表示提交是否被接受
    status = verdict.get("status") or "ok"
    passed = bool(verdict.get("passed"))
    valid = float(verdict.get("valid") or 0.0)
    score = verdict.get("combined_score")
    err = str(verdict.get("error") or "")

    # 失败分类(failure class):按"最先需要诊断信息"的顺序判定,不同类别需要不同机制
    if diag.get("no_payload"):
        # 完全没写 payload:终止/预算/管道类失败,与工程能力无关
        cls = ("NO SUBMISSION - the run never wrote a payload. This is a "
               "termination / budget / plumbing failure, not an engineering one.")
    elif status != "ok":
        # 验证器自身未跑完:视为基础设施问题,不能作为设计好坏的证据
        cls = (f"VERIFIER {status} - the verifier itself did not complete "
               f"({_clip(err, 200)}). Treat as infrastructure, not evidence "
               f"about the design.")
    elif "PAYLOAD" in err or "literal" in err.lower() or "validation" in err.lower():
        # payload 写了但验证器用不了(非字面量/缺字段/类型错):契约合规类失败,
        # 设计可能是对的却在最后一步被丢弃
        cls = ("MALFORMED SUBMISSION - a payload was written but the verifier "
               "could not use it (not a literal / missing or mistyped schema "
               "field). Whatever the design was, it was discarded at the last "
               "step. This is a contract-compliance failure.")
    elif valid < 1.0:
        # 提交能解析但违反任务硬约束:无效设计
        cls = ("INVALID DESIGN - the submission parsed but violated the task's "
               "hard constraints.")
    elif passed:
        # 通过
        cls = "PASSED."
    else:
        # 设计有效但没达到性能门槛:真正受工程能力限制的情形
        cls = ("UNDER TARGET - a valid design that did not meet the task's "
               "performance gate. This is the engineering-limited case.")

    # 第一行永远是失败分类,最具诊断价值
    lines = [f"FAILURE CLASS: {cls}"]
    # 第二行:总分与原始分、通过标志、有效性
    lines.append(
        f"SCORE: combined_score={score} (raw {verdict.get('raw_score')} of "
        f"{verdict.get('score_max')}) | passed={passed} | valid={valid}")
    if err:
        # 有验证器错误则截断后单列一行
        lines.append(f"VERIFIER ERROR: {_clip(err, 500)}")
    det = verdict.get("details")
    if det is not None:
        # details 是逐子判据的通过/失败明细:读它才知道"错在哪",而不只是"错了"
        lines.append("VERIFIER DETAILS (which sub-checks the design met and "
                     "missed -- read this to see WHAT was wrong, not just that "
                     "something was):")
        try:
            # 优先 JSON 缩进序列化后再截断
            lines.append(_clip(json.dumps(det, indent=1, default=str), details_head))
        except Exception:  # noqa: BLE001
            # 序列化失败就退化为字符串形式
            lines.append(_clip(str(det), details_head))
    # 代理活动统计:步数、code_exec 次数与失败数、写文件/读文件次数、payload 写入次数
    lines.append(
        f"AGENT ACTIVITY: {diag.get('n_steps')} steps, "
        f"{diag.get('n_code_exec')} code_exec calls "
        f"({diag.get('n_code_exec_failed')} of them errored), "
        f"{diag.get('n_write_file')} file writes, "
        f"{diag.get('n_read')} reads | wrote the payload "
        f"{diag.get('n_payload_writes')} time(s)")
    return "\n".join(lines)


def render_full(rec: dict, detail: bool = False) -> str:
    """rec = {task, meta, verdict, payload, traj, diag}. detail=True expands the
    per-message caps so content-level failures (what the agent actually computed,
    what it actually submitted) are visible."""
    # 取出记录的三个组成部分,缺失时兜底为空对象
    task = rec.get("task") or {}
    meta = rec.get("meta") or {}
    traj = rec.get("traj") or {}
    # detail 模式放宽各类截断上限,让"内容级失败"(实际算了什么、实际交了什么)可见
    caps = ({"tool_head": 4000, "args_head": 1200, "asst_head": 4000,
             "prompt_head": TASK_PROMPT_HEAD_DETAIL}
            if detail else {})
    # payload 截断上限:detail 模式 30000,普通 8000
    payload_cap = 30000 if detail else 8000
    messages = traj.get("messages") or []
    # 按段落拼装:任务头、状态行、评分真值、提交的 payload、轨迹
    return "\n".join([
        f"TASK: {task.get('id')} | domain: {task.get('domain')} | "
        f"trial: {rec.get('trial')}",
        f"STATUS: {meta.get('status')} | assistant turns: "
        f"{sum(1 for m in messages if m.get('role') == 'assistant')} | "
        f"total_tokens: {meta.get('total_tokens')} | wall_s: {meta.get('wall_s')}",
        "",
        "=== GRADING (ground truth) ===",
        render_grading(task, rec.get("verdict") or {}, rec.get("diag") or {},
                       details_head=(8000 if detail else DETAILS_HEAD)),
        "",
        "=== SUBMITTED PAYLOAD (the only graded artefact) ===",
        _clip(rec.get("payload"), payload_cap),
        "",
        "=== TRAJECTORY ===",
        render_messages(messages, **caps),
    ])


def load_trial(job_dir, task: dict, trial: int) -> dict | None:
    """Assemble one trial record from a job dir; None if it never ran."""
    # 局部导入,避免模块加载期的循环依赖
    from pathlib import Path

    # trial 目录 = job 目录 / 任务 id / t<编号>
    td = Path(job_dir) / task["id"] / f"t{trial}"
    # 没有 meta.json 说明该 trial 从未运行过
    if not (td / "meta.json").is_file():
        return None
    # 基础记录:任务、trial 编号、运行元数据
    rec = {"task": task, "trial": trial,
           "meta": json.loads((td / "meta.json").read_text())}
    # 三个证据文件:评分结果、提交的 payload、轨迹
    vp, pp, tp = td / "verdict.json", td / "payload.py", td / "traj.json"
    # 各文件存在才读,否则给空缺省值
    rec["verdict"] = json.loads(vp.read_text()) if vp.is_file() else {}
    rec["payload"] = pp.read_text(errors="replace") if pp.is_file() else ""
    rec["traj"] = json.loads(tp.read_text()) if tp.is_file() else {}
    try:
        # 把 bench/ 与本目录加进 sys.path 后导入诊断模块,计算轨迹统计
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent / "bench"))
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        import diagnostics as _D
        # 统计步数、code_exec 次数等诊断量
        rec["diag"] = _D.trial_stats(rec["traj"], rec["meta"])
    except Exception:  # noqa: BLE001 - evidence rendering must never kill a round
        # 证据渲染绝不能弄垮整轮流程,诊断失败就给空 dict
        rec["diag"] = {}
    return rec

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
"""Render one harbor trial directory into the text form the analyst, digester
and proposer read: task instruction, every episode (agent analysis/plan,
keystrokes, terminal output), run metadata and the verifier's ground truth
(reward, per-test results, test log tail)."""
# 【模块说明（中文）】把一个 harbor trial 目录渲染成 analyst/digester/proposer 读的文本：
# 任务指令、每个 episode（agent 分析/计划、按键、终端输出）、运行元数据，以及
# verifier 的 ground truth（奖励、逐测试结果、测试日志尾部）。各段都有字符截断上限，
# 防止长轨迹撑爆 LLM 上下文。

# json：解析 result.json / trajectory.json / ctrf.json
import json
# Path：路径对象
from pathlib import Path

# 三个默认截断上限：终端观测 900 字符、按键串 500 字符、agent 消息 1200 字符
OBS_HEAD, KEYSTROKES_HEAD, ASSISTANT_HEAD = 900, 500, 1200
# 任务指令最多 5000 字符；整篇渲染文本硬上限 30 万字符
INSTRUCTION_HEAD, MAX_CHARS = 5000, 300_000
# detail=True（digester 细读模式）时启用的更宽截断上限
DETAIL_CAPS = {"obs_head": 4000, "keys_head": 1500, "asst_head": 4000}


# 定义 _clip：把任意值转成字符串并截到 n 字符，截断时附"还有多少字符"的标记
def _clip(s, n):
    # 统一转字符串（容忍 None 等非字符串字段）
    s = str(s)
    # 不超限原样返回；超限取前 n 字符并标注被截掉的长度
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"


# 定义 _obs_text：从一步记录里抽取终端观测文本（拼接所有非空的 result.content）
def _obs_text(step: dict) -> str:
    # observation 字段（缺省空 dict）
    obs = step.get("observation") or {}
    # 逐个 result 取 content，跳过空内容，用换行连接
    return "\n".join(str(r.get("content")) for r in obs.get("results") or []
                     if r.get("content"))


# 定义 render_steps：把轨迹的 step 列表渲染成带步号的行序列（核心渲染函数）
def render_steps(steps: list, obs_head=OBS_HEAD, keys_head=KEYSTROKES_HEAD,
                 asst_head=ASSISTANT_HEAD) -> str:
    # lines：输出行；ep：agent episode 计数（user 消息不计）
    lines, ep = [], 0
    # 逐条 step 分派渲染
    for s in steps:
        # 该 step 的来源（user / agent / 其他）
        src = s.get("source")
        # 任务提示词：只出现一次，单独标注
        if src == "user":
            lines.append(f"[TASK PROMPT] {_clip(s.get('message'), INSTRUCTION_HEAD)}")
        # agent 步：先渲染其分析/计划消息，再渲染每个工具调用（按键）与终端观测
        elif src == "agent":
            # episode 计数递增（一个 agent step = 一个 episode）
            ep += 1
            # 有分析/计划消息才渲染（截到 asst_head）
            if s.get("message"):
                lines.append(f"[step {ep}] AGENT: {_clip(s['message'], asst_head)}")
            # 遍历该步的每个工具调用（Terminus-2 里就是按键批次）
            for tc in s.get("tool_calls") or []:
                # 工具调用参数
                args = tc.get("arguments") or {}
                # 取 keystrokes 字段；没有就把整个参数 JSON 截 200 字符兜底展示
                keys = args.get("keystrokes", json.dumps(args)[:200])
                # 渲染按键串及其等待时长（duration 影响长命令是否被等够——常见失败点）
                lines.append(f"[step {ep}] KEYSTROKES (duration={args.get('duration')}): "
                             f"{_clip(keys, keys_head)}")
            # 该步执行后的终端观测
            obs = _obs_text(s)
            # 有观测才渲染（截到 obs_head）
            if obs:
                lines.append(f"[step {ep}] TERMINAL: {_clip(obs, obs_head)}")
        # 其他来源的 step（系统消息等）：降级为一行简短展示
        else:
            lines.append(f"[{src}] {_clip(s.get('message'), 500)}")
    # 拼接全部行
    text = "\n".join(lines)
    # 超过全局硬上限时做头尾保留式截断（前 60% + 后 40%），避免整篇超长
    if len(text) > MAX_CHARS:
        # 头部取 60%、尾部取 40%
        head, tail = int(MAX_CHARS * 0.6), int(MAX_CHARS * 0.4)
        # 中间用 TRUNCATED 标记替代被删掉的字符
        text = text[:head] + f"\n...[TRUNCATED {len(text) - head - tail} chars]...\n" + text[-tail:]
    # 返回渲染文本
    return text


# 定义 render_verifier：渲染 trial 的 verifier ground truth（逐测试结果 + 测试日志尾部）
def render_verifier(trial_dir: Path) -> str:
    # 输出块列表
    parts = []
    # ctrf.json：CTRF 格式的机器可读测试报告
    ctrf = trial_dir / "verifier" / "ctrf.json"
    # 报告存在才解析
    if ctrf.exists():
        # 解析失败不能让整个渲染崩掉
        try:
            # results 段（缺省空 dict）
            res = json.loads(ctrf.read_text()).get("results") or {}
            # summary 段：通过/失败/总数
            summ = res.get("summary") or {}
            # 先给一行汇总
            parts.append(f"tests: {summ.get('passed', 0)} passed / "
                         f"{summ.get('failed', 0)} failed / {summ.get('tests', 0)} total")
            # 逐测试渲染：状态 + 测试名
            for t in res.get("tests") or []:
                # 一条测试一行
                line = f"  [{t.get('status')}] {t.get('name')}"
                # 失败测试附断言消息或 traceback（截 1200 字符）——analyst 的关键证据
                msg = t.get("message") or t.get("trace")
                if msg and t.get("status") != "passed":
                    line += f"\n    {_clip(msg, 1200)}"
                # 收集该行
                parts.append(line)
        # ctrf.json 损坏：明说读不了，而不是静默丢弃
        except json.JSONDecodeError:
            parts.append("(ctrf.json unreadable)")
    # test-stdout.txt：pytest 等测试进程的原始 stdout，取末尾 4000 字符（失败通常在尾部）
    so = trial_dir / "verifier" / "test-stdout.txt"
    if so.exists():
        parts.append(f"--- test-stdout tail ---\n{so.read_text(errors='replace')[-4000:]}")
    # 什么都没有就明说"无 verifier 输出"
    return "\n".join(parts) if parts else "(no verifier output)"


# 定义 render_full：渲染一整条 trial 记录（头部元数据 + 轨迹 + verifier ground truth）
def render_full(rec: dict, detail: bool = False) -> str:
    # trial 目录路径（load_trial 已写入 rec）
    trial_dir = Path(rec["trial_dir"])
    # result.json 与 trajectory.json 的解析结果容器
    result, traj = {}, {}
    # 两个源文件路径
    rj, tj = trial_dir / "result.json", trial_dir / "agent" / "trajectory.json"
    # 逐个尝试读取（哪个存在读哪个；解析失败保持空 dict）
    for p, target in ((rj, "result"), (tj, "traj")):
        # 文件存在才读
        if p.exists():
            # 解析并合并进对应容器；JSON 损坏则静默跳过（后面按缺省渲染）
            try:
                (result if target == "result" else traj).update(json.loads(p.read_text()))
            except json.JSONDecodeError:
                pass
    # agent 结果段（token 用量等）
    ar = result.get("agent_result") or {}
    # agent 元数据（episode 数、摘要次数）
    meta = ar.get("metadata") or {}
    # verifier 奖励段
    vr = (result.get("verifier_result") or {}).get("rewards") or {}
    # detail=True 时启用更宽的截断上限
    caps = DETAIL_CAPS if detail else {}
    # 组装三个截断参数（未启用 detail 时回落到默认值）
    kw = {"obs_head": caps.get("obs_head", OBS_HEAD),
          "keys_head": caps.get("keys_head", KEYSTROKES_HEAD),
          "asst_head": caps.get("asst_head", ASSISTANT_HEAD)}
    # 异常信息（可能为 None）
    exc = result.get("exception_info")
    # 组装输出块：头部两行（任务/trial 名；奖励与元数据）
    parts = [
        f"TASK: {result.get('task_name')} | trial: {result.get('trial_name')}",
        f"REWARD: {vr.get('reward')} | episodes: {meta.get('n_episodes')} | "
        f"summarizations: {meta.get('summarization_count')} | "
        f"input_tokens: {ar.get('n_input_tokens')} | output_tokens: {ar.get('n_output_tokens')}",
        # 异常行（有异常才有）
        f"EXCEPTION: {_clip(json.dumps(exc), 1500)}" if exc else "",
        # 空行 + 轨迹大标题 + 轨迹正文
        "", "=== TRAJECTORY ===", render_steps(traj.get("steps") or [], **kw),
        # 空行 + verifier 标题 + verifier 正文（ground truth）
        "", "=== VERIFIER (ground truth) ===", render_verifier(trial_dir),
    ]
    # 过滤空块后拼接返回
    return "\n".join(p for p in parts if p != "")

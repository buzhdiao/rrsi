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
"""Evolvable mechanism substrate for the structural levers (apex port of the


These are inert primitives: importing this module changes nothing. The
proposer WIRES a primitive into main.py to activate a lever, then the
critic/smoke/eval gates judge whether it helps. Dependency-free and
defensive — the harness runs unattended on 240 trials.

Levers:
  skill    — author harness/skills/<name>/SKILL.md and wire a skill_use
             meta-tool in tools.py so the frozen policy can discover+cat them
             (progressive disclosure). This substrate does not force a skills
             pipeline; build one when the evidence calls for it.
  memory   — Memory: fcntl-locked JSONL under APEX_STATE_DIR, semantic +
             episodic. ENTITY-FREE ONLY (critic + SKILL.md enforce): store
             general procedures/lessons, never task-specific I/O — the
             benchmark is evolved and evaluated on the same tasks, so
             persisting concrete task data across trials is memorization.
  client_tool — ToolRegistry: in-process callables the loop can run and inject.
  subagent — subcall(): one bounded, tightly-briefed extra policy call.
"""

# 启用延迟注解求值（PEP 563），类型注解在运行期不立即求值，兼容较低 Python 版本。
from __future__ import annotations

# 导入 fcntl：Unix 文件锁，用于多进程安全的记忆文件读写。
import fcntl
# 导入 json：记忆记录与工具参数的序列化/反序列化。
import json
# 导入 os：读取环境变量与 seek 常量。
import os
# 导入 re：构建防泄漏的实体检测正则。
import re
# 导入 Path：以面向对象方式操作状态目录与文件路径。
from pathlib import Path


def _state_dir() -> Path:
    # 从环境变量读取状态目录，未设置时退回 /tmp 下的默认目录。
    d = os.environ.get("APEX_STATE_DIR", "/tmp/apex_mh_state")
    # 转成 Path 对象。
    p = Path(d)
    # 递归创建目录（已存在则忽略），保证后续读写可用。
    p.mkdir(parents=True, exist_ok=True)
    # 返回状态目录路径。
    return p


# crude entity-ish guard for memory writes (defence-in-depth; critic is primary)
# 记忆写入的"实体泄漏"粗筛正则（纵深防御第一层，主要防线是 critic 审查）：
# 拦截文件系统路径、应用数据目录、任务 id、期望值/参考答案/答案字段等
# 具体任务实体，防止跨试次"记忆化"作弊。
_LEAKY = re.compile(
    r"/filesystem/|\.apps_data|task_[0-9a-f]{6}|expected|reference|"
    r"gold\s*answer|answer\s*[:=]", re.I)


# ---------------------------------------------------------------- memory ----
class Memory:
    """Append-only JSONL memory with fcntl locking under APEX_STATE_DIR. Two
    granularities: semantic (general timeless lessons) and episodic (pattern
    tied to a task TYPE/situation, still entity-free). Reads are ALWAYS
    filtered; never dump the whole store into context."""

    def __init__(self, state_dir: Path | str | None = None):
        # 使用给定状态目录；未指定时走环境变量/默认路径。
        self.dir = Path(state_dir) if state_dir else _state_dir()
        # 确保目录存在。
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str) -> Path:
        # 只允许 semantic/episodic 两种粒度，其他值一律按 semantic 处理。
        kind = kind if kind in ("semantic", "episodic") else "semantic"
        # 返回对应的 JSONL 文件路径（如 mem_semantic.jsonl）。
        return self.dir / f"mem_{kind}.jsonl"

    def write(self, kind: str, note: str, tags: list | None = None,
              dedupe: bool = True) -> bool:
        # 去掉笔记首尾空白（None 按空串处理）。
        note = (note or "").strip()
        # 空笔记或命中泄漏正则的一律拒写。
        if not note or _LEAKY.search(note):
            return False
        # 组装记录：笔记截断到 600 字符，标签最多 6 个，控制记忆膨胀。
        rec = {"note": note[:600], "tags": (tags or [])[:6]}
        # 取目标 JSONL 文件路径。
        p = self._path(kind)
        try:
            # 以追加+读模式打开文件。
            with open(p, "a+") as f:
                # 加排他锁（fcntl），保证多进程并发写的原子性。
                fcntl.flock(f, fcntl.LOCK_EX)
                if dedupe:
                    # 开启去重时先把读指针移到文件头，扫描已有记录。
                    f.seek(0)
                    for line in f:
                        try:
                            # 已存在完全相同的笔记则视为重复，拒绝写入。
                            if json.loads(line).get("note") == rec["note"]:
                                fcntl.flock(f, fcntl.LOCK_UN)
                                return False
                        except json.JSONDecodeError:
                            # 损坏行直接跳过，不中断扫描。
                            pass
                # 把写指针移到文件末尾，准备追加。
                f.seek(0, os.SEEK_END)
                # 追加一条 JSON 行（非 ASCII 原样保留）。
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # 显式解锁。
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001 - never crash the rollout
            # 任何异常都不能让 rollout 崩溃，静默返回失败。
            return False
        # 写入成功。
        return True

    def read(self, kind: str, match: str | None = None, limit: int = 8) -> list:
        # 取目标文件路径。
        p = self._path(kind)
        # 文件不存在说明还没有任何记忆，返回空列表。
        if not p.exists():
            return []
        # 收集符合条件的结果。
        out = []
        try:
            # 只读方式打开。
            with open(p) as f:
                # 加共享锁，允许多个读者并发读取。
                fcntl.flock(f, fcntl.LOCK_SH)
                for line in f:
                    try:
                        # 逐行解析 JSON 记录。
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        # 损坏行跳过。
                        continue
                    if match and match.lower() not in (
                            r.get("note", "") + " " + " ".join(r.get("tags", []))
                    ).lower():
                        # 指定 match 时做大小写不敏感的子串过滤（搜笔记+标签）。
                        continue
                    # 通过过滤的记录加入结果。
                    out.append(r)
                # 读取完毕显式解锁。
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            # 读失败同样不崩溃，返回空列表。
            return []
        # 只返回最后 limit 条（最近的记忆），避免上下文爆炸。
        return out[-limit:]

    def digest(self, kind: str = "semantic", match: str | None = None,
               limit: int = 8) -> str:
        # 读取匹配的若干条记忆。
        recs = self.read(kind, match=match, limit=limit)
        # 无记忆时返回空串（调用方无需注入任何内容）。
        if not recs:
            return ""
        # 把每条笔记渲染成 "- 内容" 的列表行。
        lines = "\n".join(f"- {r['note']}" for r in recs)
        # 返回带标题的记忆摘录文本，可直接拼进提示词。
        return f"\n\n[MEMORY — {kind} lessons from prior tasks]\n{lines}"


# ---------------------------------------------------------- client tools ----
class ToolRegistry:
    """In-process meta-tools the agent loop can run and feed back into context.
    Register callables that compute/parse/transform (NOT touch the judge)."""

    def __init__(self):
        # 工具注册表：名字 -> (可调用对象, 描述)。
        self._tools: dict = {}

    def register(self, name: str, fn, desc: str):
        # 登记一个进程内工具：可调用函数与其描述。
        self._tools[name] = (fn, desc)

    def spec(self) -> str:
        # 没有注册任何工具时返回空串。
        if not self._tools:
            return ""
        # 把所有工具渲染成 "- 名字: 描述" 的行。
        rows = "\n".join(f"- {n}: {d}" for n, (_, d) in self._tools.items())
        # 返回可注入提示词的"可用客户端工具"说明文本。
        return f"\n\n[CLIENT TOOLS you may request]\n{rows}"

    def call(self, name: str, args: dict) -> str:
        # 未注册的工具名返回错误文本。
        if name not in self._tools:
            return f"ERROR: unknown tool {name}"
        try:
            # 关键字参数解包调用工具函数，结果转字符串并截断到 4000 字符。
            return str(self._tools[name][0](**(args or {})))[:4000]
        except Exception as e:  # noqa: BLE001 - never crash the rollout
            # 工具抛异常时返回错误说明，绝不让 rollout 崩溃。
            return f"ERROR: tool {name} raised {type(e).__name__}: {e}"


# -------------------------------------------------------------- subagent ----
async def subcall(model: str, extra_args: dict, brief: str,
                  max_tokens: int = 800, timeout: int = 120) -> str:
    """One bounded extra call to the SAME frozen policy via litellm. Keep the
    brief tight ('return only X'); never delegate open-ended reasoning or a
    whole task. Returns '' on any error (never crashes the rollout)."""
    try:
        # 延迟导入 litellm，保持本模块零依赖（按需加载）。
        import litellm
        # 用同一冻结策略模型发起一次受限的子调用（subagent 杠杆）：
        # 单条 user 消息、限 max_tokens 与超时，并过滤掉 api_key 等敏感参数。
        r = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": brief}],
            max_tokens=max_tokens, timeout=timeout,
            **{k: v for k, v in (extra_args or {}).items()
               if k not in ("api_key",)} if extra_args else {})
        # 取出回复文本并去首尾空白（无内容时按空串处理）。
        return (r.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001
        # 任何错误都返回空串，保证主流程不被子调用拖垮。
        return ""

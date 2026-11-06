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
# 模块说明:结构杠杆的可进化机制基座(mechanism substrate)。这些是惰性原语:
# 导入本模块不改变任何行为。提议者(proposer)把原语"接线"进 main.py 才激活杠杆,
# 再由批评家/冒烟/评测门禁判断是否有效。零依赖且防御式——harness 要无人值守跑
# 300 个 trial。可用杠杆:
#   skill    — 写 harness/skills/<name>/SKILL.md 并在 tools.py 里接一个 skill_use
#              元工具,让冻结策略能发现并 cat 它们(渐进式披露/progressive disclosure)
#   memory   — Memory:RRSI_STATE_DIR 下 fcntl 加锁的 JSONL,语义 + 情景两类;
#              只许存实体无关内容(批评家与 SKILL.md 强制):只能存通用方法/教训,
#              不许存任务特定 I/O——进化与评测用的是同一批任务,跨 trial 持久化
#              具体任务数据等于背题
#   client_tool — ToolRegistry:进程内可调用对象,循环可运行并注入上下文
#   subagent — subcall():一次有界、简报收紧的额外策略调用
"""Evolvable mechanism substrate for the structural levers .

These are inert primitives: importing this module changes nothing. The
proposer WIRES a primitive into main.py to activate a lever, then the
critic/smoke/eval gates judge whether it helps. Dependency-free and
defensive — the harness runs unattended on 300 trials.

Levers:
  skill    — author harness/skills/<name>/SKILL.md and wire a skill_use
             meta-tool in tools.py so the frozen policy can discover+cat them
             (progressive disclosure). This substrate does not force a skills
             pipeline; build one when the evidence calls for it.
  memory   — Memory: fcntl-locked JSONL under RRSI_STATE_DIR, semantic +
             episodic. ENTITY-FREE ONLY (critic + SKILL.md enforce): store
             general procedures/lessons, never task-specific I/O — the
             benchmark is evolved and evaluated on the same tasks, so
             persisting concrete task data across trials is memorization.
  client_tool — ToolRegistry: in-process callables the loop can run and inject.
  subagent — subcall(): one bounded, tightly-briefed extra policy call.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from pathlib import Path


def _state_dir() -> Path:
    # 状态目录:环境变量 RRSI_STATE_DIR,缺省 /tmp/dr_mh_state;确保存在
    d = os.environ.get("RRSI_STATE_DIR", "/tmp/dr_mh_state")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


# crude entity-ish guard for memory writes (defence-in-depth; critic is primary)
# 记忆写入的粗粒度"实体"防泄露正则(纵深防御;批评家才是主防线):命中任务特定
# 标识、URL、"expected/gold answer" 等模式一律拒写
_LEAKY = re.compile(
    r"\b\d{15}-s\d+\b|https?://|expected|gold\s*answer|answer\s*[:=]|"
    r"\bdoi\b|et al\.", re.I)


# ---------------------------------------------------------------- memory ----
class Memory:
    """Append-only JSONL memory with fcntl locking under RRSI_STATE_DIR. Two
    granularities: semantic (general timeless lessons) and episodic (pattern
    tied to a task TYPE/situation, still entity-free). Reads are ALWAYS
    filtered; never dump the whole store into context."""

    def __init__(self, state_dir: Path | str | None = None):
        # 目录可注入,缺省取 _state_dir()
        self.dir = Path(state_dir) if state_dir else _state_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str) -> Path:
        # 只允许 semantic/episodic 两种粒度,非法值回退 semantic
        kind = kind if kind in ("semantic", "episodic") else "semantic"
        return self.dir / f"mem_{kind}.jsonl"

    def write(self, kind: str, note: str, tags: list | None = None,
              dedupe: bool = True) -> bool:
        # 写入前清洗:空串或命中泄露正则直接拒绝
        note = (note or "").strip()
        if not note or _LEAKY.search(note):
            return False
        # 截断限长:笔记 600 字符、标签最多 6 个
        rec = {"note": note[:600], "tags": (tags or [])[:6]}
        p = self._path(kind)
        try:
            with open(p, "a+") as f:
                # fcntl 排他锁保证多进程并发追加安全
                fcntl.flock(f, fcntl.LOCK_EX)
                if dedupe:
                    # 去重:已存在相同笔记则放弃写入
                    f.seek(0)
                    for line in f:
                        try:
                            if json.loads(line).get("note") == rec["note"]:
                                fcntl.flock(f, fcntl.LOCK_UN)
                                return False
                        except json.JSONDecodeError:
                            pass
                f.seek(0, os.SEEK_END)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001 - never crash the rollout
            # 记忆失败绝不能弄垮 rollout
            return False
        return True

    def read(self, kind: str, match: str | None = None, limit: int = 8) -> list:
        # 读取:共享锁;可按子串过滤(note+tags);只返回最后 limit 条
        p = self._path(kind)
        if not p.exists():
            return []
        out = []
        try:
            with open(p) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if match and match.lower() not in (
                            r.get("note", "") + " " + " ".join(r.get("tags", []))
                    ).lower():
                        continue
                    out.append(r)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            return []
        return out[-limit:]

    def digest(self, kind: str = "semantic", match: str | None = None,
               limit: int = 8) -> str:
        # 摘要成可注入上下文的文本块;无记录返回空串(不给模型噪音)
        recs = self.read(kind, match=match, limit=limit)
        if not recs:
            return ""
        lines = "\n".join(f"- {r['note']}" for r in recs)
        return f"\n\n[MEMORY — {kind} lessons from prior tasks]\n{lines}"


# ---------------------------------------------------------- client tools ----
class ToolRegistry:
    """In-process meta-tools the agent loop can run and feed back into context.
    Register callables that compute/parse/transform (NOT touch the judge)."""

    def __init__(self):
        # 名称 -> (可调用对象, 描述)
        self._tools: dict = {}

    def register(self, name: str, fn, desc: str):
        # 注册一个进程内工具(只许计算/解析/变换,不许碰评审器)
        self._tools[name] = (fn, desc)

    def spec(self) -> str:
        # 工具清单文本(可注入提示);未注册任何工具返回空串
        if not self._tools:
            return ""
        rows = "\n".join(f"- {n}: {d}" for n, (_, d) in self._tools.items())
        return f"\n\n[CLIENT TOOLS you may request]\n{rows}"

    def call(self, name: str, args: dict) -> str:
        # 调用工具:未知工具报错;结果截到 4000 字符;异常不外泄
        if name not in self._tools:
            return f"ERROR: unknown tool {name}"
        try:
            return str(self._tools[name][0](**(args or {})))[:4000]
        except Exception as e:  # noqa: BLE001 - never crash the rollout
            return f"ERROR: tool {name} raised {type(e).__name__}: {e}"


# -------------------------------------------------------------- subagent ----
async def subcall(model: str, extra_args: dict, brief: str,
                  max_tokens: int = 800, timeout: int = 120) -> str:
    """One bounded extra call to the SAME frozen policy via litellm. Keep the
    brief tight ('return only X'); never delegate open-ended reasoning or a
    whole task. Returns '' on any error (never crashes the rollout)."""
    try:
        # 局部导入 litellm;extra_args 里剔除 api_key 防止透传泄露
        import litellm
        r = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": brief}],
            max_tokens=max_tokens, timeout=timeout,
            **{k: v for k, v in (extra_args or {}).items()
               if k not in ("api_key",)} if extra_args else {})
        # 只取文本回答并去空白
        return (r.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001
        # 任何错误都返回空串:子调用失败绝不弄垮 rollout
        return ""

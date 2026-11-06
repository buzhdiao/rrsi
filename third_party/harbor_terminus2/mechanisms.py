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
"""Evolvable mechanism substrate for the four structural levers.

These are inert primitives: importing this module changes nothing. The
proposer WIRES a primitive into terminus_2.py to activate a lever, then the
critic/smoke/eval gates judge whether it helps. Kept dependency-free and
defensive — the harness runs unattended on 178 trials.

Levers:
  skill    — author harness/skills/<name>/SKILL.md; call upload_skills() in
             run()/setup() so the model can discover+cat them in-container
             (progressive disclosure). See Terminus2._build_skills_section.
  memory   — Memory: fcntl-locked host-side JSONL, semantic + episodic.
             ENTITY-FREE ONLY (critic + SKILL.md enforce): store general
             procedures/lessons, never task-specific I/O — the benchmark is
             evolved and evaluated on the same tasks, so persisting concrete
             task data across trials is memorization, not improvement.
  client_tool — ToolRegistry: host-side callables the loop can run and inject.
  subagent — subcall(): one bounded, tightly-briefed extra policy call.
"""

# ---------------------------------------------------------------------------
# 中文注释（叠加层：只增不改，既有英文 docstring/注释原样保留）
# 本模块是 RRSI 的"可演化机制基座"（evolvable mechanism substrate），实现论文中
# 四个结构杠杆（levers）的惰性原语：
#   skill       —— 把 harness/skills/<name>/SKILL.md 上传进任务容器，让模型"发现+cat"
#                  按需加载（渐进式披露 progressive disclosure）；
#   memory      —— 主机侧 fcntl 文件锁保护的 JSONL 记忆，分语义记忆（semantic：
#                  与任务无关的通用经验）与情景记忆（episodic：绑定任务"类型"的
#                  模式）两种粒度；强制 entity-free（无实体）——基准在相同任务上
#                  演化并评测，跨 trial 持久化具体任务数据属于"记忆化 memorization"
#                  而非真实能力提升；
#   client_tool —— 主机侧可注册、由代理主循环代跑并回灌上下文的工具注册表；
#   subagent    —— subcall()：一次有界、简报紧凑的额外策略调用。
# 这些原语导入后不改变任何行为：提案者（proposer）把它们"接线"进 terminus_2.py
# 才激活对应杠杆，再由 critic/smoke/eval 门禁（gates）评判是否有效。
# 设计上零依赖且防御式（harness 无人值守跑 178 个 trial）。
# ---------------------------------------------------------------------------
# 启用延迟注解求值（PEP 563），支持 `Path | str` 这类新式联合类型标注
from __future__ import annotations

# fcntl.flock：POSIX 文件锁，保证多进程并发读写记忆 JSONL 的安全
import fcntl
# JSON 编解码：记忆条目按 JSONL（每行一个 JSON 对象）存取
import json
# 读取环境变量 RRSI_STATE_DIR；os.SEEK_END 用于定位文件尾
import os
# 正则：实体泄漏（entity leak）守卫 _LEAKY
import re
# shlex.quote：把路径/命令安全引用进容器 shell，防注入与拆词
import shlex
# 路径对象：状态目录与 SKILL.md 文件定位
from pathlib import Path

# 记忆状态根目录：可用环境变量 RRSI_STATE_DIR 覆盖，默认 /tmp/tbmh_state
STATE_DIR = Path(os.environ.get("RRSI_STATE_DIR", "/tmp/tbmh_state"))

# crude entity-ish guard for memory writes (defence-in-depth; critic is primary)
# 写记忆前的"疑似实体数据"粗筛（纵深防御 defence-in-depth，主防线是 critic 门禁）：
# 命中 /app/、/tests/、pytest 节点（.py::）、expected/reference/answer 等字样的
# 笔记视为疑似任务专属数据，一律拒收
_LEAKY = re.compile(r"/app/|/tests?/|\.py::|expected|reference|answer\s*[:=]",
                    re.I)


# ---------------------------------------------------------------- memory ----
# 记忆机制（论文 lever "memory"）：跨 trial 沉淀经验的唯一持久化通道。
class Memory:
    """Append-only JSONL memory with fcntl locking. Two granularities:
      semantic  — general, timeless procedures/lessons (task-agnostic).
      episodic  — pattern tied to a task TYPE/situation (still entity-free).
    Reads are ALWAYS filtered (by kind + optional predicate/limit); never
    dump the whole store into context."""

    def __init__(self, state_dir: Path | str = STATE_DIR):
        # 记住记忆库根目录（默认取模块级 STATE_DIR，可由调用方覆盖）
        self.dir = Path(state_dir)
        # 递归创建目录（已存在则忽略），确保后续读写前目录就绪
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str) -> Path:
        # 只认 semantic/episodic 两种粒度，其余值防御性回退为 semantic
        kind = kind if kind in ("semantic", "episodic") else "semantic"
        # 返回该粒度对应的 JSONL 文件路径（semantic.jsonl / episodic.jsonl）
        return self.dir / f"{kind}.jsonl"

    # 写入一条"无实体"经验（entity-free lesson）；被拒（空笔记/疑似实体数据）或
    # 重复时返回 False。def 签名跨两行（kind/note/tags/dedupe 参数列表）。
    def write(self, kind: str, note: str, tags: list[str] | None = None,
              dedupe: bool = True) -> bool:
        """Persist one entity-free lesson. Returns False if rejected/duplicate.
        Rejects notes that look like they carry task-specific data."""
        # 归一化输入：None 视为空串并去掉首尾空白
        note = (note or "").strip()
        # 空笔记，或命中实体泄漏正则（疑似任务专属 I/O/答案）→ 拒绝写入
        if not note or _LEAKY.search(note):
            return False
        # 组装记录：笔记截断到 600 字符、标签至多 6 个，防单条记忆膨胀
        rec = {"note": note[:600], "tags": (tags or [])[:6]}
        # 取目标 JSONL 文件路径
        p = self._path(kind)
        # "a+" 追加模式打开：写始终落在文件尾，同时保留读能力用于去重
        with open(p, "a+") as f:
            # 加排他锁（LOCK_EX）：flock 保证多进程并发写入互斥
            fcntl.flock(f, fcntl.LOCK_EX)
            if dedupe:
                # 开启去重：游标移回文件头，从头逐行扫描已有记录
                f.seek(0)
                for line in f:
                    try:
                        # 已存在完全相同的 note → 判定为重复，拒绝再写
                        if json.loads(line).get("note") == rec["note"]:
                            # 释放排他锁后返回 False
                            fcntl.flock(f, fcntl.LOCK_UN)
                            return False
                    except json.JSONDecodeError:
                        # 损坏/截断的行跳过：坏数据不应中断去重扫描
                        pass
            # 去重通过：游标定位到文件末尾准备追加
            f.seek(0, os.SEEK_END)
            # 追加一行 JSON（ensure_ascii=False 保留中文等非 ASCII），以换行结尾
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # 写入完成，释放排他锁
            fcntl.flock(f, fcntl.LOCK_UN)
        # 成功持久化一条新经验
        return True

    # 读取记忆：始终按 kind 过滤，可附加关键词匹配与条数上限；绝不整库倾倒进上下文
    def read(self, kind: str, match: str | None = None, limit: int = 8) -> list:
        # 目标粒度对应的 JSONL 文件路径
        p = self._path(kind)
        # 文件尚不存在说明还没有该类记忆，直接返回空列表
        if not p.exists():
            return []
        # 收集匹配记录的结果列表
        out = []
        with open(p) as f:
            # 加共享锁（LOCK_SH）：允许并发读，但与写入者的排他锁互斥
            fcntl.flock(f, fcntl.LOCK_SH)
            for line in f:
                try:
                    # 逐行反序列化一条 JSON 记录
                    r = json.loads(line)
                except json.JSONDecodeError:
                    # 损坏行直接跳过，不让坏数据中断读取
                    continue
                # 关键词过滤（match 不区分大小写）：在 note 正文与 tags 拼接文本中查找
                # （if 条件表达式跨三行，注释置于条件起始处）
                if match and match.lower() not in (
                        r.get("note", "") + " " + " ".join(r.get("tags", []))
                ).lower():
                    # 不匹配 → 丢弃该条
                    continue
                # 通过过滤的记录加入结果
                out.append(r)
            # 读完释放共享锁
            fcntl.flock(f, fcntl.LOCK_UN)
        # 只返回最近 limit 条（尾部切片），控制注入上下文的体量
        return out[-limit:]

    # 把过滤后的经验汇总成一段可直接注入 prompt 的紧凑记忆块（def 签名跨两行）
    def digest(self, kind: str = "semantic", match: str | None = None,
               limit: int = 8) -> str:
        """Compact, injectable block of filtered lessons (or '')."""
        # 先按相同条件读出过滤后的记录
        recs = self.read(kind, match=match, limit=limit)
        # 没有任何记忆 → 返回空串（调用方据此跳过注入）
        if not recs:
            return ""
        # 把每条笔记渲染为 "- <note>" 的列表行
        lines = "\n".join(f"- {r['note']}" for r in recs)
        # 带标题的记忆区块（标注粒度与"来自既往任务"），供主循环拼进 prompt
        return f"\n\n[MEMORY — {kind} lessons from prior tasks]\n{lines}"


# ---------------------------------------------------------- client tools ----
# 客户端工具机制（论文 lever "client_tool"）：主机侧元工具，由代理主循环
# （agent loop）代为执行并把结果回灌进模型上下文（host-side tool injection）。
class ToolRegistry:
    """Host-side meta-tools the agent loop can run and feed back into context.
    Register callables that compute/parse/transform (NOT touch the verifier)."""

    def __init__(self):
        # 工具表：name -> (可调用对象, 自然语言描述)
        self._tools: dict = {}

    def register(self, name: str, fn, desc: str):
        # 登记一个工具：函数本体与其描述（描述会被模型看到，用于决定是否申请调用）
        self._tools[name] = (fn, desc)

    def spec(self) -> str:
        # 尚未注册任何工具 → 返回空串，主循环无需注入工具说明
        if not self._tools:
            return ""
        # 把每个工具渲染为 "- 名称: 描述" 的清单行
        rows = "\n".join(f"- {n}: {d}" for n, (_, d) in self._tools.items())
        # 带标题的"可请求工具"区块，注入 prompt 告知模型有哪些工具可申请
        return f"\n\n[CLIENT TOOLS you may request]\n{rows}"

    def call(self, name: str, args: dict) -> str:
        # 请求了未注册的工具名 → 返回错误文本（而非抛异常中断 rollout）
        if name not in self._tools:
            return f"ERROR: unknown tool {name}"
        try:
            # 以关键字参数展开调用工具函数，结果转字符串并截断到 4000 字符防上下文膨胀
            return str(self._tools[name][0](**(args or {})))[:4000]
        except Exception as e:  # noqa: BLE001 - never crash the rollout
            # 任何工具异常都吞掉并转为错误文本：绝不因工具崩溃拖垮整个 rollout（回合采样）
            return f"ERROR: tool {name} raised {type(e).__name__}: {e}"


# -------------------------------------------------------------- subagent ----
# 子代理机制（论文 lever "subagent"）：一次有界的额外策略调用（subcall）。
def subcall(llm, brief: str, max_tokens: int = 800) -> str:
    """One bounded extra call to the SAME frozen policy. Keep the brief tight
    ('return only X'); never delegate open-ended reasoning or a whole task.
    PRIOR: sub-calls tend to hurt on small policies (early exit); measure,
    don't assume. Returns '' on any error."""
    try:
        # 延迟导入 Chat：仅在真正发起子调用时才加载依赖
        from harbor.llms.chat import Chat
        # 用同一个（冻结的）策略 LLM 包装出聊天对象
        c = Chat(llm)
        # 发起这次简报紧凑的子调用；空结果归一化为空串并去除首尾空白
        return (c.chat(brief, max_tokens=max_tokens) or "").strip()
    except Exception:  # noqa: BLE001
        # 任何失败都静默返回空串：子调用是可选项，绝不能影响主 rollout
        return ""


# --------------------------------------------------------------- skills -----
# 技能机制（论文 lever "skill"）：把编写好的 SKILL.md 注入容器，供模型
# "发现 + cat" 按需激活（渐进式披露 progressive disclosure）。
# async def 签名跨三行（environment/repo_skills_dir/container_dir 参数列表）。
async def upload_skills(environment, repo_skills_dir: Path | str,
                        container_dir: str = "/root/skills") -> str | None:
    """Copy authored skills (harness/skills/<name>/SKILL.md) into the task
    container and return the container path to assign to self.skills_dir so
    Terminus2._build_skills_section can advertise them. Returns None if there
    are no skills. Wire this in run() before building the skills section."""
    # 技能源目录（仓库侧 harness/skills/）
    repo = Path(repo_skills_dir)
    # 找出所有 <name>/SKILL.md 并按名字排序（*/SKILL.md 只匹配一层子目录）
    mds = sorted(repo.glob("*/SKILL.md"))
    # 没有编写任何技能 → 返回 None，调用方不设置 skills_dir
    if not mds:
        return None
    try:
        # 在任务容器里创建技能根目录；shlex.quote 防路径特殊字符拆词（语句跨两行）
        await environment.exec(f"mkdir -p {shlex.quote(container_dir)}",
                               timeout_sec=10)
        for md in mds:
            # 技能名取自其父目录名（harness/skills/<name>/SKILL.md）
            name = md.parent.name
            # 容器内该技能的目标目录
            dest = f"{container_dir}/{name}"
            # 为每个技能在容器内建独立子目录
            await environment.exec(f"mkdir -p {shlex.quote(dest)}", timeout_sec=10)
            # 读出 SKILL.md 全文，准备传进容器
            body = md.read_text()
            # write via heredoc-safe base64 to avoid quoting issues
            # 延迟导入 base64（只有确有技能需要上传时才用到）
            import base64
            # 将内容 base64 编码：规避引号/换行/heredoc 等 shell 引用陷阱
            b64 = base64.b64encode(body.encode()).decode()
            # 容器内 echo 解码落盘为 SKILL.md（await 语句跨三行，限时 15 秒）
            await environment.exec(
                f"echo {shlex.quote(b64)} | base64 -d > "
                f"{shlex.quote(dest + '/SKILL.md')}", timeout_sec=15)
        # 返回容器内技能目录路径，供赋给 self.skills_dir 以广播技能清单
        return container_dir
    except Exception:  # noqa: BLE001
        # 任何容器执行失败都吞掉：技能上传失败不应中断整个 trial
        return None

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
"""The editable component set K and its structural subset K_str.

The paper tags every candidate edit with the harness component l' it modifies
(Sec. evidence-aware credit assignment). K is the fixed vocabulary of those
tags; K_str are the components that add machinery (a tool, a skill file, a
memory store, an extra policy call) rather than changing text or constants.

A proposer declares the component of each edit; the declaration is validated
against K and, when it is missing or invalid, recovered from the diff using the
domain's file-name signals so a mislabelled prompt tweak cannot pass as a skill.
"""
# 【模块说明（中文）】可编辑组件集合 K 与其结构化子集 K_str。
# 论文给候选的每个编辑打上它所修改的 harness 组件标签 l'（"证据感知的信用分配"一节）。
# K 是这些标签的固定词表；K_str 是"新增机制"类组件（一个工具、一个 skill 文件、
# 一个记忆库、一次额外的策略调用），而不是改文本或常量的组件。
# 提案者（proposer）为每个编辑声明组件；声明要对照 K 校验，缺失或非法时用 diff 和
# domain 的文件名信号恢复——防止一次改 prompt 的编辑冒充 skill 来骗取 novelty 加分
# 或满足 reserved exploration slot（保留探索槽位）。

# 延迟类型注解求值
from __future__ import annotations

# re：所有组件信号都是正则表达式，在 diff 文本上匹配
import re

# K：九个可编辑组件的固定词表（论文组件分类学）。prompt=模型可见文本；
# control_flow=控制流；config=配置；output_plumbing=输出管道；context_mgmt=上下文管理；
# client_tool=客户端工具；skill=技能文件；memory=记忆/状态；subagent=子代理
K = ["prompt", "control_flow", "config", "output_plumbing", "context_mgmt",
     "client_tool", "skill", "memory", "subagent"]
# K_STR：结构化子集——会"新增机制"的组件；novelty 项 nu(l') 只统计这四类（论文 novelty 一节）
K_STR = ["client_tool", "skill", "memory", "subagent"]

# Signals shared by every domain (structural substrate names in mechanisms.py).
# 所有 domain 共享的信号：结构化组件在 harness 机制层的代码特征（对应 mechanisms.py 中的命名）
GENERIC_SIGNALS = [
    # memory 组件的信号：Memory( 构造、remember/recall 方法调用、状态目录常量
    ("memory",      [r"\bMemory\(", r"\.remember\(", r"\.recall\(", r"_STATE_DIR"]),
    # skill 组件的信号：skills/ 目录、SkillRegistry、技能目录/上传等关键词
    ("skill",       [r"skills/", r"SkillRegistry", r"skill_use", r"skill_catalog",
                     r"upload_skills"]),
    # client_tool 组件的信号：工具注册表、工具 spec、客户端工具清单
    ("client_tool", [r"ToolRegistry", r"register_tool", r"tool_spec", r"CLIENT_TOOLS"]),
    # subagent 组件的信号：subcall( 调用、子代理命名
    ("subagent",    [r"\bsubcall\(", r"sub_agent", r"subagent"]),
]

# _STRING_LINE：匹配"改动行是字符串字面量（含 f/r/b 前缀或三引号）"的 diff 行——纯文本编辑的证据
_STRING_LINE = re.compile(r'^[+-]\s*(?:[frb]?["\']|""")')


def text_only(diff: str) -> bool:  # 定义 text_only：判断一次 diff 是否只改了模型可见文本
    """True when every changed line is a string literal or a comment: a
    model-facing text edit, i.e. `prompt`, whatever words the prose contains."""
    # docstring：当所有改动行都是字符串字面量或注释时为真——即 `prompt` 类编辑，
    # 无论文案写的是什么词（防止用花哨 prompt 文案冒充其他组件）
    changed = [l for l in diff.splitlines()  # 逐行扫描 diff
               if (l.startswith("+") or l.startswith("-"))  # 只看增删行
               and not l.startswith("+++") and not l.startswith("---") and l[1:].strip()]  # 排除 diff 文件头（+++ / ---）和纯空白行
    if not changed:  # 没有任何实质改动行
        return False  # 不算文本编辑（空 diff 由别处处理）
    return all(_STRING_LINE.match(l) or l[1:].lstrip().startswith("#") for l in changed)  # 所有改动行要么以引号字符串开头，要么是 Python 注释


def classify_diff(diff: str, domain_signals: list | None = None) -> str:  # 定义 classify_diff：从 diff 内容推断组件标签
    """First matching component in (domain signals, then generic); default prompt."""
    # docstring：先匹配 domain 专属信号，再匹配通用信号；都未命中时默认 prompt
    if text_only(diff):  # 先判纯文本编辑——它优先于一切信号，避免 prompt 文案里出现 "skills/" 字样被误判
        return "prompt"  # 归为 prompt
    for component, pats in (domain_signals or []) + GENERIC_SIGNALS:  # 按 domain 信号在前、通用信号在后的顺序
        if any(re.search(p, diff) for p in pats):  # 该组件的任一正则在 diff 中命中
            return component  # 返回第一个命中的组件（顺序即优先级）
    return "prompt"  # 兜底：没有任何结构信号时按 prompt 处理（文本类改动是常态）


def has_evidence(component: str, diff: str, domain_signals: list | None = None) -> bool:  # 定义 has_evidence：检查 diff 是否携带某组件的证据
    """Does the diff contain any signal of `component`? A declared tag is kept
    only when the diff carries evidence for it: a proposer can name a
    skill/memory/tool/subagent edit without shipping one, or relabel a
    control-flow change as an untried component to satisfy a reserved
    exploration slot, and an unverified tag corrupts T_t, U_t and the novelty
    term."""
    # docstring：diff 中是否含该组件的信号？声明的标签只有在 diff 有证据时才被保留：
    # 提案者可能给编辑贴上 skill/memory/tool/subagent 的标签却没真的写一个，
    # 或把控制流改动改标成未尝试组件来满足保留探索槽位；未验证的标签会污染
    # 已尝试集合 T_t、未尝试集合 U_t 和 novelty 项
    for comp, pats in (domain_signals or []) + GENERIC_SIGNALS:  # 遍历 domain 信号 + 通用信号
        if comp == component and any(re.search(p, diff) for p in pats):  # 找到目标组件且其任一信号在 diff 中命中
            return True  # 有证据
    return False  # 无证据：声明的标签不可信


def normalize(declared: str | None, diff: str,  # 定义 normalize：把"声明标签 + diff"规范成可信标签
              domain_signals: list | None = None) -> str:
    d = (declared or "").strip().lower()  # 声明的标签去空白、转小写（None 视为空串）
    if d in K and has_evidence(d, diff, domain_signals):  # 声明在词表内且 diff 有对应证据：信声明
        return d  # 保留声明的组件标签
    return classify_diff(diff, domain_signals)  # 否则从 diff 重新分类——这就是"按 diff 重新打标签"（loop.py 中记录 retagged 的来源）


def novelty(edit_components: list[str], incumbent_counts: dict) -> int:  # 定义 novelty：论文的新颖项 nu(l')
    """nu(l'): number of STRUCTURAL components the candidate touches that the
    incumbent has never had an accepted edit on. Only ever tie-breaks a
    candidate inside the noise band (Alg. 2 line 5)."""
    # docstring：nu(l') = 候选触及的、现任 harness 从未有过"被接受编辑"的结构化组件数。
    # 只在噪声带内作为决胜项（Alg. 2 第 5 行的 shaped 规则 w_n * nu）
    return sum(1 for c in set(edit_components)  # 对候选触及的组件去重后计数
               if c in K_STR and incumbent_counts.get(c, 0) == 0)  # 仅统计结构化组件（K_str）且现任在该组件上的被接受编辑数为 0

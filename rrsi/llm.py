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
"""Claude client for the three search roles (proposer, analyst, critic).

Claude Opus 4.8 via AnthropicVertex, round-robin over the configured GCP
projects with retry-and-rotate on failure. `cache_prefix` sends a large stable
leading block (constitution + harness source) as its own ephemeral-cached
content block so repeated turns pay to read it once.
"""
# 【模块说明（中文）】三个搜索角色（proposer/analyst/critic）共用的 Claude 客户端。
# 经 AnthropicVertex 调 Claude Opus 4.8，在多个已配置的 GCP 项目间轮转（round-robin），
# 失败时重试并换下一个项目。`cache_prefix` 把大而稳定的头部内容（宪法 + harness 源码）
# 作为独立的 ephemeral 缓存块发送——多轮对话只为读它付一次钱（prompt caching）。

# 延迟类型注解求值
from __future__ import annotations

# itertools：itertools.count 生成轮转游标（线程安全的取 next）
import itertools
# os：读环境变量（项目列表、区域、默认模型）
import os
# re：从模型输出中抢救 JSON
import re
# threading：客户端缓存字典的互斥锁（多线程并行评估时会并发调用）
import threading
# time：指数退避的 sleep
import time

# MODEL：默认搜索模型，可用环境变量 RRSI_SEARCH_MODEL 覆盖
MODEL = os.environ.get("RRSI_SEARCH_MODEL", "claude-opus-4-8")
# _PROJECTS：从 RRSI_VERTEX_PROJECTS（逗号分隔）解析出的 GCP 项目列表，区域默认 global
_PROJECTS = [
    {"project_id": p.strip(), "region": os.environ.get("RRSI_VERTEX_REGION", "global")}  # 每个项目一个 {project_id, region} 字典
    for p in os.environ.get("RRSI_VERTEX_PROJECTS", "").split(",")  # 逗号分隔拆分
    if p.strip()  # 跳过空段
]
# MAX_TOKENS：单次生成的最大输出 token
MAX_TOKENS = 20_000

# _clients：已创建的 Vertex 客户端缓存（按项目下标），避免重复构造
_clients: dict = {}
# _clients_lock：保护 _clients 的锁（ThreadPoolExecutor 并发下 lazy-init 要互斥）
_clients_lock = threading.Lock()
# _rr：全局轮转游标，每次调用 next() 前进一格——把负载均匀摊到各项目
_rr = itertools.count()
# _JSON_SUFFIX：json_only 模式追加到系统提示词的硬约束——只许输出单个 JSON 对象
_JSON_SUFFIX = ("\n\nOutput ONLY a single valid JSON object. No prose before or "
                "after, no markdown fences.")


def _client_for(idx: int):  # 定义 _client_for：取（或懒创建）第 idx 个项目的 Vertex 客户端
    from anthropic import AnthropicVertex  # 函数内导入：只有真正调用时才需要 SDK（模块可被无凭据环境导入）
    if not _PROJECTS:  # 没配置项目
        raise RuntimeError("set RRSI_VERTEX_PROJECTS to a comma-separated list of GCP "  # 明确报错并给出配置方法
                           "projects with Claude on Vertex AI enabled")
    with _clients_lock:  # 进入临界区：多线程同时首次调用时只建一个客户端
        c = _clients.get(idx)  # 先查缓存
        if c is None:  # 没有
            c = AnthropicVertex(**_PROJECTS[idx])  # 用 {project_id, region} 构造客户端
            _clients[idx] = c  # 写回缓存
        return c  # 返回共享客户端


def extract_json(text: str) -> str:  # 定义 extract_json：从模型自由文本里提取 JSON 正文
    t = text.strip()  # 去首尾空白
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)  # 优先剥掉 ```json ...``` 围栏（DOTALL 跨行匹配）
    if m:  # 有围栏
        t = m.group(1).strip()  # 取围栏内内容
    if not t.startswith("{") and not t.startswith("["):  # 开头不是 JSON 定界符（前面混入了说明文字）
        start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=-1)  # 找最早的 { 或 [ 出现位置
        if start != -1:  # 存在 JSON 起点
            t = t[start:]  # 丢弃前缀废话
    if t and t[0] == "{" and not t.endswith("}"):  # 对象没闭合（输出被截断）
        end = t.rfind("}")  # 找最后一个 }
        if end != -1:  # 存在
            t = t[:end + 1]  # 截到闭括号——容忍尾部垃圾
    if t and t[0] == "[" and not t.endswith("]"):  # 数组同理
        end = t.rfind("]")  # 找最后一个 ]
        if end != -1:  # 存在
            t = t[:end + 1]  # 截到闭括号
    return t  # 返回尽量干净的 JSON 文本（仍可能解析失败，由调用方处理）


def generate(prompt: str, system: str | None = None, max_retries: int = 6,  # 定义 generate：全部 LLM 调用的唯一出口
             json_only: bool = False, model: str | None = None,
             max_tokens: int = MAX_TOKENS, cache_prefix: str | None = None) -> str:
    mdl = model or MODEL  # 显式指定模型优先，否则用默认搜索模型
    sys_prompt = (system or "") + (_JSON_SUFFIX if json_only else "")  # 系统提示词 + （可选）JSON 硬约束后缀
    content = ([{"type": "text", "text": cache_prefix,  # 有 cache_prefix 时拆成两个块：
                 "cache_control": {"type": "ephemeral"}},  # 前块打 ephemeral 缓存标记（稳定前缀只计费一次）
                {"type": "text", "text": prompt}] if cache_prefix else prompt)  # 后块是每轮变化的正文；无前缀时直接传字符串
    n = len(_PROJECTS)  # 项目数（轮转的模）
    start = next(_rr)  # 取一个全局递增起点：并发调用散布到不同项目
    last_err: Exception | None = None  # 记录最后一次异常（最终报错用）
    for attempt in range(max_retries):  # 最多 6 次尝试
        idx = (start + attempt) % n  # 每次尝试换下一个项目（失败轮转）
        try:  # 尝试调用
            client = _client_for(idx)  # 该项目的客户端（懒创建）
            kwargs = {"model": mdl, "max_tokens": max_tokens,  # 组装请求参数
                      "messages": [{"role": "user", "content": content}]}
            if sys_prompt:  # 有系统提示词
                kwargs["system"] = sys_prompt  # 单独传 system 字段
            resp = client.messages.create(**kwargs)  # 发起 Vertex AI 调用
            text = "".join(b.text for b in resp.content  # 拼接所有文本块
                           if getattr(b, "type", "") == "text")  # 跳过非文本块（如 tool_use）
            if text:  # 拿到非空文本
                return extract_json(text) if json_only else text  # json_only 模式顺手做 JSON 清洗
            last_err = RuntimeError("empty response")  # 空响应也算失败
        except Exception as e:  # noqa: BLE001 - rotate to the other project
            # 捕获一切异常：配额/网络/项目级故障——换下一个项目重试
            last_err = e  # 记录异常
            time.sleep(min(2 ** attempt, 30))  # 指数退避（1,2,4,...秒，封顶 30 秒）避免雪上加霜
    raise RuntimeError(f"generate failed after {max_retries} tries: {last_err}")  # 重试耗尽：抛出带原因的异常


if __name__ == "__main__":  # 直接运行本文件时的自检入口
    print(generate("Reply with exactly: OK", max_tokens=16))  # 最小调用：验证凭据与网络连通

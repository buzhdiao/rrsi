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

"""Datadog metric submission for the agents runner.

Design goals (in priority order):
  1. NEVER crash the caller. A misconfigured DD client, a serialization
     bug, a network outage, or anything else cannot break the wrapping
     trajectory. The flow runs identically with or without metrics.
  2. NEVER block the event loop. Every emit returns to the caller
     immediately; serialization AND HTTP both happen on worker threads.
  3. Fire-and-forget. We do not wait for the DD intake; failed
     submissions are dropped at debug level.

Architecture:
  caller (event loop)
    └─→ asyncio.create_task(_safely(...))
          └─→ asyncio.to_thread(serialize + submit)
                └─→ DD ThreadedApiClient pool ──HTTPS──→ intake

Two layers of thread offload (asyncio default executor + the DD client's
own pool) is intentional belt-and-suspenders: even if a future SDK bump
regresses to blocking on the typed wrapper, the asyncio.to_thread layer
guarantees the event loop never sees it.

If there is no running event loop (e.g. sync caller, import-time emit),
we fall back to a direct call. ThreadedApiClient still routes HTTP via
its own pool in that path, so the network is non-blocking either way.
"""

# 延迟注解（PEP 563）：注解以字符串形式存储。
from __future__ import annotations

# 导入 asyncio：任务调度与 to_thread 线程卸载。
import asyncio
# 导入 time：取当前时间戳作为指标点。
import time
# 导入异步迭代器与可调用类型注解。
from collections.abc import AsyncIterator, Callable
# 导入 asynccontextmanager：实现 phase 异步上下文管理器。
from contextlib import asynccontextmanager
# 导入 dataclass 装饰器与 field：定义 PhaseHandle。
from dataclasses import dataclass, field
# 导入 typing.Any：通用类型注解。
from typing import Any

# 导入 Datadog 客户端配置与线程安全 API 客户端。
from datadog_api_client import Configuration, ThreadedApiClient
# 导入 v1 指标 API（分布 distribution 指标走 v1）。
from datadog_api_client.v1.api.metrics_api import MetricsApi as MetricsApiV1
# 导入分布点与负载/序列模型。
from datadog_api_client.v1.model.distribution_point import DistributionPoint
from datadog_api_client.v1.model.distribution_points_payload import (
    DistributionPointsPayload,
)
from datadog_api_client.v1.model.distribution_points_series import (
    DistributionPointsSeries,
)
# 导入 v2 指标 API（COUNT 计数指标走 v2）及其负载模型。
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_series import MetricSeries
# 导入 loguru logger：丢弃指标时打 debug 日志。
from loguru import logger

# 导入设置加载函数：读取 Datadog 密钥与环境。
from runner.utils.settings import get_settings

# 模块加载时读取一次全局设置。
settings = get_settings()

# 模块级客户端缓存：共享 HTTP 客户端、计数 API 与分布 API（未配置则为 None）。
_api_client: ThreadedApiClient | None = None
_counts_api: MetricsApi | None = None
_dists_api: MetricsApiV1 | None = None

# Client init is wrapped so a bad API key, broken Configuration, or any other
# init-time exception cannot break this module's import.
# 客户端初始化被 try 包裹：坏的 API key、损坏的 Configuration 或任何初始化
# 异常都不能破坏本模块的导入。
try:
    if settings.DATADOG_API_KEY:
        # 配置了 Datadog API key：创建客户端。
        _config = Configuration()
        # 写入 API key。
        _config.api_key["apiKeyAuth"] = settings.DATADOG_API_KEY
        # 创建线程安全共享客户端。
        _api_client = ThreadedApiClient(_config)
        # 初始化计数指标 API。
        _counts_api = MetricsApi(api_client=_api_client)
        # 初始化分布指标 API。
        _dists_api = MetricsApiV1(api_client=_api_client)
except Exception as e:
    # 初始化失败：禁用指标（各 API 置 None），只打 debug 日志。
    logger.debug(f"Datadog metrics disabled — client init failed: {e}")
    _api_client = None
    _counts_api = None
    _dists_api = None

# 所有指标共享的基础标签：环境与服务名。
BASE_TAGS = [f"env:{settings.ENV.value}", "service:rl-studio-agents"]

# Strong refs for in-flight emits so they aren't GC'd mid-flight.
# asyncio.create_task only weakly references its Task; we add to the set and
# discard via done_callback to avoid an unbounded leak.
# 持有在途上报任务的强引用，防止任务被垃圾回收。asyncio.create_task 只弱引用
# 其 Task；加入集合并在完成回调中移除，避免无限泄漏。
_inflight: set[asyncio.Task[None]] = set()


def _do_count(api: MetricsApi, metric: str, tags: list[str], value: int) -> None:
    """Serialize + submit a COUNT metric. Sync; intended for off-loop use."""
    # 构造单条 COUNT 序列：当前秒级时间戳 + 数值 + 标签。
    series = MetricSeries(
        metric=metric,
        type=MetricIntakeType.COUNT,
        points=[MetricPoint(timestamp=int(time.time()), value=float(value))],
        tags=tags,
    )
    # 提交计数指标负载到 Datadog。
    api.submit_metrics(body=MetricPayload(series=[series]))


def _do_dist(api: MetricsApiV1, metric: str, tags: list[str], v: float) -> None:
    """Serialize + submit a DISTRIBUTION metric. Sync; intended for off-loop use."""
    # 构造分布点：[时间戳, [单值]] 形式。
    point = DistributionPoint(value=[float(int(time.time())), [float(v)]])
    # 构造分布序列。
    series = DistributionPointsSeries(metric=metric, points=[point], tags=tags)
    # 提交分布指标负载到 Datadog。
    api.submit_distribution_points(body=DistributionPointsPayload(series=[series]))


async def _safely(fn: Callable[..., Any], *args: Any) -> None:
    """Run fn off the loop; swallow any Exception so nothing reaches the caller.

    Note: this catches `Exception`, NOT `BaseException`. CancelledError /
    KeyboardInterrupt / SystemExit deliberately propagate — we never want
    metrics to swallow cancellation signals or shutdown.
    """
    try:
        # 把序列化+提交放到工作线程执行，事件循环零阻塞。
        await asyncio.to_thread(fn, *args)
    except Exception as e:
        # 任何失败都只打 debug 日志，绝不抛给调用方。
        logger.debug(f"DD emit dropped: {e}")


def _fire_and_forget(fn: Callable[..., Any], *args: Any) -> None:
    """Schedule fn(*args) so the caller returns immediately. Never raises.

    When a running event loop is available (the normal case in the agent
    runner), serialization + submission both run off the loop via
    `create_task → to_thread`. Otherwise (sync caller, import-time emit)
    we fall through to a direct call — ThreadedApiClient still offloads
    HTTP to its internal pool, so the network is non-blocking either way.
    """
    try:
        # 尝试获取当前运行中的事件循环。
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 无事件循环（同步调用方/导入期上报）：退化为直接同步调用。
        try:
            fn(*args)
        except Exception as e:
            # 失败只打 debug 日志。
            logger.debug(f"DD emit dropped (sync path): {e}")
        return

    try:
        # 有事件循环：调度异步任务，序列化与 HTTP 都在线程池执行。
        task = loop.create_task(_safely(fn, *args))
        # 登记强引用防止任务被 GC。
        _inflight.add(task)
        # 任务完成后自动从集合移除（完成回调直接用集合的 discard 方法）。
        task.add_done_callback(_inflight.discard)
    except Exception as e:
        # 调度失败也只打 debug 日志。
        logger.debug(f"DD emit task-schedule failed: {e}")


def increment(metric: str, tags: list[str] | None = None, value: int = 1) -> None:
    """Fire-and-forget COUNT submit. Returns immediately. Never raises."""
    if _counts_api is None:
        # 未启用 Datadog：静默跳过。
        return
    try:
        # 合并基础标签与调用方标签。
        all_tags = BASE_TAGS + (tags or [])
        # 以发后即忘方式提交计数。
        _fire_and_forget(_do_count, _counts_api, metric, all_tags, value)
    except Exception as e:
        # 任何异常只打 debug 日志。
        logger.debug(f"DD increment dropped: {e}")


def distribution(metric: str, value: float, tags: list[str] | None = None) -> None:
    """Fire-and-forget DISTRIBUTION submit. Returns immediately. Never raises.

    Use for any quantity where you want percentiles (latencies, payload sizes,
    token counts). DD aggregates server-side into a sketch and exposes
    p50/p90/p95/p99/avg/min/max/count on the same metric name.
    """
    if _dists_api is None:
        # 未启用 Datadog：静默跳过。
        return
    try:
        # 合并基础标签与调用方标签。
        all_tags = BASE_TAGS + (tags or [])
        # 以发后即忘方式提交分布值。
        _fire_and_forget(_do_dist, _dists_api, metric, all_tags, value)
    except Exception as e:
        # 任何异常只打 debug 日志。
        logger.debug(f"DD distribution dropped: {e}")


@dataclass
class PhaseHandle:
    """Mutable handle yielded by `phase()` so the body can attach tags or
    emit related metrics under the same prefix without re-typing them.

    - `tag()`   adds a tag known only mid-block (e.g. result count).
    - `value()` emits a sibling distribution (e.g. `populate_objects`).
    - `count()` emits a sibling counter.
    """

    # 指标前缀（如 "studio.populate"）。
    prefix: str
    # 阶段名（如 "load_tasks"）。
    name: str
    # 阶段内动态累积的标签列表（默认空列表）。
    tags: list[str] = field(default_factory=list)

    def tag(self, value: str) -> None:
        # 追加一个执行中途才知道的标签（如结果数量）。
        self.tags.append(value)

    def value(
        self,
        suffix: str,
        v: float,
        extra_tags: list[str] | None = None,
    ) -> None:
        # 在同一前缀下上报一个兄弟分布指标。
        distribution(
            f"{self.prefix}.{suffix}",
            float(v),
            tags=self.tags + (extra_tags or []),
        )

    def count(
        self,
        suffix: str,
        extra_tags: list[str] | None = None,
        value: int = 1,
    ) -> None:
        # 在同一前缀下上报一个兄弟计数指标。
        increment(
            f"{self.prefix}.{suffix}",
            tags=self.tags + (extra_tags or []),
            value=value,
        )


@asynccontextmanager
async def phase(
    name: str,
    *,
    prefix: str,
    tags: list[str] | None = None,
    emit_errors: bool = True,
) -> AsyncIterator[PhaseHandle]:
    """Time one flow step end-to-end, auto-tagging success vs failure.

    Emits `{prefix}.{name}_seconds` (distribution) with `status:ok` on
    success or `status:error` on exception. When `emit_errors=True`,
    also increments `{prefix}.{name}_errors` tagged with `exc:<class>`
    on exception. The exception is always re-raised — wrap the call site
    in try/except if it should be swallowed.

    Cancellation and other BaseException-derived signals propagate
    without emitting, matching the existing teardown convention.
    """
    # 构造阶段句柄（复制初始标签，避免共享可变列表）。
    handle = PhaseHandle(prefix=prefix, name=name, tags=list(tags or []))
    # 记录阶段起始时刻。
    start = time.perf_counter()
    try:
        # 把句柄交给 with 体使用。
        yield handle
    except Exception as e:
        # 异常路径：计算已耗时。
        elapsed = time.perf_counter() - start
        # 上报耗时分布，状态标记为 error。
        distribution(
            f"{prefix}.{name}_seconds",
            elapsed,
            tags=handle.tags + ["status:error"],
        )
        if emit_errors:
            # 允许时额外上报错误计数，附异常类名标签。
            increment(
                f"{prefix}.{name}_errors",
                tags=handle.tags + [f"exc:{type(e).__name__}"],
            )
        # 始终重新抛出原异常（指标不吞错误）。
        raise
    else:
        # 成功路径：上报耗时分布，状态标记为 ok。
        distribution(
            f"{prefix}.{name}_seconds",
            time.perf_counter() - start,
            tags=handle.tags + ["status:ok"],
        )

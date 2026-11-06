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

"""
Utility decorators for the agent runner.
"""

# 导入 asyncio：提供异步睡眠，用于重试之间的指数退避等待。
import asyncio
# 导入 functools：使用 wraps 保留被装饰函数的元信息（函数名等）。
import functools
# 导入 random：为退避时间添加随机抖动（jitter），避免多客户端同步重试造成惊群。
import random
# 导入 time：用 perf_counter 做高精度计时，统计尝试耗时与总耗时。
import time
# 导入 uuid：为每个逻辑调用生成全局唯一的关联 id（call_id）。
import uuid
# 导入 Callable 类型：用于 skip_if 谓词函数的类型注解。
from collections.abc import Callable
# 导入 ContextVar：基于上下文的变量，可在异步任务中安全地携带协程局部状态。
from contextvars import ContextVar

# 导入 loguru 的 logger：结构化日志输出。
from loguru import logger

# 导入 metrics 模块的分布/计数打点函数，用于上报重试相关指标。
from runner.utils.metrics import distribution, increment

# Per-logical-call correlation id, stable across `with_retry` attempts of the
# same wrapped function call. Downstream code (e.g. `runner.utils.llm`) reads
# this to tag each LiteLLM request so Datadog can dedupe retries:
#   unique_count(@call_id where status:429) / unique_count(@call_id)
# gives the "% of logical calls that hit a 429" instead of the inflated
# per-attempt rate.
# 逻辑调用级关联 id：同一次被包装函数调用（含多次重试尝试）共享同一个 id。
# 下游（如 runner.utils.llm）读取它给每次 LiteLLM 请求打标签，让 Datadog 按
# 唯一 call_id 去重，得到"发生 429 的逻辑调用占比"而非被放大的按次统计。
llm_call_id_ctx: ContextVar[str | None] = ContextVar("llm_call_id", default=None)
# 当前重试尝试的序号（第几次尝试），供下游请求打点区分不同 attempt。
llm_attempt_ctx: ContextVar[int] = ContextVar("llm_attempt", default=1)

# Trajectory-scoped "time to first successful LLM" SLI. Set in `modal_labs.run_agent`
# at function entry; consumed by `with_retry` to emit `studio.trajectory.time_to_first_llm_seconds`
# (success) or `studio.trajectory.first_llm_giveup_seconds` (terminal failure) on the
# first wrapped LLM call only. The flag prevents repeat emits on subsequent agent turns.
# Outside `run_agent` these vars remain unset (None / False) and the emit is a no-op,
# so non-trajectory `with_retry` usages (grading, tests) don't pollute trajectory metrics.
# 轨迹级"到首次成功 LLM 调用耗时"SLI 相关上下文：由 modal_labs.run_agent 在入口
# 设置；with_retry 只在首个被包装的 LLM 调用上消费并上报
# studio.trajectory.time_to_first_llm_seconds（成功）或
# studio.trajectory.first_llm_giveup_seconds（彻底失败）。在 run_agent 之外
# 这些变量保持未设置（None / False），上报为 no-op，因此评分、测试等非轨迹
# 场景不会污染轨迹指标。
# 轨迹开始时间戳（perf_counter 时刻）。
trajectory_started_at_ctx: ContextVar[float | None] = ContextVar(
    "trajectory_started_at", default=None
)
# 轨迹相关标签列表，会附加到轨迹 SLI 打点上。
trajectory_tags_ctx: ContextVar[list[str] | None] = ContextVar(
    "trajectory_tags", default=None
)
# 标记本轨迹是否已上报过首次 LLM 结果，防止后续 agent 轮次重复上报。
first_llm_seen_ctx: ContextVar[bool] = ContextVar("first_llm_seen", default=False)


def with_retry(
    max_retries=3,
    base_backoff=1.5,
    jitter: float = 1.0,
    retry_on: tuple[type[Exception], ...] | None = None,
    skip_on: tuple[type[Exception], ...] | None = None,
    skip_if: Callable[[Exception], bool] | None = None,
):
    """
    This decorator is used to retry a function if it fails.
    It will retry the function up to the specified number of times, with a backoff between attempts.

    Args:
        max_retries: Maximum number of retry attempts
        base_backoff: Base backoff time in seconds
        jitter: Random jitter to add to backoff time
        retry_on: Tuple of exception types to retry on. If None, retries on all exceptions.
        skip_on: Tuple of exception types to never retry on, even if they match retry_on.
        skip_if: Predicate function that returns True if the exception should NOT be retried.
                 Useful for checking error messages (e.g., context window errors).
    """
    # 工厂第一层：接收重试策略参数，返回真正的装饰器。

    def decorator(func):
        # 工厂第二层：接收被装饰的异步函数，返回包装后的版本。
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # One id for the whole logical call; reused across retry attempts
            # so each retry of the same call carries the same `call_id` tag.
            # If we're already inside an enclosing `with_retry` (nested
            # decorators), inherit its call_id to keep correlation intact.
            # 为整个逻辑调用分配一个 id，并在各次重试间复用，使同一调用的每次
            # 重试都带相同 call_id 标签。若已处于外层 with_retry（装饰器嵌套），
            # 则继承外层的 call_id 以保持关联一致。
            outer_call_id = llm_call_id_ctx.get()
            # 外层不存在时才生成新的 call_id，否则沿用外层的。
            call_id = outer_call_id or uuid.uuid4().hex
            # 只有本层是"最外层"时才设置上下文变量，并记录 token 以便最终恢复。
            call_id_token = (
                llm_call_id_ctx.set(call_id) if outer_call_id is None else None
            )

            # Per-attempt + cumulative timing for the retry summary log. Used by
            # Datadog to sum "wall-time wasted on retries" across logical calls.
            # We track failed-attempt time and final-attempt time separately so
            # downstream consumers can distinguish productive vs wasted work.
            # 为重试摘要日志记录"每次尝试+累计"耗时。Datadog 用它统计各逻辑调用
            # 在重试上浪费的墙钟时间；分别跟踪失败尝试与最终尝试的耗时，便于
            # 区分有效工作与浪费的工作。
            # 整个逻辑调用序列（含所有重试）的起始时刻。
            sequence_start = time.perf_counter()
            # 累计退避等待时间（秒）。
            total_backoff_seconds = 0.0
            # 失败尝试累计耗时（不含最终那次尝试）。
            failed_attempt_seconds = 0.0
            # 最终（成功或最后失败）那次尝试的耗时。
            final_attempt_seconds = 0.0
            # 最终状态：初始为 failure，成功时改为 success。
            final_status = "failure"
            # Distinguishes a caller-handled skip (skip_on / skip_if / not in
            # retry_on) from a genuine "ran out of retries" terminal failure.
            # Both raise and leave `final_status == "failure"`, but the
            # first-LLM SLI must NOT count a skip as a give-up — the agent
            # loop will handle the exception (e.g. context-window → compact)
            # and try again, and the next attempt's outcome should be what
            # populates time_to_first_llm.
            # 区分"调用方自行处理的跳过"（skip_on / skip_if / 不在 retry_on 中）
            # 与真正的"重试耗尽"终局失败。两者都会抛异常且 final_status 均为
            # failure，但首次 LLM SLI 不应把跳过计为放弃——agent 循环会处理该
            # 异常（如上下文超限→压缩）后再次调用，应让下次尝试的结果来填充
            # time_to_first_llm。
            terminal_outcome: str = "max_retries"
            # 最终异常类名（未失败则为 None）。
            final_error_class: str | None = None
            # 实际使用的尝试次数。
            attempts_used = 0
            try:
                # 按序号 1..max_retries 依次尝试。
                for attempt in range(1, max_retries + 1):
                    # 记录已使用的尝试次数（供 finally 汇总）。
                    attempts_used = attempt
                    # 把当前尝试序号写入上下文，下游可读到 attempt 标签。
                    attempt_token = llm_attempt_ctx.set(attempt)
                    # 本次尝试的起始时刻。
                    attempt_start = time.perf_counter()
                    try:
                        # 执行真正的被包装函数（一次尝试）。
                        result = await func(*args, **kwargs)
                        # 尝试成功：记录本次尝试耗时作为最终尝试耗时。
                        final_attempt_seconds = time.perf_counter() - attempt_start
                        # 最终状态标记为成功。
                        final_status = "success"
                        # 上报成功尝试的耗时分布（按函数名与结果打标签）。
                        distribution(
                            "studio.llm.attempt_seconds",
                            final_attempt_seconds,
                            tags=[
                                f"func:{func.__name__}",
                                "outcome:success",
                            ],
                        )
                        # 成功即返回结果，结束重试循环。
                        return result
                    except Exception as e:
                        # 捕获任何异常：计算本次失败尝试的耗时。
                        attempt_duration = time.perf_counter() - attempt_start
                        # Tentatively classify this attempt as failed; if we
                        # decide below not to retry, promote it to "final".
                        # 先暂时把这次尝试计入"失败尝试耗时"；若下面决定不再
                        # 重试，则再把它改归为"最终尝试耗时"。
                        failed_attempt_seconds += attempt_duration
                        # 记录异常类名，供汇总打点使用。
                        final_error_class = type(e).__name__

                        # "skipped" = caller-handled exception (skip_on /
                        # skip_if / not in retry_on). Distinct from "failure"
                        # so dashboards don't inflate the failure rate with
                        # expected non-retriable cases (e.g. context-window).
                        # "skipped" 表示调用方自行处理的异常（skip_on / skip_if /
                        # 不在 retry_on 内）。与 "failure" 区分开，避免仪表盘把
                        # 预期中不可重试的情形（如上下文超限）计入失败率而虚高。
                        is_skipped = (
                            (skip_on is not None and isinstance(e, skip_on))
                            or (skip_if is not None and skip_if(e))
                            or (retry_on is not None and not isinstance(e, retry_on))
                        )
                        # 上报该失败尝试的耗时分布，标签区分 skipped / failure 及异常类。
                        distribution(
                            "studio.llm.attempt_seconds",
                            attempt_duration,
                            tags=[
                                f"func:{func.__name__}",
                                f"outcome:{'skipped' if is_skipped else 'failure'}",
                                f"exc:{type(e).__name__}",
                            ],
                        )

                        if is_skipped:
                            # 跳过情形：把刚才计入的失败耗时退回。
                            failed_attempt_seconds -= attempt_duration
                            # 该尝试成为最终尝试（不再重试）。
                            final_attempt_seconds = attempt_duration
                            # 终局类型标记为 skipped（区别于重试耗尽）。
                            terminal_outcome = "skipped"
                            # 原样向上抛出异常，交由调用方处理。
                            raise

                        # 判断是否已是最后一次允许的尝试。
                        is_last_attempt = attempt >= max_retries
                        if is_last_attempt:
                            # 最后一次失败：把耗时从"失败尝试"改归"最终尝试"。
                            failed_attempt_seconds -= attempt_duration
                            final_attempt_seconds = attempt_duration
                            # 以 error 级别记录重试耗尽日志（含 call_id）。
                            logger.error(
                                f"Error in {func.__name__}: {repr(e)}, after {max_retries} attempts (call_id={call_id})"
                            )
                            # 抛出最终异常，结束重试。
                            raise

                        # 指数退避：基础退避 × 2^(尝试序号-1)，逐次翻倍。
                        backoff = base_backoff * (2 ** (attempt - 1))
                        # 在 [0, jitter) 区间取随机抖动；jitter 为 0 则不抖动。
                        jitter_delay = random.uniform(0, jitter) if jitter > 0 else 0
                        # 总等待时间 = 指数退避 + 抖动。
                        delay = backoff + jitter_delay
                        # 以 warning 级别记录一次失败与即将重试的信息。
                        logger.warning(
                            f"Error in {func.__name__}: {repr(e)} (call_id={call_id}, attempt={attempt}/{max_retries})"
                        )
                        # 累计退避时间，供最终汇总。
                        total_backoff_seconds += delay
                        # 异步等待指定时长后再进行下一次尝试。
                        await asyncio.sleep(delay)
                    finally:
                        # 无论成功失败，恢复 attempt 上下文变量到进入前的值。
                        llm_attempt_ctx.reset(attempt_token)
            finally:
                # Only the outermost retry wrapper emits the per-logical-call
                # summary; nested wrappers would double-count the same call.
                # 只有最外层重试包装器才上报逻辑调用级汇总；嵌套包装器会重复
                # 统计同一调用。
                # 计算整个调用序列的总墙钟时间。
                total_wall_seconds = time.perf_counter() - sequence_start
                if outer_call_id is None:
                    # 组装汇总打点的公共标签：函数名与最终状态。
                    summary_tags = [
                        f"func:{func.__name__}",
                        f"status:{final_status}",
                    ]
                    if final_status != "success" and final_error_class is not None:
                        # 失败时追加最终异常类名标签。
                        summary_tags.append(f"exc:{final_error_class}")
                    # 上报逻辑调用总耗时分布。
                    distribution(
                        "studio.llm.total_seconds",
                        total_wall_seconds,
                        tags=summary_tags,
                    )
                    # 上报尝试次数分布。
                    distribution(
                        "studio.llm.attempts",
                        float(attempts_used),
                        tags=summary_tags,
                    )
                    if total_backoff_seconds > 0:
                        # 发生过退避：上报退避总耗时。
                        distribution(
                            "studio.llm.backoff_seconds",
                            total_backoff_seconds,
                            tags=summary_tags,
                        )
                    if failed_attempt_seconds > 0:
                        # 存在失败尝试：上报失败尝试累计耗时。
                        distribution(
                            "studio.llm.failed_attempt_seconds",
                            failed_attempt_seconds,
                            tags=summary_tags,
                        )
                    if attempts_used > 1:
                        # 发生过重试：上报重试次数（尝试数-1）计数。
                        increment(
                            "studio.llm.retries",
                            value=attempts_used - 1,
                            tags=summary_tags,
                        )
                    if final_status != "success":
                        # 最终失败：上报错误计数。
                        increment(
                            "studio.llm.errors",
                            tags=summary_tags,
                        )

                    # Trajectory-scoped "time to first LLM" SLI — fires once per
                    # trajectory on the first wrapped LLM call that reaches a
                    # *resolved* outcome: either a success, or a max_retries
                    # give-up. A caller-handled skip (skip_on / skip_if / not in
                    # retry_on) is treated as "not yet resolved" — the agent
                    # loop will catch the exception and likely call again (e.g.
                    # context-window → compact → retry), so we leave the flag
                    # unset so the next attempt populates the SLI correctly.
                    # Without this, a compact-then-success flow would emit
                    # `first_llm_never_succeeded` followed by never emitting
                    # `time_to_first_llm_seconds`, systematically over-counting
                    # give-ups and under-counting successes.
                    # 轨迹级"首次 LLM 调用"SLI——每条轨迹只在首个达到*确定*结果
                    # 的被包装 LLM 调用上触发一次：要么成功，要么重试耗尽放弃。
                    # 调用方自行处理的跳过视为"尚未定局"——agent 循环会捕获异常
                    # 并很可能再次调用（如上下文超限→压缩→重试），因此不置位
                    # 标记，让下一次尝试来正确填充 SLI。否则"先压缩后成功"的
                    # 流程会先报 first_llm_never_succeeded 且永远不再报
                    # time_to_first_llm_seconds，系统性高估放弃、低估成功。
                    trajectory_started_at = trajectory_started_at_ctx.get()
                    if (
                        trajectory_started_at is not None
                        and not first_llm_seen_ctx.get()
                        and terminal_outcome != "skipped"
                    ):
                        # 计算从轨迹开始到现在的经过时间。
                        elapsed_since_trajectory_start = (
                            time.perf_counter() - trajectory_started_at
                        )
                        # 取出轨迹标签副本，附加到轨迹 SLI 打点。
                        traj_tags = list(trajectory_tags_ctx.get() or [])
                        if final_status == "success":
                            # 首次 LLM 成功：上报"到首次成功 LLM 的耗时"。
                            distribution(
                                "studio.trajectory.time_to_first_llm_seconds",
                                elapsed_since_trajectory_start,
                                tags=traj_tags,
                            )
                            # 同时上报首次成功所用的尝试次数。
                            distribution(
                                "studio.trajectory.first_llm_attempts",
                                float(attempts_used),
                                tags=traj_tags,
                            )
                        else:
                            # 首次 LLM 彻底失败：附加异常类名标签（若已知）。
                            exc_tags = (
                                [f"exc:{final_error_class}"]
                                if final_error_class
                                else []
                            )
                            # 计数上报"首次 LLM 从未成功"事件。
                            increment(
                                "studio.trajectory.first_llm_never_succeeded",
                                tags=traj_tags + exc_tags,
                            )
                            # 上报"到首次放弃 LLM 的耗时"。
                            distribution(
                                "studio.trajectory.first_llm_giveup_seconds",
                                elapsed_since_trajectory_start,
                                tags=traj_tags + exc_tags,
                            )
                        # 置位"已见过首次 LLM 结果"标记，后续轮次不再重复上报。
                        first_llm_seen_ctx.set(True)

                # The structured retry-summary log keeps the same gate it had
                # before metrics existed: only log when something interesting
                # happened, since logs are the expensive ingestion path while
                # distributions aggregate server-side.
                # 结构化重试摘要日志保持引入指标之前的门槛：只在"发生了有意思
                # 的事"时才打日志——日志是昂贵的摄取通道，而分布指标在服务端聚合。
                # 发送条件：本层是最外层，且（发生过多于一次尝试或最终失败）。
                should_emit = outer_call_id is None and (
                    attempts_used > 1 or final_status == "failure"
                )
                if should_emit:
                    # 用 loguru 的 bind 附加结构化字段后输出 info 日志。
                    logger.bind(
                        message_type="llm_retry_summary",
                        call_id=call_id,
                        func_name=func.__name__,
                        attempts=attempts_used,
                        retries_used=max(attempts_used - 1, 0),
                        total_backoff_seconds=round(total_backoff_seconds, 4),
                        failed_attempt_seconds=round(failed_attempt_seconds, 4),
                        final_attempt_seconds=round(final_attempt_seconds, 4),
                        total_wall_seconds=round(total_wall_seconds, 4),
                        final_status=final_status,
                        final_error_class=final_error_class,
                    ).info(
                        f"llm_retry_summary call_id={call_id} func={func.__name__} "
                        f"attempts={attempts_used} status={final_status} "
                        f"backoff_s={total_backoff_seconds:.2f} "
                        f"failed_attempt_s={failed_attempt_seconds:.2f} "
                        f"final_attempt_s={final_attempt_seconds:.2f}"
                    )
                if call_id_token is not None:
                    # 本层设置过 call_id 上下文，恢复到进入前的值。
                    llm_call_id_ctx.reset(call_id_token)

        # 返回包装后的函数。
        return wrapper

    # 返回装饰器本身（with_retry(...) 的求值结果）。
    return decorator

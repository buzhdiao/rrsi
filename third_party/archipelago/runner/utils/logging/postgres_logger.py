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

# 延迟注解（PEP 563）：注解以字符串形式存储。
from __future__ import annotations

# 导入 asyncio：队列、任务与异步原语。
import asyncio
# 导入 contextlib：suppress 用于按异常类型静默处理。
import contextlib
# 导入 json：把 extra 上下文序列化为 JSON。
import json
# 导入 typing.Any：日志 dict 的通用值类型。
from typing import Any
# 导入 uuid4 并别名为 uuid：生成日志主键 id。
from uuid import uuid4 as uuid

# 导入 asyncpg：PostgreSQL 异步驱动。
import asyncpg
# 导入 loguru 模块本体：Message 类型注解。
import loguru

# 导入设置加载函数：读取 POSTGRES_URL 等配置。
from runner.utils.settings import get_settings

# 模块加载时读取一次全局设置。
settings = get_settings()

# 待写入数据库的日志队列（元素为 dict，None 作为停机哨兵）。
_log_queue: asyncio.Queue[dict[str, Any] | None] | None = None
# 后台消费任务句柄。
_worker_task: asyncio.Task[None] | None = None
# 懒初始化的异步锁，防止并发启动 worker 的竞态。
_init_lock: asyncio.Lock | None = None
# worker 持有的单一持久连接。
_conn: asyncpg.Connection | None = None
_stopping: bool = False  # Block new enqueues during shutdown
# 停机标记：置位后拒绝新日志入队。


def _generate_trajectory_log_id() -> str:
    # 生成形如 "log_<32位hex>" 的唯一日志主键。
    return f"log_{uuid().hex}"


async def _log_worker() -> None:
    """
    Background worker that processes logs from the queue using a single connection.
    Exits when it receives a `None` sentinel or is cancelled.
    """
    # 声明修改模块级连接与队列变量。
    global _conn, _log_queue

    if not settings.POSTGRES_URL or _log_queue is None:
        # 前置条件不满足（无 DSN 或队列未初始化）：打印并退出 worker。
        print("[Postgres Logger] POSTGRES_URL is not set or queue not initialized")
        return

    # 本函数局部持有的连接引用。
    conn: asyncpg.Connection | None = None
    try:
        # 建立单一持久连接：10 秒连接超时、每条命令 10 秒超时。
        conn = await asyncpg.connect(
            dsn=settings.POSTGRES_URL,
            timeout=10,  # connect timeout
            command_timeout=10,  # per-command timeout
        )
        # 发布到模块级变量，供外部查询连接状态。
        _conn = conn
        print("[Postgres Logger] Connected with single persistent connection")

        # 主循环：持续从队列取日志并写库。
        while True:
            try:
                # 阻塞式获取一条日志。
                log_data = await _log_queue.get()
            except asyncio.CancelledError:
                # worker 被取消：跳出循环。
                break

            if log_data is None:
                # 收到 None 哨兵：正常退出。
                break

            if conn is None:
                # 连接缺失（理论上不会发生）：丢弃该条，继续。
                print("[Postgres Logger] Connection not established")
                continue

            try:
                # 执行参数化 INSERT：写入轨迹日志表（log_extra 转为 jsonb 列）。
                await conn.execute(
                    """
                    INSERT INTO trajectory_logs (
                        trajectory_log_id, trajectory_id, log_timestamp,
                        log_message, log_level, log_extra
                    )
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    log_data["trajectory_log_id"],
                    log_data["trajectory_id"],
                    log_data["log_timestamp"],
                    log_data["log_message"],
                    log_data["log_level"],
                    log_data["log_extra"],
                )
            except Exception as e:
                # 单条写库失败只打印，不中断 worker。
                print(f"[Postgres Logger] Error inserting log: {repr(e)}")
            finally:
                # 标记该条目处理完成，供 queue.join() 统计。
                _log_queue.task_done()

    except Exception as e:
        # 连接或循环级别的致命错误：打印并退出。
        print(f"[Postgres Logger] Worker error: {repr(e)}")
    finally:
        try:
            if conn is not None and not conn.is_closed():
                # 连接仍打开：关闭它。
                await conn.close()
        except (asyncio.CancelledError, RuntimeError) as e:
            # 停机期间的取消/运行时错误：静默处理，只打印提示。
            print(
                f"[Postgres Logger] Suppressed close error during shutdown: {repr(e)}"
            )
        except Exception as e:
            # 其他关闭异常：打印。
            print(f"[Postgres Logger] Error during connection close: {repr(e)}")
        finally:
            # 清空模块级连接引用并打印关闭完成。
            _conn = None
            print("[Postgres Logger] Connection closed")


async def _ensure_worker_started() -> None:
    """
    Ensure the background worker is running (with locking to prevent races).
    """
    # 声明修改模块级队列、任务与锁。
    global _log_queue, _worker_task, _init_lock

    if _init_lock is None:
        # 首次调用：创建初始化锁。
        _init_lock = asyncio.Lock()

    if _log_queue is not None and _worker_task is not None and not _worker_task.done():
        # 快路径：队列与 worker 均已就绪且在运行，直接返回。
        return

    # 拿锁后再检查，避免并发重复创建。
    async with _init_lock:
        if _log_queue is None:
            # 首次初始化：创建容量 1000 的有界队列（满时丢弃日志而非阻塞）。
            _log_queue = asyncio.Queue(maxsize=1000)

        if _worker_task is None or _worker_task.done():
            # worker 未启动或已结束：重新创建后台任务。
            _worker_task = asyncio.create_task(
                _log_worker(), name="postgres-logger-worker"
            )
            print("[Postgres Logger] Started background worker")


async def postgres_sink(message: loguru.Message) -> None:
    """
    Queue a log message to be written to the database.
    This is non-blocking and safe to call even during spikes.
    Expected `message.record` interface with fields used below.
    """
    # 声明修改模块级停机标记。
    global _stopping

    # 取出 loguru 记录；缺失则直接返回。
    record = getattr(message, "record", None)
    if not record:
        return

    # 从 extra 中取出轨迹 id（日志按轨迹归属入库）。
    trajectory_id = record.get("extra", {}).get("trajectory_id")
    if not trajectory_id:
        # 无轨迹 id 的日志不入库。
        return

    if not settings.POSTGRES_URL:
        # 未配置数据库：静默跳过。
        return

    if _stopping:
        # 停机中：拒绝新入队。
        return

    try:
        # 确保队列与后台 worker 已就绪。
        await _ensure_worker_started()

        if _log_queue is None:
            # 队列仍未初始化（异常情况）：放弃本条。
            print("[Postgres Logger] Queue not initialized")
            return

        # 组装待入库的日志数据。
        log_data = {
            "trajectory_log_id": _generate_trajectory_log_id(),
            "trajectory_id": trajectory_id,
            "log_timestamp": record["time"],
            "log_message": record["message"],
            "log_level": record["level"].name,
            "log_extra": json.dumps(record["extra"], default=str),
        }

        try:
            # 非阻塞入队：高峰期也不阻塞日志调用方。
            _log_queue.put_nowait(log_data)
        except asyncio.QueueFull:
            # 队列已满：丢弃日志并提示（背压策略为丢弃）。
            print("[Postgres Logger] Queue full, dropping log")

    except Exception as e:
        # 入队环节的任何异常都不影响主流程。
        print(f"[Postgres Logger] Error queuing log: {repr(e)}")


async def teardown_postgres_logger(timeout: float = 180.0) -> None:
    """
    Flush all pending logs and shut down the worker cleanly.
    Idempotent. Call from your app's shutdown path BEFORE the loop closes.
    """
    # 声明修改停机标记、队列与任务。
    global _stopping, _log_queue, _worker_task

    # 置位停机标记：此后 sink 不再入队新日志。
    _stopping = True

    if _log_queue is None or _worker_task is None:
        # 从未初始化过：幂等直接返回。
        return

    # Wait for queue to drain with timeout
    # 等待队列排空（带超时上限）。
    try:
        with contextlib.suppress(RuntimeError):
            await asyncio.wait_for(_log_queue.join(), timeout=timeout)
    except TimeoutError:
        # 排空超时：打印并强制进入关闭流程。
        print(
            f"[Postgres Logger] Queue drain timed out after {timeout}s, forcing shutdown"
        )

    # Send shutdown signal
    # 向队列投递 None 哨兵通知 worker 退出（事件循环关闭期的 RuntimeError 静默）。
    with contextlib.suppress(RuntimeError):
        await _log_queue.put(None)

    # Wait for worker to finish
    # 等待 worker 任务结束。
    try:
        await asyncio.wait_for(_worker_task, timeout=timeout)
    except (TimeoutError, asyncio.CancelledError):
        # worker 超时未退出：强制取消并回收。
        print("[Postgres Logger] Worker shutdown timed out, cancelling task")
        _worker_task.cancel()
        with contextlib.suppress(Exception):
            await _worker_task
    finally:
        # 最终清空模块级状态，恢复可重新初始化。
        _worker_task = None
        _log_queue = None
        _stopping = False

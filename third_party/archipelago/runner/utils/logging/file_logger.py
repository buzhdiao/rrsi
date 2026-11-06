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

# 导入 json：把日志条目序列化为 JSON 行。
import json
# 导入 Path：处理日志文件路径与父目录创建。
from pathlib import Path
# 导入 Lock：线程锁，保证多线程写文件的原子性。
from threading import Lock
# 导入 TextIO：文件句柄类型注解。
from typing import TextIO

# 导入 loguru 模块本体：用于 Message 类型注解。
import loguru
# 导入 loguru 的 logger：兜底记录写入失败信息。
from loguru import logger

# 导入设置加载函数：读取日志文件路径配置。
from runner.utils.settings import get_settings

# 模块加载时读取一次全局设置。
settings = get_settings()

# 文件 logger 要求配置日志路径，否则直接报错。
if not settings.FILE_LOG_PATH:
    raise ValueError("FILE_LOG_PATH must be set to use the file logger")

# 模块级文件句柄缓存（懒加载，None 表示尚未打开）。
_file_handle: TextIO | None = None
# 保护文件句柄创建与写入的互斥锁。
_file_lock: Lock = Lock()


def _ensure_log_file() -> TextIO:
    # 声明使用模块级句柄变量。
    global _file_handle

    if _file_handle is not None:
        # 句柄已打开：直接复用。
        return _file_handle

    if settings.FILE_LOG_PATH is None:
        # 无路径配置：防御性检查，报错。
        raise ValueError("FILE_LOG_PATH must be set to use the file logger")

    # 构造日志文件路径对象。
    log_path = Path(settings.FILE_LOG_PATH)
    # 递归创建父目录（已存在则忽略）。
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # 以追加模式、UTF-8 编码打开日志文件并缓存句柄。
    _file_handle = log_path.open("a", encoding="utf-8")
    # 返回新打开的句柄。
    return _file_handle


def file_sink(message: loguru.Message) -> None:
    # 取出 loguru 的完整日志记录。
    record = message.record

    # 组装 JSON 日志条目：ISO 时间戳、级别、消息正文与 extra 上下文。
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "extra": record["extra"],
    }

    try:
        # 加锁保证多线程下句柄创建与写入的原子性。
        with _file_lock:
            # 懒加载确保文件已打开。
            log_file = _ensure_log_file()
            # 写一行 JSON（default=str 兜底不可序列化对象）。
            log_file.write(json.dumps(log_entry, default=str) + "\n")
            # 立即刷盘，避免崩溃丢失日志。
            log_file.flush()
    except Exception as exc:
        # 写文件失败不影响主流程：只在 debug 级别记录。
        logger.debug(f"Error writing log to file: {exc!r}")


async def teardown_file_logger() -> None:
    # 声明使用模块级句柄变量。
    global _file_handle

    # 加锁后检查：无人打开过文件则直接返回。
    with _file_lock:
        if _file_handle is None:
            return

        try:
            # 关闭前先刷盘，确保缓冲日志落盘。
            _file_handle.flush()
        finally:
            # 无论刷盘是否成功都要关闭句柄并重置缓存，避免悬挂句柄。
            _file_handle.close()
            _file_handle = None

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

# 导入 sys：拿到标准输出流，供 loguru 绑定控制台 sink。
import sys

# 导入 loguru 的 logger：本模块统一配置日志 sink。
from loguru import logger

# 导入环境枚举与设置加载函数。
from runner.utils.settings import Environment, get_settings

# 模块加载时读取一次全局设置。
settings = get_settings()


def setup_logger() -> None:
    # 先清空 loguru 默认 sink，从零开始按配置装配。
    logger.remove()

    if settings.DATADOG_LOGGING:
        # Datadog logger
        # 启用 Datadog sink：延迟导入，避免未启用时加载重型客户端。
        from .datadog_logger import datadog_sink  # import-check-ignore

        logger.debug("Adding Datadog logger")
        # 注册 Datadog sink：DEBUG 级别起，enqueue=True 跨线程安全。
        logger.add(datadog_sink, level="DEBUG", enqueue=True)

    if settings.REDIS_LOGGING:
        # Redis logger
        # 启用 Redis sink（延迟导入）。
        from .redis_logger import redis_sink  # import-check-ignore

        logger.debug("Adding Redis logger")
        # 注册 Redis sink：INFO 级别起。
        logger.add(redis_sink, level="INFO")

    if settings.FILE_LOGGING:
        # File logger
        # 启用文件 sink（延迟导入）。
        from .file_logger import file_sink  # import-check-ignore

        logger.debug("Adding File logger")
        # 注册文件 sink：DEBUG 级别起。
        logger.add(file_sink, level="DEBUG")

    if settings.POSTGRES_LOGGING:
        # Postgres logger
        # 启用 Postgres sink（延迟导入）。
        from .postgres_logger import postgres_sink  # import-check-ignore

        logger.debug("Adding Postgres logger")
        # 注册 Postgres sink：INFO 级别起。
        logger.add(postgres_sink, level="INFO")

    if settings.ENV == Environment.LOCAL:
        # Local logger
        # 本地环境：彩色、带变量诊断的人类可读控制台输出。
        logger.add(
            sys.stdout,
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            colorize=True,
        )
    else:
        # Structured logger
        # 非本地环境：JSON 序列化的结构化输出，便于机器采集。
        logger.add(
            sys.stdout,
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            serialize=True,
        )


async def teardown_logger() -> None:
    # 等待 loguru 队列中排队的异步日志全部处理完。
    await logger.complete()

    if settings.POSTGRES_LOGGING:
        # Postgres logger
        # 关闭 Postgres sink：刷空队列并停止后台 worker。
        from .postgres_logger import teardown_postgres_logger  # import-check-ignore

        logger.debug("Tearing down Postgres logger")
        await teardown_postgres_logger()

    if settings.FILE_LOGGING:
        # 关闭文件 sink：刷盘并关闭句柄。
        from .file_logger import teardown_file_logger  # import-check-ignore

        logger.debug("Tearing down File logger")
        await teardown_file_logger()

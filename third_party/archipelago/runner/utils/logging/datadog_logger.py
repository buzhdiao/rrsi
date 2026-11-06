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

# 导入 json：把日志消息序列化为 JSON 字符串。
import json

# 导入 loguru 模块本体：用于 Message 类型注解。
import loguru
# 导入 Datadog 客户端配置与线程安全 API 客户端。
from datadog_api_client import Configuration, ThreadedApiClient
# 导入 Datadog 日志提交 API。
from datadog_api_client.v2.api.logs_api import LogsApi
# 导入 HTTP 日志批量与单条模型。
from datadog_api_client.v2.model.http_log import HTTPLog
from datadog_api_client.v2.model.http_log_item import HTTPLogItem
# 导入 loguru 的 logger：兜底记录发送失败信息。
from loguru import logger

# 导入设置加载函数：读取 Datadog 密钥与环境配置。
from runner.utils.settings import get_settings

# 模块加载时读取一次全局设置。
settings = get_settings()

# Datadog logger 要求同时配置 API key 与 App key，否则直接报错拒绝启动。
if not settings.DATADOG_API_KEY or not settings.DATADOG_APP_KEY:
    raise ValueError(
        "DATADOG_API_KEY and DATADOG_APP_KEY must be set to use the Datadog logger"
    )

# 创建 Datadog 客户端配置对象。
configuration = Configuration()
# 写入 API key（用于数据上报鉴权）。
configuration.api_key["apiKeyAuth"] = settings.DATADOG_API_KEY
# 写入 App key（用于管理操作鉴权）。
configuration.api_key["appKeyAuth"] = settings.DATADOG_APP_KEY

# 用线程安全客户端包装配置，供日志提交 API 共享使用。
api_client = ThreadedApiClient(configuration)


def datadog_sink(message: loguru.Message):
    """Send logs to Datadog."""
    # 取出 loguru 的完整日志记录（含时间、级别、extra 等）。
    record = message.record

    try:
        # 组装 Datadog 标签：环境与轨迹 id。
        tags = {
            "env": settings.ENV.value,
            "trajectory_id": record["extra"].get("trajectory_id", None),
        }
        # 把标签拼接为 "k:v,k:v" 形式，值为 None 的标签跳过。
        ddtags = ",".join([f"{k}:{v}" for k, v in tags.items() if v is not None])

        # 组装完整日志消息体：环境、轨迹 id、级别、来源文件/行号/函数/模块、
        # 进程/线程名、extra 上下文与消息正文。
        msg = {
            "env": settings.ENV.value,
            "trajectory_id": record["extra"].get("trajectory_id", None),
            "level": record["level"].name,
            "file": record["file"].path,
            "line": record["line"],
            "function": record["function"],
            "module": record["module"],
            "process": record["process"].name,
            "thread": record["thread"].name,
            "extra": record["extra"],
            "message": record["message"],
        }

        # 构造单条 HTTP 日志项：标签 + JSON 化的消息体 + 固定服务名。
        log_item = HTTPLogItem(
            ddtags=ddtags,
            message=json.dumps(msg, default=str),
            service="rl-studio-trajectory",
        )
        # 提交日志到 Datadog（包裹为批量结构）。
        _ = LogsApi(api_client=api_client).submit_log(body=HTTPLog([log_item]))
    except Exception as e:
        # 上报失败不能影响主流程：只在 debug 级别记录错误。
        logger.debug(f"Error sending log to Datadog: {e}")

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

# 导入 json：把日志数据序列化为字符串存入 Redis。
import json

# 导入 loguru 模块本体：Message 类型注解。
import loguru

# 导入共享的 Redis 客户端与设置加载函数。
from runner.utils.redis import redis_client
from runner.utils.settings import get_settings

# 模块加载时读取一次全局设置。
settings = get_settings()


async def redis_sink(message: loguru.Message) -> None:
    # 取出 loguru 的完整日志记录。
    record = message.record

    # 从 extra 中取出轨迹 id，用于路由到对应日志流。
    trajectory_id = record["extra"].get("trajectory_id")

    if not trajectory_id:
        # 无轨迹 id 的日志不写入 Redis。
        return

    # 组装日志数据：ISO 时间戳、级别、正文与 extra 上下文。
    log_data = {
        "log_timestamp": record["time"].isoformat(),
        "log_level": record["level"].name,
        "log_message": record["message"],
        "log_extra": record["extra"],
    }

    # 按轨迹 id 拼接 Redis Stream 流名（前缀:轨迹id）。
    stream_name = f"{settings.REDIS_STREAM_PREFIX}:{trajectory_id}"

    # 向流中追加一条日志（JSON 字符串形式，default=str 兜底不可序列化对象）。
    await redis_client.xadd(stream_name, {"log": json.dumps(log_data, default=str)})
    # 刷新流过期时间为 12 小时（43200 秒），实现滚动清理。
    await redis_client.expire(stream_name, 43200)  # 12 hours

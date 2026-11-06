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

# 导入 redis 的异步子模块：提供 asyncio 兼容的 Redis 客户端。
import redis.asyncio as redis

# 导入设置加载函数：读取 Redis 连接配置。
from runner.utils.settings import get_settings

# 模块加载时读取一次全局设置。
settings = get_settings()
# 提取 Redis 主机地址。
REDIS_HOST = settings.REDIS_HOST
# 提取 Redis 端口。
REDIS_PORT = settings.REDIS_PORT
# 提取 Redis 用户名。
REDIS_USER = settings.REDIS_USER
# 提取 Redis 密码。
REDIS_PASSWORD = settings.REDIS_PASSWORD

# 四项连接信息缺一不可，否则直接报错拒绝启动。
if not REDIS_HOST or not REDIS_PORT or not REDIS_USER or not REDIS_PASSWORD:
    raise ValueError("Redis configuration is not set")

# 创建模块级共享的异步 Redis 客户端（懒连接，供日志 sink 等复用）。
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    username=REDIS_USER,
)

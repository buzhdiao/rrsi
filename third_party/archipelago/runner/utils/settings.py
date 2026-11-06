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

# 导入 Enum 基类：定义运行环境枚举。
from enum import Enum
# 导入 cache 装饰器：让 get_settings 只读一次配置并缓存结果。
from functools import cache

# 导入 pydantic-settings 的 BaseSettings：从环境变量自动装配配置。
from pydantic_settings import BaseSettings


class Environment(Enum):
    # 本地开发环境。
    LOCAL = "local"
    # 开发环境。
    DEV = "dev"
    # 演示环境。
    DEMO = "demo"
    # 生产环境。
    PROD = "prod"


class Settings(BaseSettings):
    # 运行环境，默认本地。
    ENV: Environment = Environment.LOCAL

    # Agent execution hard timeout
    # Agent 执行的硬超时（论文中 agent 轨迹的运行时长上限）。
    AGENT_TIMEOUT_SECONDS: int = 12 * 60 * 60  # 12 hours

    # RL Studio API
    # RL Studio 服务端 API 地址与密钥（提交轨迹/评分结果用）。
    RL_STUDIO_API: str | None = None
    RL_STUDIO_API_KEY: str | None = None

    # Webhook for saving results
    # 保存结果的 webhook 地址与密钥。
    SAVE_WEBHOOK_URL: str | None = None
    SAVE_WEBHOOK_API_KEY: str | None = None

    # Postgres logging
    # Postgres 日志开关与连接串。
    POSTGRES_LOGGING: bool = False
    POSTGRES_URL: str | None = None

    # Redis logging
    # Redis 日志开关、连接信息与日志流名前缀。
    REDIS_LOGGING: bool = False
    REDIS_HOST: str | None = None
    REDIS_PORT: int | None = None
    REDIS_USER: str | None = None
    REDIS_PASSWORD: str | None = None
    REDIS_STREAM_PREFIX: str = "trajectory_logs"

    # Datadog logging
    # Datadog 日志开关与双密钥。
    DATADOG_LOGGING: bool = False
    DATADOG_API_KEY: str | None = None
    DATADOG_APP_KEY: str | None = None

    # File logging
    # 文件日志开关与路径。
    FILE_LOGGING: bool = False
    FILE_LOG_PATH: str | None = None

    # LiteLLM Proxy
    # If set, all LLM requests will be routed through the proxy
    # LiteLLM 代理：设置后所有 LLM 请求都会经代理路由。
    LITELLM_PROXY_API_BASE: str | None = None
    LITELLM_PROXY_API_KEY: str | None = None

    # Scraping / web content
    # 网页抓取/搜索相关密钥：Firecrawl 抓取、SearchAPI（YouTube 字幕）、
    # Reddit 请求代理。
    ACE_FIRECRAWL_API_KEY: str | None = None
    ACE_SEARCHAPI_API_KEY: str | None = None  # YouTube transcript API
    ACE_REDDIT_PROXY: str | None = None  # Proxy for Reddit requests


@cache
def get_settings() -> Settings:
    # 实例化 Settings：pydantic-settings 自动从环境变量读取；cache 保证全程只构建一次。
    return Settings()

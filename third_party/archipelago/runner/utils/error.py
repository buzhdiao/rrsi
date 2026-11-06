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

"""Error classification logic for distinguishing system errors from model errors."""

# 导入 httpx：用于识别 HTTP 客户端相关异常并按状态码分类。
import httpx
# 导入 LiteLLM 的各类 LLM 异常，用于把异常归类为"系统错误"或"模型错误"。
from litellm.exceptions import (
    APIConnectionError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
# 导入 MCP 协议库的异常类型，用于判定 MCP 会话是否已致命失效。
from mcp import McpError

# 导入图片抓取失败异常：属于确定性输入问题，不可重试。
from runner.utils.image_fetch import ImageFetchError


def is_system_error(exception: Exception) -> bool:
    """Determine if an exception represents a system error (retryable) vs model error.

    System errors are transient infrastructure issues that can be retried.
    Model errors are non-retryable failures like context overflow.

    Returns:
        True if the exception is a system error (should use ERROR status),
        False if it's a model error (should use FAILED status).
    """
    # 第一类：限流、超时、服务不可用、连接错误、服务器内部错误——均为瞬时
    # 基础设施问题，可重试，判为系统错误。
    if isinstance(
        exception,
        (
            RateLimitError,
            Timeout,
            ServiceUnavailableError,
            APIConnectionError,
            InternalServerError,
        ),
    ):
        # 返回 True 表示系统错误（可重试，应记为 ERROR 状态）。
        return True

    # BadRequestError could be either, check the error message
    # BadRequestError 可能属于两类之一，需检查错误消息文本再判断。
    if isinstance(exception, BadRequestError):
        # 把异常消息转为小写便于子串匹配。
        error_str = str(exception).lower()
        if "exceeded your current quota" in error_str:
            # 配额超限：属于瞬时/账号层面问题，判为系统错误。
            return True  # System error
        # If it's context/token/multimodal related, it's a model error
        # 若消息涉及上下文/token/多模态能力缺失，则是模型错误（不可重试）。
        if (
            "context" in error_str
            or "token" in error_str
            or "is not a multimodal model" in error_str
            or "does not support multimodal" in error_str
        ):
            # 返回 False 表示模型错误（应记为 FAILED 状态）。
            return False  # Model error
        # 其余 BadRequest 视为配置/基础设施问题，判为系统错误。
        return True  # System error (configuration/infrastructure issue)

    # Model errors (non-retryable)
    # 上下文窗口超限（ContextWindowExceededError，对应论文中 agent 上下文预算
    # 耗尽）是确定性模型错误，重试无益。
    if isinstance(exception, ContextWindowExceededError):
        return False

    # ValueError is typically a configuration/validation error (non-retryable)
    # ValueError 通常是配置/校验错误，不可重试。
    if isinstance(exception, ValueError):
        return False

    # RuntimeError (e.g. SSE error events) — non-retryable
    # RuntimeError（如 SSE 错误事件）——不可重试。
    if isinstance(exception, RuntimeError):
        return False

    # Image fetch failures are deterministic input/contract problems
    # (oversize, unsupported scheme, bad URL, 4xx). Retrying won't fix them.
    # 图片抓取失败属于确定性的输入/契约问题（超限、协议不支持、URL 错误、4xx），
    # 重试无法解决。
    if isinstance(exception, ImageFetchError):
        return False

    # httpx HTTP errors — 5xx are retryable, 4xx are not
    # httpx 的 HTTP 状态错误——5xx 可重试，4xx 不可重试。
    if isinstance(exception, httpx.HTTPStatusError):
        # 按响应状态码判断：≥500 判为系统错误。
        return exception.response.status_code >= 500

    # httpx transport/connection errors are retryable system errors
    # httpx 的传输/连接层错误（连接、读、写失败）是可重试的系统错误。
    if isinstance(exception, (httpx.ConnectError, httpx.ReadError, httpx.WriteError)):
        return True

    # Unknown exceptions default to system error (safer to retry than fail permanently)
    # 未知异常默认判为系统错误（宁可重试也不要永久失败）。
    return True


def is_fatal_mcp_error(exception: Exception) -> bool:
    """Determine if an exception is fatal and should immediately end the agent run.

    Fatal errors indicate the MCP session/connection is dead and cannot recover.
    Non-fatal errors can be reported to the LLM and the agent can continue.

    Args:
        exception: The exception to check.

    Returns:
        True if the error is fatal (session terminated, connection dead),
        False if the error is recoverable.
    """
    # Check for MCP-specific errors
    # 先检查 MCP 专有异常。
    if isinstance(exception, McpError):
        # Check error code - handle both positive 32600 (current MCP bug) and
        # negative -32600 (JSON-RPC 2.0 standard) for forward compatibility
        # 检查错误码——同时兼容正数 32600（当前 MCP 的 bug）与负数 -32600
        # （JSON-RPC 2.0 标准），保证向前兼容。
        error_code = (
            getattr(exception.error, "code", None)
            if hasattr(exception, "error")
            else None
        )
        if error_code in (32600, -32600):
            # 32600 = Invalid Request：会话已失效，判为致命错误。
            return True

        # Fallback to string matching for robustness
        # 兜底用字符串匹配，增强健壮性。
        if "Session terminated" in str(exception):
            # 会话已终止：判为致命错误。
            return True

    # Check for FastMCP client disconnection errors
    # 再检查 FastMCP 客户端断连类错误。
    if isinstance(exception, RuntimeError):
        # 取异常消息文本。
        error_str = str(exception)
        # FastMCP raises this when the client session has been closed/corrupted
        # FastMCP 在客户端会话已被关闭/损坏时抛出该消息。
        if "Client is not connected" in error_str:
            # 客户端未连接：判为致命错误。
            return True

    # 其余情况均视为可恢复的非致命错误。
    return False

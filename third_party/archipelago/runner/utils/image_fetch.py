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

"""Fetch remote image URLs and re-encode them as data: URLs.

CLI-wrapping agent harnesses (codex_agent, claude_code_agent, gemini_cli_agent,
cursor_agent, …) all share the same problem: the CLI's image-input flag takes a
local file path, but task definitions in Studio commonly embed images as
`image_url` blocks with remote URLs (presigned S3, https). Rather than each
MCP server doing its own outbound HTTP — which would multiply SSRF surface,
egress policy, streaming/timeout handling, and per-protocol error handling —
this helper resolves remote URLs to base64-encoded data URLs on the harness
side. The MCP servers then only need to decode the data URL and write bytes
to disk.

The 5MB-per-image cap matches Codex CLI's documented soft guideline and is
a reasonable upper bound for the model context budget of any vision model.
"""

# 延迟注解（PEP 563）：注解以字符串形式存储，降低运行时开销。
from __future__ import annotations

# 导入 base64：把图片字节编码为 base64 以构造 data: URL。
import base64
# 导入 typing.Any：图片条目 dict 的通用值类型。
from typing import Any

# 导入 httpx：异步 HTTP 客户端，用于抓取远程图片。
import httpx
# 导入 loguru logger：记录抓取解析过程。
from loguru import logger

# 单张图片大小上限：5MB（与 Codex CLI 的软性上限一致）。
MAX_IMAGE_BYTES = 5 * 1024 * 1024
# 单次 HTTP 请求的超时时间（秒）。
FETCH_TIMEOUT_SECONDS = 30.0
# httpx.AsyncClient's default is 20; we want enough hops to traverse
# legitimate CDN/presigned-URL chains but not unbounded.
# httpx.AsyncClient 默认是 20；这里限制重定向跳数，既能穿过合法的
# CDN/预签名 URL 链，又不至于无上限。
MAX_REDIRECTS = 10

# Allowed schemes for inline pass-through (no fetch needed).
# 允许直接透传（无需抓取）的 URL 前缀。
_DATA_SCHEMES = ("data:",)
# Schemes the helper will resolve to bytes via outbound HTTP.
# 本助手会通过外发 HTTP 解析为字节流的 URL 前缀。
_HTTPS_SCHEMES = ("https://",)


class ImageFetchError(Exception):
    """Raised when an image URL cannot be resolved within budget."""


def _sniff_image_mime(data: bytes) -> str | None:
    """Identify a known image format from the leading bytes.

    Returns the MIME type ("image/png", "image/jpeg", "image/gif",
    "image/webp") if the magic bytes match a supported format, else None.

    Used as a fallback when the upstream response advertises a generic
    content-type (e.g. S3 stores objects as `binary/octet-stream` when no
    ContentType was set on upload — see
    rl-studio/server/packages/custom_field_files/service.py upload_files()).
    Without this, every downstream MCP server's `_extension_for_mime` maps
    the wrong MIME to `.bin`, and the resulting file is rejected by Gemini
    CLI's `@path` resolver (extension-based via mime/lite) and by Studio's
    `mcp__gateway__filesystem_read_image_file` (extension allowlist).
    """
    # PNG 魔数：\x89PNG\r\n\x1a\n，命中返回 image/png。
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    # JPEG 魔数：以 \xff\xd8\xff 开头，命中返回 image/jpeg。
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    # GIF 魔数：GIF87a 或 GIF89a 两种版本，命中返回 image/gif。
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    # WEBP = RIFF<size>WEBP; need 12 bytes to confirm.
    # WEBP 是 RIFF 容器：前 4 字节 "RIFF"、第 9-12 字节 "WEBP"，需 12 字节确认。
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # 都不匹配：返回 None 表示无法识别。
    return None


async def _fetch_streaming(
    client: httpx.AsyncClient, url: str
) -> tuple[bytes, str | None]:
    """GET `url`, following redirects manually with per-hop https validation.

    `client.stream("GET", ...)` is invoked with redirects disabled at the
    client level (see resolve_to_data_urls). We follow them ourselves so we
    can reject any hop that tries to land on a non-https URL — otherwise an
    https→http://internal-host redirect would silently bypass the SSRF
    mitigation and the docstring's https-only guarantee.

    Body is streamed with an incremental byte counter; the connection is
    aborted past MAX_IMAGE_BYTES so a hostile or misconfigured source can't
    blow memory.
    """
    # 当前请求 URL，随重定向逐步更新。
    current = url
    # 最多跟随 MAX_REDIRECTS 次重定向（循环次数多留 1 次用于实际请求）。
    for _ in range(MAX_REDIRECTS + 1):
        # 每一跳都要求 https：非 https 目标直接拒绝（SSRF 防护）。
        if not current.startswith(_HTTPS_SCHEMES):
            raise ImageFetchError(
                f"Refusing to follow redirect to non-https URL: {current[:80]}"
            )
        # 以流式方式发起 GET 请求（响应体不一次性读入内存）。
        async with client.stream(
            "GET", current, timeout=FETCH_TIMEOUT_SECONDS
        ) as response:
            # 若响应是重定向：手动解析下一跳。
            if response.is_redirect:
                # 从 Location 头读取重定向目标。
                location = response.headers.get("location")
                if not location:
                    # 缺少 Location 头：抛错，无法继续。
                    raise ImageFetchError(
                        f"Redirect from {current[:80]} has no Location header"
                    )
                # Resolve relative locations against the current URL.
                # 相对 Location 以当前 URL 为基准解析为绝对 URL。
                current = str(httpx.URL(current).join(location))
                # 继续下一轮循环，跟随重定向。
                continue

            # 非重定向：状态码非 2xx 时抛出 HTTPStatusError。
            response.raise_for_status()
            # 用分块列表累积响应体字节。
            chunks: list[bytes] = []
            # 已接收字节总数。
            total = 0
            # 流式迭代响应字节块。
            async for chunk in response.aiter_bytes():
                # 累加本块长度。
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    # 超过 5MB 上限立即中止，防止恶意源撑爆内存。
                    raise ImageFetchError(
                        f"Image exceeds {MAX_IMAGE_BYTES}-byte cap during fetch from {url[:80]}"
                    )
                # 保存本块。
                chunks.append(chunk)
            # 合并所有字节块为完整图片数据。
            raw = b"".join(chunks)
            # 从响应头提取 content-type，去掉参数部分（如 ;charset），空则置 None。
            mime = (
                response.headers.get("content-type", "").split(";")[0].strip() or None
            )
            # If the response advertised a generic MIME (e.g. S3's
            # binary/octet-stream default when no ContentType was set on
            # upload), sniff magic bytes for a real image format. Only
            # override when sniff is confident — leave non-image bytes
            # alone so we don't mislabel something.
            # 若响应头给出的是通用 MIME（如 S3 上传未设 ContentType 时的
            # binary/octet-stream 默认值），则用魔数嗅探真实图片格式。只在
            # 嗅探有把握时才覆盖——非图片字节保持原样，避免误标。
            if not mime or not mime.startswith("image/"):
                # 只取前 16 字节足够识别魔数。
                sniffed = _sniff_image_mime(raw[:16])
                if sniffed is not None:
                    # 嗅探成功：用嗅探到的 MIME 覆盖。
                    mime = sniffed
            # 返回原始字节与最终确定的 MIME 类型。
            return raw, mime

    # 重定向次数超限仍未拿到最终响应：抛错。
    raise ImageFetchError(
        f"Too many redirects (>{MAX_REDIRECTS}) starting from {url[:80]}"
    )


def _to_data_url(raw: bytes, mime: str | None) -> str:
    # MIME 未知时兜底为通用二进制流类型。
    mime = mime or "application/octet-stream"
    # 把字节做 base64 编码并转为 ASCII 字符串。
    encoded = base64.b64encode(raw).decode("ascii")
    # 组装 RFC 2397 格式的 data: URL。
    return f"data:{mime};base64,{encoded}"


async def resolve_to_data_urls(
    images: list[dict[str, Any]],
    *,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Rewrite every entry in `images` so its `url` is a `data:` URL.

    - `data:...` entries pass through unchanged.
    - `https://...` entries are fetched (streamed, size-capped) and re-encoded.
    - Other schemes (http, s3, file, …) are rejected with ImageFetchError so the
      caller sees a clear contract violation rather than a silent CLI failure
      downstream.

    Each entry's other keys (e.g. `detail`) are preserved.
    """
    # 空列表快速返回，无需创建客户端。
    if not images:
        return []

    # 记录是否由本函数自建 HTTP 客户端（决定 finally 中是否负责关闭）。
    owns_client = client is None
    if client is None:
        # follow_redirects=False so _fetch_streaming can validate the scheme
        # at each hop and refuse https → http redirects (SSRF guard).
        # 关闭自动重定向，让 _fetch_streaming 能在每一跳校验协议并拒绝
        # https → http 重定向（SSRF 防护）。
        client = httpx.AsyncClient(follow_redirects=False)

    # 存放解析结果的新列表。
    resolved: list[dict[str, Any]] = []
    try:
        # 逐条处理图片条目。
        for i, image in enumerate(images):
            # 取出条目的 url 字段。
            url = image.get("url")
            if not isinstance(url, str) or not url:
                # url 缺失或类型非法：抛契约错误。
                raise ImageFetchError(f"Image {i} has no url")

            if url.startswith(_DATA_SCHEMES):
                # data: URL 无需抓取，浅拷贝原条目直接透传。
                resolved.append(dict(image))
                continue

            if url.startswith(_HTTPS_SCHEMES):
                try:
                    # 流式抓取 https 图片（带大小上限与逐跳 https 校验）。
                    raw, mime = await _fetch_streaming(client, url)
                except (httpx.HTTPError, httpx.InvalidURL) as exc:
                    # 把底层 HTTP 异常包装为 ImageFetchError，并保留异常链。
                    raise ImageFetchError(
                        f"Failed to fetch image {i} from {url[:80]}: {exc}"
                    ) from exc
                # 记录解析成功日志（截断 URL，附字节数与 MIME）。
                logger.info(
                    f"image_fetch: resolved {url[:80]} -> data url ({len(raw)} bytes, mime={mime})"
                )
                # 复制原条目，保留除 url 外的其他键（如 detail）。
                new_entry = dict(image)
                # 把 url 替换为 base64 data: URL。
                new_entry["url"] = _to_data_url(raw, mime)
                # 加入结果列表。
                resolved.append(new_entry)
                continue

            # 其他协议（http、s3、file 等）：显式拒绝，让调用方看到清晰的契约违规。
            raise ImageFetchError(
                f"Unsupported image URL scheme for image {i}: "
                f"{url[:40]}... (expected data: or https:)"
            )
    finally:
        if owns_client:
            # 本函数自建的客户端由本函数负责异步关闭，避免连接泄漏。
            await client.aclose()

    # 返回所有条目均已是 data: URL 的列表。
    return resolved

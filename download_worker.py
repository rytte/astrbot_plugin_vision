"""独立 yt-dlp 进程；通过 stdin 接收配置，不把签名 URL 或 Cookie 写入命令行。"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import socket
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode


class DownloadRejected(ValueError):
    """可向调用者显示的下载拒绝原因。"""


class NetworkRejected(OSError):
    """保留可公开的地址拦截原因，供 yt-dlp 包装异常后读取。"""


FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")


def guard_network(allow_fake_ip: bool = False) -> None:
    # 所有 Python 网络连接（包括重定向和媒体 CDN）都检查实际 DNS 结果。
    original = socket.getaddrinfo

    def public_addresses(*args, **kwargs):
        addresses = original(*args, **kwargs)
        host = args[0] if args else kwargs.get("host")
        if isinstance(host, bytes):
            host = host.decode("ascii", errors="ignore")
        domain = (
            isinstance(host, str)
            and "." in host
            and not host.lower().rstrip(".").endswith((".local", ".localhost"))
        )
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            domain = False
        if not addresses:
            raise NetworkRejected("下载地址未返回可用的 DNS 结果")
        for item in addresses:
            address = ipaddress.ip_address(item[4][0].split("%", 1)[0])
            if address in FAKE_IP_NETWORK:
                if allow_fake_ip and domain:
                    continue
                raise NetworkRejected(
                    "DNS 返回了 198.18.0.0/15 Fake-IP 地址；如使用本机代理，请开启“兼容本机代理 Fake-IP DNS”后重试"
                )
            if not address.is_global or address.is_multicast:
                raise NetworkRejected("下载地址解析到了非公网地址，已停止访问")
        return addresses

    socket.getaddrinfo = public_addresses


def download_failure(error: Exception, stage: str = "") -> str:
    """只返回固定提示，避免上游异常中的签名链接、Cookie 等内容泄露。"""
    from yt_dlp.networking.exceptions import HTTPError, SSLError, TransportError
    from yt_dlp.utils import ContentTooShortError, PostProcessingError

    pending, seen = [error], set()
    errors = []
    while pending and len(seen) < 24:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, NetworkRejected | DownloadRejected):
            return str(current)
        errors.append(current)
        for name in ("cause", "__cause__", "__context__"):
            cause = getattr(current, name, None)
            if isinstance(cause, BaseException):
                pending.append(cause)
        details = getattr(current, "exc_info", None)
        if (
            isinstance(details, tuple)
            and len(details) > 1
            and isinstance(details[1], BaseException)
        ):
            pending.append(details[1])
    prefix = f"{stage}失败：" if stage else ""
    for current in errors:
        # yt-dlp 的部分下载错误只保留文本；只提取状态码，不复制原始消息。
        status = current.status if isinstance(current, HTTPError) else None
        match = re.search(r"\bHTTP Error ([45]\d{2})\b", str(current), re.IGNORECASE)
        if status is None and match:
            status = int(match[1])
        if status is not None:
            reason = {
                403: "服务器拒绝访问，请检查平台访问限制或 Cookie",
                404: "视频或媒体地址不存在、已失效",
                412: "平台拒绝请求，可能触发了风控，请稍后重试",
                429: "请求过于频繁，请稍后重试",
            }.get(status, "服务器响应异常，请稍后重试")
            return f"{prefix}HTTP {status}，{reason}"
    if any(
        isinstance(current, TimeoutError)
        or re.search(r"\b(?:timed out|timeout)\b", str(current), re.IGNORECASE)
        for current in errors
    ):
        return prefix + "网络连接或读取超时，请稍后重试并检查网络或代理"
    if any(isinstance(current, SSLError) for current in errors):
        return prefix + "TLS 连接或证书校验失败，请检查网络、代理及系统证书"
    if any(isinstance(current, socket.gaierror) for current in errors):
        return prefix + "DNS 解析失败，请检查网络或代理"
    if any(
        isinstance(current, ConnectionError | TransportError | ContentTooShortError)
        for current in errors
    ):
        return prefix + "网络连接失败或下载中断，请稍后重试并检查网络或代理"
    if any(isinstance(current, PostProcessingError) for current in errors):
        return prefix + "FFmpeg 处理失败，请检查 FFmpeg 配置及下载文件是否完整"
    if any("fresh cookies" in str(current).lower() for current in errors):
        return prefix + (
            "yt-dlp 提示需要有效的浏览器 Cookie（不一定需要登录）；也可能是平台风控或签名适配失效，"
            "请更新插件和 yt-dlp，必要时配置“yt-dlp Cookie 文件路径”"
        )
    return prefix + (
        "yt-dlp 无法解析或下载该视频；可能需要登录、链接已失效、网络受限或平台暂不支持"
    )


def validate_info(info: dict, max_duration: float, max_bytes: int) -> None:
    if info.get("_type", "video") != "video" or "entries" in info:
        raise DownloadRejected("不支持合集或播放列表，请发送单个视频链接")
    if info.get("is_live") or info.get("live_status") in {
        "is_live",
        "is_upcoming",
        "post_live",
    }:
        raise DownloadRejected("不支持直播或尚未完成的视频")
    duration = info.get("duration")
    if duration is not None and (
        not isinstance(duration, int | float)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise DownloadRejected("无法确定视频时长，已停止下载")
    if duration is not None and duration > max_duration:
        raise DownloadRejected("分享视频超过配置的时长上限")
    formats = info.get("requested_formats") or [info]
    if sum(item.get("filesize") or 0 for item in formats) > max_bytes:
        raise DownloadRejected("分享视频超过配置的大小上限")
    if not any(item.get("vcodec") != "none" for item in formats):
        raise DownloadRejected("分享内容不包含可用视频画面")
    # 不让 FFmpeg 或外部下载器访问网络；合并仅处理下载后的本地文件。
    for item in formats:
        if item.get("protocol") not in {
            "http",
            "https",
            "m3u8_native",
            "http_dash_segments",
        }:
            raise DownloadRejected("该视频的下载协议暂不支持")


def register_douyin_extractor(ydl) -> None:
    """在当前下载实例中补齐抖音匿名会话与 WebSign，不修改 yt-dlp 安装文件。"""
    from yt_dlp.extractor.tiktok import DouyinIE as BaseDouyinIE

    # 请求流程参考 https://github.com/yt-dlp/yt-dlp/pull/17826 。
    # 使用相同的提取器键覆盖内置实现，短链经 Generic 跳转后也走这里。
    class DouyinIE(BaseDouyinIE):
        def _visitor_id(self) -> str:
            cookies = self._get_cookies(self._WEBPAGE_HOST)
            for name in (
                "UIFID",
                "UIFID_TEMP",
                "UIFIDTEMP",
                "uifid",
                "uifid_temp",
                "uifidtemp",
            ):
                if cookie := cookies.get(name):
                    if cookie.value:
                        return cookie.value
            response = self._request_webpage(self._WEBPAGE_HOST, None)
            visitor = response.headers.get("x-tt-trace-host")
            if not visitor:
                raise DownloadRejected(
                    "抖音未返回可用访客标识；可能触发平台风控，请稍后重试或配置有效的浏览器 Cookie"
                )
            return visitor

        def _real_extract(self, url):
            video_id = self._match_id(url)
            cookies = self._get_cookies(self._WEBPAGE_HOST)
            # 保留管理员提供的 Cookie，只在缺少匿名访问凭据时补齐。
            if not cookies.get("passport_csrf_token"):
                self._set_cookie(".douyin.com", "passport_csrf_token", uuid.uuid4().hex)
            if not cookies.get("ttwid"):
                self._request_webpage(
                    self._WEBPAGE_HOST + "ttwid/register",
                    None,
                    data=json.dumps(
                        {"aid": 6383, "service": "www.douyin.com"}
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                if not self._get_cookies(self._WEBPAGE_HOST).get("ttwid"):
                    raise DownloadRejected(
                        "抖音未返回匿名访问 Cookie；可能触发平台风控，请稍后重试或配置有效的浏览器 Cookie"
                    )
            visitor = self._visitor_id()
            timestamp = str(int(time.time()))
            query = {
                "aweme_id": video_id,
                "aid": 6383,
                "uifid": visitor,
                "timestamp": timestamp,
            }
            # 签名覆盖最终发送的 query 字节，参数顺序及编码必须一致。
            query["x-secsdk-web-signature"] = hashlib.md5(
                f"{visitor}_{timestamp}_A96D855A08C0A9707F8BEF0D9A527E4E_{urlencode(query)}".encode()
            ).hexdigest()
            response = self._download_json(
                self._WEBPAGE_HOST + "aweme/v1/web/aweme/detail/",
                video_id,
                note="Downloading web detail JSON",
                query=query,
            )
            detail = (
                response.get("aweme_detail") if isinstance(response, dict) else None
            )
            if not isinstance(detail, dict) or not detail:
                raise DownloadRejected(
                    "抖音未返回视频详情；可能是视频不可用、平台风控或签名规则变化，"
                    "请检查链接并尝试更新插件或配置浏览器 Cookie"
                )
            info = self._parse_aweme_video_app(detail)
            info["http_headers"] = {
                **info.get("http_headers", {}),
                "Referer": self._WEBPAGE_HOST,
            }
            return info

    ydl.add_info_extractor(DouyinIE())


def download(config: dict) -> None:
    from yt_dlp import YoutubeDL
    from yt_dlp.downloader.external import FFmpegFD
    from yt_dlp.postprocessor.ffmpeg import FFmpegPostProcessor
    from yt_dlp.utils import DownloadError

    root = Path(config["directory"])
    limit = config["max_bytes"]
    stage = "解析视频信息"

    def postprocess(info):
        nonlocal stage
        if info.get("status") == "started":
            stage = (
                "合并音视频"
                if info.get("postprocessor") == "Merger"
                else "处理下载文件"
            )

    def progress(_):
        if (
            sum(path.stat().st_size for path in root.glob("media.*") if path.is_file())
            > limit
        ):
            raise DownloadRejected("分享视频超过配置的大小上限")

    class QuietLogger:
        def debug(self, *_):
            pass

        def warning(self, *_):
            pass

        def error(self, *_):
            pass

    options = {
        "quiet": True,
        "no_warnings": True,
        "logger": QuietLogger(),
        "noplaylist": True,
        "playlist_items": "1",
        "extract_flat": "in_playlist",
        "outtmpl": str(root / "media.%(ext)s"),
        "restrictfilenames": True,
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "max_filesize": limit,
        "socket_timeout": 15,
        "retries": 1,
        "fragment_retries": 1,
        "concurrent_fragment_downloads": 1,
        "hls_prefer_native": True,
        "fixup": "never",
        "cachedir": False,
        "proxy": "",
        "enable_file_urls": False,
        "progress_hooks": [progress],
        "postprocessor_hooks": [postprocess],
        "overwrites": True,
        "ffmpeg_location": config["ffmpeg"],
    }
    if config.get("cookie_file"):
        options["cookiefile"] = config["cookie_file"]
    # yt-dlp 只提供共同的 ffmpeg_location；沿用插件分别配置的两个可执行文件。
    original_paths = FFmpegPostProcessor._determine_executables
    original_download = FFmpegFD.real_download

    def executable_paths(processor):
        paths = original_paths(processor)
        for name in ("ffmpeg", "ffprobe"):
            if config.get(name):
                paths[name] = config[name]
        return paths

    FFmpegPostProcessor._determine_executables = executable_paths

    def reject_external_download(*_):
        raise DownloadRejected(
            "该媒体需要 FFmpeg 联网下载，暂不支持；请直接发送视频附件"
        )

    # 原生 HLS 遇到不支持的特性时会回退 FFmpeg，不能绕开网络检查。
    FFmpegFD.real_download = reject_external_download
    try:
        with YoutubeDL(options) as ydl:
            # 排除可绕开 Python DNS 检查的可选原生网络后端。
            director = ydl._request_director
            for name in list(director.handlers):
                if name != "Urllib":
                    director.handlers.pop(name).close()
            register_douyin_extractor(ydl)
            info = ydl.extract_info(config["url"], download=False)
            if not info:
                raise DownloadRejected("未找到可下载的视频")
            validate_info(info, config["max_duration"], limit)
            stage = "下载媒体"
            ydl.process_info(info)
    except DownloadError as exc:
        # 不向模型或控制台复制可能包含 URL 参数/凭据的 yt-dlp 原始异常。
        raise DownloadRejected(download_failure(exc, stage)) from exc
    finally:
        FFmpegPostProcessor._determine_executables = original_paths
        FFmpegFD.real_download = original_download
    outputs = [p for p in root.glob("media.*") if p.suffix not in {".part", ".ytdl"}]
    if len(outputs) != 1 or not 0 < outputs[0].stat().st_size <= limit:
        raise DownloadRejected("未得到符合大小限制的单个视频文件")
    print(json.dumps({"filename": outputs[0].name}))


def main() -> None:
    try:
        config = json.loads(sys.stdin.read(65536))
        # 通过 stdin 恢复宿主的导入路径，兼容桌面版插件依赖和忽略 PYTHONPATH 的解释器。
        module_paths = config.get("module_paths", [])
        sys.path[:0] = [
            path
            for path in module_paths
            if isinstance(path, str) and path and path not in sys.path
        ]
        guard_network(config.get("allow_fake_ip", False))
        download(config)
    except ImportError:
        print(
            json.dumps(
                {
                    "error": "下载进程无法导入 yt-dlp 或其依赖，请检查插件依赖安装并重载插件"
                }
            )
        )
        sys.exit(1)
    except DownloadRejected as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
    except Exception:
        print(
            json.dumps(
                {"error": "分享视频下载失败，请检查网络、Cookie 配置并尝试更新 yt-dlp"}
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

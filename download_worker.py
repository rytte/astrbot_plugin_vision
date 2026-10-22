"""独立 yt-dlp 进程；通过 stdin 接收配置，不把签名 URL 或 Cookie 写入命令行。"""

from __future__ import annotations

import ipaddress
import json
import math
import socket
import sys
from pathlib import Path


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


def download_failure(error: Exception) -> str:
    """只返回固定提示，避免上游异常中的签名链接、Cookie 等内容泄露。"""
    pending, seen = [error], set()
    while pending and len(seen) < 24:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, NetworkRejected):
            return str(current)
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
    return (
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


def download(config: dict) -> None:
    from yt_dlp import YoutubeDL
    from yt_dlp.downloader.external import FFmpegFD
    from yt_dlp.postprocessor.ffmpeg import FFmpegPostProcessor
    from yt_dlp.utils import DownloadError

    root = Path(config["directory"])
    limit = config["max_bytes"]

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
            info = ydl.extract_info(config["url"], download=False)
            if not info:
                raise DownloadRejected("未找到可下载的视频")
            validate_info(info, config["max_duration"], limit)
            ydl.process_info(info)
    except DownloadError as exc:
        # 不向模型或控制台复制可能包含 URL 参数/凭据的 yt-dlp 原始异常。
        raise DownloadRejected(download_failure(exc)) from exc
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

import asyncio
import io
import json
import os
import shutil
import socket
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.download_worker import (
    DownloadRejected,
    NetworkRejected,
    download,
    download_failure,
    guard_network,
    main,
    validate_info,
)
from astrbot_plugin_vision.downloads import VideoDownloads
from astrbot_plugin_vision.media import VideoError


@pytest.fixture
def downloads(tmp_path):
    return VideoDownloads(Settings(), tmp_path / "downloads", "ffmpeg", "ffprobe")


def test_network_guard_checks_redirect_and_cdn_dns(monkeypatch):
    resolver = socket.getaddrinfo
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *_: [(2, 1, 6, "", ("10.0.0.2", 443))]
    )
    guard_network()
    with pytest.raises(OSError, match="非公网"):
        socket.getaddrinfo("cdn.example", 443)
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *_: [(2, 1, 6, "", ("8.8.8.8", 443))]
    )
    guard_network()
    assert socket.getaddrinfo("example.com", 443)[0][4][0] == "8.8.8.8"
    monkeypatch.setattr(socket, "getaddrinfo", resolver)


@pytest.mark.parametrize(
    "enabled, host, address, allowed",
    [
        (False, "cdn.example", "198.18.0.2", False),
        (True, "cdn.example", "198.18.0.2", True),
        (True, "cdn.example", "198.19.255.254", True),
        (True, "198.18.0.2", "198.18.0.2", False),
        (True, "cdn.example", "10.0.0.1", False),
        (True, "cdn.example", "127.0.0.1", False),
        (True, "cdn.example", "169.254.169.254", False),
        (True, "cdn.example", "224.0.0.1", False),
    ],
)
def test_fake_ip_compatibility_only_allows_domain_synthetic_addresses(
    monkeypatch, enabled, host, address, allowed
):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *_: [(2, 1, 6, "", (address, 443))]
    )
    guard_network(enabled)
    if allowed:
        assert socket.getaddrinfo(host, 443)[0][4][0] == address
    else:
        with pytest.raises(NetworkRejected):
            socket.getaddrinfo(host, 443)


def test_wrapped_network_failure_keeps_safe_reason_and_redacts_raw_error():
    from yt_dlp.utils import DownloadError, ExtractorError

    blocked = NetworkRejected("DNS 返回了 Fake-IP 地址")
    extraction = ExtractorError(
        "https://example.com/?sign=secret Cookie=private", cause=blocked
    )
    error = DownloadError(
        "signed-secret-url", exc_info=(type(extraction), extraction, None)
    )
    assert download_failure(error) == "DNS 返回了 Fake-IP 地址"
    assert "secret" not in download_failure(DownloadError("private-secret"))


@pytest.mark.parametrize(
    "extra",
    [
        {"_type": "playlist", "entries": []},
        {"is_live": True},
        {"live_status": "is_upcoming"},
        {"duration": 1801},
        {"duration": float("inf")},
        {"filesize": 1001},
        {"vcodec": "none"},
        {"protocol": "m3u8"},
        {"protocol": "rtmp"},
    ],
)
def test_metadata_limits_before_download(extra):
    info = {"duration": 3, "vcodec": "avc1", "protocol": "https"} | extra
    with pytest.raises(ValueError):
        validate_info(info, 1800, 1000)


async def test_cache_concurrent_reuse_and_authorization_identity(downloads):
    async def fake(url, directory):
        await asyncio.sleep(0.01)
        directory.mkdir()
        path = directory / "media.mp4"
        path.write_bytes(b"video")
        return path

    downloads._download = AsyncMock(side_effect=fake)

    async def read():
        async with downloads.acquire("https://example.com/v?sig=secret") as path:
            identity = path.stat().st_mtime_ns
            downloads.cleanup()
            assert path.exists()
            return path, identity

    first, second = await asyncio.gather(read(), read())
    assert first == second
    assert downloads._download.await_count == 1
    assert not downloads.active
    assert not downloads.locks
    assert "secret" not in str(first[0])


async def test_cleanup_pins_active_and_never_removes_other_files(downloads):
    directory = downloads.root / "cached"
    directory.mkdir()
    path = directory / "media.mp4"
    path.write_bytes(b"video")
    other = downloads.root.parent / "attachment.mp4"
    other.write_bytes(b"source")
    with downloads.pin(path):
        os.utime(directory, (1, 1))
        downloads.cleanup()
        assert path.exists()
    downloads.cleanup()
    assert not directory.exists()
    assert other.read_bytes() == b"source"


def test_cleanup_capacity_oldest_first(downloads):
    downloads.settings = replace(downloads.settings, download_cache_max_mb=1)
    for i in range(3):
        directory = downloads.root / str(i)
        directory.mkdir()
        (directory / "media.mp4").write_bytes(b"x" * 600000)
        os.utime(directory, (time.time() - 10 + i, time.time() - 10 + i))
    downloads.cleanup()
    assert [p.name for p in downloads.root.iterdir()] == ["2"]


async def test_cookie_copy_and_worker_payload_are_private(
    downloads, tmp_path, monkeypatch
):
    original = tmp_path / "admin-cookies.txt"
    original.write_text("private-original-cookie", encoding="utf-8")
    downloads.settings = replace(downloads.settings, download_cookie_file=str(original))
    captured = {}

    class Process:
        returncode = 0

        async def communicate(self, payload):
            captured.update(json.loads(payload))
            cookie = Path(captured["cookie_file"])
            assert cookie.read_text() == "private-original-cookie"
            cookie.write_text("updated-cookie", encoding="utf-8")
            (Path(captured["directory"]) / "media.mp4").write_bytes(b"video")
            return b'{"filename":"media.mp4"}', None

    async def spawn(*args, **kwargs):
        assert "secret" not in str(args)
        assert str(original) not in str(args)
        return Process()

    monkeypatch.setattr(
        "astrbot_plugin_vision.downloads.asyncio.create_subprocess_exec", spawn
    )
    async with downloads.acquire("https://example.com/watch?sig=secret") as path:
        assert path.read_bytes() == b"video"
        assert list(path.parent.iterdir()) == [path]
        assert captured["url"].endswith("sig=secret")
        assert captured["ffprobe"] == "ffprobe"
        assert captured["allow_fake_ip"] is False
        assert captured["module_paths"] == [
            path for path in sys.path if isinstance(path, str) and path
        ]
    assert original.read_text() == "private-original-cookie"
    assert not downloads.staging


def test_worker_redacts_unexpected_error_details(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    monkeypatch.setattr(
        "astrbot_plugin_vision.download_worker.guard_network", lambda *_: None
    )

    def failure(_):
        raise ValueError("secret-signed-url-and-password")

    monkeypatch.setattr("astrbot_plugin_vision.download_worker.download", failure)
    with pytest.raises(SystemExit):
        main()
    output = capsys.readouterr().out
    assert "secret" not in output
    assert "error" in json.loads(output)


def test_cleanup_removes_stale_partial_but_keeps_running_download(downloads):
    stale = downloads.root / ".partial-abandoned"
    current = downloads.root / ".partial-running"
    stale.mkdir()
    current.mkdir()
    downloads.staging.add(current)
    downloads.cleanup()
    assert not stale.exists()
    assert current.exists()


async def test_real_worker_rejects_internal_address_and_cleans_partial(downloads):
    # 真正启动隔离进程，DNS 检查发生在 yt-dlp 的网络请求内。
    with pytest.raises(VideoError):
        await asyncio.wait_for(
            downloads._download("http://127.0.0.1/video.mp4", downloads.root / "bad"),
            20,
        )
    assert not list(downloads.root.iterdir())


async def test_isolated_worker_inherits_host_dependency_paths(downloads, monkeypatch):
    original = asyncio.create_subprocess_exec

    async def isolated(executable, *args, **kwargs):
        # 不加载 site-packages，也忽略 PYTHONPATH，模拟依赖只在宿主 sys.path 中可见。
        return await original(executable, "-I", "-S", *args, **kwargs)

    monkeypatch.setattr(
        "astrbot_plugin_vision.downloads.asyncio.create_subprocess_exec", isolated
    )
    with pytest.raises(VideoError, match="非公网地址"):
        await asyncio.wait_for(
            downloads._download("http://127.0.0.1/video.mp4", downloads.root / "bad"),
            20,
        )
    assert not list(downloads.root.iterdir())


@pytest.mark.parametrize("hls_fallback", [False, True])
async def test_real_ytdlp_downloads_sample_video(
    sample_video,
    tmp_path,
    capsys,
    monkeypatch,
    hls_fallback,
):
    # 本地测试服务替代变化频繁的平台，不启用 worker.main 的公网限制。
    blob = sample_video.read_bytes()
    if hls_fallback:
        from yt_dlp.downloader.hls import HlsFD

        blob = (
            b"#EXTM3U\n#EXT-X-TARGETDURATION:3\n#EXTINF:3,\nvideo.ts\n#EXT-X-ENDLIST\n"
        )
        monkeypatch.setattr(HlsFD, "can_download", lambda *_: False)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/vnd.apple.mpegurl" if hls_fallback else "video/mp4",
            )
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = tmp_path / "result"
    root.mkdir()
    try:
        operation = asyncio.to_thread(
            download,
            {
                "url": f"http://127.0.0.1:{server.server_port}/video."
                + ("m3u8" if hls_fallback else "mp4"),
                "directory": str(root),
                "max_bytes": 1024**2,
                "max_duration": 60,
                "ffmpeg": shutil.which("ffmpeg"),
                "ffprobe": shutil.which("ffprobe"),
            },
        )
        if hls_fallback:
            with pytest.raises(DownloadRejected, match="FFmpeg 联网"):
                await operation
        else:
            await operation
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    if not hls_fallback:
        result = json.loads(capsys.readouterr().out)
        assert (root / result["filename"]).read_bytes() == blob


@pytest.mark.parametrize("cancel", [False, True])
async def test_subprocess_timeout_cancel_kills_and_removes_partial(
    downloads, monkeypatch, cancel
):
    original = asyncio.create_subprocess_exec
    started = asyncio.Event()
    processes = []

    async def spawn(*args, **kwargs):
        process = await original(
            sys.executable,
            "-c",
            "import sys,time; sys.stdin.read(); time.sleep(60)",
            **kwargs,
        )
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(
        "astrbot_plugin_vision.downloads.asyncio.create_subprocess_exec", spawn
    )
    # Windows taskkill 必须执行真实命令。
    real_stop = downloads._stop

    async def stop(process):
        monkeypatch.setattr(
            "astrbot_plugin_vision.downloads.asyncio.create_subprocess_exec", original
        )
        await real_stop(process)

    monkeypatch.setattr(downloads, "_stop", stop)
    task = asyncio.create_task(
        downloads._download("https://example.com/video", downloads.root / "video")
    )
    await asyncio.wait_for(started.wait(), 10)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, 0.05)
    assert processes[0].returncode is not None
    assert not list(downloads.root.iterdir())

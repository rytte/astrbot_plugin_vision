from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from .config import Settings
from .media import VideoError
from .sources import normalize_url


class VideoDownloads:
    """只管理插件下载的源视频；使用期间固定缓存，取消时终止整个进程树。"""

    def __init__(self, settings: Settings, root: Path, ffmpeg: str, ffprobe: str):
        self.settings, self.ffmpeg, self.ffprobe = settings, ffmpeg, ffprobe
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.active: Counter[Path] = Counter()
        self.locks: dict[str, asyncio.Lock] = {}
        self.staging: set[Path] = set()

    @asynccontextmanager
    async def acquire(self, url: str):
        url = normalize_url(url)
        cookie = self.settings.download_cookie_file
        identity = ""
        if cookie:
            try:
                stat = Path(cookie).stat()
                identity = f"{cookie}:{stat.st_size}:{stat.st_mtime_ns}"
            except OSError as exc:
                raise VideoError("无法读取配置的 yt-dlp Cookie 文件") from exc
        key = hashlib.sha256(f"{url}:{identity}".encode()).hexdigest()
        directory = self.root / key
        lock = self.locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                self.active[directory] += 1
                try:
                    files = list(directory.glob("media.*"))
                    path = files[0] if len(files) == 1 else None
                    if (
                        path is None
                        or not path.is_file()
                        or time.time() - directory.stat().st_mtime
                        > self.settings.cache_ttl
                        or not 0
                        < path.stat().st_size
                        <= self.settings.download_max_mb * 1024**2
                    ):
                        if directory.exists():
                            shutil.rmtree(directory)
                        path = await self._download(url, directory)
                    directory.touch()
                    yield path
                finally:
                    self.active[directory] -= 1
                    if not self.active[directory]:
                        del self.active[directory]
        finally:
            # 不删除有等待者的锁，避免同一 URL 的并发下载。
            if not lock.locked() and not getattr(lock, "_waiters", None):
                self.locks.pop(key, None)

    @contextmanager
    def pin(self, path: Path):
        directory = path.parent
        if directory.parent == self.root:
            directory.touch()
        self.active[directory] += 1
        try:
            yield
        finally:
            self.active[directory] -= 1
            if not self.active[directory]:
                del self.active[directory]

    async def _stop(self, process):
        if process.returncode is not None:
            return
        try:
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                await killer.wait()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.wait()

    async def _download(self, url: str, directory: Path) -> Path:
        staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=self.root))
        self.staging.add(staging)
        process = None
        try:
            # yt-dlp 会写回 Cookie；只向临时副本写入，不改管理员原文件。
            cookie_file = ""
            if self.settings.download_cookie_file:
                cookie_file = str(staging / "cookies.txt")
                shutil.copyfile(self.settings.download_cookie_file, cookie_file)
            options = (
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if os.name == "nt"
                else {"start_new_session": True}
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).with_name("download_worker.py")),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                **options,
            )
            payload = json.dumps(
                {
                    "url": url,
                    "directory": str(staging),
                    "ffmpeg": self.ffmpeg,
                    "ffprobe": self.ffprobe,
                    "cookie_file": cookie_file,
                    "allow_fake_ip": self.settings.download_allow_fake_ip,
                    # 桌面版把插件依赖目录动态加入 sys.path，子进程不会自动继承。
                    "module_paths": [
                        path for path in sys.path if isinstance(path, str) and path
                    ],
                    "max_bytes": self.settings.download_max_mb * 1024**2,
                    "max_duration": self.settings.download_max_duration,
                }
            ).encode()

            async def communicate():
                output, _ = await process.communicate(payload)
                if len(output) > 65536:
                    raise VideoError("视频下载输出异常")
                return output

            async def watch_size():
                while process.returncode is None:
                    size = 0
                    for path in staging.iterdir():
                        try:
                            if path.is_file():
                                size += path.stat().st_size
                        except FileNotFoundError:
                            # yt-dlp/FFmpeg 可能刚完成重命名或合并。
                            continue
                    # 合并期间同时保留输入和输出，硬限制为单片上限的三倍。
                    if size > self.settings.download_max_mb * 1024**2 * 3:
                        raise VideoError("视频下载临时文件超过大小限制")
                    await asyncio.sleep(0.2)

            reader = asyncio.create_task(communicate())
            watcher = asyncio.create_task(watch_size())
            try:
                await asyncio.gather(reader, watcher)
                response = json.loads(reader.result())
            finally:
                reader.cancel()
                watcher.cancel()
                await asyncio.gather(reader, watcher, return_exceptions=True)
            if process.returncode or response.get("error"):
                raise VideoError(response.get("error") or "视频下载进程异常退出")
            path = staging / response["filename"]
            if path.parent != staging or not path.name.startswith("media."):
                raise VideoError("视频下载输出路径无效")
            if (
                not path.is_file()
                or not 0
                < path.stat().st_size
                <= self.settings.download_max_mb * 1024**2
            ):
                raise VideoError("下载的视频为空或超过大小上限")
            if cookie_file:
                Path(cookie_file).unlink(missing_ok=True)
            staging.rename(directory)
            return directory / path.name
        except (ValueError, KeyError) as exc:
            raise VideoError("视频下载进程返回了无效结果") from exc
        finally:
            if process is not None:
                await self._stop(process)
            if staging.exists():
                shutil.rmtree(staging)
            self.staging.discard(staging)

    def cleanup(self):
        entries = []
        now = time.time()
        for directory in self.root.iterdir():
            if directory.name.startswith(".partial-") and directory not in self.staging:
                shutil.rmtree(directory)
                continue
            if (
                not directory.is_dir()
                or directory in self.active
                or directory.name.startswith(".")
            ):
                continue
            if now - directory.stat().st_mtime > self.settings.cache_ttl:
                shutil.rmtree(directory)
            else:
                size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
                entries.append((directory.stat().st_mtime, directory, size))
        total = sum(size for _, _, size in entries)
        for _, directory, size in sorted(entries):
            if total <= self.settings.download_cache_max_mb * 1024**2:
                break
            shutil.rmtree(directory)
            total -= size

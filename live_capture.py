"""直接采集 AstrBot 宿主设备或媒体直链，不启动 HTTP 服务。"""

from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass

from .live import MAX_FRAME_BYTES, LiveSession, LiveSessions
from .media import VideoError


@dataclass(frozen=True)
class CaptureInput:
    video: tuple[str, ...]
    audio: tuple[str, ...] | None
    native_audio: bool = False


async def device_listing(ffmpeg: str, backend: str) -> str:
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-list_devices",
        "true",
        "-f",
        backend,
        "-i",
        "dummy" if backend == "dshow" else "",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), 5)
        return stderr.decode("utf-8", errors="replace")[:32000]
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()


async def capture_input(session: LiveSession, settings, ffmpeg: str) -> CaptureInput:
    source = session.options.source
    # 本机输入至少每秒一帧，确保低频预设也能尽快输出首帧。
    input_fps = str(max(1, math.ceil(1 / session.options.sample_interval)))
    if source == "stream":
        if not session.source_url:
            raise VideoError("缺少直播媒体直链")
        incoming = (
            "-rw_timeout",
            "10000000",
            "-protocol_whitelist",
            "http,https,tcp,tls,rtsp,rtp,udp,rtmp,rtmps,crypto",
            "-i",
            session.source_url,
        )
        return CaptureInput(incoming, incoming)
    camera, audio = settings.live_camera_device, settings.live_audio_device
    if sys.platform.startswith("linux"):
        if source == "screen":
            display = os.environ.get("DISPLAY", "")
            if not display or os.environ.get("XDG_SESSION_TYPE") == "wayland":
                raise VideoError(
                    "本机屏幕采集需要可访问的 X11 桌面；无桌面的服务器或纯 Wayland 会话请使用媒体URL"
                )
            video = ("-f", "x11grab", "-framerate", input_fps, "-i", display)
        else:
            video = ("-f", "v4l2", "-i", camera or "/dev/video0")
        # 优先复用系统 PulseAudio/PipeWire 录音工具，不要求 FFmpeg 编译 pulse。
        recorder = shutil.which("parec")
        audio_command = (
            (
                recorder,
                "--raw",
                "--format=s16le",
                "--rate=16000",
                "--channels=1",
                "--device="
                + (
                    audio
                    or (
                        "@DEFAULT_MONITOR@"
                        if source == "screen"
                        else "@DEFAULT_SOURCE@"
                    )
                ),
            )
            if recorder
            else None
        )
        return CaptureInput(video, audio_command, True)
    if sys.platform == "win32":
        if source == "screen":
            video = ("-f", "gdigrab", "-framerate", input_fps, "-i", "desktop")
        else:
            if not camera:
                devices = await device_listing(ffmpeg, "dshow")
                match = re.search(r'"([^"]+)" \(video\)', devices)
                if not match:
                    raise VideoError(
                        "未找到摄像头，请配置 live_camera_device 为 DirectShow 设备名称"
                    )
                camera = match.group(1)
            video = ("-f", "dshow", "-i", "video=" + camera)
        return CaptureInput(
            video, ("-f", "dshow", "-i", "audio=" + audio) if audio else None
        )
    if sys.platform == "darwin":
        if source == "screen":
            devices = await device_listing(ffmpeg, "avfoundation")
            match = re.search(r"\[(\d+)\] Capture screen", devices)
            if not match:
                raise VideoError("未找到屏幕设备，请授予 AstrBot/FFmpeg 屏幕录制权限")
            device = match.group(1)
        else:
            device = camera or "0"
        video = ("-f", "avfoundation")
        if source == "screen":
            video += ("-framerate", input_fps)
        video += ("-i", device + ":none")
        audio_input = audio or ("0" if source == "camera" else "")
        return CaptureInput(
            video,
            ("-f", "avfoundation", "-i", "none:" + audio_input)
            if audio_input
            else None,
        )
    raise VideoError("当前系统暂不支持本机设备采集，请使用媒体URL")


async def read_capture(
    sessions: LiveSessions, session: LiveSession, ffmpeg: str
) -> None:
    video_process = audio_process = audio_task = None
    started = time.monotonic()
    try:
        incoming = await capture_input(session, sessions.settings, ffmpeg)
        options = session.options
        max_edge = options.max_edge
        fps = format(1 / options.sample_interval, ".10g")
        video_process = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            *incoming.video,
            "-an",
            "-vf",
            f"fps={fps}:round=up:start_time=0,scale='min(iw,{max_edge})':'min(ih,{max_edge})':"
            "force_original_aspect_ratio=decrease",
            "-threads",
            "1",
            "-c:v",
            "mjpeg",
            "-f",
            "image2pipe",
            "pipe:1",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def audio_loop():
            nonlocal audio_process
            try:
                if not incoming.audio:
                    session.audio_status = "unavailable"
                    return
                command = (
                    incoming.audio
                    if incoming.native_audio
                    else (
                        ffmpeg,
                        "-nostdin",
                        "-loglevel",
                        "error",
                        *incoming.audio,
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-f",
                        "s16le",
                        "pipe:1",
                    )
                )
                audio_process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                position = max(0.0, time.monotonic() - started)
                while not session.stopped:
                    try:
                        pcm = await asyncio.wait_for(
                            audio_process.stdout.readexactly(160000), 20
                        )
                    except asyncio.IncompleteReadError as exc:
                        pcm = exc.partial[: len(exc.partial) // 2 * 2]
                        if (
                            len(pcm) >= 3200
                            and not session.stopped
                            and not session.paused
                        ):
                            sessions.add_audio(
                                session, pcm, position, position + len(pcm) / 32000
                            )
                        else:
                            session.audio_status = "unavailable"
                        return
                    end = position + len(pcm) / 32000
                    if not session.paused:
                        sessions.add_audio(session, pcm, position, end)
                    position = end
            except (OSError, TimeoutError, ValueError):
                session.audio_status = "unavailable"

        if session.options.audio:
            audio_task = asyncio.create_task(audio_loop())
        pending = bytearray()
        while not session.stopped:
            part = await asyncio.wait_for(
                video_process.stdout.read(65536), max(15, options.fresh_seconds)
            )
            if not part:
                raise VideoError("画面来源已结束或无法读取")
            pending.extend(part)
            if len(pending) > MAX_FRAME_BYTES * 2:
                raise VideoError("采集画面超过大小限制")
            while (begin := pending.find(b"\xff\xd8")) >= 0 and (
                end := pending.find(b"\xff\xd9", begin + 2)
            ) >= 0:
                jpeg = bytes(pending[begin : end + 2])
                del pending[: end + 2]
                if not session.paused:
                    await sessions.add_frame(session, jpeg, time.monotonic() - started)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        session.error = (
            str(exc)
            if isinstance(exc, VideoError)
            else "采集失败，请检查 FFmpeg、设备权限和媒体地址"
        )
    finally:
        session.ready.set()
        if audio_task:
            audio_task.cancel()
            await asyncio.gather(audio_task, return_exceptions=True)
        for process in (video_process, audio_process):
            if process and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()

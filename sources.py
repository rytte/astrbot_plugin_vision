from __future__ import annotations

import html
import ipaddress
import json
import re
from urllib.parse import urlsplit, urlunsplit

from astrbot.api.message_components import Json, Plain, Reply, Share

URL_PATTERN = re.compile(r"https?://[^\s<>\"'\[\]，。；！？、（）【】]+", re.I)
CARD_KEYS = {
    "url",
    "jumpurl",
    "jump_url",
    "qqdocurl",
    "source_url",
    "sourceurl",
    "web_url",
    "weburl",
    "link",
    "href",
    "playurl",
    "play_url",
}
IMAGE_KEYS = {
    "image",
    "images",
    "cover",
    "covers",
    "thumbnail",
    "thumb",
    "preview",
    "icon",
    "avatar",
    "picture",
    "pic",
    "image_url",
    "imageurl",
}


def normalize_url(value: str) -> str:
    """保留分享签名参数，只接受无凭据的公网 HTTP(S) 地址。DNS 在下载进程检查。"""
    if not isinstance(value, str) or len(value) > 8192:
        raise ValueError("分享链接无效或过长")
    value = html.unescape(value).strip()
    if any(ord(char) < 33 for char in value) or "\\" in value:
        raise ValueError("分享链接包含无效字符")
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError as exc:
        raise ValueError("分享链接格式无效") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and port not in {80, 443})
        or host == "localhost"
        or host.endswith((".localhost", ".local"))
    ):
        raise ValueError("只支持无凭据、使用标准端口的公网 HTTP(S) 分享链接")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global or address.is_multicast:
            raise ValueError("不支持内网或本机分享链接")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path, parsed.query, "")
    )


def display_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))[:500]


def extract_share_urls(components: list, raw_message=None) -> list[str]:
    """有限扫描原始卡片及引用，忽略封面、图标等资源字段。"""
    found: dict[str, None] = {}
    budget = 512

    def add(text):
        if not isinstance(text, str):
            return
        for match in URL_PATTERN.finditer(html.unescape(text[:65536])):
            try:
                url = normalize_url(match.group().rstrip(".,;!?)"))
            except ValueError:
                continue
            if (
                urlsplit(url)
                .path.lower()
                .endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico"))
            ):
                continue
            if len(found) < 8:
                found[url] = None

    def card(value, depth=0):
        nonlocal budget
        budget -= 1
        if budget < 0 or depth > 8:
            return
        if isinstance(value, str) and len(value) <= 65536:
            try:
                value = json.loads(value)
            except (ValueError, RecursionError):
                return
        if isinstance(value, dict):
            for key, item in list(value.items())[:64]:
                if str(key).lower() in IMAGE_KEYS:
                    continue
                if str(key).lower() in CARD_KEYS:
                    add(item)
                elif isinstance(item, dict | list):
                    card(item, depth + 1)
        elif isinstance(value, list):
            for item in value[:64]:
                card(item, depth + 1)

    def raw(value, depth=0):
        if depth > 3:
            return
        if isinstance(value, dict):
            kind, data = value.get("type"), value.get("data", {})
            if isinstance(data, dict):
                if kind == "text":
                    add(data.get("text"))
                elif kind == "share":
                    add(data.get("url"))
                elif kind == "json":
                    card(data.get("data"))
                elif kind == "xml" and isinstance(data.get("data"), str):
                    for match in re.finditer(
                        r"""(?:url|jumpUrl)\s*=\s*["']([^"']+)["']""",
                        data["data"][:65536],
                        re.I,
                    ):
                        add(match.group(1))
            for key in ("message", "content"):
                raw(value.get(key), depth + 1)
        elif isinstance(value, list):
            for item in value[:80]:
                raw(item, depth + 1)

    # 原始消息优先，避免其他插件生成的摘要丢失签名参数。
    # aiocqhttp.Event 是 dict 子类，缺失属性也会返回 None，不能用 hasattr 判定。
    if not isinstance(raw_message, dict | list):
        model_dump = getattr(raw_message, "model_dump", None)
        if callable(model_dump):
            raw_message = model_dump()
    raw(raw_message)
    queue = list(components[:64])
    for component in queue[:64]:
        if isinstance(component, Reply):
            queue.extend((component.chain or [])[:16])
    for component in queue[:80]:
        if isinstance(component, Plain):
            add(component.text)
        elif isinstance(component, Share):
            add(component.url)
        elif isinstance(component, Json):
            card(component.data)
    return list(found)

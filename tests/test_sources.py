import json
from types import SimpleNamespace

import pytest
from aiocqhttp import Event as OneBotEvent
from astrbot.api.message_components import Json, Plain, Reply, Share
from astrbot_plugin_vision.sources import display_url, extract_share_urls, normalize_url
from pydantic import BaseModel


class RawMessageModel(BaseModel):
    message: list[dict]


@pytest.mark.parametrize("container", [dict, OneBotEvent, RawMessageModel])
def test_raw_event_containers_preserve_signed_card_url(container):
    url = "https://www.bilibili.com/video/BV1234567890?share_source=qq&sign=secret"
    payload = {
        "message": [
            {
                "type": "json",
                "data": {"data": json.dumps({"meta": {"detail_1": {"qqdocurl": url}}})},
            }
        ]
    }
    raw = container(**payload)
    assert extract_share_urls([], raw) == [url]


def test_noncallable_model_dump_still_extracts_components():
    url = "https://b23.tv/abc"
    assert extract_share_urls(
        [Json({"url": url})], SimpleNamespace(model_dump=None)
    ) == [url]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("BV186pc65E68", "https://www.bilibili.com/video/BV186pc65E68"),
        ("bv186pc65E68", "https://www.bilibili.com/video/BV186pc65E68"),
        ("看看【BV186pc65E68】讲了什么", "https://www.bilibili.com/video/BV186pc65E68"),
        ("av170001", "https://www.bilibili.com/video/av170001"),
        ("看看AV170001吧", "https://www.bilibili.com/video/av170001"),
        ("b23.tv/9unK6EN", "https://b23.tv/9unK6EN"),
        ("B23.TV/9unK6EN", "https://b23.tv/9unK6EN"),
        ("看看（b23.tv/9unK6EN）。", "https://b23.tv/9unK6EN"),
        ("`b23.tv/9unK6EN`", "https://b23.tv/9unK6EN"),
        (
            "b23.tv/9unK6EN?share_source=qq&sign=secret",
            "https://b23.tv/9unK6EN?share_source=qq&sign=secret",
        ),
    ],
)
def test_bilibili_identifiers_and_schemeless_short_links(text, expected):
    assert extract_share_urls([Plain(text)]) == [expected]


def test_mixed_bilibili_sources_keep_order_deduplicate_and_preserve_url_parameters():
    text = (
        "BV186pc65E68 https://www.bilibili.com/video/BV186pc65E68 "
        "b23.tv/9unK6EN https://b23.tv/9unK6EN av170001 AV170001 "
        "https://www.bilibili.com/video/BV186pc65E68?p=2&sign=secret "
        "https://example.com/?id=av123"
    )
    raw = {"message": [{"type": "text", "data": {"text": text}}]}
    assert extract_share_urls([Plain(text)], raw) == [
        "https://www.bilibili.com/video/BV186pc65E68",
        "https://b23.tv/9unK6EN",
        "https://www.bilibili.com/video/av170001",
        "https://www.bilibili.com/video/BV186pc65E68?p=2&sign=secret",
        "https://example.com/?id=av123",
    ]


def test_bilibili_sources_in_replies_and_cards():
    components = [
        Reply(id="1", chain=[Plain("BV186pc65E68")]),
        Json({"url": "b23.tv/9unK6EN", "cover": "b23.tv/ignored"}),
        Share(url="av170001", title="视频"),
    ]
    assert extract_share_urls(components) == [
        "https://b23.tv/9unK6EN",
        "https://www.bilibili.com/video/av170001",
        "https://www.bilibili.com/video/BV186pc65E68",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "BV186pc65E6",
        "BV186pc65E680",
        "BV186pc65E6_",
        "xBV186pc65E68",
        "BV186pc65E68_suffix",
        "BV186pc65E68.mp4",
        "nav170001",
        "av170001suffix",
        "av0",
        "av-1",
        "av123.4",
        "av170001.mp4",
        "evil.b23.tv/9unK6EN",
        "notb23.tv/9unK6EN",
        "b23.tv.evil/9unK6EN",
        "https://user:pass@b23.tv/9unK6EN",
        "ftp://b23.tv/9unK6EN",
        "file:///BV186pc65E68",
        "https://localhost/BV186pc65E68",
    ],
)
def test_invalid_or_embedded_bilibili_sources_are_not_extracted(text):
    assert extract_share_urls([Plain(text)]) == []


def test_text_card_reference_and_raw_preserve_signed_url():
    signed = "https://www.xiaohongshu.com/explore/abc?xsec_token=secret&xsec_source=pc"
    card = {
        "meta": {
            "detail_1": {
                "qqdocurl": signed,
                "preview": "https://cdn.example/cover",
                "icon": {"url": "https://cdn.example/icon"},
            }
        }
    }
    components = [
        Plain("分享： https://b23.tv/ABC123，看看这个。https://b23.tv/ABC123"),
        Json(card),
        Share(
            url="https://v.douyin.com/abc/",
            title="视频",
            image="https://cdn.example/icon",
        ),
        Reply(id="1", chain=[Plain("https://youtu.be/test")]),
    ]
    raw = {"message": [{"type": "json", "data": {"data": json.dumps(card)}}]}
    urls = extract_share_urls(components, raw)
    assert urls == [
        signed,
        "https://b23.tv/ABC123",
        "https://v.douyin.com/abc/",
        "https://youtu.be/test",
    ]
    assert "secret" not in display_url(signed)


def test_xml_card_html_entities_and_covers_are_ignored():
    raw = {
        "message": [
            {
                "type": "xml",
                "data": {
                    "data": '<msg url="https://example.com/watch?v=1&amp;sig=2" cover="https://cdn.example/cover"/>'
                },
            }
        ]
    }
    assert extract_share_urls([Plain("https://cdn.example/cover.jpg")], raw) == [
        "https://example.com/watch?v=1&sig=2"
    ]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/video",
        "http://localhost/a",
        "http://127.0.0.1/a",
        "http://[::1]/a",
        "http://10.0.0.1/a",
        "http://169.254.169.254/",
        "http://198.18.0.1/",
        "http://198.19.255.254/",
        "https://user:pass@example.com/v",
        "https://example.com:8443/v",
        "https://example.com/\nvideo",
        "https://example.com\\a",
    ],
)
def test_unsafe_sources(url):
    with pytest.raises(ValueError):
        normalize_url(url)


def test_limits_and_broken_cards():
    nested = {"url": "https://example.com/too-deep"}
    for _ in range(20):
        nested = {"meta": nested}
    components = [
        Json(nested),
        Json({"data": "{"}),
        Plain(" ".join(f"https://example.com/{i}" for i in range(100))),
    ]
    assert len(extract_share_urls(components)) == 8
    assert "too-deep" not in str(extract_share_urls(components))

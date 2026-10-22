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

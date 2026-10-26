import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from astrbot_plugin_vision.download_worker import (
    DownloadRejected,
    register_douyin_extractor,
)
from yt_dlp import YoutubeDL
from yt_dlp.extractor.generic import GenericIE
from yt_dlp.extractor.tiktok import DouyinIE

VIDEO_ID = "7687928731218773865"
VIDEO_URL = f"https://www.douyin.com/video/{VIDEO_ID}"
SHORT_URL = "https://v.douyin.com/fBwKc1GH4UU/"


@pytest.fixture
def douyin(monkeypatch):
    with YoutubeDL({"quiet": True, "no_warnings": True, "cachedir": False}) as ydl:
        register_douyin_extractor(ydl)
        extractor = ydl.get_info_extractor("Douyin")

        def webpage(url, *args, **kwargs):
            if url.endswith("/ttwid/register"):
                assert json.loads(kwargs["data"]) == {
                    "aid": 6383,
                    "service": "www.douyin.com",
                }
                extractor._set_cookie(".douyin.com", "ttwid", "issued-ttwid")
            return SimpleNamespace(headers={"x-tt-trace-host": "test-visitor"})

        requests = Mock(side_effect=webpage)
        metadata = Mock(
            return_value={
                "aweme_detail": {
                    "aweme_id": VIDEO_ID,
                    "desc": "sample",
                    "video": {
                        "width": 320,
                        "height": 180,
                        "duration": 68000,
                        "play_addr": {
                            "url_list": ["https://cdn.example/video.mp4"],
                        },
                    },
                }
            }
        )
        monkeypatch.setattr(extractor, "_request_webpage", requests)
        monkeypatch.setattr(extractor, "_download_json", metadata)
        yield ydl, extractor, requests, metadata


@pytest.mark.parametrize("url", [VIDEO_URL, SHORT_URL])
def test_anonymous_douyin_extraction_signs_details_and_passes_media_referer(
    douyin, monkeypatch, url
):
    ydl, extractor, requests, metadata = douyin
    monkeypatch.setattr(
        GenericIE,
        "_real_extract",
        lambda self, url: self.url_result(VIDEO_URL),
    )
    monkeypatch.setattr(
        "astrbot_plugin_vision.download_worker.time.time", lambda: 1793020000
    )
    info = ydl.extract_info(url, download=False)
    assert info["id"] == VIDEO_ID
    assert info["duration"] == 68
    assert all(
        item["http_headers"]["Referer"] == "https://www.douyin.com/"
        for item in info["formats"]
    )
    assert metadata.call_args.kwargs["query"] == {
        "aweme_id": VIDEO_ID,
        "aid": 6383,
        "uifid": "test-visitor",
        "timestamp": "1793020000",
        "x-secsdk-web-signature": "da222189872bcb02cb961d632f1ab6e8",
    }
    assert requests.call_count == 2
    cookies = extractor._get_cookies("https://www.douyin.com/")
    assert cookies["ttwid"].value == "issued-ttwid"
    assert cookies["passport_csrf_token"].value


@pytest.mark.parametrize("visitor_name", ["UIFID", "UIFID_TEMP", "uifid"])
def test_configured_douyin_cookies_are_preserved_and_visitor_is_reused(
    douyin, visitor_name
):
    ydl, extractor, requests, metadata = douyin
    original = {
        "ttwid": "configured-ttwid",
        "passport_csrf_token": "configured-csrf",
        "sessionid": "private-session",
        visitor_name: "configured-visitor",
    }
    for name, value in original.items():
        extractor._set_cookie(".douyin.com", name, value)
    ydl.extract_info(VIDEO_URL, download=False)
    requests.assert_not_called()
    assert metadata.call_args.kwargs["query"]["uifid"] == "configured-visitor"
    cookies = extractor._get_cookies("https://www.douyin.com/")
    assert {name: cookies[name].value for name in original} == original


@pytest.mark.parametrize("missing", ["ttwid", "visitor"])
def test_missing_anonymous_credentials_stop_before_requesting_details(douyin, missing):
    ydl, extractor, requests, metadata = douyin
    if missing == "visitor":
        extractor._set_cookie(".douyin.com", "ttwid", "issued-ttwid")
    requests.side_effect = None
    requests.return_value = SimpleNamespace(headers={})
    with pytest.raises(DownloadRejected, match="抖音未返回"):
        ydl.extract_info(VIDEO_URL, download=False)
    metadata.assert_not_called()


@pytest.mark.parametrize(
    "response", [None, {}, {"aweme_detail": []}, {"status_msg": "private-secret"}]
)
def test_missing_details_report_platform_failure_without_raw_response(douyin, response):
    ydl, _, _, metadata = douyin
    metadata.return_value = response
    with pytest.raises(DownloadRejected, match="抖音未返回视频详情") as caught:
        ydl.extract_info(VIDEO_URL, download=False)
    assert "private-secret" not in str(caught.value)


def test_douyin_override_is_local_to_the_download_instance(douyin):
    ydl, extractor, _, _ = douyin
    assert type(extractor) is not DouyinIE
    assert isinstance(extractor, DouyinIE)
    assert not extractor.suitable("https://www.bilibili.com/video/BV1234567890")
    with YoutubeDL({"quiet": True, "cachedir": False}) as other:
        assert type(other.get_info_extractor("Douyin")) is DouyinIE
        assert type(ydl.get_info_extractor("Generic")) is type(
            other.get_info_extractor("Generic")
        )

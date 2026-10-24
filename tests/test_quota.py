import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.quota import BEIJING_TZ, DailyQuota, DailyQuotaExceeded


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "5", None, float("inf")])
def test_daily_limit_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="daily_analysis_limit"):
        Settings.from_mapping({"user_quota": {"daily_analysis_limit": value}})


def test_quota_resets_at_beijing_midnight_and_prunes_old_usage(tmp_path, monkeypatch):
    clock = Mock()
    monkeypatch.setattr("astrbot_plugin_vision.quota.datetime", clock)
    # UTC 15:59:59 对应北京时间 23:59:59。
    before = datetime(2026, 10, 24, 15, 59, 59, tzinfo=timezone.utc)
    clock.now.side_effect = lambda tz: before.astimezone(tz)
    quota = DailyQuota(tmp_path / "quota.sqlite3")
    assert quota.charge("qq", "alice", "first", 1) == ("2026-10-24", True)
    with pytest.raises(DailyQuotaExceeded):
        quota.charge("qq", "alice", "second", 1)
    after = before + timedelta(seconds=1)
    clock.now.side_effect = lambda tz: after.astimezone(tz)
    assert quota.charge("qq", "alice", "second", 1) == ("2026-10-25", True)
    clock.now.assert_called_with(BEIJING_TZ)
    with sqlite3.connect(quota.path) as connection:
        assert connection.execute("SELECT day FROM analysis_usage").fetchall() == [
            ("2026-10-25",)
        ]


def test_atomic_quota_across_instances_and_idempotent_question(tmp_path):
    path = tmp_path / "quota.sqlite3"

    def attempt(index):
        try:
            return index, DailyQuota(path).charge("qq", "alice", str(index), 5)
        except DailyQuotaExceeded:
            return index, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(20)))
    accepted = [(index, value) for index, value in results if value is not None]
    assert len(accepted) == 5
    index, (day, _) = accepted[0]
    quota = DailyQuota(path)
    # 已达上限时重试同一问题仍然放行，且不多记一次。
    assert quota.charge("qq", "alice", str(index), 5) == (day, False)
    quota.refund(day, "qq", "alice", str(index))
    assert quota.charge("qq", "alice", "replacement", 5) == (day, True)
    with pytest.raises(DailyQuotaExceeded):
        quota.charge("qq", "alice", "over-limit", 5)

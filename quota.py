"""按北京时间自然日持久化用户分析额度，同一问题只计一次。"""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

BEIJING_TZ = timezone(timedelta(hours=8))


class DailyQuotaExceeded(Exception):
    def __init__(self, limit: int):
        super().__init__(
            f"今日视频分析次数已达上限（{limit} 次），请于北京时间次日 00:00 后再试。"
        )


class DailyQuota:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS analysis_usage ("
                "day TEXT NOT NULL, platform TEXT NOT NULL, user_id TEXT NOT NULL, "
                "request_id TEXT NOT NULL, "
                "PRIMARY KEY (day, platform, user_id, request_id)) WITHOUT ROWID"
            )
            yield connection

    def charge(
        self, platform: str, user_id: str, request_id: str, limit: int
    ) -> tuple[str, bool]:
        with self._connection() as connection:
            # 同一事务中检查并记账，跨会话并发也不能越过上限。
            connection.execute("BEGIN IMMEDIATE")
            day = datetime.now(BEIJING_TZ).date().isoformat()
            connection.execute("DELETE FROM analysis_usage WHERE day < ?", (day,))
            owner = (day, platform, user_id)
            if connection.execute(
                "SELECT 1 FROM analysis_usage "
                "WHERE day = ? AND platform = ? AND user_id = ? AND request_id = ?",
                (*owner, request_id),
            ).fetchone():
                return day, False
            used = connection.execute(
                "SELECT COUNT(*) FROM analysis_usage "
                "WHERE day = ? AND platform = ? AND user_id = ?",
                owner,
            ).fetchone()[0]
            if used >= limit:
                raise DailyQuotaExceeded(limit)
            connection.execute(
                "INSERT INTO analysis_usage (day, platform, user_id, request_id) "
                "VALUES (?, ?, ?, ?)",
                (*owner, request_id),
            )
            return day, True

    def refund(self, day: str, platform: str, user_id: str, request_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM analysis_usage "
                "WHERE day = ? AND platform = ? AND user_id = ? AND request_id = ?",
                (day, platform, user_id, request_id),
            )

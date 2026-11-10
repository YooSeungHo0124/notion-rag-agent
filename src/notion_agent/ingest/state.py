"""Sync state (SQLite): which page versions are already in the index."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    page_id          TEXT PRIMARY KEY,
    last_edited_time TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    indexed_at       TEXT NOT NULL,
    status           TEXT NOT NULL,
    chunk_count      INTEGER NOT NULL DEFAULT 0
)
"""


@dataclass(frozen=True)
class PageState:
    page_id: str
    last_edited_time: str  # ISO 8601
    content_hash: str
    indexed_at: str
    status: str  # "indexed"
    chunk_count: int = 0


class SyncState:
    """Small SQLite-backed store. Use `":memory:"` for tests."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, page_id: str) -> PageState | None:
        row = self._conn.execute(
            "SELECT page_id, last_edited_time, content_hash, indexed_at, status, chunk_count"
            " FROM pages WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        return PageState(*row) if row else None

    def set(
        self,
        page_id: str,
        last_edited_time: datetime | str,
        content_hash: str,
        *,
        status: str = "indexed",
        chunk_count: int = 0,
    ) -> None:
        edited = (
            last_edited_time.isoformat()
            if isinstance(last_edited_time, datetime)
            else last_edited_time
        )
        now = datetime.now(UTC).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?)",
                (page_id, edited, content_hash, now, status, chunk_count),
            )

    def delete(self, page_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM pages WHERE page_id = ?", (page_id,))

    def known_ids(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT page_id FROM pages")}

    def close(self) -> None:
        self._conn.close()

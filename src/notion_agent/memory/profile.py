"""Long-term memory: durable facts/preferences about me, stored locally (SQLite).

Short-term memory (the conversation) lives in the LangGraph checkpoint per thread_id.
Long-term facts are written only when I explicitly ask the agent to remember something
(`remember` tool) and are injected into every planner prompt. They never go to Notion.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

MAX_FACTS_IN_PROMPT = 30


class ProfileStore:
    def __init__(self, path: Path | str) -> None:
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, fact TEXT UNIQUE, "
            "created_at TEXT)"
        )
        self.db.commit()

    def add(self, fact: str) -> int:
        fact = fact.strip()
        cur = self.db.execute(
            "INSERT OR IGNORE INTO facts(fact, created_at) VALUES (?, ?)",
            (fact, datetime.now(UTC).isoformat()),
        )
        self.db.commit()
        return cur.lastrowid or 0

    def list(self) -> list[dict[str, object]]:
        rows = self.db.execute("SELECT id, fact, created_at FROM facts ORDER BY id").fetchall()
        return [{"id": r[0], "fact": r[1], "created_at": r[2]} for r in rows]

    def delete(self, fact_id: int) -> bool:
        cur = self.db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        self.db.commit()
        return cur.rowcount > 0

    def as_prompt(self) -> str:
        facts = self.list()[-MAX_FACTS_IN_PROMPT:]
        return "\n".join(f"- {f['fact']}" for f in facts) or "(없음)"

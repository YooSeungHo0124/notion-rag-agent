"""Shared helpers for evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from notion_agent.config import PROJECT_ROOT, get_settings
from notion_agent.rag.store import QdrantStore

EVAL_DIR = PROJECT_ROOT / "eval"
DATASET = EVAL_DIR / "dataset.jsonl"
RUNS = EVAL_DIR / "runs"


def open_store() -> QdrantStore:
    s = get_settings()
    return QdrantStore(s.file.retrieval.collection, url=s.qdrant_url, path=s.qdrant_path)


def all_chunks(store: QdrantStore) -> list[dict[str, Any]]:
    out, offset = [], None
    while True:
        points, offset = store.client.scroll(
            store.collection, limit=512, offset=offset, with_payload=True, with_vectors=False
        )
        out.extend({"chunk_id": str(p.id), **(p.payload or {})} for p in points)
        if offset is None:
            return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

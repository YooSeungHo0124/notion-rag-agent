"""Source viewer: one Notion page as a flat list of blocks (ids match chunk.block_ids).

Notion pages cannot be iframed (x-frame-options: SAMEORIGIN), so the UI renders the page
itself and highlights the cited blocks. Content is fetched live from the Notion API (so edits
after the last sync are visible) with a short in-memory cache; if Notion is unreachable we
fall back to the indexed chunks.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

CACHE_TTL_S = 300.0


class PageLoader:
    def __init__(
        self,
        fetch_live: Callable[[str], dict[str, Any]] | None,
        fetch_index: Callable[[str], dict[str, Any] | None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fetch_live, self.fetch_index, self.clock = fetch_live, fetch_index, clock
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(
        self, page_id: str, refresh: bool = False, live: bool = True
    ) -> dict[str, Any] | None:
        if not live:  # instant: indexed chunks only (UI swaps in the live page when ready)
            idx = self.fetch_index(page_id)
            return idx | {"source": "index"} if idx else None
        now = self.clock()
        with self._lock:
            hit = self._cache.get(page_id)
        if hit and not refresh and now - hit[0] < CACHE_TTL_S:
            return hit[1]
        page: dict[str, Any] | None = None
        if self.fetch_live is not None:
            try:
                page = self.fetch_live(page_id) | {"source": "notion"}
            except Exception as e:  # noqa: BLE001 - fall back to the index
                log.warning("live fetch failed for %s: %s", page_id, e)
        if page is None and self.fetch_live is None:
            # no Notion token: index only. (With a token we don't fall back here: the live
            # path runs without the store lock, and the UI already shows the index version.)
            idx = self.fetch_index(page_id)
            page = idx | {"source": "index"} if idx else None
        if page is not None:
            with self._lock:
                self._cache[page_id] = (now, page)
        return page


def live_fetcher(reader: Any) -> Callable[[str], dict[str, Any]]:
    """Build a fetcher from a NotionReader (pages.retrieve + top-level blocks)."""
    from notion_agent.ingest.notion_reader import extract_title, properties_to_text

    def fetch(page_id: str) -> dict[str, Any]:
        meta = reader._call(reader.client.pages.retrieve, page_id=page_id)
        blocks = []
        props = properties_to_text(meta)
        if props:
            blocks.append({"id": meta["id"], "type": "properties", "text": props, "level": None})
        blocks += [{"id": b.block_id, "type": b.type, "text": b.text, "level": b.heading_level}
                   for b in reader.fetch_blocks(page_id)]
        return {
            "page_id": meta["id"],
            "title": extract_title(meta) or "(제목 없음)",
            "url": meta.get("url") or f"https://www.notion.so/{page_id.replace('-', '')}",
            "last_edited_time": meta.get("last_edited_time"),
            "blocks": blocks,
        }

    return fetch


def index_fetcher(store: Any) -> Callable[[str], dict[str, Any] | None]:
    """Reassemble a page from its indexed chunks (one pseudo-block per chunk)."""

    def fetch(page_id: str) -> dict[str, Any] | None:
        from qdrant_client import models

        pts, _ = store.client.scroll(
            store.collection, limit=500, with_payload=True, with_vectors=False,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="page_id", match=models.MatchValue(value=page_id))]))
        rows = sorted((p.payload or {} for p in pts), key=lambda r: r.get("chunk_index", 0))
        if not rows:
            return None
        return {
            "page_id": page_id,
            "title": rows[0].get("page_title"),
            "url": rows[0].get("page_url"),
            "path": rows[0].get("path", []),
            "last_edited_time": str(rows[0].get("last_edited_time")),
            "blocks": [{"id": (r.get("block_ids") or [f"chunk-{i}"])[0], "type": "chunk",
                        "text": r.get("text", ""), "level": None,
                        "block_ids": r.get("block_ids", [])} for i, r in enumerate(rows)],
        }

    return fetch

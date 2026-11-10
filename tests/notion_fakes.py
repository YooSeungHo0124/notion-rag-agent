"""Offline fake of the notion-client endpoints used by NotionReader."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class FakeAPIError(Exception):
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.headers = headers or {}


def rt(text: str, code: bool = False) -> list[dict[str, Any]]:
    return [{"plain_text": text, "annotations": {"code": code}}]


def blk(bid: str, type_: str, text: str = "", children: bool = False, **extra: Any) -> dict:
    data: dict[str, Any] = {"rich_text": rt(text)}
    data.update(extra.get(type_, {}))
    return {"object": "block", "id": bid, "type": type_, "has_children": children, type_: data}


def page(
    pid: str, title: str, edited: str = "2026-01-01T00:00:00.000Z", **kw: Any
) -> dict[str, Any]:
    props: dict[str, Any] = {"Name": {"type": "title", "title": rt(title)}}
    props.update(kw.pop("props", {}))
    return {
        "object": "page",
        "id": pid,
        "url": f"https://www.notion.so/{pid}",
        "created_time": "2025-12-01T00:00:00.000Z",
        "last_edited_time": edited,
        "archived": False,
        "properties": props,
        **kw,
    }


class FakeNotion:
    """children: block_id -> list of blocks; pages: id -> page; page_size forces pagination."""

    def __init__(self, page_size: int = 2) -> None:
        self.page_store: dict[str, dict] = {}
        self.children: dict[str, list[dict]] = {}
        self.db_store: dict[str, dict] = {}
        self.ds_rows: dict[str, list[dict]] = {}
        self.page_size = page_size
        self.calls: list[str] = []
        self.fail_queue: list[Exception] = []
        self.blocks = SimpleNamespace(children=SimpleNamespace(list=self._children_list))
        self.pages = SimpleNamespace(retrieve=self._page_retrieve)
        self.databases = SimpleNamespace(retrieve=self._db_retrieve)
        self.data_sources = SimpleNamespace(query=self._ds_query)

    def _maybe_fail(self) -> None:
        if self.fail_queue:
            raise self.fail_queue.pop(0)

    def _slice(self, items: list[dict], start_cursor: str | None) -> dict:
        start = int(start_cursor or 0)
        end = start + self.page_size
        more = end < len(items)
        return {
            "results": items[start:end],
            "has_more": more,
            "next_cursor": str(end) if more else None,
        }

    def _children_list(self, block_id: str, start_cursor: str | None = None, **_: Any) -> dict:
        self.calls.append(f"children:{block_id}:{start_cursor}")
        self._maybe_fail()
        return self._slice(self.children.get(block_id, []), start_cursor)

    def _page_retrieve(self, page_id: str) -> dict:
        self.calls.append(f"page:{page_id}")
        self._maybe_fail()
        if page_id not in self.page_store:
            raise FakeAPIError(404)
        return self.page_store[page_id]

    def _db_retrieve(self, database_id: str) -> dict:
        self.calls.append(f"db:{database_id}")
        if database_id not in self.db_store:
            raise FakeAPIError(404)
        return self.db_store[database_id]

    def _ds_query(self, data_source_id: str, start_cursor: str | None = None, **_: Any) -> dict:
        self.calls.append(f"ds:{data_source_id}:{start_cursor}")
        return self._slice(self.ds_rows.get(data_source_id, []), start_cursor)

    def search(self, start_cursor: str | None = None, **_: Any) -> dict:
        self.calls.append(f"search:{start_cursor}")
        return self._slice(list(self.page_store.values()), start_cursor)

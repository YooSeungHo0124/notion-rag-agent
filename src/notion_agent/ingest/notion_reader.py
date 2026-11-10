"""Read pages from Notion (official `notion-client`) and convert them to `NotionPage`.

Targets Notion API version 2025-09-03 (databases expose `data_sources`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

from notion_agent.ingest.models import Block, NotionPage

log = logging.getLogger(__name__)

MAX_RETRIES = 5
MIN_INTERVAL_S = 0.34  # ~3 requests / second (Notion average rate limit)
_LIST_TYPES = {"bulleted_list_item", "numbered_list_item", "to_do"}


def _status_of(exc: Exception) -> int | None:
    return getattr(exc, "status", None)


def _retry_after(exc: Exception, attempt: int) -> float:
    headers = getattr(exc, "headers", None) or {}
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value is not None:
            return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    return min(2.0**attempt, 30.0)


def rich_text(items: list[dict[str, Any]] | None) -> str:
    """Concatenate Notion rich_text into plain text (inline code kept as `code`)."""
    out = []
    for it in items or []:
        text = it.get("plain_text", "")
        if it.get("annotations", {}).get("code") and text:
            text = f"`{text}`"
        out.append(text)
    return "".join(out)


def extract_title(page: dict[str, Any]) -> str:
    """Title from whichever property has type `title` (name varies per database)."""
    for prop in (page.get("properties") or {}).values():
        if prop.get("type") == "title":
            return rich_text(prop.get("title")).strip()
    # child_database / database objects carry a top-level title
    if isinstance(page.get("title"), list):
        return rich_text(page["title"]).strip()
    return ""


def extract_tags(page: dict[str, Any]) -> list[str]:
    """Tags from all select / multi_select properties (deduplicated, order kept)."""
    tags: list[str] = []
    for prop in (page.get("properties") or {}).values():
        t = prop.get("type")
        if t == "multi_select":
            tags.extend(o["name"] for o in prop.get("multi_select") or [] if o.get("name"))
        elif t == "select" and prop.get("select") and prop["select"].get("name"):
            tags.append(prop["select"]["name"])
    return list(dict.fromkeys(tags))


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def block_to_text(block: dict[str, Any], children_text: str = "", number: int = 1) -> str | None:
    """Render one block (children already rendered) as markdown-ish text.

    Returns None for unsupported / empty-meaningless blocks.
    """
    t = block.get("type", "")
    data = block.get(t) or {}
    txt = rich_text(data.get("rich_text"))
    body: str | None
    if t in ("heading_1", "heading_2", "heading_3"):
        body = f"{'#' * int(t[-1])} {txt}"
    elif t == "paragraph":
        body = txt
    elif t == "bulleted_list_item":
        body = f"- {txt}"
    elif t == "numbered_list_item":
        body = f"{number}. {txt}"
    elif t == "to_do":
        body = f"- [{'x' if data.get('checked') else ' '}] {txt}"
    elif t == "code":
        lang = data.get("language") or ""
        body = f"```{lang}\n{txt}\n```"
    elif t == "quote":
        body = _indent(txt, "> ") if txt else ">"
    elif t == "callout":
        icon = (data.get("icon") or {}).get("emoji") or ""
        body = f"> {icon} {txt}".rstrip() if icon else f"> {txt}"
    elif t == "toggle":
        body = txt
    elif t == "table_row":
        cells = [rich_text(c).replace("|", "\\|") for c in data.get("cells", [])]
        body = "| " + " | ".join(cells) + " |"
    elif t == "table":
        body = ""
    elif t == "child_page":
        body = f"[하위 페이지: {data.get('title', '')}]"
    elif t == "child_database":
        body = f"[데이터베이스: {data.get('title', '')}]"
    elif t == "equation":
        body = f"$${data.get('expression', '')}$$"
    elif t == "bookmark" or t == "embed" or t == "link_preview":
        cap = rich_text(data.get("caption"))
        body = f"[{cap or 'link'}]({data.get('url', '')})"
    elif t == "divider":
        body = "---"
    else:
        log.debug("skipping unsupported block type %s", t)
        body = None

    if children_text:
        if t in _LIST_TYPES:
            children_text = _indent(children_text)
        body = f"{body}\n{children_text}" if body else children_text
    if body is None or not body.strip():
        return None
    return body


class NotionReader:
    """Enumerate and fetch Notion pages with pagination, recursion and 429 handling.

    `client` is a `notion_client.Client` (or any object with the same endpoints, for tests).
    """

    def __init__(
        self,
        client: Any,
        root_ids: list[str] | None = None,
        *,
        min_interval_s: float = MIN_INTERVAL_S,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.root_ids = list(root_ids or [])
        self.min_interval_s = min_interval_s
        self._sleep = sleep
        self._clock = clock
        self._last_call = 0.0
        self._titles: dict[str, str] = {}  # page/database id -> title (for paths)

    @classmethod
    def from_settings(cls, settings: Any | None = None) -> NotionReader:
        """Build a reader with a real client (SDK retries disabled; we retry ourselves)."""
        from notion_client import Client

        from notion_agent.config import get_settings

        s = settings or get_settings()
        if not s.notion_token:
            raise RuntimeError("NOTION_TOKEN is not set (.env)")
        return cls(Client(auth=s.notion_token, retry=False), s.root_ids)

    # -- low level ---------------------------------------------------------
    def _call(self, fn: Callable[..., Any], **kwargs: Any) -> Any:
        """Call an endpoint with throttling and retry on 429 / 5xx."""
        for attempt in range(MAX_RETRIES + 1):
            wait = self.min_interval_s - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            self._last_call = self._clock()
            try:
                return fn(**kwargs)
            except Exception as exc:
                status = _status_of(exc)
                retriable = status == 429 or (status is not None and 500 <= status < 600)
                if not retriable or attempt == MAX_RETRIES:
                    raise
                delay = _retry_after(exc, attempt)
                log.warning("notion %s, retrying in %.1fs", status, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _paginate(self, fn: Callable[..., Any], **kwargs: Any) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            params = dict(kwargs, page_size=100)
            if cursor:
                params["start_cursor"] = cursor
            resp = self._call(fn, **params)
            yield from resp.get("results", [])
            if not resp.get("has_more"):
                return
            cursor = resp.get("next_cursor")

    def list_children(self, block_id: str) -> Iterator[dict[str, Any]]:
        return self._paginate(self.client.blocks.children.list, block_id=block_id)

    def _query_database(self, database_id: str) -> Iterator[dict[str, Any]]:
        db = self._call(self.client.databases.retrieve, database_id=database_id)
        self._titles[database_id] = extract_title(db)
        for ds in db.get("data_sources") or []:
            yield from self._paginate(self.client.data_sources.query, data_source_id=ds["id"])

    # -- enumeration -------------------------------------------------------
    def iter_page_metas(self) -> Iterator[tuple[dict[str, Any], list[str]]]:
        """Yield (raw page object, ancestor titles) for every reachable page, once each."""
        seen: set[str] = set()
        if not self.root_ids:
            for obj in self._paginate(
                self.client.search, filter={"property": "object", "value": "page"}
            ):
                if obj.get("object") == "page" and obj["id"] not in seen:
                    seen.add(obj["id"])
                    self._titles[obj["id"]] = extract_title(obj)
                    yield obj, []
            return
        for root in self.root_ids:
            yield from self._walk_root(root, seen)

    def _walk_root(
        self, root_id: str, seen: set[str]
    ) -> Iterator[tuple[dict[str, Any], list[str]]]:
        try:
            page = self._call(self.client.pages.retrieve, page_id=root_id)
        except Exception as exc:
            if _status_of(exc) not in (400, 404):
                raise
            # root may be a database
            yield from self._walk_database(root_id, [], seen)
            return
        yield from self._walk_page(page, [], seen)

    def _walk_page(
        self, page: dict[str, Any], path: list[str], seen: set[str]
    ) -> Iterator[tuple[dict[str, Any], list[str]]]:
        if page["id"] in seen:
            return
        seen.add(page["id"])
        title = extract_title(page)
        self._titles[page["id"]] = title
        yield page, path
        if page.get("archived") or page.get("in_trash"):
            return
        yield from self._walk_children(page["id"], [*path, title], seen)

    def _walk_children(
        self, block_id: str, path: list[str], seen: set[str]
    ) -> Iterator[tuple[dict[str, Any], list[str]]]:
        for child in self.list_children(block_id):
            t = child.get("type")
            if t == "child_page":
                page = self._call(self.client.pages.retrieve, page_id=child["id"])
                yield from self._walk_page(page, path, seen)
            elif t == "child_database":
                yield from self._walk_database(child["id"], path, seen)
            elif child.get("has_children") and t not in ("synced_block",):
                # child pages can be nested inside toggles / columns
                yield from self._walk_children(child["id"], path, seen)

    def _walk_database(
        self, database_id: str, path: list[str], seen: set[str]
    ) -> Iterator[tuple[dict[str, Any], list[str]]]:
        rows = list(self._query_database(database_id))
        db_path = [*path, self._titles.get(database_id, "")]
        for row in rows:
            if row.get("object") == "page":
                yield from self._walk_page(row, db_path, seen)

    # -- content -----------------------------------------------------------
    def fetch_blocks(self, page_id: str) -> list[Block]:
        """Top-level blocks of a page; nested children are flattened into their parent text."""
        return self._render_level(page_id)

    def _render_level(self, parent_id: str) -> list[Block]:
        blocks: list[Block] = []
        number = 0
        for raw in self.list_children(parent_id):
            t = raw.get("type", "")
            number = number + 1 if t == "numbered_list_item" else 0
            children_text = ""
            if raw.get("has_children") and t not in ("child_page", "child_database"):
                kids = self._render_level(raw["id"])
                children_text = "\n".join(b.text for b in kids)
            text = block_to_text(raw, children_text, number or 1)
            if text is None:
                continue
            level = int(t[-1]) if t.startswith("heading_") else None
            blocks.append(Block(block_id=raw["id"], type=t, text=text, heading_level=level))
        return blocks

    def to_notion_page(
        self, page: dict[str, Any], path: list[str], *, with_blocks: bool = True
    ) -> NotionPage:
        archived = bool(page.get("archived") or page.get("in_trash"))
        return NotionPage(
            page_id=page["id"],
            title=extract_title(page),
            url=page.get("url") or f"https://www.notion.so/{page['id'].replace('-', '')}",
            path=[p for p in path if p],
            tags=extract_tags(page),
            created_time=_parse_dt(page["created_time"]),
            last_edited_time=_parse_dt(page["last_edited_time"]),
            archived=archived,
            blocks=self.fetch_blocks(page["id"]) if with_blocks and not archived else [],
        )

    def iter_pages(self) -> Iterator[NotionPage]:
        """Fully fetched pages (metadata + blocks)."""
        for raw, path in self.iter_page_metas():
            yield self.to_notion_page(raw, path)

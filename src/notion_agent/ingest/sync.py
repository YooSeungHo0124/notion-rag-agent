"""Incremental sync: Notion -> PageSink (vector index), tracked in SyncState.

Consistency rule: the sink is called first and state is committed only after the sink
succeeded, so a crash leaves the page "dirty" and it is retried on the next run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from notion_agent.ingest.models import NotionPage
from notion_agent.ingest.state import SyncState

log = logging.getLogger(__name__)


class PageSink(Protocol):
    """Index writer implemented by F2 (e.g. `notion_agent.rag.sink.QdrantPageSink`)."""

    def upsert_page(self, page: NotionPage) -> int:
        """Replace all chunks of `page` in the index; return the chunk count."""
        ...

    def delete_page(self, page_id: str) -> None:
        """Remove all chunks of `page_id` from the index (no-op if absent)."""
        ...


class PageSource(Protocol):
    """What `sync` needs from a reader (`NotionReader` satisfies it)."""

    def iter_page_metas(self) -> Iterator[tuple[dict[str, Any], list[str]]]: ...

    def to_notion_page(
        self, page: dict[str, Any], path: list[str], *, with_blocks: bool = True
    ) -> NotionPage: ...


@dataclass
class SyncReport:
    scanned: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)
    changed_ids: list[str] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"scanned={self.scanned} changed={self.changed} unchanged={self.unchanged} "
            f"deleted={self.deleted} failed={self.failed} chunks={self.chunks}"
        )


class PrintSink:
    """Dry-run sink: prints what would happen, writes nothing."""

    def __init__(self, echo: Any = print) -> None:
        self.echo = echo

    def upsert_page(self, page: NotionPage) -> int:
        title = page.title or "(untitled)"
        self.echo(f"[upsert] {title} ({page.page_id}) {len(page.blocks)} blocks")
        return 0

    def delete_page(self, page_id: str) -> None:
        self.echo(f"[delete] {page_id}")


def _delete(sink: PageSink, state: SyncState, page_id: str, report: SyncReport, commit: bool):
    try:
        sink.delete_page(page_id)
    except Exception as exc:  # noqa: BLE001 - recorded, run continues
        report.failed += 1
        report.errors.append(f"delete {page_id}: {exc}")
        return
    if commit:
        state.delete(page_id)
    report.deleted += 1
    report.deleted_ids.append(page_id)


def sync(
    reader: PageSource,
    sink: PageSink,
    state: SyncState,
    full: bool = False,
    *,
    commit: bool = True,
) -> SyncReport:
    """Bring the index in line with Notion. `full=True` reprocesses every page.

    `commit=False` (dry-run) leaves the state DB untouched.
    """
    report = SyncReport()
    known = state.known_ids()
    alive: set[str] = set()
    enumeration_ok = True

    try:
        metas = reader.iter_page_metas()
        for raw, path in metas:
            page_id = raw["id"]
            report.scanned += 1
            if raw.get("archived") or raw.get("in_trash"):
                alive.add(page_id)  # handled here, not again in the deletion pass
                if page_id in known:
                    _delete(sink, state, page_id, report, commit)
                continue
            alive.add(page_id)
            prev = state.get(page_id)
            try:
                meta = reader.to_notion_page(raw, path, with_blocks=False)
                edited = meta.last_edited_time.isoformat()
                if not full and prev and prev.last_edited_time == edited:
                    report.unchanged += 1
                    continue
                page = reader.to_notion_page(raw, path)
                digest = page.content_hash()
                if not full and prev and prev.content_hash == digest:
                    # edited timestamp moved but content identical: no re-embedding
                    if commit:
                        state.set(page_id, edited, digest, chunk_count=prev.chunk_count)
                    report.unchanged += 1
                    continue
                n = sink.upsert_page(page)
                if commit:
                    state.set(page_id, edited, digest, chunk_count=n)
                report.changed += 1
                report.chunks += n
                report.changed_ids.append(page_id)
            except Exception as exc:  # noqa: BLE001 - one bad page must not abort the run
                log.exception("sync failed for page %s", page_id)
                report.failed += 1
                report.errors.append(f"{page_id}: {exc}")
    except Exception as exc:  # noqa: BLE001 - enumeration broke; skip deletions to be safe
        enumeration_ok = False
        report.failed += 1
        report.errors.append(f"enumeration aborted: {exc}")

    if enumeration_ok:
        for page_id in sorted(known - alive):
            _delete(sink, state, page_id, report, commit)

    log.info("sync done: %s", report.summary())
    return report

"""Index a folder of Markdown files as pages (page_id `doc:<file stem>`).

Used for the fictional company docs in `scenario/docs` that sit next to the demo ERP,
without writing anything into the user's real Notion. Pages are tagged `company` so they
can be filtered, and they never enter the Notion sync state, so a Notion sync leaves
them alone.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from notion_agent.ingest.models import Block, NotionPage

DOC_PREFIX = "doc:"
TAG = "company"
_HEAD = re.compile(r"^(#{1,3})\s+(.*)$")


def md_to_page(path: Path, root: Path) -> NotionPage:
    text = path.read_text(encoding="utf-8")
    pid = DOC_PREFIX + path.stem
    title, blocks, para = path.stem, [], []

    def bid(i: int) -> str:
        return hashlib.md5(f"{pid}:{i}".encode()).hexdigest()

    def flush() -> None:
        if para:
            blocks.append(Block(block_id=bid(len(blocks)), type="paragraph",
                                text="\n".join(para)))
            para.clear()

    for line in text.splitlines():
        m = _HEAD.match(line)
        if m:
            flush()
            level, head = len(m.group(1)), m.group(2).strip()
            if level == 1 and title == path.stem:
                title = head
                continue
            blocks.append(Block(block_id=bid(len(blocks)), type=f"heading_{level}", text=head,
                                heading_level=level))
        elif line.strip():
            para.append(line.rstrip())
        else:
            flush()
    flush()
    mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)  # dated docs: use the doc date
    edited = datetime.fromisoformat(m.group(1)).replace(tzinfo=UTC) if m else mtime
    rel = path.relative_to(root.parent) if root.parent in path.parents else path
    return NotionPage(page_id=pid, title=title, url=f"file://{rel}", path=["회사 문서"],
                      tags=[TAG], created_time=edited, last_edited_time=edited, blocks=blocks)


def load_docs(root: Path) -> list[NotionPage]:
    return [md_to_page(p, root) for p in sorted(root.glob("*.md")) if p.name != "README.md"]


def index_docs(root: Path, sink) -> int:  # type: ignore[no-untyped-def]
    pages = load_docs(root)
    n = 0
    for p in pages:
        sink.delete_page(p.page_id)  # replace old chunks of an edited doc
        n += sink.upsert_page(p)
    return n

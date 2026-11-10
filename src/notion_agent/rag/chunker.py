"""Heading-aware chunker: NotionPage -> list[Chunk] with context prefix and block provenance."""

from __future__ import annotations

from dataclasses import dataclass, field

from notion_agent.config import ChunkingConfig
from notion_agent.ingest.models import Block, Chunk, NotionPage, make_chunk_id


@dataclass
class _Piece:
    block_id: str
    text: str
    is_code: bool


@dataclass
class _Section:
    heading_path: list[str]
    pieces: list[_Piece] = field(default_factory=list)


def _sections(blocks: list[Block]) -> list[_Section]:
    """Group blocks into sections keyed by their heading path (heading text is included)."""
    stack: list[tuple[int, str]] = []
    sections: list[_Section] = [_Section([])]
    for b in blocks:
        if not b.text.strip():
            continue
        if b.heading_level is not None:
            stack = [(lvl, t) for lvl, t in stack if lvl < b.heading_level]
            stack.append((b.heading_level, b.text.strip()))
            sections.append(_Section([t for _, t in stack]))
        sections[-1].pieces.append(_Piece(b.block_id, b.text, b.type == "code"))
    return [s for s in sections if s.pieces]


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Hard-split an oversized text, preferring newline boundaries."""
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + max_chars // 2, end)
            if nl > start:
                end = nl + 1
        out.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def _join(pieces: list[_Piece]) -> str:
    return "\n\n".join(p.text for p in pieces)


def _tail(text: str, n: int) -> str:
    if n <= 0 or len(text) <= n:
        return text if n > 0 else ""
    cut = text[-n:]
    sp = cut.find(" ")
    return cut[sp + 1 :] if 0 <= sp < n // 2 else cut


def _prefix(page: NotionPage, heading_path: list[str]) -> str:
    parts = [page.title, *page.path, *heading_path]
    return f"[{' > '.join(p for p in parts if p)} | {page.last_edited_time:%Y-%m-%d}]\n"


def chunk_page(page: NotionPage, cfg: ChunkingConfig) -> list[Chunk]:
    """Split a page along headings, packing blocks up to ``cfg.max_chars`` with overlap.

    Code blocks are never split unless a single block exceeds ``max_chars``. Overlap
    (tail of previous chunk) is only carried within a section and never from code.
    """
    raw: list[tuple[list[str], list[str], str]] = []  # (heading_path, block_ids, text)
    max_c, ov = cfg.max_chars, cfg.overlap_chars
    for sec in _sections(page.blocks):
        cur: list[_Piece] = []
        carry = ""
        for p in sec.pieces:
            if len(p.text) > max_c:
                if cur:
                    raw.append((sec.heading_path, [x.block_id for x in cur], _join(cur)))
                    cur = []
                for part in _split_long(p.text, max_c, 0 if p.is_code else ov):
                    raw.append((sec.heading_path, [p.block_id], part))
                carry = ""
                continue
            if cur and len(_join(cur)) + 2 + len(p.text) > max_c:
                text = _join(cur)
                raw.append((sec.heading_path, [x.block_id for x in cur], text))
                carry = "" if cur[-1].is_code else _tail(text, ov)
                cur = []
                if carry and not p.is_code and len(carry) + 2 + len(p.text) <= max_c:
                    # overlap piece: provenance stays with the previous chunk
                    cur.append(_Piece("", carry, False))
            cur.append(p)
        if cur:
            raw.append((sec.heading_path, [x.block_id for x in cur], _join(cur)))

    chunks: list[Chunk] = []
    for idx, (hp, ids, text) in enumerate(raw):
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(page.page_id, idx, text),
                page_id=page.page_id,
                page_title=page.title,
                page_url=page.url,
                path=page.path,
                heading_path=hp,
                block_ids=list(dict.fromkeys(i for i in ids if i)),
                text=text,
                context_text=_prefix(page, hp) + text,
                tags=page.tags,
                last_edited_time=page.last_edited_time,
                chunk_index=idx,
            )
        )
    return chunks

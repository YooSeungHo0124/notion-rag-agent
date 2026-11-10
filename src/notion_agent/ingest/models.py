"""Shared data contract between ingestion (F1) and RAG (F2).

Ingestion produces `NotionPage` objects; the chunker consumes them and produces
`Chunk` objects that keep block-level provenance so citations can deep-link to the
exact Notion block (https://www.notion.so/<page_id_nodash>#<block_id_nodash>).
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from pydantic import BaseModel, Field


class Block(BaseModel):
    block_id: str
    type: str  # notion block type, e.g. "paragraph", "heading_2", "code", "to_do"
    text: str  # plain text / markdown rendering of the block (children flattened)
    heading_level: int | None = None  # 1..3 for headings, else None


class NotionPage(BaseModel):
    page_id: str
    title: str
    url: str
    path: list[str] = Field(default_factory=list)  # ancestor titles, root first
    tags: list[str] = Field(default_factory=list)
    created_time: datetime
    last_edited_time: datetime
    archived: bool = False
    blocks: list[Block] = Field(default_factory=list)

    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.title.encode())
        h.update("\x1f".join(sorted(self.tags)).encode())  # tag-only edits must re-index
        h.update("\x1f".join(self.path).encode())
        for b in self.blocks:
            h.update(b.block_id.encode())
            h.update(b.text.encode())
        return h.hexdigest()


def block_url(page_id: str, block_id: str) -> str:
    return f"https://www.notion.so/{page_id.replace('-', '')}#{block_id.replace('-', '')}"


class Chunk(BaseModel):
    chunk_id: str  # deterministic: see make_chunk_id
    page_id: str
    page_title: str
    page_url: str
    path: list[str]
    heading_path: list[str]
    block_ids: list[str]
    text: str  # raw chunk text
    context_text: str  # text with "[title > path > heading | date]" prefix, used for embedding
    tags: list[str] = Field(default_factory=list)
    last_edited_time: datetime
    chunk_index: int

    @property
    def cite_url(self) -> str:
        return block_url(self.page_id, self.block_ids[0]) if self.block_ids else self.page_url


def make_chunk_id(page_id: str, chunk_index: int, text: str) -> str:
    """Deterministic id -> idempotent upserts. Returned as a UUID string (Qdrant point id)."""
    digest = hashlib.sha256(f"{page_id}:{chunk_index}:{text}".encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"

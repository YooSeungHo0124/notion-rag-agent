"""Deterministic fake encoders/reranker for RAG tests (no model downloads)."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime

from qdrant_client import models

from notion_agent.ingest.models import Block, NotionPage


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _h(tok: str, mod: int) -> int:
    return int(hashlib.md5(tok.encode()).hexdigest(), 16) % mod


class FakeDense:
    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in _tokens(text):
            v[_h(t, self.dim)] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeSparse:
    def _vec(self, text: str) -> models.SparseVector:
        w: dict[int, float] = {}
        for t in _tokens(text):
            k = _h(t, 250000)
            w[k] = w.get(k, 0.0) + 1.0
        ks = sorted(w)
        return models.SparseVector(indices=ks, values=[w[k] for k in ks])

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._vec(text)


class FakeReranker:
    """Score = fraction of query tokens present in the passage."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(_tokens(query))
        return [len(q & set(_tokens(p))) / max(len(q), 1) for p in passages]


def make_page(
    page_id: str,
    title: str,
    blocks: list[tuple[str, str, int | None]],
    tags: list[str] | None = None,
    edited: datetime | None = None,
) -> NotionPage:
    t = edited or datetime(2026, 1, 1, tzinfo=UTC)
    return NotionPage(
        page_id=page_id,
        title=title,
        url=f"https://www.notion.so/{page_id}",
        path=["Root"],
        tags=tags or [],
        created_time=t,
        last_edited_time=t,
        blocks=[
            Block(block_id=f"{page_id}-b{i}", type=typ, text=text, heading_level=lvl)
            for i, (typ, text, lvl) in enumerate(blocks)
        ],
    )

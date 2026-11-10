"""Hybrid retrieval + cross-encoder rerank."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from notion_agent.config import RetrievalConfig
from notion_agent.ingest.models import block_url
from notion_agent.observability.tracing import documents, span
from notion_agent.rag.embedder import DenseEncoder, SparseEncoder
from notion_agent.rag.reranker import RerankerProtocol
from notion_agent.rag.store import QdrantStore, SearchFilters


class Hit(BaseModel):
    chunk_id: str
    page_id: str
    page_title: str
    page_url: str
    path: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    text: str
    context_text: str = ""
    tags: list[str] = Field(default_factory=list)
    last_edited_time: datetime
    chunk_index: int = 0
    fusion_score: float
    rerank_score: float
    answerable: bool = True

    @property
    def cite_url(self) -> str:
        if ":" in self.page_id:  # note:/doc:/erp: sources are not Notion pages
            return self.page_url
        return block_url(self.page_id, self.block_ids[0]) if self.block_ids else self.page_url


class Retriever:
    """query -> hybrid top candidate_k -> rerank -> top k."""

    def __init__(
        self,
        store: QdrantStore,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        reranker: RerankerProtocol,
        cfg: RetrievalConfig,
    ) -> None:
        self.store, self.dense, self.sparse, self.reranker, self.cfg = (
            store,
            dense,
            sparse,
            reranker,
            cfg,
        )

    def retrieve(
        self, query: str, k: int | None = None, filters: SearchFilters | None = None
    ) -> list[Hit]:
        """Return reranked hits; all marked ``answerable=False`` if best score < threshold."""
        k = k or self.cfg.final_k
        with span("retrieve", "RETRIEVER", input=query, k=k) as root:
            with span("hybrid_search", "RETRIEVER", input=query,
                      candidate_k=self.cfg.candidate_k) as sp:
                cands = self.store.hybrid_search(
                    self.dense.embed_query(query),
                    self.sparse.embed_query(query),
                    self.cfg.candidate_k,
                    filters,
                )
                sp.set(n_candidates=len(cands))
            if not cands:
                return []
            with span("rerank", "RERANKER", input=query, reranker__top_k=k) as sp:
                scores = self.reranker.score(
                    query, [c.payload.get("context_text") or c.payload["text"] for c in cands]
                )
                ranked = sorted(zip(cands, scores, strict=True), key=lambda x: x[1],
                                reverse=True)[:k]
                sp.set(top_score=float(ranked[0][1]))
            answerable = ranked[0][1] >= self.cfg.min_rerank_score
            hits = [
                Hit(**c.payload, fusion_score=c.score, rerank_score=s, answerable=answerable)
                for c, s in ranked
            ]
            root.set(answerable=answerable, **documents("retrieval__documents", [
                {"id": h.chunk_id, "score": round(h.rerank_score, 4),
                 "content": h.text[:1000], "metadata": {"title": h.page_title}}
                for h in hits]))
            return hits

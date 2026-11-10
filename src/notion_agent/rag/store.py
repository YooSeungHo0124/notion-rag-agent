"""Qdrant storage with named dense+sparse vectors and hybrid (RRF) search."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from notion_agent.ingest.models import Chunk

log = logging.getLogger(__name__)

DENSE, SPARSE = "dense", "sparse"
RRF_K = 60


@dataclass
class SearchFilters:
    """Optional retrieval filters (all combined with AND)."""

    page_ids: list[str] | None = None
    tags: list[str] | None = None  # match any
    edited_after: datetime | None = None
    edited_before: datetime | None = None


@dataclass
class ScoredChunk:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def build_filter(f: SearchFilters | None) -> models.Filter | None:
    if f is None:
        return None
    must: list[models.Condition] = []
    if f.page_ids:
        must.append(models.FieldCondition(key="page_id", match=models.MatchAny(any=f.page_ids)))
    if f.tags:
        must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=f.tags)))
    if f.edited_after or f.edited_before:
        must.append(
            models.FieldCondition(
                key="last_edited_time",
                range=models.DatetimeRange(gte=f.edited_after, lte=f.edited_before),
            )
        )
    return models.Filter(must=must) if must else None


def rrf_fuse(rankings: list[list[ScoredChunk]], limit: int, k: int = RRF_K) -> list[ScoredChunk]:
    """Client-side Reciprocal Rank Fusion (same formula as Qdrant: sum 1/(k+rank))."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
            payloads.setdefault(hit.id, hit.payload)
    order = sorted(scores, key=lambda i: scores[i], reverse=True)[:limit]
    return [ScoredChunk(i, scores[i], payloads[i]) for i in order]


class QdrantStore:
    """Thin wrapper around a Qdrant collection of Notion chunks."""

    def __init__(
        self,
        collection: str,
        url: str = "",
        path: Path | str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection = collection
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url)
        else:
            if path is None:
                raise ValueError("either url or path is required")
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        self.server_fusion = True  # flipped off if the backend rejects prefetch+RRF

    def ensure_collection(self, dense_dim: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams()},
        )
        for key, schema in (
            ("page_id", models.PayloadSchemaType.KEYWORD),
            ("tags", models.PayloadSchemaType.KEYWORD),
            ("last_edited_time", models.PayloadSchemaType.DATETIME),
        ):
            with warnings.catch_warnings():  # local mode ignores payload indexes (harmless)
                warnings.simplefilter("ignore", UserWarning)
                self.client.create_payload_index(self.collection, key, field_schema=schema)

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[models.SparseVector],
    ) -> int:
        if not (len(chunks) == len(dense) == len(sparse)):
            raise ValueError("chunks/dense/sparse length mismatch")
        if not chunks:
            return 0
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector={DENSE: d, SPARSE: s},
                payload=c.model_dump(mode="json"),
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)
        return len(points)

    def delete_by_page(self, page_id: str) -> None:
        flt = models.Filter(
            must=[models.FieldCondition(key="page_id", match=models.MatchValue(value=page_id))]
        )
        self.client.delete(
            self.collection, points_selector=models.FilterSelector(filter=flt), wait=True
        )

    def count(self, page_id: str | None = None) -> int:
        flt = None
        if page_id:
            flt = models.Filter(
                must=[models.FieldCondition(key="page_id", match=models.MatchValue(value=page_id))]
            )
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def _single(
        self, vec: Any, using: str, limit: int, flt: models.Filter | None
    ) -> list[ScoredChunk]:
        res = self.client.query_points(
            self.collection,
            query=vec,
            using=using,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [ScoredChunk(str(p.id), p.score, p.payload or {}) for p in res.points]

    def _client_side(
        self, dense: list[float], sparse: models.SparseVector, limit: int, flt: models.Filter | None
    ) -> list[ScoredChunk]:
        pre = max(limit * 2, limit)
        rankings = [self._single(dense, DENSE, pre, flt)]
        if sparse.indices:
            rankings.append(self._single(sparse, SPARSE, pre, flt))
        return rrf_fuse(rankings, limit)

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: models.SparseVector,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[ScoredChunk]:
        """Dense+sparse prefetch fused with server-side RRF; client-side RRF fallback."""
        flt = build_filter(filters)
        if not self.server_fusion:
            return self._client_side(query_dense, query_sparse, limit, flt)
        prefetch = [models.Prefetch(query=query_dense, using=DENSE, limit=limit * 2, filter=flt)]
        if query_sparse.indices:
            prefetch.append(
                models.Prefetch(query=query_sparse, using=SPARSE, limit=limit * 2, filter=flt)
            )
        try:
            res = self.client.query_points(
                self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        except (NotImplementedError, ValueError, TypeError) as e:
            log.warning("server-side RRF unsupported (%s); using client-side fusion", e)
            self.server_fusion = False
            return self._client_side(query_dense, query_sparse, limit, flt)
        return [ScoredChunk(str(p.id), p.score, p.payload or {}) for p in res.points]

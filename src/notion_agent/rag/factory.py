"""Build RAG components from settings."""

from __future__ import annotations

from notion_agent.config import Settings, get_settings
from notion_agent.rag.embedder import DenseEmbedder, SparseEmbedder
from notion_agent.rag.reranker import Reranker
from notion_agent.rag.retriever import Retriever
from notion_agent.rag.sink import QdrantPageSink
from notion_agent.rag.store import QdrantStore

_cache: dict[str, object] = {}


def _get(key: str, make):  # type: ignore[no-untyped-def]
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def _store(s: Settings) -> QdrantStore:
    return _get(
        "store",
        lambda: QdrantStore(s.file.retrieval.collection, url=s.qdrant_url, path=s.qdrant_path),
    )  # type: ignore[return-value]


def _dense(s: Settings) -> DenseEmbedder:
    e = s.file.embedding
    return _get("dense", lambda: DenseEmbedder(e.dense_model, e.device, e.batch_size))  # type: ignore[return-value]


def _sparse(s: Settings) -> SparseEmbedder:
    e = s.file.embedding
    return _get("sparse", lambda: SparseEmbedder(e.sparse_model, e.device, e.batch_size))  # type: ignore[return-value]


def build_sink(settings: Settings | None = None) -> QdrantPageSink:
    s = settings or get_settings()
    return QdrantPageSink(_store(s), _dense(s), _sparse(s), s.file.chunking)


def build_retriever(settings: Settings | None = None) -> Retriever:
    s = settings or get_settings()
    store, dense = _store(s), _dense(s)
    store.ensure_collection(dense.dim)
    r = s.file.retrieval
    return Retriever(store, dense, _sparse(s), build_verifier(s), r)  # type: ignore[arg-type]


def build_verifier(settings: Settings | None = None) -> Reranker:
    """The cross-encoder, shared by reranking and sentence-level grounding checks."""
    s = settings or get_settings()
    r = s.file.retrieval
    return _get("reranker", lambda: Reranker(r.reranker_model, s.file.embedding.device))  # type: ignore[return-value]

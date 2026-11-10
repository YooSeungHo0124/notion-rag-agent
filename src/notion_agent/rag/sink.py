"""PageSink implementation that indexes NotionPages into Qdrant."""

from __future__ import annotations

from notion_agent.config import ChunkingConfig
from notion_agent.ingest.models import NotionPage
from notion_agent.rag.chunker import chunk_page
from notion_agent.rag.embedder import DenseEncoder, SparseEncoder
from notion_agent.rag.store import QdrantStore


class QdrantPageSink:
    """Replace-on-write sink: delete a page's chunks, then chunk+embed+upsert."""

    def __init__(
        self,
        store: QdrantStore,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        chunking: ChunkingConfig,
    ) -> None:
        self.store, self.dense, self.sparse, self.chunking = store, dense, sparse, chunking
        self._ready = False

    def _ensure(self) -> None:
        if not self._ready:
            self.store.ensure_collection(self.dense.dim)
            self._ready = True

    def upsert_page(self, page: NotionPage) -> int:
        self._ensure()
        chunks = [] if page.archived else chunk_page(page, self.chunking)
        texts = [c.context_text for c in chunks]
        dense = self.dense.embed_documents(texts)
        sparse = self.sparse.embed_documents(texts)
        self.store.delete_by_page(page.page_id)  # after embedding, so a failure keeps old data
        return self.store.upsert_chunks(chunks, dense, sparse)

    def delete_page(self, page_id: str) -> None:
        self._ensure()
        self.store.delete_by_page(page_id)

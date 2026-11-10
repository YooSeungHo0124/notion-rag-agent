from datetime import UTC, datetime

import pytest
from rag_fakes import FakeDense, FakeReranker, FakeSparse, make_page

from notion_agent.config import ChunkingConfig, RetrievalConfig
from notion_agent.rag.retriever import Retriever
from notion_agent.rag.sink import QdrantPageSink
from notion_agent.rag.store import QdrantStore, ScoredChunk, SearchFilters, rrf_fuse

PAGES = [
    make_page(
        "pa",
        "Kubernetes notes",
        [("paragraph", "kubectl rollout restart deployment pods", None)],
        tags=["devops"],
        edited=datetime(2025, 3, 1, tzinfo=UTC),
    ),
    make_page(
        "pb",
        "Recipe",
        [("paragraph", "김치찌개 레시피 돼지고기 두부 김치", None)],
        tags=["food"],
        edited=datetime(2026, 5, 1, tzinfo=UTC),
    ),
    make_page(
        "pc",
        "Reading list",
        [("paragraph", "books about distributed systems consensus raft", None)],
        tags=["books", "devops"],
        edited=datetime(2026, 8, 1, tzinfo=UTC),
    ),
]


@pytest.fixture
def env(tmp_path):
    store = QdrantStore("t", path=tmp_path / "q")
    sink = QdrantPageSink(
        store, FakeDense(), FakeSparse(), ChunkingConfig(max_chars=300, overlap_chars=20)
    )
    for p in PAGES:
        sink.upsert_page(p)
    retr = Retriever(
        store,
        FakeDense(),
        FakeSparse(),
        FakeReranker(),
        RetrievalConfig(candidate_k=10, final_k=2, min_rerank_score=0.5),
    )
    yield store, sink, retr
    store.client.close()


def test_upsert_idempotent_and_delete(env):
    store, sink, _ = env
    n = store.count()
    assert sink.upsert_page(PAGES[0]) == store.count("pa")
    assert store.count() == n
    sink.delete_page("pa")
    assert store.count("pa") == 0 and store.count() == n - 1


def test_reupsert_changed_page_removes_stale_chunks(env):
    store, sink, _ = env
    changed = make_page("pa", "Kubernetes notes", [("paragraph", "completely new text", None)])
    sink.upsert_page(changed)
    assert store.count("pa") == 1
    assert store.client.scroll("t", limit=10)[0]


def test_hybrid_search_and_retriever(env):
    _, _, retr = env
    hits = retr.retrieve("kubectl rollout restart")
    assert hits[0].page_id == "pa" and hits[0].answerable
    assert hits[0].cite_url.endswith("#pab0")
    assert retr.retrieve("김치찌개 레시피")[0].page_title == "Recipe"


def test_filters(env):
    _, _, retr = env
    q = "books distributed kubectl"
    assert {h.page_id for h in retr.retrieve(q, k=5, filters=SearchFilters(tags=["books"]))} == {
        "pc"
    }
    assert {h.page_id for h in retr.retrieve(q, k=5, filters=SearchFilters(page_ids=["pa"]))} == {
        "pa"
    }
    after = SearchFilters(edited_after=datetime(2026, 1, 1, tzinfo=UTC))
    assert "pa" not in {h.page_id for h in retr.retrieve(q, k=5, filters=after)}


def test_unanswerable(env):
    _, _, retr = env
    hits = retr.retrieve("quantum chromodynamics lattice")
    assert hits and not any(h.answerable for h in hits)


def test_client_side_fallback_matches(env):
    store, _, retr = env
    server = [h.page_id for h in retr.retrieve("kubectl rollout restart", k=3)]
    store.server_fusion = False
    client = [h.page_id for h in retr.retrieve("kubectl rollout restart", k=3)]
    assert server[0] == client[0] == "pa"


def test_rrf_fuse():
    a = [ScoredChunk("x", 1), ScoredChunk("y", 0.5)]
    b = [ScoredChunk("y", 9), ScoredChunk("z", 1)]
    assert [h.id for h in rrf_fuse([a, b], 3)][0] == "y"

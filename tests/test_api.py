import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from rag_fakes import FakeDense, FakeReranker, FakeSparse, make_page

from notion_agent.api.app import Services, create_app
from notion_agent.config import ChunkingConfig, RetrievalConfig
from notion_agent.ingest.sync import SyncReport
from notion_agent.llm.base import LLMClient, LLMResponse
from notion_agent.qa.grounded import GroundedQA
from notion_agent.rag.retriever import Retriever
from notion_agent.rag.sink import QdrantPageSink
from notion_agent.rag.store import QdrantStore

PAGES = [
    make_page(
        "pa",
        "Kubernetes notes",
        [("heading_2", "Ops", 2), ("paragraph", "kubectl rollout restart deployment pods", None)],
        tags=["devops"],
    ),
    make_page(
        "pb",
        "Recipe",
        [("paragraph", "kimchi stew pork tofu kimchi", None)],
        tags=["food"],
        edited=datetime(2026, 5, 1, tzinfo=UTC),
    ),
]


class CitingLLM(LLMClient):
    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None, json_schema=None):
        self.prompts.append(prompt)
        return LLMResponse(
            text="",
            latency_ms=7,
            structured={
                "answerable": True,
                "sentences": [
                    {"text": "kubectl rollout restart deployment pods", "citations": ["S1"]},
                    {"text": "the moon is cheese", "citations": ["S1", "S99"]},
                ],
            },
        )


@pytest.fixture
def env():
    store = QdrantStore("t", client=QdrantClient(location=":memory:"))
    sink = QdrantPageSink(store, FakeDense(), FakeSparse(), ChunkingConfig(max_chars=300))
    for p in PAGES:
        sink.upsert_page(p)
    retr = Retriever(
        store,
        FakeDense(),
        FakeSparse(),
        FakeReranker(),
        RetrievalConfig(candidate_k=10, final_k=2, min_rerank_score=0.5),
    )
    llm = CitingLLM()
    calls: list[bool] = []

    def run_sync(full: bool) -> SyncReport:
        calls.append(full)
        return SyncReport(scanned=2, unchanged=2)

    svc = Services(
        store=store,
        qa=GroundedQA(retr, llm, FakeReranker()),
        run_sync=run_sync,
        last_sync=lambda: "2026-11-01T00:00:00+00:00",
    )
    app = create_app(lambda: svc, background_load=False, static_dir=None)
    with TestClient(app) as c:
        yield c, llm, calls


def test_health(env):
    c, _, _ = env
    h = c.get("/api/health").json()
    assert h["ready"] and h["pages"] == 2 and h["chunks"] >= 2
    assert h["last_sync"].startswith("2026-11-01")


def test_sources_grouped_by_page(env):
    c, _, _ = env
    src = c.get("/api/sources").json()["sources"]
    assert {s["page_id"] for s in src} == {"pa", "pb"}
    pa = next(s for s in src if s["page_id"] == "pa")
    assert pa["title"] == "Kubernetes notes" and pa["tags"] == ["devops"] and pa["chunks"] >= 1
    assert c.get("/api/sources?refresh=true").json()["sources"] == src


def test_ask_returns_grounded_answer(env):
    c, _, _ = env
    a = c.post("/api/ask", json={"question": "kubectl rollout restart"}).json()
    assert a["answerable"] and a["confidence"] in {"high", "medium", "low"}
    s1, s2 = a["sentences"]
    assert s1["support"] == "supported" and s1["citations"] == [1]
    assert s2["support"] == "unsupported" and a["dropped_citations"] == 1
    src = a["sources"][0]
    assert src["page_id"] == "pa" and src["cite_url"].startswith("https://www.notion.so/pa#")
    assert src["heading_path"] == ["Ops"] and "kubectl" in src["text"]
    assert a["latency_ms"] == 7 and 0 <= a["supported_ratio"] <= 1


def test_ask_filters_by_page(env):
    c, llm, _ = env
    a = c.post("/api/ask", json={"question": "kubectl rollout restart", "page_ids": ["pb"]}).json()
    assert not a["answerable"] and llm.prompts == []  # only recipe searched -> abstain


def test_ask_validates(env):
    c, _, _ = env
    assert c.post("/api/ask", json={"question": ""}).status_code == 422


def test_sync_runs_in_background(env):
    c, _, calls = env
    r = c.post("/api/sync", json={"full": True})
    assert r.status_code == 202
    for _ in range(50):
        st = c.get("/api/sync/status").json()
        if not st["running"]:
            break
        time.sleep(0.05)
    assert st["ok"] and "scanned=2" in st["summary"] and calls == [True]


def test_not_ready_returns_503():
    def boom() -> Services:
        raise RuntimeError("no gpu")

    app = create_app(boom, background_load=False, static_dir=None)
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert not h["ready"] and "no gpu" in h["error"]
        assert c.post("/api/ask", json={"question": "x"}).status_code == 503


def test_serves_spa(tmp_path):
    (tmp_path / "index.html").write_text("<html>ui</html>")
    app = create_app(
        lambda: (_ for _ in ()).throw(RuntimeError()), background_load=False, static_dir=tmp_path
    )
    with TestClient(app) as c:
        assert "ui" in c.get("/").text and "ui" in c.get("/some/route").text
        assert c.get("/api/nope").status_code == 404


def test_demo_services_load():
    from notion_agent.api.demo import load_demo_services

    app = create_app(load_demo_services, background_load=False, static_dir=None)
    with TestClient(app) as c:
        assert c.get("/api/health").json()["pages"] == 3

import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from rag_fakes import FakeDense, FakeReranker, FakeSparse, make_page

from notion_agent.api.app import Services, create_app
from notion_agent.config import ChunkingConfig, RetrievalConfig
from notion_agent.llm.base import LLMClient, LLMResponse
from notion_agent.qa.grounded import GroundedQA
from notion_agent.qa.studio import MINDMAP_SCHEMA, Studio, load_scope, pick_material
from notion_agent.rag.retriever import Hit, Retriever
from notion_agent.rag.sink import QdrantPageSink
from notion_agent.rag.store import QdrantStore

PAGES = [
    make_page(
        "pa",
        "Kubernetes notes",
        [("heading_2", "Ops", 2), ("paragraph", "kubectl rollout restart deployment pods", None)],
        edited=datetime(2026, 3, 2, tzinfo=UTC),
    ),
    make_page(
        "pb",
        "Recipe",
        [("paragraph", "kimchi stew pork tofu kimchi", None)],
        edited=datetime(2026, 5, 1, tzinfo=UTC),
    ),
]


class StudioLLM(LLMClient):
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt, *, system=None, json_schema=None):
        self.calls.append((prompt, system))
        if json_schema is MINDMAP_SCHEMA:
            return LLMResponse(
                "",
                structured={
                    "root": {
                        "label": "notes",
                        "citations": [],
                        "children": [
                            {"label": "k8s", "citations": ["S2", "S99"], "children": []},
                            {"label": "food", "citations": ["[S1]"], "children": []},
                            {"label": "  ", "citations": [], "children": []},
                        ],
                    }
                },
            )
        return LLMResponse(
            "",
            latency_ms=5,
            structured={
                "answerable": True,
                "followups": ["q1"],
                "blocks": [
                    {
                        "type": "summary",
                        "text": "kubectl rollout restart deployment pods",
                        "citations": ["S2"],
                    },
                    {
                        "type": "table",
                        "columns": ["날짜", "무슨 일"],
                        "rows": [["2026-03-02", "kubectl rollout restart"]],
                        "citations": ["S2"],
                    },
                    {"type": "text", "text": "the moon is cheese", "citations": ["S1"]},
                ],
            },
        )


def _hit(pid: str, idx: int, text: str, day: int) -> Hit:
    return Hit(
        chunk_id=f"{pid}{idx}",
        page_id=pid,
        page_title=pid,
        page_url="u",
        text=text,
        last_edited_time=datetime(2026, 1, day, tzinfo=UTC),
        chunk_index=idx,
        fusion_score=0,
        rerank_score=1,
    )


def test_pick_material_many_pages_recent_first(monkeypatch):
    monkeypatch.setattr("notion_agent.qa.studio.FAIR_MIN", 60)
    hits = [
        _hit("old", 0, "a" * 30, 1),
        _hit("new", 1, "b" * 30, 9),
        _hit("new", 0, "c" * 30, 9),
        _hit("big", 0, "d" * 500, 5),
    ]
    picked, cov = pick_material(hits, budget=100)  # 100/3 < FAIR_MIN -> recency
    assert [h.chunk_id for h in picked] == ["new0", "new1", "old0"]  # doc order, recent first
    assert cov.pages_total == 3 and cov.pages_used == 2 and cov.truncated


def test_pick_material_few_pages_share_budget_fairly(monkeypatch):
    monkeypatch.setattr("notion_agent.qa.studio.FAIR_MIN", 10)
    big = [_hit("big", i, "x" * 40, 9) for i in range(10)]  # 400 chars, most recent
    small = [_hit("small", i, "y" * 20, 1) for i in range(2)]  # 40 chars
    picked, cov = pick_material(big + small, budget=200)
    ids = [h.chunk_id for h in picked]
    assert ids[-2:] == ["small0", "small1"]  # small page fully included
    assert sum(1 for i in ids if i.startswith("big")) == 4  # 160 chars left for big
    assert cov.pages_used == 2 and cov.truncated


def test_pick_material_everything_fits():
    picked, cov = pick_material([_hit("p", i, "x" * 10, 1) for i in range(3)], budget=100)
    assert len(picked) == 3 and not cov.truncated


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
    llm = StudioLLM()
    qa = GroundedQA(retr, llm, FakeReranker())
    svc = Services(store=store, qa=qa, studio=Studio(qa))
    app = create_app(lambda: svc, background_load=False, static_dir=None)
    with TestClient(app) as c:
        yield c, llm, store


def _sse(r) -> list[tuple[str, dict]]:
    out = []
    for ev in r.text.strip().split("\n\n"):
        name, data = ev.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_load_scope_filters_pages(env):
    _, _, store = env
    assert {h.page_id for h in load_scope(store, ["pa"])} == {"pa"}
    assert {h.page_id for h in load_scope(store, None)} == {"pa", "pb"}


def test_studio_document_is_verified(env):
    c, llm, _ = env
    ev = _sse(c.post("/api/studio/stream", json={"kind": "timeline"}))
    names = [e for e, _ in ev]
    assert names[:2] == ["collecting", "collected"] and names[-1] == "result"
    assert "generating" in names and "verifying" in names
    res = ev[-1][1]
    a = res["answer"]
    summary, table, moon = a["sentences"]
    assert summary["support"] == "supported" and table["support"] == "supported"
    assert table["note"] == ""  # 2026-03-02 comes from the source's edit date
    assert moon["support"] == "unsupported"
    assert res["coverage"]["pages_used"] == 2 and a["followups"] == ["q1"]
    assert "Studio 규칙" in llm.calls[0][1] and "수정일 2026-05-01" in llm.calls[0][0]


def test_studio_mindmap_cleans_citations(env):
    c, _, _ = env
    res = _sse(c.post("/api/studio/stream", json={"kind": "mindmap", "page_ids": []}))[-1][1]
    kids = res["tree"]["children"]
    assert [k["label"] for k in kids] == ["k8s", "food"]
    assert kids[0]["citations"] == [1] and kids[1]["citations"] == [2]  # S99 dropped, renumbered
    assert len(res["sources"]) == 2


def test_studio_rejects_unknown_kind(env):
    c, _, _ = env
    assert c.post("/api/studio/stream", json={"kind": "poem"}).status_code == 422


def test_review_questions_are_not_judged(env):
    c, llm, _ = env
    orig = llm.complete

    def complete(prompt, *, system=None, json_schema=None):
        r = orig(prompt, system=system, json_schema=json_schema)
        r.structured["blocks"].append(
            {"type": "text", "text": "Q. 무엇을 재시작하나?", "citations": ["S2"]}
        )
        return r

    llm.complete = complete
    res = _sse(c.post("/api/studio/stream", json={"kind": "study"}))[-1][1]["answer"]
    q = res["sentences"][-1]
    assert q["text"].startswith("Q.") and q["support"] == "supported"
    assert res["supported_ratio"] == round(2 / 3, 4)  # summary+table ok, moon not; Q excluded


def test_quiz_is_cleaned_and_verified_per_question(env):
    c, llm, _ = env
    orig = llm.complete

    def complete(prompt, *, system=None, json_schema=None):
        from notion_agent.qa.studio import QUIZ_SCHEMA

        if json_schema is QUIZ_SCHEMA:
            return LLMResponse("", structured={"questions": [
                {"question": "파드를 재시작하는 명령은?", "answer": 1, "citations": ["S2"],
                 "options": ["kubectl get", "kubectl rollout restart deployment", "ls", "rm"],
                 "explanation": "kubectl rollout restart deployment pods"},
                {"question": "달은 무엇으로 되어 있나?", "answer": 0, "citations": ["S1"],
                 "options": ["cheese", "rock", "gas", "ice"], "explanation": "the moon is cheese"},
                {"question": "broken", "answer": 7, "citations": [], "options": ["a", "b", "c"],
                 "explanation": ""},
            ]})
        return orig(prompt, system=system, json_schema=json_schema)

    llm.complete = complete
    res = _sse(c.post("/api/studio/stream", json={"kind": "quiz"}))[-1][1]
    q = res["quiz"]
    assert len(q) == 2  # out-of-range answer dropped
    assert q[0]["support"] == "supported" and q[1]["support"] == "unsupported"
    assert q[0]["citations"] == [1] and len(res["sources"]) == 2

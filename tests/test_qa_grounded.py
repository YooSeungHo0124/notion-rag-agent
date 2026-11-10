from datetime import UTC, datetime

from notion_agent.config import RetrievalConfig
from notion_agent.ingest.notion_reader import properties_to_text
from notion_agent.llm.base import LLMClient, LLMResponse
from notion_agent.qa.grounded import GroundedQA
from notion_agent.rag.retriever import Hit

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def hit(i: int, text: str, score: float = 0.9, answerable: bool = True) -> Hit:
    return Hit(
        chunk_id=f"c{i}", page_id=f"p{i}", page_title=f"page{i}", page_url=f"https://n/{i}",
        block_ids=[f"b{i}"], text=text, last_edited_time=NOW,
        fusion_score=0.1, rerank_score=score, answerable=answerable,
    )


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.cfg = RetrievalConfig()

    def retrieve(self, q, k=None, filters=None):
        return self.hits


class StubLLM(LLMClient):
    def __init__(self, structured):
        self.structured = structured
        self.calls = 0

    def complete(self, prompt, *, system=None, json_schema=None):
        self.calls += 1
        return LLMResponse(text="", structured=self.structured, latency_ms=5)


class OverlapVerifier:
    def score(self, query, passages):
        q = set(query.split())
        return [len(q & set(p.split())) / max(len(q), 1) for p in passages]


def test_abstains_without_calling_llm_when_retrieval_weak():
    llm = StubLLM({})
    qa = GroundedQA(StubRetriever([hit(1, "x", 0.05, answerable=False)]), llm, OverlapVerifier())
    a = qa.answer("q")
    assert not a.answerable and llm.calls == 0


def test_invalid_citations_dropped_and_sources_renumbered():
    hits = [hit(1, "batch size reduced to 8"), hit(2, "used gradient checkpointing")]
    llm = StubLLM({
        "answerable": True,
        "sentences": [
            {"text": "used gradient checkpointing", "citations": ["S2", "S9"]},
            {"text": "batch size reduced to 8", "citations": ["[S1]"]},
        ],
    })
    a = GroundedQA(StubRetriever(hits), llm, OverlapVerifier()).answer("q")
    assert a.dropped_citations == 1
    assert [s.citations for s in a.sentences] == [[1], [2]]
    assert [h.chunk_id for h in a.sources] == ["c2", "c1"]
    assert all(s.support == "supported" for s in a.sentences)
    assert a.confidence == "high"


def test_unsupported_sentence_lowers_confidence():
    hits = [hit(1, "batch size reduced to 8")]
    llm = StubLLM({
        "answerable": True,
        "sentences": [
            {"text": "batch size reduced to 8", "citations": ["S1"]},
            {"text": "the moon is made of cheese", "citations": ["S1"]},
            {"text": "no citation here", "citations": []},
        ],
    })
    a = GroundedQA(StubRetriever(hits), llm, OverlapVerifier()).answer("q")
    assert [s.support for s in a.sentences] == ["supported", "unsupported", "unsupported"]
    assert a.confidence == "low"


def test_properties_to_text_skips_title_and_empty():
    page = {"properties": {
        "이름": {"type": "title", "title": [{"plain_text": "T"}]},
        "상태": {"type": "status", "status": {"name": "진행 중"}},
        "메모": {"type": "rich_text", "rich_text": [{"plain_text": "OOM 해결"}]},
        "빈칸": {"type": "rich_text", "rich_text": []},
        "완료": {"type": "checkbox", "checkbox": True},
        "기간": {"type": "date", "date": {"start": "2026-09-01", "end": "2026-09-03"}},
    }}
    assert properties_to_text(page).splitlines() == [
        "상태: 진행 중", "메모: OOM 해결", "완료: 예", "기간: 2026-09-01 ~ 2026-09-03",
    ]

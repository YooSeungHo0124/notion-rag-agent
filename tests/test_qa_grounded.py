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


def test_blocks_table_chart_code_verified_against_sources():
    hits = [hit(1, "batch 4 gives 1.8 fps and batch 8 gives 3.2 fps"),
            hit(2, "docker build -t app .\ndocker run app")]
    llm = StubLLM({"answerable": True, "blocks": [
        {"type": "text", "text": "batch 4 gives 1.8 fps", "citations": ["S1"]},
        {"type": "table", "columns": ["batch", "fps"], "rows": [["4", "1.8"], ["8", "3.2"]],
         "citations": ["S1"]},
        {"type": "chart", "chart_kind": "bar", "labels": ["4", "8"], "values": [1.8, 9.9],
         "unit": "fps", "citations": ["S1"]},
        {"type": "code", "lang": "bash", "code": "docker build -t app .\ndocker run app",
         "citations": ["S2"]},
        {"type": "table", "columns": [], "rows": [], "citations": ["S1"]},  # unusable -> dropped
    ]})
    a = GroundedQA(StubRetriever(hits), llm, OverlapVerifier()).answer("fps?")
    types = [s.type for s in a.sentences]
    assert types == ["text", "table", "chart", "code"]
    table, chart, code = a.sentences[1], a.sentences[2], a.sentences[3]
    assert table.support == "supported" and table.data["rows"][1] == ["8", "3.2"]
    assert chart.support == "unsupported" and "9.9" in chart.note  # invented number caught
    assert code.support == "supported" and code.data["lang"] == "bash"


def test_progress_events_emitted_in_order():
    events = []
    llm = StubLLM({"answerable": True, "blocks": [
        {"type": "text", "text": "batch size reduced to 8", "citations": ["S1"]}]})
    qa = GroundedQA(StubRetriever([hit(1, "batch size reduced to 8")]), llm, OverlapVerifier())
    qa.answer("q", on_event=lambda e, d: events.append(e))
    assert events == ["retrieving", "retrieved", "generating", "verifying"]


def test_missing_citation_for_table_numbers_is_attached_and_leadin_inherits():
    hits = [hit(1, "batch 4 gives 1.8 fps"), hit(2, "batch 8 gives 3.2 fps")]
    llm = StubLLM({"answerable": True, "blocks": [
        {"type": "text", "text": "비교는 다음과 같습니다.", "citations": []},
        {"type": "table", "columns": ["batch", "fps"], "rows": [["4", "1.8"], ["8", "3.2"]],
         "citations": ["S1"]},
    ]})
    a = GroundedQA(StubRetriever(hits), llm, OverlapVerifier()).answer("fps?")
    lead, table = a.sentences
    assert table.support == "supported" and sorted(table.citations) == [1, 2]
    assert lead.support == "supported" and len(a.sources) == 2


def test_rounded_numbers_accepted_but_invented_rejected():
    from notion_agent.qa.grounded import _num_in

    src = {"771.8", "1.38", "20"}
    assert all(_num_in(n, src) for n in ("772", "771", "1.4", "20"))
    assert not _num_in("770", src) and not _num_in("21", src) and not _num_in("1.5", src)


def test_summary_heading_callout_flow_and_followups():
    hits = [hit(1, "camera stream goes to decoder then yolo then event server")]
    llm = StubLLM({"answerable": True, "followups": ["a?", "b?", "c?", "d?"], "blocks": [
        {"type": "summary", "text": "camera stream goes to decoder then yolo",
         "citations": ["S1"]},
        {"type": "heading", "text": "구조", "citations": []},
        {"type": "flow", "nodes": ["camera", "decoder", "yolo", "event server"],
         "node_notes": ["stream"], "citations": ["S1"]},
        {"type": "callout", "level": "bogus", "text": "event server then yolo",
         "citations": ["S1"]},
        {"type": "flow", "nodes": ["only one"], "citations": ["S1"]},  # unusable -> dropped
    ]})
    a = GroundedQA(StubRetriever(hits), llm, OverlapVerifier()).answer("pipeline?")
    assert [s.type for s in a.sentences] == ["summary", "heading", "flow", "callout"]
    head, flow, call = a.sentences[1], a.sentences[2], a.sentences[3]
    assert head.support == "supported" and head.citations == []
    assert flow.data["notes"] == ["stream", "", "", ""]
    assert "camera(stream) → decoder" in flow.text
    assert call.data["level"] == "note"
    assert a.followups == ["a?", "b?", "c?"]
    # heading is structure, not a claim: excluded from the supported ratio
    assert a.supported_ratio == sum(s.support == "supported" for s in a.sentences if
                                    s.type != "heading") / 3

"""Studio: documents generated from a whole scope of notes (NotebookLM-style).

Unlike Q&A (top-k chunks for one question), Studio reads every chunk of the selected pages
(up to a character budget) and writes a briefing / study guide / FAQ / timeline / source
guide. The output uses the same typed blocks as answers and goes through the SAME
block-level verification, so every claim still carries citations and a support level.

The quiz is verified per question: "question + correct option + explanation" is checked
against the cited sources, so a question whose answer key is not backed by the notes is
flagged instead of quietly testing the user on something the notes never said.

The mind map is the exception: it is a structure of short labels, which a cross-encoder
cannot meaningfully verify, so its nodes carry citations only (shown as "unverified").
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from notion_agent.llm import LLMClient
from notion_agent.observability.tracing import span
from notion_agent.qa.grounded import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    Answer,
    GroundedQA,
    _confidence,
)
from notion_agent.rag.retriever import Hit
from notion_agent.rag.store import QdrantStore, SearchFilters, build_filter

Kind = Literal["briefing", "study", "faq", "timeline", "guide", "mindmap", "quiz"]
KINDS: tuple[str, ...] = ("briefing", "study", "faq", "timeline", "guide", "mindmap", "quiz")

MAX_CHARS = 60_000  # material budget per generation (~20k tokens)
TOPIC_K = 40  # chunks taken by retrieval when a topic narrows the scope

TITLES = {
    "briefing": "브리핑 문서",
    "study": "학습 가이드",
    "faq": "자주 묻는 질문",
    "timeline": "타임라인",
    "guide": "소스 가이드",
    "mindmap": "마인드맵",
    "quiz": "퀴즈",
}

INSTRUCTIONS = {
    "briefing": """브리핑 문서를 쓴다. 이 기록들을 처음 보는 사람이 5분 안에 파악하도록.
구성: summary(전체 결론 2문장) → heading '핵심 내용' 아래 주제별 text →
heading '주요 수치·결정' 아래 table 또는 text(소스에 있는 값만) →
heading '남은 문제' 아래 callout(warning)으로 미해결·한계(소스에 있을 때만).""",
    "study": """학습 가이드를 쓴다. 이 기록의 내용을 복습·이해하기 위한 자료.
구성: summary(무엇을 배우는 자료인지) → heading '핵심 개념' 아래 table(columns: 용어, 설명) →
heading '동작 원리' 아래 steps 또는 flow → heading '복습 질문' 아래 각 질문을 text 한 블록씩
('Q. ...' 형식, 답은 쓰지 않는다. 기록으로 답할 수 있는 질문만).""",
    "faq": """FAQ를 쓴다. 이 기록을 보고 사람들이 실제로 물어볼 질문 6~10개와 답.
구성: summary(이 FAQ가 다루는 범위) → 각 질문마다 heading(질문 문장) 다음에 답 text 블록 1~3개.
답은 소스에 있는 내용만, 명령어·값은 `코드`로.""",
    "timeline": """타임라인을 쓴다. 기록에 나타난 사건·변경·결정을 시간순으로.
구성: summary(기간과 큰 흐름) → table(columns: 날짜, 무슨 일, 비고) 오래된 것부터.
날짜는 본문에 적힌 날짜를 우선한다.
없으면 소스 머리의 수정일을 쓰고 비고에 '수정일 기준'이라고 쓴다.
같은 날짜 사건이 많으면 묶는다. 표에 없는 해석·평가는 덧붙이지 않는다.""",
    "guide": """소스 가이드를 쓴다. 이 페이지가 무엇에 관한 기록인지 한눈에 보여준다.
구성: summary(이 페이지가 무엇인지 2문장) → heading '핵심 주제' 아래 주제마다 text 한 블록
(**주제명** — 한 줄 설명). 짧게. followups에는 이 페이지로 답할 수 있는 좋은 질문 4~5개.""",
}

STUDIO_RULES = """
Studio 규칙(위 블록 규칙보다 우선):
- 아래 소스 전체가 대상이다. 특정 질문에 답하는 것이 아니라 요청한 문서를 만든다.
- 소스에 없는 내용, 일반 상식, 추측은 쓰지 않는다. 모든 블록에 citations.
- answerable은 소스에 의미 있는 내용이 있으면 true.
- followups: 이 문서를 본 뒤 내 노트에 물어볼 만한 질문 3개(소스 가이드는 4~5개)."""

MINDMAP_SYSTEM = """너는 사용자의 노션 기록으로 마인드맵을 만든다.
- 루트는 전체 주제 한 마디. 1단계 가지 3~7개, 각 가지 아래 2~5개, 최대 3단계.
- label은 짧은 명사구(15자 이내). 소스에 있는 개념만. 각 노드 citations에 근거 소스 id.
- 비슷한 개념은 합치고, 같은 수준의 가지는 같은 종류(기능/단계/구성요소 등)로 맞춘다."""


QUIZ_SYSTEM = """너는 사용자의 노션 기록으로 복습용 객관식 퀴즈를 만든다.
- 문제 5~8개. 각 문제는 보기 4개, 정답 1개(answer는 0부터 시작하는 보기 번호).
- 정답과 해설은 소스에 그대로 있는 사실만. 명령어·수치·이름처럼 기억할 가치가 있는 것을 묻는다.
- 오답 보기는 그럴듯하지만 소스 기준으로 확실히 틀린 것. '모두 맞다' 같은 보기 금지.
- explanation은 정답의 근거를 1~2문장으로. citations에 근거 소스 id(S1 형식).
- 정답 위치를 고르게 섞는다."""

QUIZ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "integer"},
            "explanation": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["question", "options", "answer", "explanation", "citations"],
        "additionalProperties": False,
    }}},
    "required": ["questions"],
    "additionalProperties": False,
}


def clean_quiz(raw: list[dict]) -> list[dict]:
    """Drop malformed questions: need >=3 distinct options and an in-range answer."""
    out = []
    for q in raw:
        opts = [o.strip() for o in q.get("options") or [] if str(o).strip()]
        a = q.get("answer")
        if (q.get("question", "").strip() and len(opts) >= 3 and len(set(opts)) == len(opts)
                and isinstance(a, int) and 0 <= a < len(opts)):
            out.append({"question": q["question"].strip(), "options": opts, "answer": a,
                        "explanation": q.get("explanation", "").strip(),
                        "citations": q.get("citations") or []})
    return out


def _node(depth: int) -> dict[str, Any]:
    props: dict[str, Any] = {
        "label": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    }
    if depth < 3:
        props["children"] = {"type": "array", "items": _node(depth + 1)}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


MINDMAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"root": _node(0)},
    "required": ["root"],
    "additionalProperties": False,
}


@dataclass
class Coverage:
    pages_total: int = 0
    pages_used: int = 0
    chunks_used: int = 0
    truncated: bool = False
    topic: str = ""


@dataclass
class StudioResult:
    kind: str
    title: str
    answer: Answer | None = None  # document kinds
    tree: dict[str, Any] | None = None  # mindmap
    sources: list[Hit] = field(default_factory=list)  # mindmap / quiz sources
    quiz: list[dict] | None = None
    coverage: Coverage = field(default_factory=Coverage)
    latency_ms: int = 0


def _date(h: Hit) -> str:
    return f"{h.last_edited_time:%Y-%m-%d}"


def src_with_date(h: Hit) -> str:
    """Verification text: the edit date is part of what the model saw for this source."""
    return f"(수정일 {_date(h)}) {h.text}"


def load_scope(store: QdrantStore, page_ids: list[str] | None) -> list[Hit]:
    """Every indexed chunk of the given pages (all pages when None)."""
    client = store.client
    if not client.collection_exists(store.collection):
        return []
    flt = build_filter(SearchFilters(page_ids=page_ids)) if page_ids else None
    out: list[Hit] = []
    offset = None
    while True:
        pts, offset = client.scroll(
            store.collection,
            scroll_filter=flt,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in pts:
            if p.payload and p.payload.get("text"):
                out.append(Hit(**p.payload, fusion_score=0.0, rerank_score=1.0))
        if offset is None:
            break
    return out


FAIR_MIN = 1500  # chars per page below which fair sharing is pointless


def pick_material(hits: list[Hit], budget: int = MAX_CHARS) -> tuple[list[Hit], Coverage]:
    """Choose chunks within the character budget; chunks keep document order per page.

    - few pages (each could get >= FAIR_MIN): the budget is shared fairly (water-filling),
      so a big page cannot crowd out the others the user picked;
    - many pages: whole pages, most recently edited first, until the budget is spent.
    """
    pages: dict[str, list[Hit]] = {}
    for h in hits:
        pages.setdefault(h.page_id, []).append(h)
    for cs in pages.values():
        cs.sort(key=lambda c: c.chunk_index)
    size = {p: sum(len(c.text) for c in cs) for p, cs in pages.items()}
    recent = sorted(pages, key=lambda p: max(c.last_edited_time for c in pages[p]), reverse=True)
    total = sum(size.values())
    if total <= budget:
        picked = [c for p in recent for c in pages[p]]
        return picked, Coverage(len(pages), len(pages), len(picked), False)

    allot: dict[str, int] = {}
    if pages and budget // len(pages) >= FAIR_MIN:
        left, rest = budget, sorted(pages, key=size.__getitem__)
        while rest:
            share = left // len(rest)
            p = rest.pop(0)
            allot[p] = min(size[p], share)
            left -= allot[p]
    else:
        used = 0
        for p in recent:
            if used + size[p] <= budget:
                allot[p] = size[p]
                used += size[p]
        if not allot:  # the most recent page alone is too big: take its beginning
            allot[recent[0]] = budget

    picked: list[Hit] = []
    for p in recent:
        if p not in allot:
            continue
        used = 0
        for c in pages[p]:
            if used + len(c.text) > allot[p] and used > 0:
                break
            picked.append(c)
            used += len(c.text)
    n_used = len({c.page_id for c in picked})
    return picked, Coverage(len(pages), n_used, len(picked), True)


def format_material(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        where = " > ".join([h.page_title, *h.heading_path])
        parts.append(f"[S{i}] ({where} | 수정일 {_date(h)})\n{h.text}")
    return "\n\n".join(parts)


def _clean_tree(node: dict[str, Any], n_src: int, used: list[int], depth: int = 0) -> dict:
    cites = []
    for c in node.get("citations") or []:
        k = str(c).strip().strip("[]").upper()
        if k.startswith("S") and k[1:].isdigit() and 1 <= int(k[1:]) <= n_src:
            i = int(k[1:])
            if i not in used:
                used.append(i)
            cites.append(i)
    kids = (node.get("children") or []) if depth < 3 else []
    return {
        "label": str(node.get("label", "")).strip()[:40],
        "citations": list(dict.fromkeys(cites)),
        "children": [
            _clean_tree(k, n_src, used, depth + 1)
            for k in kids[:8]
            if str(k.get("label", "")).strip()
        ],
    }


class Studio:
    def __init__(
        self,
        qa: GroundedQA,
        llm: LLMClient | None = None,
        loader: Callable[[list[str] | None], list[Hit]] | None = None,
    ) -> None:
        self.qa = qa
        self.llm = llm or qa.llm
        self._load = loader or (lambda ids: load_scope(qa.retriever.store, ids))

    def material(self, page_ids: list[str] | None, topic: str = "") -> tuple[list[Hit], Coverage]:
        """Needs the store/GPU lock (scroll + optional rerank)."""
        if topic.strip():
            flt = SearchFilters(page_ids=page_ids) if page_ids else None
            hits = self.qa.retriever.retrieve(topic.strip(), k=TOPIC_K, filters=flt)
            hits = [h for h in hits if h.rerank_score >= 0.05]
            hits, cov = pick_material(hits)
            cov.topic = topic.strip()
            return hits, cov
        return pick_material(self._load(page_ids or None))

    def generate(
        self,
        kind: str,
        hits: list[Hit],
        cov: Coverage,
        emit: Callable[[str, dict], None] | None = None,
    ) -> StudioResult:
        """LLM part (no lock needed). Verification runs the reranker: call `verify` under it."""
        emit = emit or (lambda *_: None)
        title = TITLES[kind]
        if not hits:
            return StudioResult(kind, title, answer=Answer(title, False), coverage=cov)
        t0 = time.perf_counter()
        prompt = f"<sources>\n{format_material(hits)}\n</sources>\n\n"
        emit("generating", {"chunks": len(hits), "pages": cov.pages_used})
        with span(f"studio:{kind}", "CHAIN", input=kind, chunks=len(hits)):
            if kind == "mindmap":
                resp = self.llm.complete(
                    prompt + "이 기록 전체의 마인드맵을 만든다.",
                    system=MINDMAP_SYSTEM,
                    json_schema=MINDMAP_SCHEMA,
                )
                used: list[int] = []
                root = (resp.structured or {}).get("root") or {}
                tree = _clean_tree(root, len(hits), used)
                remap = {o: n for n, o in enumerate(used, 1)}
                _renumber(tree, remap)
                return StudioResult(
                    kind,
                    title,
                    tree=tree,
                    sources=[hits[i - 1] for i in used],
                    coverage=cov,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
            if kind == "quiz":
                resp = self.llm.complete(prompt + "이 기록으로 복습 퀴즈를 만든다.",
                                         system=QUIZ_SYSTEM, json_schema=QUIZ_SCHEMA)
                r = StudioResult(kind, title, coverage=cov,
                                 quiz=clean_quiz((resp.structured or {}).get("questions") or []))
                r.latency_ms = int((time.perf_counter() - t0) * 1000)
                return r
            resp = self.llm.complete(
                prompt + f"요청: {title}을(를) 만든다.\n{INSTRUCTIONS[kind]}",
                system=SYSTEM_PROMPT + STUDIO_RULES,
                json_schema=ANSWER_SCHEMA,
            )
        data = resp.structured or {}
        r = StudioResult(kind, title, coverage=cov)
        r.answer = Answer(title, False, latency_ms=resp.latency_ms)
        r._raw = data  # type: ignore[attr-defined]
        r.latency_ms = int((time.perf_counter() - t0) * 1000)
        return r

    def verify(
        self, r: StudioResult, hits: list[Hit], emit: Callable[[str, dict], None] | None = None
    ) -> StudioResult:
        if r.kind == "quiz":
            return _verify_quiz_impl(self.qa, r, hits, emit)
        data = getattr(r, "_raw", None)
        if r.kind == "mindmap" or not data:
            return r
        raw = data.get("blocks") or []
        if not data.get("answerable") or not raw:
            return r
        (emit or (lambda *_: None))("verifying", {"blocks": len(raw)})
        a = self.qa._verify(
            r.title, raw, hits, 1.0, r.answer.latency_ms if r.answer else 0, src_text=src_with_date
        )
        a.followups = [f.strip() for f in data.get("followups") or [] if f.strip()][:5]
        # review questions ("Q. ...") ask, they claim nothing: not judged, not in the ratio
        claims = []
        for x in a.sentences:
            if x.type == "text" and x.text.lstrip().startswith("Q."):
                x.support, x.support_score, x.note = "supported", 1.0, ""
            elif x.type != "heading":
                claims.append(x)
        if claims:
            a.supported_ratio = sum(x.support == "supported" for x in claims) / len(claims)
            a.confidence = _confidence(1.0, a.supported_ratio)
        r.answer = a
        return r


def _verify_quiz_impl(qa: GroundedQA, r: StudioResult, hits: list[Hit],
                     emit: Callable[[str, dict], None] | None) -> StudioResult:
    if not r.quiz:
        return r
    (emit or (lambda *_: None))("verifying", {"blocks": len(r.quiz)})
    claims = [{"type": "text", "citations": q["citations"],
               "text": f"{q['question']} 정답: {q['options'][q['answer']]}. {q['explanation']}"}
              for q in r.quiz]
    a = qa._verify(r.title, claims, hits, 1.0, 0, src_text=src_with_date)
    for q, s in zip(r.quiz, a.sentences, strict=True):
        q["citations"], q["support"] = s.citations, s.support
        q["support_score"] = round(s.support_score, 4)
    r.sources = a.sources
    return r


def _renumber(node: dict[str, Any], remap: dict[int, int]) -> None:
    node["citations"] = [remap[c] for c in node["citations"] if c in remap]
    for k in node["children"]:
        _renumber(k, remap)


def result_to_json(r: StudioResult) -> dict[str, Any]:
    from notion_agent.qa.serialize import answer_to_json, hit_to_json

    out: dict[str, Any] = {
        "kind": r.kind,
        "title": r.title,
        "coverage": r.coverage.__dict__,
        "latency_ms": r.latency_ms,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if r.kind == "mindmap":
        out["tree"] = r.tree
        out["sources"] = [hit_to_json(h) for h in r.sources]
    elif r.kind == "quiz":
        out["quiz"] = r.quiz or []
        out["sources"] = [hit_to_json(h) for h in r.sources]
    else:
        out["answer"] = answer_to_json(r.answer) if r.answer else None
    return out

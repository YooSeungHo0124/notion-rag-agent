"""Grounded question answering with sentence-level citations and verification.

Flow (docs/08 section 3):
  1. retrieve -> if nothing relevant (rerank below threshold) abstain without calling the LLM
  2. LLM answers with structured output: sentences, each citing source ids [S1..Sn].
     The LLM never writes URLs; links are attached by our code, so they cannot be fabricated.
  3. verify: drop citations to ids that were not provided, then score every sentence
     against the text it cites with the cross-encoder -> supported / partial / unsupported
  4. confidence = f(retrieval score, supported ratio)

"Confidence" means *grounded in my notes*, not *true*.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from notion_agent.llm.base import LLMClient
from notion_agent.observability.tracing import span
from notion_agent.rag.reranker import RerankerProtocol
from notion_agent.rag.retriever import Hit, Retriever
from notion_agent.rag.store import SearchFilters

Support = Literal["supported", "partial", "unsupported"]
Confidence = Literal["high", "medium", "low", "none"]

SUPPORTED_MIN = 0.5
PARTIAL_MIN = 0.2

SYSTEM_PROMPT = """너는 사용자의 개인 노션 기록만을 근거로 답하는 어시스턴트다.
규칙:
- 제공된 소스([S1], [S2] ...)에 있는 내용만 사용한다. 일반 상식으로 빈칸을 채우지 않는다.
- 답변을 문장 단위로 나누고, 각 문장에 근거가 되는 소스 id를 citations에 넣는다.
- 소스에 답이 없으면 answerable=false로 하고 sentences는 비워 둔다.
- 기록끼리 내용이 다르면 날짜와 함께 둘 다 언급한다.
- 질문이 특정 조건(설정값·날짜·대상)을 물으면 그 조건에 해당하는 값을 먼저 정확히 답하고, 다른 조건의 값은 섞지 않는다.
- 질문이 묻는 항목(수치·원인·방법)을 소스에 있는 범위에서 빠짐없이 답한다. 질문과 무관한 부가 설명은 덧붙이지 않는다.
- 질문과 같은 언어로 간결하게 답한다. URL은 쓰지 않는다.

출력은 blocks 배열이다. 좋은 답변의 구성:
1) 첫 블록은 summary: 질문에 대한 결론 1~2문장(가장 중요한 값/답을 바로).
2) 그 다음 상세. 답이 길고 주제가 둘 이상이면 heading으로 구획을 나눈다(짧은 답에는 heading 쓰지 않는다).
3) 기록끼리 값이 다르거나, 기록이 오래됐거나, 질문 일부는 기록에 없으면 callout으로 알린다.
4) followups: 이 답을 본 사용자가 이어서 물어볼 만한 질문 2~3개(내 노트로 답할 수 있을 법한 것).

블록 종류:
- summary : 결론 1~2문장. text 필드.
- heading : 짧은 소제목. text 필드. citations는 빈 배열.
- text    : 한 문장. text 필드. 문장마다 별도 블록. 핵심어는 **굵게**, 명령어·파일명·변수는 `코드`로 감싼다.
- callout : level("note"|"warning"|"conflict") + text. 주의·불일치·한계 안내.
- flow    : 구성요소 간 흐름/구조. nodes(순서대로 3~7개 짧은 이름), 각 노드 설명은 node_notes(같은 길이, 짧게).
- table : 여러 항목을 같은 기준으로 비교할 때. columns(헤더), rows(문자열 2차원 배열).
- steps : 순서가 있는 절차. items(단계 문자열 배열).
- code  : 명령어·코드·설정 파일. lang, code. 소스에 있는 그대로 옮긴다.
- chart : 소스에 있는 수치 2개 이상을 비교할 때만. chart_kind("bar"), labels, values(숫자), unit.
          수치는 소스에 적힌 값만 쓴다. 계산·추정한 값 금지.
모든 블록에 citations를 단다. 표·단계·코드·차트 앞에는 짧은 text 블록으로 무엇인지 설명한다."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "followups": {"type": "array", "items": {"type": "string"}},
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": [
                        "summary", "heading", "text", "callout", "table", "steps", "code",
                        "chart", "flow"]},
                    "level": {"type": "string", "enum": ["note", "warning", "conflict"]},
                    "nodes": {"type": "array", "items": {"type": "string"}},
                    "node_notes": {"type": "array", "items": {"type": "string"}},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "text": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                    "items": {"type": "array", "items": {"type": "string"}},
                    "lang": {"type": "string"},
                    "code": {"type": "string"},
                    "chart_kind": {"type": "string", "enum": ["bar"]},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "values": {"type": "array", "items": {"type": "number"}},
                    "unit": {"type": "string"},
                },
                "required": ["type", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answerable", "blocks", "followups"],
    "additionalProperties": False,
}


BLOCK_TYPES = ("summary", "heading", "text", "callout", "table", "steps", "code", "chart", "flow")
UNVERIFIED_TYPES = ("heading",)  # structure only, makes no claim


@dataclass
class Sentence:
    """One answer block. `text` is its plain-text form (used for verification and eval);
    `type`/`data` drive the UI component (Generative UI)."""

    text: str
    citations: list[int]  # 1-based indexes into Answer.sources
    support: Support
    support_score: float
    type: str = "text"
    data: dict = field(default_factory=dict)
    note: str = ""  # e.g. numbers not found in the cited sources


_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _num_key(x: str) -> str:
    return x.replace(",", "").rstrip("0").rstrip(".") if "." in x else x.replace(",", "")


def _num_in(n: str, src: set[str]) -> bool:
    """Exact match, or `n` is a rounding of a source number (771 ~ 771.8, 1.4 ~ 1.38)."""
    if n in src:
        return True
    try:
        v = float(n)
    except ValueError:
        return False
    dec = len(n.split(".")[1]) if "." in n else 0
    tol = 10**-dec + 1e-9  # rounding or truncation to the shown precision
    for m in src:
        try:
            mv = float(m)
        except ValueError:
            continue
        if abs(mv - v) <= tol and (len(m.split(".")[1]) if "." in m else 0) > dec:
            return True
    return False


def block_payload(b: dict) -> tuple[str, dict] | None:
    """Validate a raw LLM block -> (plain text, component data); None if unusable."""
    t = b.get("type", "text")
    if t in ("text", "summary", "heading"):
        txt = (b.get("text") or "").strip()
        return (txt, {}) if txt else None
    if t == "callout":
        txt = (b.get("text") or "").strip()
        lvl = b.get("level") if b.get("level") in ("note", "warning", "conflict") else "note"
        return (txt, {"level": lvl}) if txt else None
    if t == "flow":
        nodes = [n.strip() for n in b.get("nodes") or [] if n.strip()][:8]
        if len(nodes) < 2:
            return None
        notes = [(x or "").strip() for x in (b.get("node_notes") or [])][: len(nodes)]
        notes += [""] * (len(nodes) - len(notes))
        txt = " → ".join(f"{n}({m})" if m else n for n, m in zip(nodes, notes, strict=True))
        return txt, {"nodes": nodes, "notes": notes}
    if t == "table":
        cols, rows = b.get("columns") or [], [r for r in b.get("rows") or [] if r]
        if not cols or not rows:
            return None
        rows = [(r + [""] * len(cols))[: len(cols)] for r in rows]
        txt = "\n".join(" | ".join(r) for r in [cols, *rows])
        return txt, {"columns": cols, "rows": rows}
    if t == "steps":
        items = [i.strip() for i in b.get("items") or [] if i.strip()]
        return ("\n".join(items), {"items": items}) if items else None
    if t == "code":
        code = (b.get("code") or "").rstrip()
        return (code, {"lang": b.get("lang") or "", "code": code}) if code else None
    if t == "chart":
        labels, values = b.get("labels") or [], b.get("values") or []
        n = min(len(labels), len(values))
        if n < 2:
            return None
        labels, values = labels[:n], values[:n]
        unit = b.get("unit") or ""
        txt = ", ".join(f"{lab} {v:g}{unit}" for lab, v in zip(labels, values, strict=True))
        return txt, {"kind": "bar", "labels": labels, "values": values, "unit": unit}
    return None


@dataclass
class Answer:
    question: str
    answerable: bool
    sentences: list[Sentence] = field(default_factory=list)
    sources: list[Hit] = field(default_factory=list)  # only sources actually cited
    confidence: Confidence = "none"
    retrieval_score: float = 0.0
    supported_ratio: float = 0.0
    dropped_citations: int = 0  # citations to ids we never provided
    latency_ms: int = 0
    followups: list[str] = field(default_factory=list)
    closest: Hit | None = None  # best (weak) hit when abstaining, so the UI can offer it


def _format_sources(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        where = " > ".join([h.page_title, *h.heading_path])
        parts.append(f"[S{i}] ({where} | {h.last_edited_time:%Y-%m-%d})\n{h.text}")
    return "\n\n".join(parts)


def _classify(score: float) -> Support:
    if score >= SUPPORTED_MIN:
        return "supported"
    if score >= PARTIAL_MIN:
        return "partial"
    return "unsupported"


def _confidence(retrieval_score: float, supported_ratio: float) -> Confidence:
    if retrieval_score >= SUPPORTED_MIN and supported_ratio >= 0.8:
        return "high"
    if supported_ratio >= 0.5:
        return "medium"
    return "low"


class GroundedQA:
    def __init__(self, retriever: Retriever, llm: LLMClient, verifier: RerankerProtocol) -> None:
        self.retriever = retriever
        self.llm = llm
        self.verifier = verifier

    def answer(
        self, question: str, filters: SearchFilters | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        extra: list[Hit] | None = None,
    ) -> Answer:
        emit = on_event or (lambda *_: None)
        emit("retrieving", {})
        hits = self.retriever.retrieve(question, filters=filters)
        if extra:
            hits = self.merge_extra(question, hits, extra)
        emit("retrieved", {"n": len(hits), "top": round(hits[0].rerank_score, 3) if hits else 0,
                           "titles": list(dict.fromkeys(h.page_title for h in hits))[:5]})
        return self.answer_from_hits(question, hits, on_event=emit)

    def merge_extra(self, question: str, hits: list[Hit], extra: list[Hit]) -> list[Hit]:
        """User memos as extra sources: scored by the same reranker, kept only if they pass
        the answerability threshold, then merged by score (they never displace more than
        two Notion hits)."""
        cfg = self.retriever.cfg
        scores = self.verifier.score(question, [e.text for e in extra])
        keep = sorted(
            (e.model_copy(update={"rerank_score": float(sc)})
             for e, sc in zip(extra, scores, strict=True) if sc >= cfg.min_rerank_score),
            key=lambda e: e.rerank_score, reverse=True)[:2]
        if not keep:
            return hits
        merged = sorted([*hits, *keep], key=lambda h: h.rerank_score, reverse=True)
        merged = merged[: max(len(hits), cfg.final_k) + len(keep)]
        ok = merged[0].rerank_score >= cfg.min_rerank_score
        return [h.model_copy(update={"answerable": ok}) for h in merged]

    def answer_from_hits(
        self, question: str, hits: list[Hit], on_event: Callable[[str, dict], None] | None = None
    ) -> Answer:
        """Grounded answer over already-retrieved hits (used by the agent's writer node)."""
        with span("grounded_answer", "CHAIN", input=question, n_hits=len(hits)) as sp:
            a = self._answer_from_hits(question, hits, on_event or (lambda *_: None))
            sp.output(" ".join(s.text for s in a.sentences) or "(abstain)")
            sp.set(answerable=a.answerable, confidence=a.confidence,
                   supported_ratio=round(a.supported_ratio, 3),
                   retrieval_score=round(a.retrieval_score, 3),
                   dropped_citations=a.dropped_citations)
            return a

    def _answer_from_hits(
        self, question: str, hits: list[Hit], emit: Callable[[str, dict], None]
    ) -> Answer:
        top = hits[0].rerank_score if hits else 0.0
        if not hits or not hits[0].answerable:
            return Answer(question=question, answerable=False, retrieval_score=top,
                          closest=hits[0] if hits else None)

        emit("generating", {})
        prompt = f"<sources>\n{_format_sources(hits)}\n</sources>\n\n질문: {question}"
        resp = self.llm.complete(prompt, system=SYSTEM_PROMPT, json_schema=ANSWER_SCHEMA)
        data = resp.structured or {}
        raw = data.get("blocks") or data.get("sentences") or []
        if not data.get("answerable") or not raw:
            return Answer(question, False, retrieval_score=top, latency_ms=resp.latency_ms,
                          closest=hits[0])
        emit("verifying", {"blocks": len(raw)})
        a = self._verify(question, raw, hits, top, resp.latency_ms)
        a.followups = [f.strip() for f in data.get("followups") or [] if f.strip()][:3]
        return a

    def _verify(
        self, question: str, raw: list[dict], hits: list[Hit], top: float, latency_ms: int,
        src_text: Callable[[Hit], str] = lambda h: h.text,
    ) -> Answer:
        valid_ids = {f"S{i}": i for i in range(1, len(hits) + 1)}
        dropped = 0
        cited_order: list[int] = []  # original hit index -> display order
        sentences: list[Sentence] = []

        for item in raw:
            ids = []
            for c in item.get("citations", []):
                key = c.strip().strip("[]").upper()
                if key in valid_ids:
                    ids.append(valid_ids[key])
                else:
                    dropped += 1
            ids = list(dict.fromkeys(ids))
            parsed = block_payload(item)
            if parsed is None:
                continue
            text, bdata = parsed
            btype = item.get("type", "text")
            score = 0.0
            if btype in UNVERIFIED_TYPES:
                sentences.append(Sentence(text, [], "supported", 1.0, btype, bdata))
                continue
            if ids:
                score = max(self.verifier.score(text[:2000], [src_text(hits[i - 1]) for i in ids]))
            note = ""
            if btype in ("table", "chart", "code") and ids:
                # numbers / code must literally come from the cited sources
                src = " ".join(src_text(hits[i - 1]) for i in ids)
                if btype == "code":
                    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                    ok = sum(ln in src for ln in lines) / max(len(lines), 1)
                    score = max(score, ok) if ok >= 0.8 else min(score, ok)
                else:
                    srcnums = {_num_key(n) for n in _NUM.findall(src)}
                    nums = [_num_key(n) for n in _NUM.findall(text)]
                    missing = [n for n in nums if not _num_in(n, srcnums)]
                    if missing:  # model forgot a citation? attach retrieved hits holding them
                        for j, h in enumerate(hits, 1):
                            if j in ids:
                                continue
                            hn = {_num_key(n) for n in _NUM.findall(src_text(h))}
                            if any(_num_in(n, hn) for n in missing):
                                ids.append(j)
                                srcnums |= hn
                        missing = [n for n in nums if not _num_in(n, srcnums)]
                    if missing:
                        note = f"출처에 없는 수치: {', '.join(missing[:5])}"
                        score = min(score, PARTIAL_MIN - 0.01)
                    elif nums:
                        score = max(score, SUPPORTED_MIN)
            for i in ids:
                if i not in cited_order:
                    cited_order.append(i)
            sentences.append(Sentence(text, ids, _classify(score), score, btype, bdata, note))

        # a short lead-in sentence ("설정은 다음과 같습니다.") right before a table/steps/code/
        # chart is judged together with that block instead of on its own
        for prev, nxt in zip(sentences, sentences[1:], strict=False):
            if (prev.type == "text" and nxt.type != "text" and len(prev.text) <= 60
                    and nxt.type not in UNVERIFIED_TYPES and prev.support_score < nxt.support_score):
                prev.support_score, prev.support = nxt.support_score, nxt.support
                prev.citations = prev.citations or list(nxt.citations)
                for i in prev.citations:
                    if i not in cited_order:
                        cited_order.append(i)

        # renumber citations to the order sources first appear in the answer
        remap = {orig: n for n, orig in enumerate(cited_order, 1)}
        for s in sentences:
            s.citations = [remap[i] for i in s.citations]

        claims = [s for s in sentences if s.type not in UNVERIFIED_TYPES]
        supported = sum(s.support == "supported" for s in claims)
        ratio = supported / len(claims) if claims else 0.0
        return Answer(
            question=question,
            answerable=True,
            sentences=sentences,
            sources=[hits[i - 1] for i in cited_order],
            confidence=_confidence(top, ratio),
            retrieval_score=top,
            supported_ratio=ratio,
            dropped_citations=dropped,
            latency_ms=latency_ms,
        )

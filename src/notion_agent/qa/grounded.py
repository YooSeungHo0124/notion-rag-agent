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

from dataclasses import dataclass, field
from typing import Literal

from notion_agent.llm.base import LLMClient
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
- 질문과 같은 언어로 간결하게 답한다. URL은 쓰지 않는다."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answerable", "sentences"],
    "additionalProperties": False,
}


@dataclass
class Sentence:
    text: str
    citations: list[int]  # 1-based indexes into Answer.sources
    support: Support
    support_score: float


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

    def answer(self, question: str, filters: SearchFilters | None = None) -> Answer:
        return self.answer_from_hits(question, self.retriever.retrieve(question, filters=filters))

    def answer_from_hits(self, question: str, hits: list[Hit]) -> Answer:
        """Grounded answer over already-retrieved hits (used by the agent's writer node)."""
        top = hits[0].rerank_score if hits else 0.0
        if not hits or not hits[0].answerable:
            return Answer(question=question, answerable=False, retrieval_score=top)

        prompt = f"<sources>\n{_format_sources(hits)}\n</sources>\n\n질문: {question}"
        resp = self.llm.complete(prompt, system=SYSTEM_PROMPT, json_schema=ANSWER_SCHEMA)
        data = resp.structured or {}
        if not data.get("answerable") or not data.get("sentences"):
            return Answer(question, False, retrieval_score=top, latency_ms=resp.latency_ms)

        return self._verify(question, data["sentences"], hits, top, resp.latency_ms)

    def _verify(
        self, question: str, raw: list[dict], hits: list[Hit], top: float, latency_ms: int
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
            text = item.get("text", "").strip()
            if not text:
                continue
            score = 0.0
            if ids:
                score = max(self.verifier.score(text, [hits[i - 1].text for i in ids]))
            for i in ids:
                if i not in cited_order:
                    cited_order.append(i)
            sentences.append(Sentence(text, ids, _classify(score), score))

        # renumber citations to the order sources first appear in the answer
        remap = {orig: n for n, orig in enumerate(cited_order, 1)}
        for s in sentences:
            s.citations = [remap[i] for i in s.citations]

        supported = sum(s.support == "supported" for s in sentences)
        ratio = supported / len(sentences) if sentences else 0.0
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

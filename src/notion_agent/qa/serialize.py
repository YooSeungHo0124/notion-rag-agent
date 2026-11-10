"""JSON serialisation of answers (shared by API and agent state)."""

from __future__ import annotations

from typing import Any

from notion_agent.qa.grounded import Answer
from notion_agent.rag.retriever import Hit


def hit_to_json(h: Hit) -> dict[str, Any]:
    return {
        "chunk_id": h.chunk_id,
        "page_id": h.page_id,
        "title": h.page_title,
        "path": h.path,
        "heading_path": h.heading_path,
        "cite_url": h.cite_url,
        "page_url": h.page_url,
        "text": h.text,
        "tags": h.tags,
        "last_edited_time": h.last_edited_time.isoformat(),
        "rerank_score": round(h.rerank_score, 4),
    }


def answer_to_json(a: Answer) -> dict[str, Any]:
    return {
        "question": a.question,
        "answerable": a.answerable,
        "sentences": [
            {
                "text": s.text,
                "citations": s.citations,
                "support": s.support,
                "support_score": round(s.support_score, 4),
                "type": s.type,
                "data": s.data,
                "note": s.note,
            }
            for s in a.sentences
        ],
        "sources": [hit_to_json(h) for h in a.sources],
        "confidence": a.confidence,
        "retrieval_score": round(a.retrieval_score, 4),
        "supported_ratio": round(a.supported_ratio, 4),
        "dropped_citations": a.dropped_citations,
        "latency_ms": a.latency_ms,
        "followups": a.followups,
    }

"""Built-in tools: note search / page read / timeline (read) and Notion writes (approval).

Writes are additive only: append blocks or create a child page. No delete/archive tool is
exposed to the agent at all.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from notion_agent.rag.retriever import Hit, Retriever
from notion_agent.rag.store import SearchFilters
from notion_agent.tools.registry import Tool, ToolRegistry


class SearchArgs(BaseModel):
    query: str = Field(description="검색할 질문/키워드 (재작성된 검색어 권장)")
    k: int = Field(5, ge=1, le=10)
    tags: list[str] = Field(default_factory=list, description="태그 필터(선택)")


class PageArgs(BaseModel):
    page_id: str


class TimelineArgs(BaseModel):
    since: str | None = Field(None, description="ISO 날짜, 이 날짜 이후 수정된 페이지")
    limit: int = Field(15, ge=1, le=50)


class AppendArgs(BaseModel):
    page_id: str = Field(description="내용을 덧붙일 노션 페이지 id")
    text: str = Field(description="추가할 본문(줄바꿈으로 문단 구분)")


class CreatePageArgs(BaseModel):
    parent_page_id: str
    title: str
    text: str = ""


def _paragraphs(text: str) -> list[dict[str, Any]]:
    return [
        {"object": "block", "type": "paragraph",
         "paragraph": {"rich_text": [{"type": "text", "text": {"content": line[:2000]}}]}}
        for line in text.split("\n") if line.strip()
    ][:100]


def hit_brief(h: Hit) -> dict[str, Any]:
    return {"chunk_id": h.chunk_id, "page_id": h.page_id, "title": h.page_title,
            "heading": " > ".join(h.heading_path), "score": round(h.rerank_score, 3),
            "text": h.text[:600]}


def build_registry(
    retriever: Retriever,
    notion: Any | None = None,
    *,
    hit_sink: dict[str, Hit] | None = None,
) -> ToolRegistry:
    """`hit_sink` collects every retrieved Hit so the writer can cite them."""
    reg = ToolRegistry()
    store = retriever.store

    def search_notes(a: SearchArgs) -> list[dict[str, Any]]:
        f = SearchFilters(tags=a.tags) if a.tags else None
        hits = retriever.retrieve(a.query, k=a.k, filters=f)
        if hit_sink is not None:
            for h in hits:
                hit_sink.setdefault(h.chunk_id, h)
        return [hit_brief(h) for h in hits]

    def _scroll(flt: Any, limit: int) -> list[dict[str, Any]]:
        pts, _ = store.client.scroll(store.collection, scroll_filter=flt, limit=limit,
                                     with_payload=True, with_vectors=False)
        return [p.payload or {} for p in pts]

    def get_page(a: PageArgs) -> dict[str, Any]:
        from qdrant_client import models

        rows = _scroll(models.Filter(must=[models.FieldCondition(
            key="page_id", match=models.MatchValue(value=a.page_id))]), 200)
        if not rows:
            return {"error": "page not indexed"}
        rows.sort(key=lambda r: r.get("chunk_index", 0))
        body = "\n\n".join(r.get("text", "") for r in rows)
        return {"title": rows[0].get("page_title"), "path": rows[0].get("path"),
                "last_edited_time": rows[0].get("last_edited_time"), "text": body[:6000]}

    def get_timeline(a: TimelineArgs) -> list[dict[str, Any]]:
        pages: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            pts, offset = store.client.scroll(store.collection, limit=512, offset=offset,
                                              with_payload=True, with_vectors=False)
            for p in pts:
                pl = p.payload or {}
                pages.setdefault(pl.get("page_id", ""), {
                    "page_id": pl.get("page_id"), "title": pl.get("page_title"),
                    "last_edited_time": pl.get("last_edited_time")})
            if offset is None:
                break
        items = [v for v in pages.values() if v["last_edited_time"]]
        if a.since:
            since = datetime.fromisoformat(a.since)
            items = [v for v in items
                     if datetime.fromisoformat(str(v["last_edited_time"])).replace(tzinfo=None)
                     >= since.replace(tzinfo=None)]
        items.sort(key=lambda v: str(v["last_edited_time"]), reverse=True)
        return items[: a.limit]

    reg.register(Tool("search_notes", "내 노션 노트 하이브리드 검색(+rerank). 근거 수집용.",
                      SearchArgs, search_notes))
    reg.register(Tool("get_page", "page_id 페이지 전체 본문(인덱스 기준) 조회.",
                      PageArgs, get_page))
    reg.register(Tool("get_timeline", "최근 수정된 페이지 목록(날짜순).",
                      TimelineArgs, get_timeline))

    if notion is not None:
        def append_to_page(a: AppendArgs) -> dict[str, Any]:
            r = notion.blocks.children.append(block_id=a.page_id, children=_paragraphs(a.text))
            return {"appended_blocks": len(r.get("results", []))}

        def create_page(a: CreatePageArgs) -> dict[str, Any]:
            r = notion.pages.create(
                parent={"page_id": a.parent_page_id},
                properties={"title": {"title": [{"text": {"content": a.title}}]}},
                children=_paragraphs(a.text),
            )
            return {"page_id": r.get("id"), "url": r.get("url")}

        reg.register(Tool("append_to_page", "기존 페이지 끝에 문단 추가.", AppendArgs,
                          append_to_page, write=True))
        reg.register(Tool("create_page", "부모 페이지 아래 새 하위 페이지 생성.", CreatePageArgs,
                          create_page, write=True))
    return reg

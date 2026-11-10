"""F3: `my-memory` MCP server — exposes my Notion knowledge to any MCP client (stdio).

Read-only on purpose: search_notes / get_page / get_timeline wrap the same ToolRegistry the
LangGraph agent uses, so behaviour and validation are identical. Notion writes stay inside the
agent, behind its approval gate.

Register in Claude Code:
    claude mcp add my-memory -- uv --directory /path/to/repo run nra mcp

Note: Qdrant runs in embedded mode (single-process lock) — stop `nra serve` while a client
uses this server, or point QDRANT_URL at the docker-compose Qdrant to share it.
"""

from __future__ import annotations

import functools
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from notion_agent.tools.registry import ToolRegistry

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

server = MCPServer(
    "my-memory",
    instructions="사용자 개인 노션 노트 검색. 답할 때 결과의 title/heading을 출처로 밝혀라.",
)


@functools.cache
def _registry() -> ToolRegistry:
    from notion_agent.rag.factory import build_retriever
    from notion_agent.tools.builtin import build_registry

    return build_registry(build_retriever(), notion=None)  # notion=None -> no write tools


def _call(name: str, **args: Any) -> Any:
    r = _registry().run(name, {k: v for k, v in args.items() if v is not None})
    if not r.ok:
        raise ValueError(r.error)
    return r.output


@server.tool(annotations=READ_ONLY)
def search_notes(query: str, k: int = 5, tags: list[str] | None = None) -> list[dict[str, Any]]:
    """내 노션 노트를 하이브리드 검색(dense+sparse, rerank)해 관련 조각을 점수순으로 돌려준다."""
    return _call("search_notes", query=query, k=k, tags=tags)


@server.tool(annotations=READ_ONLY)
def get_page(page_id: str) -> dict[str, Any]:
    """page_id 페이지의 본문 전체(인덱스 기준)를 돌려준다."""
    return _call("get_page", page_id=page_id)


@server.tool(annotations=READ_ONLY)
def get_timeline(since: str | None = None, limit: int = 15) -> list[dict[str, Any]]:
    """최근 수정된 페이지 목록. since는 ISO 날짜(예: 2026-10-01)."""
    return _call("get_timeline", since=since, limit=limit)


def main() -> None:
    server.run("stdio")

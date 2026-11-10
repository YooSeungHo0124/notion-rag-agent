import asyncio

from notion_agent import mcp_server


def test_mcp_lists_read_only_tools():
    tools = asyncio.run(mcp_server.server.list_tools())
    names = {t.name for t in tools}
    assert names == {"search_notes", "get_page", "get_timeline"}
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)

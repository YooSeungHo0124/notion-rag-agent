"""Build the real agent (shares cached retriever/verifier with the rest of the app)."""

from __future__ import annotations

import sqlite3
from typing import Any

from notion_agent.agent.graph import NotionAgent
from notion_agent.config import Settings, get_settings
from notion_agent.qa.grounded import GroundedQA


def build_agent(
    settings: Settings | None = None, qa: GroundedQA | None = None, *, notion: Any | None = None
) -> NotionAgent:
    from langgraph.checkpoint.sqlite import SqliteSaver

    from notion_agent.llm import get_llm
    from notion_agent.rag.factory import build_retriever, build_verifier
    from notion_agent.tools.builtin import build_registry

    s = settings or get_settings()
    llm = get_llm()
    qa = qa or GroundedQA(build_retriever(s), llm, build_verifier(s))
    if notion is None and s.notion_token:
        from notion_client import Client

        notion = Client(auth=s.notion_token)
    from notion_agent.memory.profile import ProfileStore

    data = s.state_db.parent
    data.mkdir(parents=True, exist_ok=True)
    profile = ProfileStore(data / "profile.sqlite")
    sink: dict = {}
    reg = build_registry(qa.retriever, notion, hit_sink=sink, profile=profile)
    if s.odoo_url:
        from notion_agent.erp.odoo import OdooClient, OdooConfig
        from notion_agent.erp.tools import build_erp_tools

        build_erp_tools(reg, OdooClient(OdooConfig(s.odoo_url, s.odoo_db, s.odoo_user,
                                                   s.odoo_password)), sink)
    saver = SqliteSaver(sqlite3.connect(data / "agent_checkpoints.sqlite", check_same_thread=False))
    return NotionAgent(llm, reg, qa, sink, checkpointer=saver, profile=profile)

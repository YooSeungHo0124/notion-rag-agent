"""Agent graph: planner loop, step limit, approval gate (interrupt/resume), registry validation."""

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from test_qa_grounded import OverlapVerifier, StubRetriever, hit

from notion_agent.agent.graph import NotionAgent
from notion_agent.llm.base import LLMClient, LLMResponse
from notion_agent.qa.grounded import GroundedQA
from notion_agent.tools.registry import Tool, ToolRegistry


class ScriptLLM(LLMClient):
    """Returns planner actions in order; answers writer calls with a fixed sentence."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.plans = 0

    def complete(self, prompt, *, system=None, json_schema=None):
        props = (json_schema or {}).get("properties", {})
        if "action" in props:
            self.plans += 1
            act = self.actions.pop(0) if self.actions else ("final", "{}")
            return LLMResponse("", {"thought": "t", "action": act[0], "args_json": act[1]})
        return LLMResponse("", {"answerable": True, "sentences": [
            {"text": "batch size reduced to 8", "citations": ["S1"]}]})


class Q(BaseModel):
    query: str


class W(BaseModel):
    page_id: str
    text: str


def make(actions, max_steps=4):
    hits = [hit(1, "batch size reduced to 8")]
    sink: dict = {}
    written: list = []
    reg = ToolRegistry()

    def search(a: Q):
        for h in hits:
            sink[h.chunk_id] = h
        return [{"chunk_id": h.chunk_id, "text": h.text} for h in hits]

    reg.register(Tool("search_notes", "search", Q, search))
    reg.register(Tool("append_to_page", "append", W, lambda a: written.append(a) or {"ok": 1},
                      write=True))
    llm = ScriptLLM(actions)
    qa = GroundedQA(StubRetriever(hits), llm, OverlapVerifier())
    agent = NotionAgent(llm, reg, qa, sink, checkpointer=InMemorySaver(), max_steps=max_steps)
    return agent, llm, written


def test_search_then_final_gives_cited_answer():
    agent, llm, _ = make([("search_notes", '{"query": "OOM batch"}'), ("final", "{}")])
    r = agent.run("GPU OOM 어떻게 해결?", "t1")
    assert r["status"] == "done" and r["done_reason"] == "final"
    assert [s["tool"] for s in r["steps"]] == ["search_notes"]
    assert r["answer"]["answerable"] and r["answer"]["sources"][0]["chunk_id"] == "c1"


def test_step_limit_stops_loop():
    agent, llm, _ = make([("search_notes", '{"query": "x"}')] * 10, max_steps=2)
    r = agent.run("q", "t2")
    assert r["done_reason"] == "step_limit" and len(r["steps"]) == 2


def test_invalid_args_become_observation_not_crash():
    agent, _, _ = make([("search_notes", '{"wrong": 1}'), ("final", "{}")])
    r = agent.run("q", "t3")
    assert r["steps"][0]["ok"] is False and "invalid call" in r["steps"][0]["observation"]


def test_write_requires_approval_and_rejection_skips_it():
    agent, _, written = make([("append_to_page", '{"page_id": "p1", "text": "hi"}'),
                              ("final", "{}")])
    r = agent.run("노션에 hi 추가해줘", "t4")
    assert r["status"] == "awaiting_approval" and r["approval"]["tool"] == "append_to_page"
    assert written == []
    r = agent.resume("t4", approved=False)
    assert r["status"] == "done" and written == []
    assert r["steps"][-1]["observation"] == "사용자가 실행을 거부함"


def test_write_runs_after_approval_with_edited_args():
    agent, _, written = make([("append_to_page", '{"page_id": "p1", "text": "hi"}'),
                              ("final", "{}")])
    agent.run("노션에 hi 추가해줘", "t5")
    r = agent.resume("t5", approved=True, args={"page_id": "p1", "text": "edited"})
    assert r["status"] == "done" and written[0].text == "edited"
    assert r["answer"]["kind"] == "action"

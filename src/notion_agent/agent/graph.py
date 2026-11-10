"""LangGraph supervisor agent: planner -> tools (approval gate for writes) -> writer.

    START -> planner --(read tool)--> tools -> planner ...
                     --(write tool)--> approval (interrupt) -> tools -> planner
                     --(final / step limit)--> writer -> END

- planner : Claude picks the next action as schema-validated JSON (our own tool calling).
            Rewrites the user's question into search queries, decides when evidence suffices.
- tools   : ToolRegistry.run — args validated by pydantic, errors become observations.
- approval: `interrupt()` pauses the graph; the UI/CLI resumes with {"approved": bool}.
            Write tools can never run without passing through this node.
- writer  : GroundedQA over all hits collected so far -> cited, verified answer.
State is checkpointed (SQLite) so a pending approval survives a server restart.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from notion_agent.llm.base import LLMClient
from notion_agent.qa.grounded import GroundedQA
from notion_agent.qa.serialize import answer_to_json
from notion_agent.rag.retriever import Hit
from notion_agent.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_STEPS = 4

PLANNER_SYSTEM = """너는 개인 노션 에이전트의 supervisor다. 사용자의 요청을 해결하기 위해 다음 행동 하나를 고른다.
도구:
{tools}

규칙:
- 질문이면 search_notes로 근거를 모은다. 검색어는 사용자 문장을 그대로 쓰지 말고 핵심 키워드/동의어로 재작성한다.
- 첫 검색 결과 score가 낮거나 주제가 어긋나면 다른 검색어로 한 번 더 찾는다(최대 {max_steps}단계).
- 특정 페이지 전체가 필요하면 get_page, '최근/요즘' 류는 get_timeline.
- 사용자가 명시적으로 노션에 기록/추가/생성을 요청한 경우에만 쓰기 도구를 쓴다.
- 충분한 근거를 모았거나 더 할 일이 없으면 action="final".
- args_json은 해당 도구 args의 JSON 문자열. final이면 "{{}}"."""


class AgentState(TypedDict, total=False):
    question: str
    steps: list[dict[str, Any]]  # {agent, tool, args, ok, observation, ms}
    pending: dict[str, Any] | None  # proposed {tool, args, thought}
    answer: dict[str, Any] | None
    done_reason: str


def _plan_schema(tool_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action": {"type": "string", "enum": [*tool_names, "final"]},
            "args_json": {"type": "string"},
        },
        "required": ["thought", "action", "args_json"],
        "additionalProperties": False,
    }


def _history(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "(아직 없음)"
    out = []
    for i, s in enumerate(steps, 1):
        obs = json.dumps(s["observation"], ensure_ascii=False, default=str)[:1500]
        out.append(f"[{i}] {s['tool']}({json.dumps(s['args'], ensure_ascii=False)}) -> {obs}")
    return "\n".join(out)


class NotionAgent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        qa: GroundedQA,
        hit_sink: dict[str, Hit],
        checkpointer: Any | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.llm, self.reg, self.qa, self.hits = llm, registry, qa, hit_sink
        self.max_steps = max_steps
        self.graph = self._build().compile(checkpointer=checkpointer)

    # -- nodes -----------------------------------------------------------------
    def _planner(self, st: AgentState) -> Command[Literal["tools", "approval", "writer"]]:
        steps = st.get("steps", [])
        if len(steps) >= self.max_steps:
            return Command(goto="writer", update={"done_reason": "step_limit"})
        system = PLANNER_SYSTEM.format(tools=self.reg.describe(), max_steps=self.max_steps)
        prompt = f"사용자 요청: {st['question']}\n\n지금까지의 행동과 관찰:\n{_history(steps)}"
        resp = self.llm.complete(prompt, system=system,
                                 json_schema=_plan_schema(list(self.reg.tools)))
        plan = resp.structured or {"action": "final", "args_json": "{}", "thought": "no plan"}
        action = plan["action"]
        log.info("planner: %s %s | %s", action, plan.get("args_json"), plan.get("thought"))
        if action == "final":
            return Command(goto="writer", update={"done_reason": "final"})
        try:
            args = json.loads(plan.get("args_json") or "{}")
        except json.JSONDecodeError:
            args = {}
        pending = {"tool": action, "args": args, "thought": plan.get("thought", "")}
        tool = self.reg.get(action)
        goto = "approval" if tool and tool.write else "tools"
        return Command(goto=goto, update={"pending": pending})

    def _approval(self, st: AgentState) -> Command[Literal["tools", "planner"]]:
        p = st["pending"] or {}
        decision = interrupt({"type": "approval", "tool": p.get("tool"), "args": p.get("args"),
                              "thought": p.get("thought")})
        if isinstance(decision, dict) and decision.get("approved"):
            if isinstance(decision.get("args"), dict):  # user may edit args before approving
                p = {**p, "args": decision["args"]}
            return Command(goto="tools", update={"pending": {**p, "approved": True}})
        step = {"agent": "NotionOps", "tool": p.get("tool"), "args": p.get("args"), "ok": False,
                "observation": "사용자가 실행을 거부함", "ms": 0}
        return Command(goto="planner",
                       update={"steps": [*st.get("steps", []), step], "pending": None})

    def _tools(self, st: AgentState) -> dict[str, Any]:
        p = st["pending"] or {}
        tool = self.reg.get(p.get("tool", ""))
        if tool and tool.write and not p.get("approved"):  # defence in depth
            obs, ok, ms = "approval missing", False, 0
        else:
            r = self.reg.run(p.get("tool", ""), p.get("args") or {})
            obs, ok, ms = (r.output if r.ok else r.error), r.ok, r.latency_ms
        agent = "NotionOps" if tool and tool.write else "Retriever"
        step = {"agent": agent, "tool": p.get("tool"), "args": p.get("args"), "ok": ok,
                "observation": obs, "ms": ms}
        return {"steps": [*st.get("steps", []), step], "pending": None}

    def _writer(self, st: AgentState) -> dict[str, Any]:
        steps = st.get("steps", [])
        hits = sorted(self.hits.values(), key=lambda h: h.rerank_score, reverse=True)[:8]
        writes = [s for s in steps if s["agent"] == "NotionOps"]
        if not hits and writes:  # pure action request: report what happened
            return {"answer": {"kind": "action", "steps": writes}}
        if not hits:  # planner skipped search: fall back to a direct search
            hits = self.qa.retriever.retrieve(st["question"])
        a = self.qa.answer_from_hits(st["question"], hits)
        return {"answer": {"kind": "answer", **answer_to_json(a)}}

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("planner", self._planner)
        g.add_node("approval", self._approval)
        g.add_node("tools", self._tools)
        g.add_node("writer", self._writer)
        g.add_edge(START, "planner")
        g.add_edge("tools", "planner")
        g.add_edge("writer", END)
        return g

    # -- API -------------------------------------------------------------------
    def _result(self, cfg: dict[str, Any], t0: float) -> dict[str, Any]:
        snap = self.graph.get_state(cfg)
        vals = snap.values
        intr = [i.value for t in snap.tasks for i in (t.interrupts or [])]
        return {
            "thread_id": cfg["configurable"]["thread_id"],
            "status": "awaiting_approval" if intr else "done",
            "approval": intr[0] if intr else None,
            "steps": vals.get("steps", []),
            "answer": vals.get("answer"),
            "done_reason": vals.get("done_reason"),
            "total_ms": int((time.time() - t0) * 1000),
        }

    def run(self, question: str, thread_id: str) -> dict[str, Any]:
        t0 = time.time()
        self.hits.clear()
        cfg = {"configurable": {"thread_id": thread_id}}
        self.graph.invoke({"question": question, "steps": [], "pending": None,
                           "answer": None}, cfg)
        return self._result(cfg, t0)

    def resume(self, thread_id: str, approved: bool,
               args: dict[str, Any] | None = None) -> dict[str, Any]:
        t0 = time.time()
        cfg = {"configurable": {"thread_id": thread_id}}
        self.graph.invoke(Command(resume={"approved": approved, "args": args}), cfg)
        return self._result(cfg, t0)

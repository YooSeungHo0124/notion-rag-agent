"""Tool registry: pydantic arg models -> JSON schema, validated execution.

The LLM never executes anything. It proposes `{tool, args}`; `ToolRegistry.run` validates
the args against the tool's pydantic model and calls our Python function. Tools marked
`write=True` must be approved by the user before `run` is called (enforced by the agent
graph via LangGraph interrupt, see notion_agent.agent.graph).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]
    write: bool = False  # mutates Notion -> needs human approval

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "write": self.write,
            "input_schema": self.args_model.model_json_schema(),
        }


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    latency_ms: int = 0


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self.tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def describe(self) -> str:
        """Compact tool catalogue for the planner prompt."""
        lines = []
        for t in self.tools.values():
            props = t.args_model.model_json_schema().get("properties", {})
            lines.append(f"- {t.name}{' [쓰기·승인 필요]' if t.write else ''}: {t.description}\n"
                         f"  args: {json.dumps(props, ensure_ascii=False)}")
        return "\n".join(lines)

    def validate(self, name: str, args: dict[str, Any]) -> BaseModel:
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool {name!r}; available: {sorted(self.tools)}")
        return tool.args_model.model_validate(args)

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        t0 = time.time()
        try:
            parsed = self.validate(name, args)
            out = self.tools[name].fn(parsed)
            return ToolResult(True, out, latency_ms=int((time.time() - t0) * 1000))
        except (KeyError, ValidationError) as e:
            return ToolResult(False, error=f"invalid call: {e}")
        except Exception as e:  # tool failure is an observation, not a crash
            return ToolResult(False, error=f"{type(e).__name__}: {e}",
                              latency_ms=int((time.time() - t0) * 1000))

"""Provider-agnostic LLM interface.

Every LLM call in the app goes through `LLMClient`, so the backend can be switched
between the Claude subscription CLI (personal use) and the Anthropic API by config.
Tool execution is never delegated to the provider: the client only returns text or
schema-validated JSON, and our own executor runs tools (see notion_agent.tools).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class LLMError(RuntimeError):
    """Raised when the provider call fails or returns an unusable response."""


@dataclass
class LLMResponse:
    text: str
    structured: dict[str, Any] | None = None
    model: str = ""
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Single-turn completion. With `json_schema`, `structured` holds the parsed object."""

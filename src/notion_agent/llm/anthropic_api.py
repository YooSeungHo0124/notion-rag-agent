"""LLMClient backed by the Anthropic Messages API (official `anthropic` SDK).

The backend for containers / shared deployments, where the personal `claude -p`
subscription login must not be used. Same contract as ClaudeCLIClient: text in, text or
schema-validated JSON out (`output_config.format`), no provider-side tool execution.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from notion_agent.llm.base import LLMClient, LLMError, LLMResponse
from notion_agent.observability.tracing import span

ALIASES = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5",
           "haiku": "claude-haiku-5-5"}
DEFAULT_SYSTEM = "You are a helpful assistant."


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Structured outputs need additionalProperties:false on every object."""
    s = dict(schema)
    if s.get("type") == "object":
        s.setdefault("additionalProperties", False)
        s["properties"] = {k: _strict(v) for k, v in s.get("properties", {}).items()}
    if isinstance(s.get("items"), dict):
        s["items"] = _strict(s["items"])
    for key in ("anyOf", "oneOf"):
        if key in s:
            s[key] = [_strict(x) for x in s[key]]
    return s


class AnthropicAPIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "sonnet", timeout_s: int = 180,
                 max_retries: int = 2, max_tokens: int = 8000, client: Any = None) -> None:
        if client is None:
            if not api_key:
                raise LLMError("ANTHROPIC_API_KEY is empty (llm_provider=anthropic_api)")
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s,
                                         max_retries=max_retries)
        self.client = client
        self.model = ALIASES.get(model, model)
        self.max_tokens = max_tokens

    def _params(self, prompt: str, system: str | None,
                json_schema: dict[str, Any] | None) -> dict[str, Any]:
        p: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                             "system": system or DEFAULT_SYSTEM,
                             "messages": [{"role": "user", "content": prompt}]}
        if json_schema is not None:
            p["output_config"] = {"format": {"type": "json_schema",
                                             "schema": _strict(json_schema)}}
        return p

    def _finish(self, msg: Any, json_schema: dict[str, Any] | None, t0: float) -> LLMResponse:
        if msg.stop_reason == "refusal":
            raise LLMError("model refused")
        text = "".join(b.text for b in msg.content if b.type == "text")
        structured = None
        if json_schema is not None:
            if msg.stop_reason == "max_tokens":
                raise LLMError("structured output truncated (max_tokens)")
            try:
                structured = json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMError(f"invalid JSON from model: {e}") from e
        u = msg.usage
        return LLMResponse(text=text, structured=structured, model=msg.model,
                           latency_ms=int((time.time() - t0) * 1000),
                           usage={"input_tokens": u.input_tokens,
                                  "output_tokens": u.output_tokens})

    def complete(self, prompt: str, *, system: str | None = None,
                 json_schema: dict[str, Any] | None = None) -> LLMResponse:
        return self.complete_stream(prompt, system=system, json_schema=json_schema)

    def complete_stream(self, prompt: str, *, system: str | None = None,
                        json_schema: dict[str, Any] | None = None,
                        on_partial: Callable[[str], None] | None = None) -> LLMResponse:
        with span("anthropic.messages", "LLM", input=prompt, llm__model_name=self.model,
                  llm__system=system) as sp:
            t0 = time.time()
            try:
                with self.client.messages.stream(**self._params(prompt, system, json_schema)) as st:
                    acc = ""
                    for chunk in st.text_stream:
                        acc += chunk
                        if on_partial:
                            on_partial(acc)
                    msg = st.get_final_message()
            except LLMError:
                raise
            except Exception as e:  # SDK API/connection errors
                raise LLMError(f"Anthropic API error: {e}") from e
            r = self._finish(msg, json_schema, t0)
            sp.output(r.structured if r.structured is not None else r.text)
            sp.set(llm__token_count__prompt=r.usage["input_tokens"],
                   llm__token_count__completion=r.usage["output_tokens"])
            return r

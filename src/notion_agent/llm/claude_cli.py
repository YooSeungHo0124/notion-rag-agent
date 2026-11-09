"""LLMClient backed by the Claude Code CLI in headless mode (`claude -p`).

Uses the host's Claude subscription login. Personal, local use only: serving other
people through a subscription login is not allowed, switch to `anthropic_api` for that.

Every call disables Claude Code's built-in tools and MCP servers (`--tools ""`,
`--strict-mcp-config`) so the model can only return text/JSON. Tool calls are decided
by the model but executed by our own code, which keeps approval gates and tracing
on our side.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

from notion_agent.llm.base import LLMClient, LLMError, LLMResponse

DEFAULT_SYSTEM = "You are a helpful assistant."


class ClaudeCLIClient(LLMClient):
    def __init__(
        self,
        model: str = "sonnet",
        timeout_s: int = 180,
        max_retries: int = 2,
        binary: str = "claude",
    ) -> None:
        path = shutil.which(binary)
        if path is None:
            raise LLMError(f"`{binary}` not found on PATH; install Claude Code and log in.")
        self.binary = path
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def build_command(
        self, prompt: str, system: str | None, json_schema: dict[str, Any] | None
    ) -> list[str]:
        cmd = [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--system-prompt",
            system or DEFAULT_SYSTEM,
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]
        if json_schema is not None:
            cmd += ["--json-schema", json.dumps(json_schema, ensure_ascii=False)]
        return cmd

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        cmd = self.build_command(prompt, system, json_schema)
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout_s, check=False
                )
                return self._parse(proc, json_schema, int((time.monotonic() - start) * 1000))
            except (subprocess.TimeoutExpired, LLMError) as exc:
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        raise LLMError(f"claude CLI failed after {self.max_retries + 1} attempts: {last_err}")

    def _parse(
        self,
        proc: subprocess.CompletedProcess[str],
        json_schema: dict[str, Any] | None,
        latency_ms: int,
    ) -> LLMResponse:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            detail = proc.stderr[:500] or proc.stdout[:500]
            raise LLMError(f"non-JSON output (exit={proc.returncode}): {detail}") from exc

        if payload.get("is_error") or payload.get("subtype") != "success":
            raise LLMError(f"claude CLI error: {payload.get('subtype')} {payload.get('result')}")

        structured = payload.get("structured_output")
        if json_schema is not None and not isinstance(structured, dict):
            raise LLMError("structured output requested but not returned")

        usage = payload.get("usage") or {}
        return LLMResponse(
            text=payload.get("result", ""),
            structured=structured,
            model=next(iter(payload.get("modelUsage") or {}), self.model),
            latency_ms=latency_ms,
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "api_ms": payload.get("duration_api_ms", 0),
            },
        )

"""LLMClient backed by the Claude Code CLI in headless mode (`claude -p`).

Uses the host's Claude subscription login. Personal, local use only: serving other
people through a subscription login is not allowed, switch to `anthropic_api` for that.

Every call disables Claude Code's built-in tools and MCP servers (`--tools ""`,
`--strict-mcp-config`) so the model can only return text/JSON. Tool calls are decided
by the model but executed by our own code, which keeps approval gates and tracing
on our side.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from notion_agent.llm.base import LLMClient, LLMError, LLMResponse
from notion_agent.observability.tracing import span

DEFAULT_SYSTEM = "You are a helpful assistant."
STDIN_THRESHOLD = 100_000


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
        # argv strings are capped at 128 KiB (MAX_ARG_STRLEN); long prompts go via stdin
        cmd = [self.binary, "-p"] + ([] if self._via_stdin(prompt) else [prompt]) + [
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

    @staticmethod
    def _via_stdin(prompt: str) -> bool:
        return len(prompt.encode()) > STDIN_THRESHOLD

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        with span("claude -p", "LLM", input=prompt, llm__model_name=self.model,
                  llm__system=system, llm__invocation_parameters={"json_schema": bool(json_schema)},
                  ) as sp:
            r = self._complete(prompt, system, json_schema)
            sp.output(r.structured if r.structured is not None else r.text)
            sp.set(llm__token_count__prompt=r.usage.get("input_tokens", 0),
                   llm__token_count__completion=r.usage.get("output_tokens", 0),
                   latency_ms=r.latency_ms)
            return r

    def complete_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        on_partial: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """stream-json with partial messages; falls back to `complete` on any stream error."""
        if on_partial is None:
            return self.complete(prompt, system=system, json_schema=json_schema)
        with span("claude -p (stream)", "LLM", input=prompt, llm__model_name=self.model,
                  llm__system=system) as sp:
            try:
                r = self._stream(prompt, system, json_schema, on_partial)
            except (LLMError, OSError, subprocess.TimeoutExpired) as exc:
                sp.set(stream_error=str(exc)[:200])
                r = self._complete(prompt, system, json_schema)
            sp.output(r.structured if r.structured is not None else r.text)
            sp.set(llm__token_count__prompt=r.usage.get("input_tokens", 0),
                   llm__token_count__completion=r.usage.get("output_tokens", 0),
                   latency_ms=r.latency_ms)
            return r

    def _stream(
        self, prompt: str, system: str | None, json_schema: dict[str, Any] | None,
        on_partial: Callable[[str], None],
    ) -> LLMResponse:
        cmd = self.build_command(prompt, system, json_schema)
        i = cmd.index("--output-format")
        cmd[i + 1 : i + 2] = ["stream-json", "--verbose", "--include-partial-messages"]
        start = time.monotonic()
        proc = subprocess.Popen(  # noqa: S603 - fixed binary, no shell
            cmd, stdin=subprocess.PIPE if self._via_stdin(prompt) else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        timer = threading.Timer(self.timeout_s, proc.kill)
        timer.start()
        try:
            if self._via_stdin(prompt):
                assert proc.stdin is not None
                proc.stdin.write(prompt)
                proc.stdin.close()
            buf, final = "", None
            assert proc.stdout is not None
            for line in proc.stdout:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "stream_event":
                    d = (ev.get("event") or {}).get("delta") or {}
                    piece = d.get("partial_json" if json_schema else "text") or ""
                    if piece:
                        buf += piece
                        # a UI callback must not kill the call
                        with contextlib.suppress(Exception):
                            on_partial(buf)
                elif ev.get("type") == "result":
                    final = ev
            proc.wait()
        finally:
            timer.cancel()
        if final is None:
            raise LLMError(f"stream ended without result (exit={proc.returncode})")
        done = subprocess.CompletedProcess(cmd, proc.returncode, json.dumps(final), "")
        return self._parse(done, json_schema, int((time.monotonic() - start) * 1000))

    def _complete(
        self, prompt: str, system: str | None, json_schema: dict[str, Any] | None
    ) -> LLMResponse:
        cmd = self.build_command(prompt, system, json_schema)
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout_s, check=False,
                    input=prompt if self._via_stdin(prompt) else None,
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

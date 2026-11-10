import json
import subprocess

import pytest

from notion_agent.llm.base import LLMError
from notion_agent.llm.claude_cli import ClaudeCLIClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    return ClaudeCLIClient(model="sonnet", max_retries=0)


def _proc(payload, code=0):
    out = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=out, stderr="")


def test_command_disables_builtin_tools_and_mcp(client):
    cmd = client.build_command("hi", None, {"type": "object"})
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in cmd
    assert "--no-session-persistence" in cmd
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == {"type": "object"}


def test_parses_structured_output(client, monkeypatch):
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": '{"answer":5}',
        "structured_output": {"answer": 5},
        "usage": {"input_tokens": 2, "output_tokens": 6},
        "modelUsage": {"claude-sonnet-5-5": {}},
    }
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc(payload))
    resp = client.complete("2+3", json_schema={"type": "object"})
    assert resp.structured == {"answer": 5}
    assert resp.model == "claude-sonnet-5-5"


def test_error_payload_raises(client, monkeypatch):
    payload = {"type": "result", "subtype": "error_max_turns", "is_error": True}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc(payload))
    with pytest.raises(LLMError):
        client.complete("x")


def test_missing_structured_output_raises(client, monkeypatch):
    payload = {"type": "result", "subtype": "success", "is_error": False, "result": "5"}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc(payload))
    with pytest.raises(LLMError):
        client.complete("x", json_schema={"type": "object"})


def test_non_json_output_raises(client, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc("boom", code=1))
    with pytest.raises(LLMError):
        client.complete("x")


def test_long_prompt_goes_via_stdin(client, monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"], seen["input"] = cmd, kw.get("input")
        return _proc('{"subtype":"success","result":"ok"}')

    monkeypatch.setattr(subprocess, "run", run)
    long = "가" * 50_000  # 150 KB in UTF-8 > argv limit
    client.complete(long)
    assert long not in seen["cmd"] and seen["input"] == long
    client.complete("short")
    assert "short" in seen["cmd"] and seen["input"] is None

from types import SimpleNamespace as NS

import pytest

from notion_agent.llm.anthropic_api import AnthropicAPIClient, _strict
from notion_agent.llm.base import LLMError


class FakeStream:
    def __init__(self, text, stop="end_turn"):
        self.text, self.stop = text, stop
        self.text_stream = [text[: len(text) // 2], text[len(text) // 2:]]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return NS(content=[NS(type="text", text=self.text)], stop_reason=self.stop,
                  model="claude-sonnet-5-5", usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    def __init__(self, text, stop="end_turn"):
        self.calls = []
        self.messages = NS(stream=lambda **p: (self.calls.append(p), FakeStream(text, stop))[1])


def test_structured_output_uses_output_config_and_streams_partials():
    fc = FakeClient('{"a": 1}')
    c = AnthropicAPIClient("", model="sonnet", client=fc)
    parts = []
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    r = c.complete_stream("q", json_schema=schema, on_partial=parts.append)
    assert r.structured == {"a": 1} and parts[-1] == '{"a": 1}'
    p = fc.calls[0]
    assert p["model"] == "claude-sonnet-5-5"
    assert p["output_config"]["format"]["schema"]["additionalProperties"] is False


def test_truncated_or_refused_output_raises():
    with pytest.raises(LLMError):
        c = AnthropicAPIClient("", client=FakeClient('{"a"', "max_tokens"))
        c.complete("q", json_schema={"type": "object"})
    with pytest.raises(LLMError):
        AnthropicAPIClient("", client=FakeClient("", "refusal")).complete("q")


def test_strict_recurses_into_nested_objects():
    s = _strict({"type": "object", "properties": {"xs": {"type": "array", "items": {
        "type": "object", "properties": {"b": {"type": "string"}}}}}})
    assert s["properties"]["xs"]["items"]["additionalProperties"] is False


def test_missing_key_is_a_clear_error():
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicAPIClient("")

"""F6: OpenTelemetry tracing -> Arize Phoenix (OpenInference semantic conventions).

    docker compose up -d phoenix       # UI http://127.0.0.1:6006
    NRA_TRACE=1 nra serve              # or nra ask / nra agent / python -m eval.run_eval

LangGraph nodes are traced by the OpenInference LangChain instrumentor. Our own components
(`claude -p` LLM calls, hybrid search, rerank, grounding verification, tools) get manual spans
via `span()`. Everything is a no-op when tracing is off or the `trace` extra isn't installed.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://127.0.0.1:6006/v1/traces"
_enabled = False

try:
    from opentelemetry import trace as _otel
except ImportError:  # trace extra not installed
    _otel = None  # type: ignore[assignment]


def setup_tracing(project: str = "notion-rag-agent", *, force: bool = False) -> bool:
    """Configure the global tracer provider once. Returns True if tracing is on."""
    global _enabled
    if _enabled:
        return True
    if not (force or os.getenv("NRA_TRACE") in {"1", "true", "yes"}) or _otel is None:
        return False
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = os.getenv("NRA_TRACE_ENDPOINT", DEFAULT_ENDPOINT)
    provider = TracerProvider(resource=Resource.create({
        "service.name": project, "openinference.project.name": project}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    _otel.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _enabled = True
    log.info("tracing -> %s (project %s)", endpoint, project)
    return True


def flush() -> None:
    if _enabled and _otel is not None:
        _otel.get_tracer_provider().force_flush()  # type: ignore[attr-defined]


def _attr(v: Any) -> Any:
    if isinstance(v, str | bool | int | float):
        return v
    return json.dumps(v, ensure_ascii=False, default=str)[:20000]


class _Span:
    def __init__(self, s: Any) -> None:
        self._s = s

    def set(self, **attrs: Any) -> None:
        if self._s is not None:
            for k, v in attrs.items():
                if v is not None:
                    self._s.set_attribute(k.replace("__", "."), _attr(v))

    def output(self, value: Any) -> None:
        self.set(**{"output__value": value})


@contextmanager
def span(name: str, kind: str = "CHAIN", input: Any = None, **attrs: Any) -> Iterator[_Span]:
    """Manual OpenInference span. kind: LLM | RETRIEVER | RERANKER | TOOL | CHAIN | EVALUATOR.

    Attribute names use `__` for dots, e.g. llm__model_name="sonnet".
    """
    if not _enabled or _otel is None:
        yield _Span(None)
        return
    tracer = _otel.get_tracer("notion_agent")
    with tracer.start_as_current_span(name) as s:
        sp = _Span(s)
        sp.set(openinference__span__kind=kind)
        if input is not None:
            sp.set(input__value=input)
        sp.set(**attrs)
        try:
            yield sp
        except Exception as e:
            s.record_exception(e)
            s.set_status(_otel.Status(_otel.StatusCode.ERROR, str(e)))
            raise


def documents(prefix: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten docs into OpenInference retrieval.documents.N.document.* attributes."""
    out: dict[str, Any] = {}
    for i, d in enumerate(docs):
        for k, v in d.items():
            out[f"{prefix}__{i}__document__{k}"] = v
    return out

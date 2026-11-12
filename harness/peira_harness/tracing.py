"""Tracing setup. One Langfuse client per process, exporting nested OTel spans."""
from __future__ import annotations

import os
import threading

from opentelemetry.sdk.trace import ReadableSpan

_lock = threading.Lock()
_client = None
_instrumented: set[str] = set()

# HTTP/DB client auto-instrumentation is noise for agent traces.
# com.anthropic.sdk: anthropic>=2.x emits its own span per request. Frameworks already emit the LLM span
# (OpenInference or native), so the SDK span would duplicate every generation and double the cost.
_NOISE_SCOPE_PREFIXES = (
    "com.anthropic.sdk",
    "opentelemetry.instrumentation.httpx",
    "opentelemetry.instrumentation.requests",
    "opentelemetry.instrumentation.urllib",
    "opentelemetry.instrumentation.urllib3",
    "opentelemetry.instrumentation.aiohttp",
    "opentelemetry.instrumentation.asyncio",
    "opentelemetry.instrumentation.sqlite3",
    "opentelemetry.instrumentation.sqlalchemy",
    "opentelemetry.instrumentation.grpc",
)


def _should_export(span: ReadableSpan) -> bool:
    scope = span.instrumentation_scope.name if span.instrumentation_scope else ""
    return not scope.startswith(_NOISE_SCOPE_PREFIXES)


def client():
    """Return the process-wide Langfuse client (creates it on first use)."""
    global _client
    with _lock:
        if _client is None:
            # OpenInference writes one attribute per message part; the OTel default of 128 drops
            # llm.model_name on long conversations, which leaves generations unpriced. Must be set
            # before the tracer provider exists.
            os.environ.setdefault("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "4096")
            os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")
            from langfuse import Langfuse

            _client = Langfuse(
                environment=os.environ.get("PEIRA_ENVIRONMENT", "experiments"),
                release=os.environ.get("PEIRA_RELEASE", "v1"),
                should_export_span=_should_export,
                flush_at=200,
                flush_interval=2.0,
            )
        return _client


def langchain_callbacks() -> list:
    """Callbacks for LangChain / LangGraph / deepagents. Spans nest under the runner's root."""
    from langfuse.langchain import CallbackHandler

    client()
    return [CallbackHandler()]


def instrument(instrumentor, **kwargs) -> None:
    """Instrument an OpenInference (or other OTel) instrumentor once per process.

    Extra kwargs go to ``instrumentor.instrument`` (e.g. ``use_event_listener=True`` for CrewAI).
    """
    key = type(instrumentor).__module__ + "." + type(instrumentor).__name__
    with _lock:
        if key in _instrumented:
            return
        _instrumented.add(key)
    client()  # make sure Langfuse registered its span processor on the global provider
    from opentelemetry import trace

    instrumentor.instrument(tracer_provider=trace.get_tracer_provider(), **kwargs)


def shutdown() -> None:
    global _client
    with _lock:
        c = _client
    if c is not None:
        c.flush()
        c.shutdown()

"""Shared helpers for the pydantic_llamaindex family.

PydanticAI agents use PydanticAI's built-in OpenTelemetry instrumentation (tracing="native-otel"),
bound to the global tracer provider that the Langfuse client registers.
LlamaIndex agents use the OpenInference LlamaIndex instrumentor (tracing="openinference").
Each agent has exactly one source of LLM spans: the framework. The Anthropic SDK is not instrumented.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
import uuid
from typing import Any, Callable

from peira_harness import llm, tracing

# anthropic>=2.x emits its own OTel span per API call (scope com.anthropic.sdk.python).
# The framework already emits the LLM generation, so turn the SDK spans off: one LLM span source.
os.environ["ANTHROPIC_OPEN_TELEMETRY"] = "false"

MAX_TOKENS = llm.MAX_TOKENS


# --------------------------------------------------------------------------- PydanticAI


def pyd_setup() -> None:
    """Bind PydanticAI instrumentation to the global OTel provider (Langfuse span processor)."""
    tracing.client()
    from opentelemetry import trace
    from pydantic_ai import Agent, InstrumentationSettings

    Agent.instrument_all(
        InstrumentationSettings(tracer_provider=trace.get_tracer_provider(), include_binary_content=False)
    )


@functools.cache
def pyd_model(model: str = llm.HAIKU):
    """PydanticAI Anthropic model bound to the LiteLLM proxy (Anthropic format)."""
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # The harness client retries 4 times (the SDK default is 2): connection drops through the proxy are transient.
    provider = AnthropicProvider(anthropic_client=llm.async_anthropic_client(timeout=180))
    return AnthropicModel(model, provider=provider, settings=AnthropicModelSettings(max_tokens=MAX_TOKENS))


# --------------------------------------------------------------------------- LlamaIndex


def li_setup() -> None:
    """Instrument LlamaIndex with OpenInference and set an Anthropic default LLM with no embeddings."""
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    tracing.instrument(LlamaIndexInstrumentor())
    from llama_index.core import Settings

    Settings.llm = li_llm()
    # No OpenAI embeddings anywhere in this family: every index is a SummaryIndex or keyword index.
    Settings.embed_model = None


# Helper methods on the LLM class that make no API call. The LlamaIndex dispatcher wraps them in spans,
# and OpenInference tags every span on an LLM instance with the model name, so Langfuse would show
# each one as an extra GENERATION without usage. Unwrapping them keeps one GENERATION per API call.
# Convenience wrappers (complete -> chat, predict -> chat, chat_with_tools -> chat) are unwrapped too,
# otherwise the wrapper and the inner chat call both become GENERATIONs with usage and cost counts twice.
_LLM_HELPERS_WITHOUT_CALLS = (
    "_prepare_chat_with_tools", "_prepare_chat_with_tools_compat", "get_tool_calls_from_response",
    "complete", "acomplete", "stream_complete", "astream_complete",
    "predict", "apredict", "stream", "astream", "predict_and_call", "apredict_and_call",
    "chat_with_tools", "achat_with_tools", "stream_chat_with_tools", "astream_chat_with_tools",
    "structured_predict", "astructured_predict", "stream_structured_predict", "astream_structured_predict",
)


@functools.cache
def _proxy_anthropic_cls():
    from llama_index.llms.anthropic import Anthropic

    class ProxyAnthropic(Anthropic):
        @classmethod
        def class_name(cls) -> str:
            return "Anthropic"

    for name in _LLM_HELPERS_WITHOUT_CALLS:
        fn = getattr(Anthropic, name, None)
        if fn is None or not hasattr(fn, "__wrapped__"):
            continue
        # Peel every layer (dispatcher span and callback-event decorators). Leaving the event
        # decorator would attach the wrapper's LLM events to the caller's span instead.
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        setattr(ProxyAnthropic, name, fn)
    return ProxyAnthropic


def li_llm(model: str = llm.HAIKU, **kw):
    """LlamaIndex Anthropic LLM bound to the proxy. The package drops temperature for *-5-5 models."""
    kw.setdefault("max_tokens", MAX_TOKENS)
    kw.setdefault("max_retries", 4)
    kw.setdefault("timeout", 180)
    return _proxy_anthropic_cls()(model=model, base_url=llm.anthropic_base_url(), api_key=llm.proxy_key(), **kw)


# --------------------------------------------------------------------------- tools


def begin_task(agent_id: str, task: dict) -> None:
    """Scope tool-service writes (refunds, bookings, ticket updates) to this one run."""
    from peira_harness import tools

    tools.set_run_id(f"{agent_id}:{task['id']}:{uuid.uuid4().hex[:8]}")


def to_text(value: Any, limit: int = 6000) -> str:
    """Render a tool result as compact text for the model."""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, default=str, ensure_ascii=False)
        except TypeError:
            s = str(value)
    return s if len(s) <= limit else s[:limit] + f"... [truncated {len(s) - limit} chars]"


def safe_tool(fn: Callable) -> Callable:
    """Wrap a sync harness tool as an async function that returns errors as text.

    Tool errors (HTTP 404 for an unknown id, bad SQL) are data the agent must react to,
    so they go back to the model instead of failing the run. The wrapper keeps the
    signature and docstring so frameworks build the same tool schema.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            out = await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as e:  # noqa: BLE001 - surfaced to the model
            if type(e).__name__ == "ToolsServiceError":
                raise  # infrastructure failure: fail the run, do not fake a tool answer
            return f"ERROR from {fn.__name__}: {type(e).__name__}: {str(e)[:500]}"
        if isinstance(out, dict) and out.get("error") and len(out) <= 2:
            return f"ERROR from {fn.__name__}: {out['error']}"
        return to_text(out)

    wrapper.__signature__ = inspect.signature(fn).replace(return_annotation=str)
    # Tool results are rendered as text, so advertise a str return type.
    wrapper.__annotations__ = {**getattr(fn, "__annotations__", {}), "return": str}
    return wrapper

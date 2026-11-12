"""Shared setup for the openai_agents family.

- All model calls use OpenAIChatCompletionsModel against the LiteLLM proxy (OpenAI format).
- Spans come only from the OpenInference OpenAIAgentsInstrumentor. It replaces the SDK's
  default trace processor, so nothing is ever exported to OpenAI's trace backend.
- The openai client itself is NOT instrumented: the SDK's generation spans are the only
  LLM spans (one per model call).
"""
from __future__ import annotations

import threading

from agents import (
    ModelSettings,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
    set_default_openai_client,
)
from peira_harness import llm, tracing

HAIKU = llm.HAIKU
SONNET = llm.SONNET

_lock = threading.Lock()
_client = None
_ready = False


def _openai_client():
    global _client
    with _lock:
        if _client is None:
            _client = llm.async_openai_client()
        return _client


def setup_family() -> None:
    """Idempotent: instrument once, route the SDK default client to the proxy."""
    global _ready
    if _ready:
        return
    from agents.tracing import get_trace_provider
    from agents.tracing.processors import BatchTraceProcessor
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    # Anything that falls back to the SDK default client must still hit the proxy.
    set_default_openai_client(_openai_client(), use_for_tracing=False)
    set_default_openai_api("chat_completions")

    # exclusive_processor=True (the default) removes the SDK's OpenAI backend exporter.
    tracing.instrument(OpenAIAgentsInstrumentor())

    processors = getattr(getattr(get_trace_provider(), "_multi_processor", None), "_processors", ())
    if any(isinstance(p, BatchTraceProcessor) for p in processors):
        raise RuntimeError("OpenAI trace exporter is still registered; refusing to run")
    _ready = True


def model(name: str = HAIKU) -> OpenAIChatCompletionsModel:
    """A chat-completions model bound to the LiteLLM proxy."""
    return OpenAIChatCompletionsModel(model=name, openai_client=_openai_client())


def settings(max_tokens: int = llm.MAX_TOKENS, **kw) -> ModelSettings:
    """Model settings that respect the proxy's model rules (no temperature/top_p)."""
    return ModelSettings(max_tokens=max_tokens, include_usage=True, **kw)


def shared_tools(*names: str) -> list:
    """Wrap functions from peira_harness.tools as SDK FunctionTools (sync tools run in threads)."""
    from agents import function_tool
    from peira_harness import tools as T

    out = []
    for n in names:
        fn = getattr(T, n)
        out.append(function_tool(fn))  # sync: runs in a thread; the harness client has its own timeouts
    return out



def refusal_as_answer(fn):
    """Return a model refusal as the agent's answer instead of a crash.

    Claude refuses some adversarial requests at the API level (stop_reason "refusal"), which the
    SDK raises as ModelRefusalError. For these agents a refusal is the correct final answer.
    """
    import functools

    from agents.exceptions import ModelRefusalError

    @functools.wraps(fn)
    async def wrapper(task: dict) -> str:
        try:
            return await fn(task)
        except ModelRefusalError as e:
            return f"[model refused] {e.refusal}"

    return wrapper

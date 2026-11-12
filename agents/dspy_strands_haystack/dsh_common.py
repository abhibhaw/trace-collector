"""Shared factories for the dspy_strands_haystack family.

The runner puts this directory on sys.path, so agents import it as `dsh_common`.
Every factory points at the LiteLLM proxy through peira_harness.llm.
No factory sets temperature, top_p, top_k, thinking, or tool_choice.
"""
from __future__ import annotations

import os
import threading

# Haystack and DSPy phone home by default. Turn it off before they are imported.
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("DSPY_TELEMETRY_DISABLED", "1")

from peira_harness import llm, tracing  # noqa: E402

_lock = threading.Lock()
_done: set[str] = set()


def _once(key: str) -> bool:
    with _lock:
        if key in _done:
            return False
        _done.add(key)
        return True


# --------------------------------------------------------------------------- DSPy

def _dspy_usage_callback():
    """DSPy callback that copies lm15 token usage onto the OpenInference LLM span.

    openinference-instrumentation-dspy records the LM call as an LLM span but no token counts.
    `with_callbacks` runs inside that span, so `get_current_span()` is the LM span.
    This keeps exactly one generation per LM call (no second SDK-level instrumentor).
    """
    import dspy
    from opentelemetry import trace

    class UsageToSpan(dspy.utils.callback.BaseCallback):
        def on_lm_end(self, call_id, outputs, exception=None):  # noqa: D401
            usage = getattr(outputs, "usage", None)
            if usage is None:
                return
            span = trace.get_current_span()
            if not span.is_recording():
                return
            inp, out = getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
            if inp is None:
                # lm15 leaves input_tokens unknown when the wire omits the cache counters; fall back to the raw usage.
                raw = getattr(outputs, "provider_data", None)
                raw_usage = raw.get("usage") if isinstance(raw, dict) else None
                if isinstance(raw_usage, dict) and raw_usage.get("input_tokens") is not None:
                    inp = sum(int(raw_usage.get(k) or 0) for k in
                              ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            tot = getattr(usage, "total_tokens", None)
            if inp is not None:
                span.set_attribute("llm.token_count.prompt", int(inp))
            if out is not None:
                span.set_attribute("llm.token_count.completion", int(out))
            if tot is None and inp is not None and out is not None:
                tot = inp + out
            if tot is not None:
                span.set_attribute("llm.token_count.total", int(tot))
            cr = getattr(usage, "cache_read_tokens", None)
            if cr:
                span.set_attribute("llm.token_count.prompt_details.cache_read", int(cr))

    return UsageToSpan()


def dspy_lm(model: str = llm.HAIKU, max_tokens: int = llm.MAX_TOKENS):
    """A dspy.LM on the proxy's Anthropic route.

    - api_base needs the /v1 suffix: the native lm15 engine appends only /messages.
    - cache=False so every task makes real, traced calls.
    - reasoning_effort=None stops ChainOfThought/ReAct from turning on native thinking
      (`thinking: adaptive` + `output_config.effort`); the reasoning stays a visible text field.
    - temperature stays None, so it is never sent.
    """
    import dspy

    return dspy.LM(
        f"anthropic/{model}",
        api_base=f"{llm.anthropic_base_url()}/v1",
        api_key=llm.proxy_key(),
        max_tokens=max_tokens,
        cache=False,
        num_retries=4,
        reasoning_effort=None,
    )


def setup_dspy(model: str = llm.HAIKU) -> None:
    """Instrument DSPy once and configure the global LM and usage callback."""
    if not _once("dspy"):
        return
    import dspy
    from openinference.instrumentation.dspy import DSPyInstrumentor

    tracing.instrument(DSPyInstrumentor())
    dspy.configure(lm=dspy_lm(model), callbacks=[_dspy_usage_callback()])


# ------------------------------------------------------------------------ Strands

def setup_strands() -> None:
    """Bind Strands' native OTel tracer to the global provider that Langfuse uses.

    Strands creates spans with the global tracer provider. The Langfuse client must exist first,
    so its span processor is on that provider. No extra exporter is added.
    """
    if not _once("strands"):
        return
    tracing.client()
    from opentelemetry import trace
    from strands.telemetry import StrandsTelemetry

    StrandsTelemetry(tracer_provider=trace.get_tracer_provider())


def strands_model(model: str = llm.HAIKU, max_tokens: int = llm.MAX_TOKENS):
    """Strands AnthropicModel on the proxy. Build one per task (it holds an async client)."""
    from strands.models.anthropic import AnthropicModel

    return AnthropicModel(
        client_args={"base_url": llm.anthropic_base_url(), "api_key": llm.proxy_key(), "max_retries": 4},
        model_id=model,
        max_tokens=max_tokens,
    )


# ----------------------------------------------------------------------- Haystack

def setup_haystack() -> None:
    """Instrument Haystack once with OpenInference.

    The instrumentor wraps `run` of every component class registered at that moment,
    so call this after the agent module has imported all its Haystack components
    (setup() runs after module import, which is enough).
    """
    if not _once("haystack"):
        return
    from haystack import tracing as hs_tracing
    from openinference.instrumentation.haystack import HaystackInstrumentor

    tracing.instrument(HaystackInstrumentor())
    # Haystack 3.x removed the ToolInvoker component: the Agent runs tools itself and reports them only
    # through Haystack's own tracer hook. Bridge just those tool spans to OTel (LLM spans stay OpenInference).
    hs_tracing.enable_tracing(_HaystackToolSpanTracer())
    hs_tracing.tracer.is_content_tracing_enabled = True


def _haystack_tool_tracer_classes():
    import contextlib
    import json

    from haystack.tracing.tracer import NullSpan, Span, Tracer
    from opentelemetry import trace as otel

    def _s(v) -> str:
        if isinstance(v, str):
            return v[:8000]
        try:
            return json.dumps(v, default=str)[:8000]
        except Exception:  # noqa: BLE001
            return str(v)[:8000]

    class _ToolSpan(Span):
        def __init__(self, span):
            self._span = span

        def set_tag(self, key, value):
            if key == "haystack.agent.step.tool.input":
                self._span.set_attribute("input.value", _s(value))
            elif key == "haystack.agent.step.tool.output":
                self._span.set_attribute("output.value", _s(value))
            elif key == "haystack.tool.error" and value:
                self._span.set_status(otel.Status(otel.StatusCode.ERROR))

        def raw_span(self):
            return self._span

    class HaystackToolSpanTracer(Tracer):
        """Turns `haystack.agent.step.tool` operations into OpenInference TOOL spans; ignores everything else."""

        def __init__(self):
            self._tracer = otel.get_tracer("dsh.haystack.tools")

        @contextlib.contextmanager
        def trace(self, operation_name, tags=None, parent_span=None):
            if operation_name != "haystack.agent.step.tool":
                yield NullSpan()
                return
            tags = tags or {}
            name = tags.get("haystack.tool.name", "tool")
            with self._tracer.start_as_current_span(name) as span:
                span.set_attribute("openinference.span.kind", "TOOL")
                span.set_attribute("tool.name", name)
                if tags.get("haystack.tool.description"):
                    span.set_attribute("tool.description", str(tags["haystack.tool.description"])[:1000])
                yield _ToolSpan(span)

        def current_span(self):
            return NullSpan()

    return HaystackToolSpanTracer


def _HaystackToolSpanTracer():  # noqa: N802 - factory that keeps haystack imports lazy
    return _haystack_tool_tracer_classes()()


def haystack_generator(model: str = llm.HAIKU, tools=None, max_tokens: int = llm.MAX_TOKENS):
    """AnthropicChatGenerator on the proxy.

    anthropic-haystack has no base_url argument; the Anthropic SDK reads ANTHROPIC_BASE_URL.
    We set it explicitly here instead of relying on the runner's setdefault.
    """
    from haystack.utils import Secret
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

    os.environ["ANTHROPIC_BASE_URL"] = llm.anthropic_base_url()
    return AnthropicChatGenerator(
        api_key=Secret.from_token(llm.proxy_key()),
        model=model,
        generation_kwargs={"max_tokens": max_tokens},
        tools=tools,
        max_retries=4,
    )

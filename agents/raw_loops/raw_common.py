"""Shared helpers for the raw_loops family: hand-written agent loops with manual Langfuse spans.

LLM generations come from OpenInference instrumentors (anthropic or litellm).
Everything else (loop turns, phases, sub-agents, tool calls) is a manual Langfuse observation.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import uuid
from typing import Any, Callable, Iterable

from anthropic import beta_tool

from peira_harness import llm, tracing

MAX_TOKENS = llm.MAX_TOKENS
_RESULT_CHARS = 12000  # cap tool output that goes back to the model


# ---------------------------------------------------------------- instrumentation
def instrument_anthropic() -> None:
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    tracing.instrument(AnthropicInstrumentor())


def instrument_litellm() -> None:
    from openinference.instrumentation.litellm import LiteLLMInstrumentor

    tracing.instrument(LiteLLMInstrumentor())


def lf():
    return tracing.client()


# The anthropic SDK (>=1.14) starts OTel spans of its own (API call, tool_runner, tool use).
# This family traces by hand, so SDK spans are off; LLM spans come from OpenInference only.
def anthropic_client():
    return llm.anthropic_client(open_telemetry=False)


def async_anthropic_client():
    return llm.async_anthropic_client(open_telemetry=False)


@contextlib.contextmanager
def phase(name: str, as_type: str = "chain", input: Any = None, metadata: Any = None):
    """A manual Langfuse child observation. LLM calls made inside nest under it."""
    with lf().start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata) as obs:
        yield obs


# ---------------------------------------------------------------- text helpers
def text_of(resp) -> str:
    """Join the text blocks of an Anthropic Message (thinking blocks may come first)."""
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


class ModelRefused(RuntimeError):
    """The model stopped with stop_reason == "refusal" (safety classifier)."""


def refusal_text(resp) -> str | None:
    """A user-facing note if the response is a refusal, else None."""
    if getattr(resp, "stop_reason", None) != "refusal":
        return None
    det = getattr(resp, "stop_details", None)
    cat = getattr(det, "category", None) or "unspecified"
    return f"I can't help with this request (declined by the model's safety system, category: {cat})."


def extract_xml(text: str, tag: str) -> str:
    m = re.search(f"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_all_xml(text: str, tag: str) -> list[str]:
    return [m.strip() for m in re.findall(f"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)]


def to_text(value: Any) -> str:
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(value)
    if len(s) > _RESULT_CHARS:
        s = s[:_RESULT_CHARS] + f"\n...[truncated {len(s) - _RESULT_CHARS} chars]"
    return s


def lit_text(resp) -> str:
    """Text of a litellm ModelResponse."""
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------- tools
class Toolbox:
    """Typed Python functions exposed as Anthropic tools. Each call is a Langfuse `tool` span."""

    def __init__(self, fns: Iterable[Callable], rename: dict[str, str] | None = None):
        self.tools = {}
        for fn in fns:
            t = beta_tool(fn)
            if rename and fn.__name__ in rename:
                t.name = rename[fn.__name__]
            self.tools[t.name] = t

    def schemas(self) -> list[dict]:
        return [t.to_dict() for t in self.tools.values()]

    def names(self) -> list[str]:
        return list(self.tools)

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Run one tool inside a tool span. Returns (text, is_error)."""
        with lf().start_as_current_observation(name=name, as_type="tool", input=args) as obs:
            tool = self.tools.get(name)
            if tool is None:
                msg = f"Error: unknown tool {name!r}. Available: {', '.join(self.tools)}"
                obs.update(output=msg, level="WARNING", status_message="unknown tool")
                return msg, True
            try:
                out = tool.call(args)
            except Exception as e:  # noqa: BLE001 - tool errors go back to the model as is_error results
                msg = f"Error: {type(e).__name__}: {e}"[:2000]
                obs.update(output=msg, level="ERROR", status_message=msg[:500])
                return msg, True
            return _finish_tool(obs, out)

    async def acall(self, name: str, args: dict) -> tuple[str, bool]:
        """Async variant: the sync HTTP tool runs in a worker thread (context is copied)."""
        with lf().start_as_current_observation(name=name, as_type="tool", input=args) as obs:
            tool = self.tools.get(name)
            if tool is None:
                msg = f"Error: unknown tool {name!r}. Available: {', '.join(self.tools)}"
                obs.update(output=msg, level="WARNING", status_message="unknown tool")
                return msg, True
            try:
                out = await asyncio.to_thread(tool.call, args)
            except Exception as e:  # noqa: BLE001
                msg = f"Error: {type(e).__name__}: {e}"[:2000]
                obs.update(output=msg, level="ERROR", status_message=msg[:500])
                return msg, True
            return _finish_tool(obs, out)


def _finish_tool(obs, out: Any) -> tuple[str, bool]:
    """peira_harness.tools returns {"error": ...} for normal failures instead of raising."""
    text = to_text(out)
    if isinstance(out, dict) and out.get("error") and len(out) <= 2:
        obs.update(output=text, level="WARNING", status_message=str(out["error"])[:500])
        return text, True
    obs.update(output=text)
    return text, False


def begin_task(agent_id: str, task: dict) -> None:
    """Scope tools-service writes (refunds, ticket updates, emails) to this run only."""
    from peira_harness import tools as T

    T.set_run_id(f"{agent_id}:{task.get('id')}:{uuid.uuid4().hex[:8]}")


def tool_results(box: Toolbox, resp) -> list[dict]:
    """Execute every tool_use block of a response and return all tool_result blocks (one user turn)."""
    out = []
    for b in resp.content:
        if b.type != "tool_use":
            continue
        text, err = box.call(b.name, dict(b.input or {}))
        r = {"type": "tool_result", "tool_use_id": b.id, "content": text}
        if err:
            r["is_error"] = True
        out.append(r)
    return out


async def atool_results(box: Toolbox, resp) -> list[dict]:
    """Async: run all tool_use blocks of one response concurrently."""
    uses = [b for b in resp.content if b.type == "tool_use"]
    results = await asyncio.gather(*(box.acall(b.name, dict(b.input or {})) for b in uses))
    out = []
    for b, (text, err) in zip(uses, results):
        r = {"type": "tool_result", "tool_use_id": b.id, "content": text}
        if err:
            r["is_error"] = True
        out.append(r)
    return out


class StepLimitExceeded(RuntimeError):
    pass


def run_tool_loop(
    client,
    *,
    model: str,
    system: str,
    messages: list[dict],
    box: Toolbox,
    max_turns: int = 8,
    max_tokens: int = MAX_TOKENS,
    turn_span_prefix: str = "turn",
) -> str:
    """Classic manual tool-use loop (sync). Each model turn is a chain span with its tool spans inside."""
    for i in range(1, max_turns + 1):
        with phase(f"{turn_span_prefix}-{i}", "chain") as turn:
            # Last allowed turn: tools stay declared but the model must answer from what it has.
            extra = {"tool_choice": {"type": "none"}} if i == max_turns else {}
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, system=system, tools=box.schemas(), messages=messages, **extra
            )
            if (refused := refusal_text(resp)) is not None:
                turn.update(output=refused, level="WARNING", status_message="model refusal")
                return refused
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                final = text_of(resp)
                turn.update(output=final, metadata={"stop_reason": resp.stop_reason})
                if resp.stop_reason == "max_tokens" and not final:
                    raise RuntimeError("model hit max_tokens without producing an answer")
                return final
            results = tool_results(box, resp)
            turn.update(output={"tool_calls": [r["tool_use_id"] for r in results]}, metadata={"stop_reason": "tool_use"})
            messages.append({"role": "user", "content": results})
    raise StepLimitExceeded(f"no final answer after {max_turns} model turns")


async def arun_tool_loop(
    client,
    *,
    model: str,
    system: str,
    messages: list[dict],
    box: Toolbox,
    max_turns: int = 6,
    max_tokens: int = MAX_TOKENS,
    turn_span_prefix: str = "turn",
) -> str:
    """Async manual tool-use loop; parallel tool calls in one turn run concurrently."""
    for i in range(1, max_turns + 1):
        with phase(f"{turn_span_prefix}-{i}", "chain") as turn:
            # Last allowed turn: tools stay declared but the model must answer from what it has.
            extra = {"tool_choice": {"type": "none"}} if i == max_turns else {}
            resp = await client.messages.create(
                model=model, max_tokens=max_tokens, system=system, tools=box.schemas(), messages=messages, **extra
            )
            if (refused := refusal_text(resp)) is not None:
                turn.update(output=refused, level="WARNING", status_message="model refusal")
                return refused
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                final = text_of(resp)
                turn.update(output=final, metadata={"stop_reason": resp.stop_reason})
                return final
            results = await atool_results(box, resp)
            turn.update(output={"tool_calls": len(results)}, metadata={"stop_reason": "tool_use"})
            messages.append({"role": "user", "content": results})
    raise StepLimitExceeded(f"no final answer after {max_turns} model turns")

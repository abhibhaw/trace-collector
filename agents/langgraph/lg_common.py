"""Shared helpers for the langgraph family.

The runner puts agents/langgraph on sys.path, so every agent can `import lg_common`.
"""
from __future__ import annotations

import json
import os

# The Anthropic SDK (>= 2.x) emits its own OTel GENERATION spans (scope com.anthropic.sdk.python).
# LangChain callbacks are this family's single source of LLM spans, so turn the SDK's off.
os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")

from typing import Any, Callable, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from peira_harness import llm, tracing

HAIKU = llm.HAIKU
SONNET = llm.SONNET

M = TypeVar("M", bound=BaseModel)


def model(name: str = HAIKU, **kw):
    """ChatAnthropic bound to the LiteLLM proxy (no temperature, max_tokens 4096)."""
    return llm.chat_anthropic(name, **kw)


def start_run(agent_id: str, task: dict) -> str:
    """A unique id for this run (checkpointer thread id). The runner already scopes tools-service writes."""
    import uuid

    return f"{agent_id}:{task.get('id')}:{uuid.uuid4().hex[:8]}"


def run_config(recursion_limit: int = 25, **configurable: Any) -> dict:
    """RunnableConfig with Langfuse callbacks. Spans nest under the runner's root observation."""
    cfg: dict[str, Any] = {"callbacks": tracing.langchain_callbacks(), "recursion_limit": recursion_limit}
    if configurable:
        cfg["configurable"] = configurable
    return cfg


def text(msg: Any) -> str:
    """Plain text of an AIMessage (Anthropic content may be a list of blocks, with thinking)."""
    if isinstance(msg, BaseMessage):
        content = msg.content
    else:
        content = msg
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)
    return str(content)


def wrap(fn: Callable, name: str | None = None, description: str | None = None) -> BaseTool:
    """Wrap a plain typed Python function (docstring = description) as a LangChain tool."""
    return StructuredTool.from_function(fn, name=name or fn.__name__, description=description)


def shared(*names: str) -> list[BaseTool]:
    """LangChain tools for functions in peira_harness.tools, by function name."""
    from peira_harness import tools as T

    out = []
    for n in names:
        fn = getattr(T, n, None)
        if fn is None:
            raise AttributeError(f"peira_harness.tools has no function {n!r}")
        out.append(wrap(fn))
    return out


def structured(chat, schema: type[M], *, include_raw: bool = False):
    """Structured output that is safe on both proxy models.

    Haiku may force tool choice; Sonnet may not (HTTP 400), so Sonnet uses json_schema.
    """
    if getattr(chat, "model", "") == SONNET:
        return chat.with_structured_output(schema, method="json_schema", include_raw=include_raw)
    return chat.with_structured_output(schema, include_raw=include_raw)


def to_json(x: Any, limit: int = 6000) -> str:
    """Compact JSON for prompts and tool messages, truncated."""
    try:
        s = json.dumps(x, default=str, ensure_ascii=False)
    except TypeError:
        s = str(x)
    return s if len(s) <= limit else s[:limit] + " ...[truncated]"

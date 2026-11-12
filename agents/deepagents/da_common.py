"""Shared helpers for the deepagents family.

- `model()` returns a proxy-bound ChatAnthropic (never sets temperature or thinking).
- `lc_tools()` wraps `peira_harness.tools` functions as LangChain tools.
- `invoke()` runs a compiled deep agent with Langfuse callbacks and a recursion limit.
- `seed_files()` builds the StateBackend `files` payload (virtual filesystem, never disk).
- `final_answer()` picks the deliverable: a written report file if any, else the last AI text.

Every agent in this family uses the default `StateBackend`, so the built-in file tools
(ls, read_file, write_file, edit_file, glob, grep, delete) work on graph state only and the
`execute` tool is not offered. Nothing touches the pod's disk or shell.
"""
from __future__ import annotations

import os
import uuid

# anthropic>=2.8 emits its own OTel GENERATION ("com.anthropic.sdk.python") under each LangChain
# generation. The contract allows one source of LLM spans, so keep LangChain's and switch the SDK's off.
# Must run before any Anthropic client is constructed (clients read it once at construction).
os.environ["ANTHROPIC_OPEN_TELEMETRY"] = "false"

from typing import Annotated, Any, Callable, Iterable

from langchain.tools import InjectedState

from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool

from peira_harness import llm, tracing

HAIKU = llm.HAIKU
SONNET = llm.SONNET


def model(name: str = HAIKU, max_tokens: int = llm.MAX_TOKENS):
    m = llm.chat_anthropic(name, max_tokens=max_tokens)
    # Warm lazily cached attributes before the instance is shared across runner threads.
    # Without this, two concurrent first calls race on `_serialized` (a functools.cached_property that
    # writes into the pydantic __dict__) and one fails with "dictionary changed size during iteration".
    _ = m._serialized
    return m


def todos() -> TodoListMiddleware:
    """Planning (`write_todos`) is opt-in since deepagents 0.7; every agent here opts in."""
    return TodoListMiddleware()


def call_limit(n: int, thread_limit: int | None = None) -> ModelCallLimitMiddleware:
    """Hard per-run cap on model calls. `end` stops gracefully with a limit notice.

    `thread_limit` also caps calls across resumes of one checkpointed thread (HITL agents).
    """
    return ModelCallLimitMiddleware(run_limit=n, thread_limit=thread_limit, exit_behavior="end")


def lc_tools(*names: str) -> list[BaseTool]:
    """Wrap shared tools-service functions by name (see harness/peira_harness/tools.py)."""
    from peira_harness import tools as T

    return [StructuredTool.from_function(safe(getattr(T, n)), name=n) for n in names]


def safe(fn: Callable) -> Callable:
    """Return tool failures (not found, invalid input, upstream down) to the model as text.

    The agent can then recover, as it would with a real API error. Agent bugs still raise,
    because only exceptions from inside the tool call are caught.
    """
    import functools
    import json

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from peira_harness.tools import ToolsServiceError

        try:
            out = fn(*args, **kwargs)
        except ToolsServiceError:
            raise  # infrastructure outage: fail the run, do not let the model improvise
        except Exception as e:  # noqa: BLE001 - surfaced to the model as a tool error
            return f"Error: {type(e).__name__}: {e}"[:2000]
        return out if isinstance(out, str) else json.dumps(out, default=str)[:15000]

    return wrapper


def lc_tool(fn: Callable, name: str | None = None, description: str | None = None) -> BaseTool:
    return StructuredTool.from_function(fn, name=name or fn.__name__, description=description)


def seed_files(files: dict[str, str]) -> dict[str, dict]:
    from deepagents.backends.utils import create_file_data

    return {path: create_file_data(content) for path, content in files.items()}


def load_seed(agent_file: str, sub: str = "seed", prefix: str = "/") -> dict[str, str]:
    """Read `<agent dir>/seed/**` (markdown memory and skills) into virtual paths."""
    from pathlib import Path

    root = Path(agent_file).resolve().parent / sub
    out = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            out[prefix + f.relative_to(root).as_posix()] = f.read_text()
    return out


def config(recursion_limit: int, thread_id: str | None = None, run_name: str | None = None) -> dict:
    cfg: dict[str, Any] = {"callbacks": tracing.langchain_callbacks(), "recursion_limit": recursion_limit}
    if run_name:
        cfg["run_name"] = run_name
    if thread_id:
        cfg["configurable"] = {"thread_id": thread_id}
    return cfg


def new_thread() -> str:
    return str(uuid.uuid4())


def invoke(agent, user_input: str, *, recursion_limit: int = 60, files: dict[str, str] | None = None,
           extra_state: dict | None = None, thread_id: str | None = None, run_name: str | None = None) -> dict:
    state: dict[str, Any] = {"messages": [{"role": "user", "content": user_input}]}
    if files:
        state["files"] = seed_files(files)
    if extra_state:
        state.update(extra_state)
    return agent.invoke(state, config=config(recursion_limit, thread_id, run_name))


def text_of(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")


def last_ai_text(result: dict) -> str:
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and not m.tool_calls:
            t = text_of(m).strip()
            if t:
                return t
    return ""


def file_text(result: dict, path: str) -> str | None:
    files = result.get("files") or {}
    data = files.get(path)
    if data is None:
        # Agents sometimes drop the leading slash or nest the file; match on basename.
        base = path.rsplit("/", 1)[-1]
        for p, d in files.items():
            if p.rsplit("/", 1)[-1] == base:
                data = d
                break
    if data is None:
        return None
    content = data.get("content") if isinstance(data, dict) else data
    if isinstance(content, list):  # older FileData stored lines
        content = "\n".join(content)
    return content


LIMIT_MARKER = "call limits exceeded"


def hit_limit(result: dict) -> bool:
    msgs = result.get("messages") or []
    return bool(msgs) and isinstance(msgs[-1], AIMessage) and LIMIT_MARKER in text_of(msgs[-1]).lower()


def final_answer(result: dict, report_paths: Iterable[str] = ()) -> str:
    """Return the deliverable. Raise if the agent produced nothing usable.

    A model-call-limit notice is never treated as an answer: if the cap was hit, the
    run only succeeds when a report file was already written.
    """
    limited = hit_limit(result)
    summary = "" if limited else last_ai_text(result)
    for p in report_paths:
        body = file_text(result, p)
        if body and body.strip():
            return body if not summary else f"{body}\n\n---\n{summary}"
    if summary:
        return summary
    if limited:
        raise RuntimeError("model call limit reached before the agent produced an answer")
    raise RuntimeError("agent produced no final answer and no report file")


def _state_file(state: dict, path: str) -> str | None:
    files = state.get("files") or {}
    d = files.get(path)
    if d is None:
        return None
    c = d.get("content") if isinstance(d, dict) else d
    return "\n".join(c) if isinstance(c, list) else c


def make_run_python_file_tool() -> BaseTool:
    """`run_python_file`: run a script from the virtual filesystem in the tools-service sandbox.

    Replaces the deepagents `execute` tool (sandbox backends) of the source examples. The
    script and any data files are read from graph state, shipped to the sandboxed
    `python_exec` tool, and materialised there under their basenames. Nothing runs locally.
    """
    import json

    from langchain_core.tools import tool

    from peira_harness import tools as T

    @tool
    def run_python_file(path: str, state: Annotated[dict, InjectedState], data_paths: list[str] | None = None) -> str:
        """Execute a Python script that you saved with write_file, in an isolated sandbox.

        Args:
            path: Virtual path of the script, e.g. /workspace/analysis.py.
            data_paths: Optional virtual paths of data files (CSV, JSON, text) the script needs.
                Each is available to the script in its working directory under its basename.
        """
        script = _state_file(state, path)
        if script is None:
            return f"Error: {path} does not exist. Write it with write_file first."
        prelude = []
        for dp in data_paths or []:
            body = _state_file(state, dp)
            if body is None:
                return f"Error: data file {dp} does not exist."
            name = dp.rsplit("/", 1)[-1]
            prelude.append(f"open({name!r}, 'w').write({body!r})")
        code = "\n".join(prelude + [script])
        try:
            out = T.python_exec(code)
        except T.ToolsServiceError:
            raise
        except Exception as e:  # noqa: BLE001 - surfaced to the model as a tool error
            return f"Error: {type(e).__name__}: {e}"[:2000]
        return out if isinstance(out, str) else json.dumps(out, default=str)[:12000]

    return run_python_file


def general_purpose(model_, tools: list | None = None, limit: int = 4) -> dict:
    """Explicit `general-purpose` sub-agent with a call cap (replaces the uncapped default)."""
    return {
        "name": "general-purpose",
        "description": "General-purpose agent for researching complex questions or doing an isolated multi-step sub-task. Give it the full context it needs; it returns only its final answer.",
        "system_prompt": "Complete the delegated task concisely. Return only the result the caller needs.",
        "tools": tools or [],
        "model": model_,
        "middleware": [call_limit(limit)],
    }


def begin_run(agent_id: str, task: dict) -> None:
    """Scope tools-service writes (ticket updates, bookings, emails) to this task run."""
    from peira_harness import tools as T

    T.set_run_id(f"{agent_id}:{task['id']}:{uuid.uuid4().hex[:8]}")

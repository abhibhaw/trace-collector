"""Shared helpers for the google_adk family.

- `claude(model)`: ADK's native AnthropicLlm bound to the LiteLLM proxy (Anthropic format).
- `setup_tracing()`: OpenInference GoogleADKInstrumentor. It also silences ADK's own tracers,
  so each model call yields exactly one LLM span (no duplicate generate_content spans).
- `run_agent(...)`: fresh InMemoryRunner + session per task, optional simulated user turns
  (auto-approver), and a per-task LLM call budget enforced by an ADK plugin.
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import uuid
from typing import Any, Callable, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.genai import types

from peira_harness import llm, tracing
from peira_harness import tools as _tools

# OpenInference writes one span attribute per message part; long ADK conversations exceed the
# OTel default of 128 attributes per span, and the dropped ones can include llm.model_name
# (then Langfuse cannot price the generation). Raise the limit before the tracer provider exists.
os.environ.setdefault("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "4096")

HAIKU = llm.HAIKU
SONNET = llm.SONNET
DEFAULT_MAX_LLM_CALLS = 25
# "Today" of the tools-service mock business world (orders, tickets, calendars, prices).
MOCK_TODAY = "2026-11-13"

_models: dict[str, AnthropicLlm] = {}


def claude(model: str = HAIKU) -> AnthropicLlm:
    """ADK AnthropicLlm that talks to the proxy's /v1/messages endpoint.

    No temperature/top_p/thinking is ever set (proxy model rules). max_tokens=4096.
    """
    if model not in _models:
        _models[model] = AnthropicLlm(
            model=model,
            max_tokens=llm.MAX_TOKENS,
            # anthropic>=2.8 emits its own OTel span per request (scope com.anthropic.sdk.python),
            # which would duplicate the instrumentor's call_llm GENERATION. Turn it off.
            client=llm.async_anthropic_client(open_telemetry=False),
        )
    return _models[model]


def async_tool(fn: Callable) -> Callable:
    """Wrap a blocking tool so ADK awaits it in a worker thread (keeps the event loop free for
    concurrent tasks). Name, docstring, signature and type hints are preserved for ADK's schema."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    return wrapper


class _AsyncTools:
    """`AT.web_search` etc.: async ADK tool versions of every peira_harness.tools function."""

    def __init__(self) -> None:
        self._cache: dict[str, Callable] = {}

    def __getattr__(self, name: str) -> Callable:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._cache:
            self._cache[name] = async_tool(getattr(_tools, name))
        return self._cache[name]


AT = _AsyncTools()


def setup_tracing() -> None:
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor

    tracing.instrument(GoogleADKInstrumentor())


class LlmCallBudgetExceeded(RuntimeError):
    pass


class CallBudgetPlugin(BasePlugin):
    """Counts model calls across the whole run, including AgentTool child runs
    (plugins propagate into AgentTool sub-runners), and stops runaway loops."""

    def __init__(self, max_calls: int):
        super().__init__(name="call_budget")
        self.max_calls = max_calls
        self.calls = 0

    async def before_model_callback(self, *, callback_context, llm_request):
        self.calls += 1
        if self.calls > self.max_calls:
            raise LlmCallBudgetExceeded(
                f"LLM call budget of {self.max_calls} exceeded in agent {callback_context.agent_name}"
            )
        return None


def _text_of(content: Optional[types.Content]) -> str:
    if not content or not content.parts:
        return ""
    return "".join(p.text for p in content.parts if p.text and not p.thought)


AutoUser = Callable[[int, str, dict], Optional[str]]
"""(turn_index, last_agent_text, session_state) -> next user message, or None to stop."""


async def run_agent(
    root_agent: BaseAgent,
    task: dict,
    *,
    app_name: str,
    initial_state: Optional[dict] = None,
    auto_user: Optional[AutoUser] = None,
    max_turns: int = 1,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    final_from_state: Optional[str] = None,
) -> str:
    """Run one task in a fresh session and return the final answer text.

    auto_user simulates the human in human-in-the-loop flows (approvals, clarifications).
    final_from_state: if set and present in session state at the end, return that value.
    """
    _tools.set_run_id(f"{app_name}:{task.get('id', 'x')}:{uuid.uuid4().hex[:8]}")
    budget = CallBudgetPlugin(max_llm_calls)
    runner = InMemoryRunner(agent=root_agent, app_name=app_name, plugins=[budget])
    user_id = f"user-{task.get('id', 'x')}"
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=str(uuid.uuid4()),
        state=dict(initial_state or {}),
    )
    message = task["input"]
    transcript: list[str] = []
    last_text = ""
    for turn in range(max_turns):
        turn_texts: list[str] = []
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
            run_config=RunConfig(max_llm_calls=max_llm_calls),
        ):
            if event.author != "user" and event.content and not event.partial:
                t = _text_of(event.content)
                if t.strip() and not event.get_function_calls():
                    turn_texts.append(t)
        if turn_texts:
            last_text = turn_texts[-1]
            transcript.append(last_text)
        if auto_user is None or turn == max_turns - 1:
            break
        sess = await runner.session_service.get_session(app_name=app_name, user_id=user_id, session_id=session.id)
        nxt = auto_user(turn, last_text, dict(sess.state))
        if not nxt:
            break
        message = nxt

    if final_from_state:
        sess = await runner.session_service.get_session(app_name=app_name, user_id=user_id, session_id=session.id)
        val = sess.state.get(final_from_state)
        if val:
            return val if isinstance(val, str) else json.dumps(val, indent=2, default=str)
    if not last_text.strip():
        raise RuntimeError("agent produced no final text response")
    return last_text


def asks_question(text: str) -> bool:
    """Heuristic: does the agent's last message wait for user input?"""
    tail = text.strip()[-600:]
    return "?" in tail


def parse_json_block(text: str) -> Any:
    """Extract the first JSON object/array from model text (fenced or bare)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    raw = m.group(1) if m else text
    start = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON found")
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(raw[start:])
    return obj


def load_local(agent_file: str, name: str = "prompts"):
    """Import a sibling module (e.g. prompts.py) of an agent under a unique module name."""
    import importlib.util
    import pathlib
    import sys

    path = pathlib.Path(agent_file).with_name(f"{name}.py")
    modname = f"_adk_{path.parent.name.replace('-', '_')}_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod

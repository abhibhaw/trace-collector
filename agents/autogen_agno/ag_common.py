"""Shared helpers for the autogen_agno family (AutoGen AgentChat and Agno agents).

Named ag_common (not autogen.py / agno.py / tools.py) so it never shadows a framework module
on sys.path. The runner puts the family dir on sys.path, so agents import it directly.
"""
from __future__ import annotations

import os

# Must be set before agno / autogen are imported anywhere in the process.
os.environ.setdefault("AGNO_TELEMETRY", "false")
# The SingleThreadedAgentRuntime spans ("autogen publish/process ...") are message-bus plumbing.
# OpenInference already gives team (CHAIN), agent (AGENT) and tool (TOOL) spans.
os.environ.setdefault("AUTOGEN_DISABLE_RUNTIME_TRACING", "true")
# anthropic>=2.8 has built-in OTel tracing (scope com.anthropic.sdk.python) that emits a second
# generation span for every messages.create on the global tracer provider. The framework / OpenInference
# instrumentor is the single LLM-span source, so turn the SDK's own tracing off.
os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")
# OpenInference flattens every chat message into span attributes; the OTel default of 128 drops most of them.
os.environ.setdefault("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "2048")
os.environ.setdefault("OTEL_ATTRIBUTE_COUNT_LIMIT", "2048")

from peira_harness import llm, tracing  # noqa: E402

HAIKU = llm.HAIKU
SONNET = llm.SONNET


# --------------------------------------------------------------------------------------------
# AutoGen
# --------------------------------------------------------------------------------------------
def _silence_autogen_core_spans() -> None:
    """autogen-core emits its own GenAI spans (`execute_tool`, `invoke_agent`, `create_agent`) on the
    global tracer provider. The OpenInference AutoGen instrumentor already opens TOOL and AGENT spans
    around the same calls, so the core spans would double every tool and agent turn in the tree.
    Replace them with no-op context managers where AgentChat imported them."""
    import contextlib

    import autogen_agentchat.agents._base_chat_agent as base_agent
    import autogen_agentchat.teams._group_chat._chat_agent_container as container
    import autogen_core.tools._base as base_tool

    @contextlib.contextmanager
    def _noop(*_a, **_kw):
        yield None

    base_tool.trace_tool_span = _noop
    for mod in (base_agent, container):
        for name in ("trace_invoke_agent_span", "trace_create_agent_span"):
            if hasattr(mod, name):
                setattr(mod, name, _noop)


def setup_autogen_tracing() -> None:
    """Agent/team/tool spans from the OpenInference AutoGen instrumentor, LLM generations from the
    OpenInference Anthropic instrumentor.

    The AutoGen instrumentor only emits LLM spans for OpenAI-format model clients. These agents use
    the Anthropic-native client, so the Anthropic SDK instrumentor is the single source of LLM spans
    (one GENERATION per messages.create call, with token usage)."""
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor

    _silence_autogen_core_spans()
    autogen_instrumentor = AutogenAgentChatInstrumentor()
    tracing.instrument(autogen_instrumentor)
    tracing.instrument(AnthropicInstrumentor())
    _instrument_autogen_gaps(autogen_instrumentor)


_GAPS_DONE = False


def _instrument_autogen_gaps(instrumentor) -> None:
    """Spans the OpenInference AutoGen instrumentor misses:

    - CodeExecutorAgent overrides on_messages_stream, so the BaseChatAgent wrapper never sees it; wrap it
      with the instrumentor's own AGENT wrapper.
    - MagenticOneGroupChat's orchestrator is not a chat agent; its planning and ledger LLM calls would sit
      loose under the team span. Wrap its phases in CHAIN spans."""
    global _GAPS_DONE
    if _GAPS_DONE:
        return
    _GAPS_DONE = True
    import functools

    from openinference.instrumentation.autogen_agentchat._wrappers import _BaseChatAgentOnMessagesStreamWrapper
    from opentelemetry import trace
    from wrapt import wrap_function_wrapper

    wrap_function_wrapper(
        "autogen_agentchat.agents._code_executor_agent",
        "CodeExecutorAgent.on_messages_stream",
        _BaseChatAgentOnMessagesStreamWrapper(instrumentor._tracer),
    )

    from autogen_agentchat.teams._group_chat._magentic_one import _magentic_one_orchestrator as mo

    tracer = trace.get_tracer("ag_common.magentic_one")
    phases = {
        "handle_start": "orchestrator.plan",
        "_update_task_ledger": "orchestrator.replan",
        "_orchestrate_step": "orchestrator.progress_ledger",
        "_prepare_final_answer": "orchestrator.final_answer",
    }
    for method, span_name in phases.items():
        original = getattr(mo.MagenticOneOrchestrator, method)

        def make(orig, name):
            @functools.wraps(orig)
            async def wrapper(self, *args, **kwargs):
                with tracer.start_as_current_span(name, attributes={"openinference.span.kind": "CHAIN"}):
                    return await orig(self, *args, **kwargs)

            return wrapper

        setattr(mo.MagenticOneOrchestrator, method, make(original, span_name))


def autogen_model(model: str = HAIKU, max_tokens: int = llm.MAX_TOKENS):
    """AutoGen Anthropic-native model client bound to the LiteLLM proxy.

    model_info is explicit because AutoGen does not know the 5.5 model ids. json_output and
    structured_output are False so no team forces a JSON response format (MagenticOne then
    parses JSON from plain text, as it does for non-JSON models)."""
    from autogen_core.models import ModelFamily
    from autogen_ext.models.anthropic import AnthropicChatCompletionClient

    client = AnthropicChatCompletionClient(
        model=model,
        base_url=llm.anthropic_base_url(),
        api_key=llm.proxy_key(),
        max_tokens=max_tokens,
        max_retries=4,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
            "family": ModelFamily.CLAUDE_4_SONNET,  # closest known Claude family; only used for capability checks
            "multiple_system_messages": False,
        },
    )
    _strip_sampling_params(client)
    return client


def _strip_sampling_params(client) -> None:
    """Make AutoGen's Anthropic requests valid for the 5.5 models, per client instance:

    - AutoGen always sends temperature=1.0. anthropic>=2 removed the sampling parameters from
      messages.create (and the models reject them), so drop temperature / top_p / top_k.
    - The models reject assistant prefill. In a Swarm, the agent that receives a handoff sees a context that
      ends with the previous agent's assistant turn ("Transferred to ..."), so append a short user turn.
    - Empty or whitespace-only text blocks (e.g. an empty agent reply forwarded by MagenticOne) are rejected;
      replace them with a placeholder.

    The class-level OpenInference wrapper still sees the call, because this calls the bound method."""
    messages_api = client._client.messages
    create = messages_api.create

    async def create_sanitized(*args, **kwargs):
        for k in ("temperature", "top_p", "top_k"):
            kwargs.pop(k, None)
        msgs = kwargs.get("messages")
        if isinstance(msgs, list) and msgs:
            kwargs["messages"] = _sanitize_messages(msgs)
        return await create(*args, **kwargs)

    messages_api.create = create_sanitized


def _sanitize_messages(msgs: list) -> list:
    out = []
    for m in msgs:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, str):
            if not content.strip():
                m["content"] = "(no content)"
        elif isinstance(content, list):
            blocks = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text" and not str(b.get("text", "")).strip():
                    b = {**b, "text": "(no content)"}
                blocks.append(b)
            m["content"] = blocks or "(no content)"
        out.append(m)
    if out[-1].get("role") == "assistant":
        out.append({"role": "user", "content": "Continue."})
    return out


def autogen_tools(*fns):
    """Wrap plain typed Python functions as AutoGen FunctionTools (docstring = description)."""
    from autogen_core.tools import FunctionTool

    return [FunctionTool(f, description=(f.__doc__ or f.__name__).strip(), name=f.__name__) for f in fns]


def final_text(result, prefer_source: str | None = None) -> str:
    """Pick the final answer from an AutoGen TaskResult: the last text message, optionally the
    last one from a given agent. Strips TERMINATE / APPROVE markers."""
    from autogen_agentchat.messages import BaseTextChatMessage

    msgs = [m for m in result.messages if isinstance(m, BaseTextChatMessage) and m.source != "user"]
    if prefer_source:
        own = [m for m in msgs if m.source == prefer_source]
        msgs = own or msgs
    if not msgs:
        raise RuntimeError(f"team produced no text answer (stop_reason={result.stop_reason!r})")
    text = msgs[-1].content
    for marker in ("TERMINATE", "APPROVE"):
        text = text.replace(marker, "")
    return text.strip()


# --------------------------------------------------------------------------------------------
# Agno
# --------------------------------------------------------------------------------------------
def setup_agno_tracing() -> None:
    """Agent, team, workflow, tool and LLM spans from the OpenInference Agno instrumentor.
    The Anthropic SDK is NOT instrumented for Agno agents: Agno's model wrapper is the single
    source of LLM spans."""
    from openinference.instrumentation.agno import AgnoInstrumentor

    tracing.instrument(AgnoInstrumentor())


def agno_claude(model: str = HAIKU, max_tokens: int = llm.MAX_TOKENS):
    """Agno Claude model bound to the LiteLLM proxy. No temperature, no thinking config."""
    from agno.models.anthropic import Claude

    return Claude(
        id=model,
        max_tokens=max_tokens,
        api_key=llm.proxy_key(),
        client_params={"base_url": llm.anthropic_base_url(), "max_retries": 4},
    )



def peira_python_executor():
    """An AutoGen CodeExecutor that runs Python blocks in the tools service sandbox (python_exec)
    instead of a local or Docker executor. Each execution is traced as an OpenInference TOOL span,
    because CodeExecutorAgent has no tool span of its own."""
    import asyncio
    import json
    from typing import List

    from autogen_core import CancellationToken
    from autogen_core.code_executor import CodeBlock, CodeExecutor, CodeResult
    from opentelemetry import trace

    from peira_harness import tools as T

    tracer = trace.get_tracer("ag_common.python_exec")

    class PeiraPythonExecutor(CodeExecutor):
        async def execute_code_blocks(
            self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
        ) -> CodeResult:
            outputs: list[str] = []
            exit_code = 0
            for block in code_blocks:
                if block.language.lower() not in ("python", "py", "python3", ""):
                    outputs.append(f"unsupported language {block.language!r}: only python runs in this sandbox")
                    exit_code = 1
                    continue
                with tracer.start_as_current_span(
                    "python_exec",
                    attributes={
                        "openinference.span.kind": "TOOL",
                        "tool.name": "python_exec",
                        "input.value": block.code,
                        "input.mime_type": "text/plain",
                    },
                ) as span:
                    res = await asyncio.to_thread(T.python_exec, block.code)
                    text, code = _python_exec_result(res)
                    span.set_attribute("output.value", text if isinstance(text, str) else json.dumps(text))
                outputs.append(text)
                if code != 0:
                    exit_code = code
                    break
            return CodeResult(exit_code=exit_code, output="\n".join(outputs))

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def restart(self) -> None:
            return None

    return PeiraPythonExecutor()


def _python_exec_result(res) -> tuple[str, int]:
    """Normalize the python_exec tool result to (output text, exit code)."""
    if isinstance(res, dict):
        out = "".join(str(res.get(k) or "") for k in ("stdout", "stderr", "error"))
        code = res.get("exit_code")
        if code is None:
            code = 1 if (res.get("error") or res.get("stderr")) and not res.get("stdout") else 0
        return out.strip() or "(no output)", int(code)
    return str(res), 0


def agno_output(response) -> str:
    """Return the text of an Agno run/team/workflow output. Agno records failures as status=ERROR
    instead of raising; the contract says to raise, so the runner marks the trace as failed."""
    status = str(getattr(response, "status", "") or "")
    content = getattr(response, "content", None)
    if status.endswith("ERROR") or status.endswith("error"):
        raise RuntimeError(f"agno run failed: {content}")
    if content is None or (isinstance(content, str) and not content.strip()):
        raise RuntimeError(f"agno run returned no content (status={status})")
    if not isinstance(content, str):
        try:
            return content.model_dump_json(indent=2)
        except AttributeError:
            return str(content)
    return content


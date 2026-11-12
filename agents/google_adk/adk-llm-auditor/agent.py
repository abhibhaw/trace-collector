"""LLM Auditor: SequentialAgent(critic -> reviser) that fact-checks an LLM answer.

Derived from google/adk-samples contrib/python/llm-auditor.
Google Search grounding is replaced by peira_harness web_search + fetch_url. The critic's
grounding-reference callback is rebuilt from the web_search results that the tool callback
collects into session state.
"""
from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

import adk_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

prompts = C.load_local(__file__)

AGENT = AgentMeta(
    id="adk-llm-auditor",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/b4fa327b91a1a2c3d710f7476ea723f42d1c717c/contrib/python/llm-auditor",
    description="SequentialAgent critic->reviser that verifies claims in an LLM answer with web search and minimally rewrites it.",
    complexity="simple",
    model=C.HAIKU,
    tracing="openinference",
)

_END_OF_EDIT_MARK = "---END-OF-EDIT---"


def _collect_references(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any):
    """after_tool_callback: remember every source the critic looked at (replaces grounding_chunks)."""
    refs = dict(tool_context.state.get("references", {}))
    if tool.name == "web_search" and isinstance(tool_response, dict):
        for r in tool_response.get("results", [])[:5]:
            if r.get("url"):
                refs.setdefault(r["url"], r.get("title") or r["url"])
    elif tool.name == "fetch_url" and isinstance(tool_response, dict) and tool_response.get("url"):
        refs.setdefault(tool_response["url"], tool_response.get("title") or tool_response["url"])
    tool_context.state["references"] = refs
    return None


def _render_reference(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse:
    """after_model_callback: append the consulted references to the critic's final findings."""
    content = llm_response.content
    if not content or not content.parts or any(p.function_call for p in content.parts):
        return llm_response
    refs = callback_context.state.get("references") or {}
    if refs:
        lines = "".join(f"* [{title}]({url})\n" for url, title in list(refs.items())[:12])
        content.parts.append(types.Part(text="\n\nReference:\n\n" + lines))
    if all(p.text is not None for p in content.parts):
        content.parts[0].text = "\n".join(p.text for p in content.parts)
        del content.parts[1:]
    return llm_response


def _remove_end_of_edit_mark(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse:
    del callback_context
    if not llm_response.content or not llm_response.content.parts:
        return llm_response
    for idx, part in enumerate(llm_response.content.parts):
        if part.text and _END_OF_EDIT_MARK in part.text:
            del llm_response.content.parts[idx + 1 :]
            part.text = part.text.split(_END_OF_EDIT_MARK, 1)[0]
            break
    return llm_response


critic_agent = LlmAgent(
    model=C.claude(C.HAIKU),
    name="critic_agent",
    instruction=prompts.CRITIC_PROMPT.replace("search the web", "search the web with the web_search tool (and fetch_url for a page)")
    + "\nLimit yourself to at most 6 tool calls in total.\n",
    tools=[C.AT.web_search, C.AT.fetch_url],
    after_tool_callback=_collect_references,
    after_model_callback=_render_reference,
    output_key="critic_findings",
)

reviser_agent = LlmAgent(
    model=C.claude(C.HAIKU),
    name="reviser_agent",
    instruction=prompts.REVISER_PROMPT,
    after_model_callback=_remove_end_of_edit_mark,
    output_key="revised_answer",
)

root_agent = SequentialAgent(
    name="llm_auditor",
    description=(
        "Evaluates LLM-generated answers, verifies actual accuracy using the"
        " web, and refines the response to ensure alignment with real-world"
        " knowledge."
    ),
    sub_agents=[critic_agent, reviser_agent],
)


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    return await C.run_agent(root_agent, task, app_name="llm_auditor", max_llm_calls=12)

"""Software bug triage assistant: one agent with ticket-database tools, a web-search sub-agent
exposed through AgentTool, and a Stack Exchange search tool.

Derived from google/adk-samples python/agents/software-bug-assistant (removed from main in #2591; pinned).
Adaptations: MCP Toolbox ticket tools (Postgres) -> tools-service tickets_search/get/update plus
the KB and employee directory; Google Search sub-agent -> web_search + fetch_url sub-agent;
LangChain StackExchangeTool -> web_search restricted to stackoverflow/stackexchange; the GitHub
MCP toolset (needs a token) is dropped.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

import adk_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

prompts = C.load_local(__file__)

AGENT = AgentMeta(
    id="adk-software-bug-assistant",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/software-bug-assistant",
    description="Bug triage agent over the support-ticket database with KB lookup, a web-search sub-agent tool and Stack Exchange search.",
    complexity="simple",
    model=C.HAIKU,
    tracing="openinference",
)


def get_current_date() -> dict:
    """
    Get the current date in the format YYYY-MM-DD
    """
    # The ticket database lives in the tools-service mock world, whose "today" is fixed.
    return {"current_date": C.MOCK_TODAY}


def stack_exchange(query: str) -> dict:
    """Search Stack Exchange (StackOverflow, SuperUser, ...) for past questions and answers.

    Args:
        query: The technical question or error message.

    Returns:
        A dict with matching questions (title, url, snippet).
    """
    res = T.web_search(query=f"{query} site:stackoverflow.com OR site:stackexchange.com", max_results=5)
    if "error" in res:
        return res
    return {"query": query, "results": res.get("results", []), "source": res.get("source")}


search_agent = LlmAgent(
    model=C.claude(C.HAIKU),
    name="search_agent",
    description="Searches the web for known issues, CVEs and vendor advisories.",
    instruction="""
    You're a specialist in web search. Use web_search (and fetch_url for one key page if needed),
    at most 3 tool calls, and answer concisely with source URLs.
    """,
    tools=[C.AT.web_search, C.AT.fetch_url],
)

root_agent = LlmAgent(
    model=C.claude(C.HAIKU),
    name="software_assistant",
    instruction=prompts.agent_instruction,
    tools=[
        get_current_date,
        AgentTool(search_agent),
        C.async_tool(stack_exchange),
        C.AT.tickets_search,
        C.AT.tickets_get,
        C.AT.tickets_update,
        C.AT.kb_search,
        C.AT.kb_get_article,
        C.AT.hr_search_employees,
    ],
)


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    return await C.run_agent(root_agent, task, app_name="software_bug_assistant", max_llm_calls=15)

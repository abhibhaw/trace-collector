"""Manager CodeAgent + managed web-search ToolCallingAgent, from the smolagents multi-agent tutorial.

Source shape (docs/source/en/examples/multiagents.md):
    manager CodeAgent (code interpreter, no tools) -> managed ToolCallingAgent "web_search_agent"
    with a web search tool and a hand-written `visit_webpage` tool.
Adaptation: WebSearchTool is replaced by the tools-service web search, `visit_webpage` calls the
tools-service fetch (markdown-ish text). Authorized imports are reduced to safe stdlib modules.
"""
from __future__ import annotations

import re
import uuid

import smol_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T
from smolagents import CodeAgent, ToolCallingAgent, tool

AGENT = AgentMeta(
    id="smol-web-multiagent",
    framework="smolagents",
    source_url="https://github.com/huggingface/smolagents/blob/main/docs/source/en/examples/multiagents.md",
    description="Manager CodeAgent that delegates web research to a managed web-search ToolCallingAgent and computes the answer in code.",
    complexity="medium",
    model="claude-haiku-5-5",
    tracing="openinference",
)


@tool
def web_search(query: str) -> str:
    """Performs a web search for your query, then returns a string of the top search results.

    Args:
        query: The search query to perform.
    """
    res = T.web_search(query, max_results=6)
    if "error" in res:
        return f"Search error: {res['error']}"
    lines = [f"## Search Results ({res.get('source', 'web')})"]
    for r in res.get("results", []):
        lines.append(f"[{r.get('title')}]({r.get('url')})\n{r.get('snippet', '')}")
    return "\n\n".join(lines) if len(lines) > 1 else "No results found. Try a less restrictive query."


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.
    """
    res = T.fetch_url(url, max_chars=8000)
    if "error" in res:
        return f"Error fetching the webpage: {res['error']}"
    content = re.sub(r"\n{3,}", "\n\n", res.get("content", "")).strip()
    return f"# {res.get('title', '')}\n\n{content}"


def setup() -> None:
    C.setup_tracing()


def build() -> CodeAgent:
    model = C.make_model(AGENT.model)
    web_agent = ToolCallingAgent(
        tools=[web_search, visit_webpage],
        model=model,
        max_steps=4,
        name="web_search_agent",
        description="Runs web searches for you. Give it your query as an argument.",
        verbosity_level=0,
    )
    return CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent],
        additional_authorized_imports=["math", "statistics", "datetime"],
        max_steps=5,
        verbosity_level=0,
        # Managed agents run inside the code executor: lift its 30 s default.
        executor_kwargs={"timeout_seconds": 300},
    )


def run(task: dict) -> str:
    T.set_run_id(f"{AGENT.id}:{task['id']}:{uuid.uuid4().hex[:8]}")
    return C.last_text(build().run(task["input"]))

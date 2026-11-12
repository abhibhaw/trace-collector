"""Academic research coordinator with two AgentTool specialists.

Derived from google/adk-samples python/agents/academic-research (removed from main in #2591;
pinned to the last commit that has it). The PDF upload is replaced by arxiv_search / fetch_url
lookups of the seminal paper; Google Search is replaced by arxiv_search + web_search.
"""
from __future__ import annotations

import datetime

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

import adk_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

prompts = C.load_local(__file__)

AGENT = AgentMeta(
    id="adk-academic-research",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/academic-research",
    description="Coordinator analyzes a seminal paper, then calls a citing-paper search agent and a research-directions agent as tools.",
    complexity="simple",
    model=C.HAIKU,
    tracing="openinference",
)

academic_websearch_agent = LlmAgent(
    model=C.claude(C.HAIKU),
    name="academic_websearch_agent",
    description="Finds recent papers that cite or build on a given seminal paper (arXiv + web search).",
    instruction=prompts.ACADEMIC_WEBSEARCH_PROMPT,
    output_key="recent_citing_papers",
    tools=[C.AT.arxiv_search, C.AT.web_search],
)

academic_newresearch_agent = LlmAgent(
    model=C.claude(C.HAIKU),
    name="academic_newresearch_agent",
    description="Proposes novel future research directions from a seminal paper and its recent citing papers.",
    instruction=prompts.ACADEMIC_NEWRESEARCH_PROMPT,
)

root_agent = LlmAgent(
    name="academic_coordinator",
    model=C.claude(C.HAIKU),
    description=(
        "analyzing seminal papers provided by the users, "
        "providing research advice, locating current papers "
        "relevant to the seminal paper, generating suggestions "
        "for new research directions, and accessing web resources "
        "to acquire knowledge"
    ),
    instruction=prompts.ACADEMIC_COORDINATOR_PROMPT,
    output_key="seminal_paper",
    tools=[
        C.AT.arxiv_search,
        C.AT.fetch_url,
        AgentTool(agent=academic_websearch_agent),
        AgentTool(agent=academic_newresearch_agent),
    ],
)


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    state = {
        "current_date": datetime.date.today().isoformat(),
        # Templated into the sub-agent prompts; filled by the coordinator's output_key at the end.
        "seminal_paper": "the seminal paper named in the coordinator's request",
        "recent_citing_papers": "(see the request)",
    }
    return await C.run_agent(root_agent, task, app_name="academic_research", initial_state=state, max_llm_calls=20)

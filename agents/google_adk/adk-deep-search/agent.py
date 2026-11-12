"""Deep search: plan -> approve -> SequentialAgent(section planner, researcher,
LoopAgent(evaluator, escalation checker, follow-up searcher), cited report composer).

Derived from google/adk-samples core/python/deep-search. Adaptations:
- Google Search grounding -> peira_harness web_search / fetch_url. The grounding-metadata source
  collector is rebuilt as an after_tool_callback that assigns src-N ids to search results, so
  the citation-replacement callback still works.
- Human plan approval is simulated by a second user turn ("looks good, run it").
- Worker agents use claude-haiku-5-5; critic agents (evaluator, report composer) use
  claude-sonnet-5-5, mirroring the source's worker_model / critic_model split.
- research_evaluator: Sonnet rejects forced tool_choice, which ADK uses to implement output_schema
  on Claude, so the evaluator returns JSON text that an after_agent_callback validates against
  the Feedback model and stores in state.
- BuiltInPlanner(thinking_config) is removed (proxy forbids setting thinking budgets).
- LoopAgent max_iterations=2 and search caps keep a run under ~25 LLM calls.
"""
from __future__ import annotations

import datetime
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal
from urllib.parse import urlparse

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from pydantic import BaseModel, Field

import adk_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="adk-deep-search",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/b4fa327b91a1a2c3d710f7476ea723f42d1c717c/core/python/deep-search",
    description="Interactive research planner that, once the plan is approved, runs a sequential pipeline with an evaluate-and-refine LoopAgent and a cited report composer.",
    complexity="complex",
    model=C.SONNET,
    tracing="openinference",
)

WORKER = C.claude(C.HAIKU)
CRITIC = C.claude(C.SONNET)
MAX_SEARCH_ITERATIONS = 2
TODAY = datetime.date.today().strftime("%Y-%m-%d")


# --- Structured Output Models ---
class SearchQuery(BaseModel):
    """Model representing a specific search query for web search."""

    search_query: str = Field(description="A highly specific and targeted query for web search.")


class Feedback(BaseModel):
    """Model for providing evaluation feedback on research quality."""

    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if the research is sufficient, 'fail' if it needs revision."
    )
    comment: str = Field(
        description="Detailed explanation of the evaluation, highlighting strengths and/or weaknesses of the research."
    )
    follow_up_queries: list[SearchQuery] | None = Field(
        default=None,
        description="A list of specific, targeted follow-up search queries needed to fix research gaps. This should be null or empty if the grade is 'pass'.",
    )


# --- Callbacks ---
def collect_research_sources_callback(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> None:
    """Collects web sources from search/fetch results into state (replaces grounding_chunks).

    Each new URL gets a short id `src-N`; the composer cites with `<cite source="src-N"/>`.
    The ids are also injected into the tool response so the model can refer to them.
    """
    url_to_short_id = dict(tool_context.state.get("url_to_short_id", {}))
    sources = dict(tool_context.state.get("sources", {}))
    items: list[dict] = []
    if tool.name == "web_search" and isinstance(tool_response, dict):
        items = tool_response.get("results", [])
    elif tool.name == "fetch_url" and isinstance(tool_response, dict):
        items = [{"url": tool_response.get("url"), "title": tool_response.get("title"), "snippet": (tool_response.get("content") or "")[:200]}]
    for item in items:
        url = item.get("url")
        if not url:
            continue
        if url not in url_to_short_id:
            short_id = f"src-{len(url_to_short_id) + 1}"
            url_to_short_id[url] = short_id
            domain = urlparse(url).netloc
            sources[short_id] = {
                "short_id": short_id,
                "title": item.get("title") or domain,
                "url": url,
                "domain": domain,
                "supported_claims": [],
            }
        sid = url_to_short_id[url]
        if item.get("snippet"):
            sources[sid]["supported_claims"].append({"text_segment": item["snippet"][:240], "confidence": 0.5})
        item["source_id"] = sid
    tool_context.state["url_to_short_id"] = url_to_short_id
    tool_context.state["sources"] = sources
    return None


def parse_evaluation_callback(callback_context: CallbackContext) -> None:
    """Turns the evaluator's JSON text into a validated Feedback dict in state."""
    raw = callback_context.state.get("research_evaluation")
    if isinstance(raw, dict):
        return None
    try:
        fb = Feedback.model_validate(C.parse_json_block(raw or ""))
        callback_context.state["research_evaluation"] = fb.model_dump()
    except Exception as e:  # noqa: BLE001 - a malformed evaluation counts as "fail"
        logging.warning("research_evaluator returned invalid JSON: %s", e)
        callback_context.state["research_evaluation"] = {
            "grade": "fail",
            "comment": f"Evaluator output was not valid Feedback JSON: {e}",
            "follow_up_queries": [],
        }
    return None


def citation_replacement_callback(callback_context: CallbackContext) -> genai_types.Content:
    """Replaces citation tags in a report with Markdown-formatted links."""
    final_report = callback_context.state.get("final_cited_report", "")
    sources = callback_context.state.get("sources", {})

    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if not (source_info := sources.get(short_id)):
            logging.warning(f"Invalid citation tag found and removed: {match.group(0)}")
            return ""
        display_text = source_info.get("title", source_info.get("domain", short_id))
        return f" [{display_text}]({source_info['url']})"

    processed_report = re.sub(
        r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>',
        tag_replacer,
        final_report,
    )
    processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report)
    callback_context.state["final_report_with_citations"] = processed_report
    return genai_types.Content(parts=[genai_types.Part(text=processed_report)])


def remember_plan_callback(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any):
    """Keeps the latest plan from the plan_generator tool in state['research_plan']."""
    if tool.name == "plan_generator" and isinstance(tool_response, str) and tool_response.strip():
        tool_context.state["research_plan"] = tool_response
    return None


# --- Custom Agent for Loop Control ---
class EscalationChecker(BaseAgent):
    """Checks research evaluation and escalates to stop the loop if grade is 'pass'."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        evaluation_result = ctx.session.state.get("research_evaluation")
        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(f"[{self.name}] Research evaluation passed. Escalating to stop loop.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(f"[{self.name}] Research evaluation failed or not found. Loop will continue.")
            yield Event(author=self.name)


# --- AGENT DEFINITIONS ---
plan_generator = LlmAgent(
    model=WORKER,
    name="plan_generator",
    description="Generates or refine the existing 5 line action-oriented research plan, using minimal search only for topic clarification.",
    instruction=f"""
    You are a research strategist. Your job is to create a high-level RESEARCH PLAN, not a summary. If there is already a RESEARCH PLAN in the session state,
    improve upon it based on the user feedback.

    RESEARCH PLAN(SO FAR):
    {{ research_plan? }}

    **GENERAL INSTRUCTION: CLASSIFY TASK TYPES**
    Your plan must clearly classify each goal for downstream execution. Each bullet point should start with a task type prefix:
    - **`[RESEARCH]`**: For goals that primarily involve information gathering, investigation, analysis, or data collection (these require search tool usage by a researcher).
    - **`[DELIVERABLE]`**: For goals that involve synthesizing collected information, creating structured outputs (e.g., tables, charts, summaries, reports), or compiling final output artifacts (these are executed AFTER research tasks, often without further search).

    **INITIAL RULE: Your initial output MUST start with a bulleted list of 5 action-oriented research goals or key questions, followed by any *inherently implied* deliverables.**
    - All initial 5 goals will be classified as `[RESEARCH]` tasks.
    - A good goal for `[RESEARCH]` starts with a verb like "Analyze," "Identify," "Investigate."
    - A bad output is a statement of fact like "The event was in April 2024."
    - **Proactive Implied Deliverables (Initial):** If any of your initial 5 `[RESEARCH]` goals inherently imply a standard output or deliverable, you MUST add these as additional, distinct goals immediately after the initial 5, prefixed with `[DELIVERABLE][IMPLIED]`.

    **REFINEMENT RULE**:
    - **Integrate Feedback & Mark Changes:** When incorporating user feedback, make targeted modifications to existing bullet points. Add `[MODIFIED]` to the existing task type and status prefix (e.g., `[RESEARCH][MODIFIED]`). New goals get `[RESEARCH][NEW]` or `[DELIVERABLE][NEW]`.
    - **Maintain Order:** Strictly maintain the original sequential order of existing bullet points.

    **TOOL USE IS STRICTLY LIMITED:**
    Your goal is to create a generic, high-quality plan *without searching*.
    Only use `web_search` (at most once) if a topic is ambiguous or time-sensitive and you absolutely cannot create a plan without a key piece of identifying information.
    You are explicitly forbidden from researching the *content* or *themes* of the topic. That is the next agent's job.
    Current date: {TODAY}
    """,
    tools=[C.AT.web_search],
)


section_planner = LlmAgent(
    model=WORKER,
    name="section_planner",
    description="Breaks down the research plan into a structured markdown outline of report sections.",
    instruction="""
    You are an expert report architect. Using the research topic and the plan from the 'research_plan' state key, design a logical structure for the final report.
    Note: Ignore all the tag nanes ([MODIFIED], [NEW], [RESEARCH], [DELIVERABLE]) in the research plan.
    Your task is to create a markdown outline with 4-6 distinct sections that cover the topic comprehensively without overlap.
    Make sure your outline is clear and easy to follow.
    Do not include a "References" or "Sources" section in your outline. Citations will be handled in-line.

    RESEARCH PLAN:
    { research_plan? }
    """,
    output_key="report_sections",
)


section_researcher = LlmAgent(
    model=WORKER,
    name="section_researcher",
    description="Performs the crucial first pass of web research.",
    instruction="""
    You are a highly capable and diligent research and synthesis agent. Execute the research plan in the `research_plan` state key with fidelity:
    first gather information, then synthesize it.

    RESEARCH PLAN:
    { research_plan? }

    **Phase 1: Information Gathering (`[RESEARCH]` Tasks)**
    *   For each `[RESEARCH]` goal, formulate 1-2 targeted search queries and run them with the `web_search` tool.
        Issue several searches in parallel in one step. Use at most 8 web_search calls in total, and fetch_url at most twice for a key primary source.
    *   Summarize the results for each goal. Each search result carries a `source_id` (e.g. src-3); keep these ids next to the facts they support.

    **Phase 2: Synthesis and Output Creation (`[DELIVERABLE]` Tasks)**
    *   Only after all research goals: produce every `[DELIVERABLE]` artifact (tables, summaries) using ONLY the Phase 1 summaries. No new searches.

    **Final Output:** The complete set of research summaries (with source ids) AND all deliverable artifacts.
    """,
    tools=[C.AT.web_search, C.AT.fetch_url],
    output_key="section_research_findings",
    after_tool_callback=collect_research_sources_callback,
)

research_evaluator = LlmAgent(
    model=CRITIC,
    name="research_evaluator",
    description="Critically evaluates research and generates follow-up queries.",
    instruction=f"""
    You are a meticulous quality assurance analyst evaluating the research findings in 'section_research_findings'.

    RESEARCH FINDINGS:
    {{ section_research_findings? }}

    **CRITICAL RULES:**
    1. Assume the given research topic is correct. Do not question or try to verify the subject itself.
    2. Your ONLY job is to assess the quality, depth, and completeness of the research provided *for that topic*.
    3. Focus on evaluating: Comprehensiveness of coverage, logical flow and organization, use of credible sources, depth of analysis, and clarity of explanations.
    4. Do NOT fact-check or question the fundamental premise or timeline of the topic.
    5. If suggesting follow-up queries, they should dive deeper into the existing topic, not question its validity.

    Be critical about the QUALITY of research. If you find significant gaps in depth or coverage, assign a grade of "fail",
    write a detailed comment about what's missing, and generate 3-4 specific follow-up queries to fill those gaps.
    If the research thoroughly covers the topic, grade "pass".

    Current date: {TODAY}
    Your response must be a single, raw JSON object (no prose) validating against this schema:
    {{"grade": "pass" | "fail", "comment": string, "follow_up_queries": [{{"search_query": string}}] | null}}
    """,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="research_evaluation",
    after_agent_callback=parse_evaluation_callback,
)

enhanced_search_executor = LlmAgent(
    model=WORKER,
    name="enhanced_search_executor",
    description="Executes follow-up searches and integrates new findings.",
    instruction="""
    You are a specialist researcher executing a refinement pass.
    You have been activated because the previous research was graded as 'fail'.

    EVALUATION: { research_evaluation? }
    EXISTING FINDINGS: { section_research_findings? }

    1.  Review the evaluation to understand the feedback and required fixes.
    2.  Execute EVERY query listed in 'follow_up_queries' using the 'web_search' tool, in parallel in one step.
    3.  Synthesize the new findings and COMBINE them with the existing findings. Keep the `source_id` of every fact.
    4.  Your output MUST be the new, complete, and improved set of research findings.
    """,
    tools=[C.AT.web_search],
    output_key="section_research_findings",
    after_tool_callback=collect_research_sources_callback,
)

report_composer = LlmAgent(
    model=CRITIC,
    name="report_composer_with_citations",
    include_contents="none",
    description="Transforms research data and a markdown outline into a final, cited report.",
    instruction="""
    Transform the provided data into a polished, professional, and meticulously cited research report.

    ---
    ### INPUT DATA
    *   Research Plan: `{research_plan?}`
    *   Research Findings: `{section_research_findings?}`
    *   Citation Sources: `{sources?}`
    *   Report Structure: `{report_sections?}`

    ---
    ### CRITICAL: Citation System
    To cite a source, you MUST insert a special citation tag directly after the claim it supports.

    **The only correct format is:** `<cite source="src-ID_NUMBER" />`

    ---
    ### Final Instructions
    Generate a comprehensive report using ONLY the `<cite source="src-ID_NUMBER" />` tag system for all citations.
    The final report must strictly follow the structure provided in the **Report Structure** markdown outline.
    Do not include a "References" or "Sources" section; all citations must be in-line.
    """,
    output_key="final_cited_report",
    after_agent_callback=citation_replacement_callback,
)

research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Executes a pre-approved research plan. It performs iterative research, evaluation, and composes a final, cited report.",
    sub_agents=[
        section_planner,
        section_researcher,
        LoopAgent(
            name="iterative_refinement_loop",
            max_iterations=MAX_SEARCH_ITERATIONS,
            sub_agents=[
                research_evaluator,
                EscalationChecker(name="escalation_checker"),
                enhanced_search_executor,
            ],
        ),
        report_composer,
    ],
)

root_agent = LlmAgent(
    name="interactive_planner_agent",
    model=WORKER,
    description="The primary research assistant. It collaborates with the user to create a research plan, and then executes it upon approval.",
    instruction=f"""
    You are a research planning assistant. Your primary function is to convert ANY user request into a research plan.

    **CRITICAL RULE: Never answer a question directly or refuse a request.** Your one and only first step is to use the `plan_generator` tool to propose a research plan for the user's topic.
    If the user asks a question, you MUST immediately call `plan_generator` to create a plan to answer the question.

    Your workflow is:
    1.  **Plan:** Use `plan_generator` to create a draft plan and present it to the user.
    2.  **Refine:** Incorporate user feedback until the plan is approved.
    3.  **Execute:** Once the user gives EXPLICIT approval (e.g., "looks good, run it"), you MUST delegate the task to the `research_pipeline` agent, passing the approved plan.

    Current date: {TODAY}
    Do not perform any research yourself. Your job is to Plan, Refine, and Delegate.
    """,
    sub_agents=[research_pipeline],
    tools=[AgentTool(plan_generator)],
    after_tool_callback=remember_plan_callback,
)


def _approver(turn: int, last_text: str, state: dict) -> str | None:
    """Simulated human: approve the proposed plan once, then stop."""
    if state.get("final_report_with_citations"):
        return None
    if turn == 0:
        return "Looks good, run it."
    return None


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    return await C.run_agent(
        root_agent,
        task,
        app_name="deep_search",
        auto_user=_approver,
        max_turns=2,
        max_llm_calls=28,
        final_from_state="final_report_with_citations",
    )

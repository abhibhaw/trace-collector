"""Open Deep Research (langchain-ai/open_deep_research, src/open_deep_research/deep_researcher.py).

Kept from the source: the 4-stage top graph (clarify_with_user -> write_research_brief ->
research_supervisor subgraph -> final_report_generation), the supervisor subgraph (supervisor <->
supervisor_tools with ConductResearch / ResearchComplete / think_tool), parallel researcher
subgraphs launched with asyncio.gather (researcher <-> researcher_tools -> compress_research),
override_reducer state, iteration limits, overflow handling, and all prompts (odr_prompts.py).
Changed: Tavily (+ per-result summarisation) is replaced by keyless web search, URL fetch and
Wikipedia; MCP is dropped. Models follow ODR's split config: the research/supervisor and final
report models are claude-sonnet-5-5, the researcher and compression models are claude-haiku-5-5.
Limits are lowered to bound cost (about 25 LLM calls worst case). Structured outputs on Sonnet use
json_schema mode because the proxy forbids forced tool choice on Sonnet.
"""

import asyncio
import operator
from datetime import datetime
from typing import Annotated, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import lg_common as C
from odr_prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from peira_harness import AgentMeta

AGENT = AgentMeta(
    id="lg-open-deep-research",
    framework="langgraph",
    source_url="https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py",
    description="Deep research: clarify, brief, supervisor fanning out parallel researcher subgraphs, compression, final report.",
    complexity="complex",
    model=C.SONNET,
    tracing="langchain",
)


class Configuration:
    allow_clarification = True
    max_structured_output_retries = 2
    max_concurrent_research_units = 2
    max_researcher_iterations = 2
    max_react_tool_calls = 3
    research_model = C.SONNET  # supervisor + brief + clarify
    researcher_model = C.HAIKU  # the parallel research sub-agents
    compression_model = C.HAIKU
    final_report_model = C.SONNET
    mcp_prompt = ""


# --------------------------------------------------------------------- structured outputs
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""

    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""


class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(description="Whether the user needs to be asked a clarifying question.")
    question: str = Field(description="A question to ask the user to clarify the report scope")
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information."
    )


class ResearchQuestion(BaseModel):
    research_brief: str = Field(description="A research question that will be used to guide the research.")


# --------------------------------------------------------------------- state
def override_reducer(current_value, new_value):
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """InputState is only 'messages'."""


class AgentState(MessagesState):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str


class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherState(TypedDict):
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherOutputState(BaseModel):
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


# --------------------------------------------------------------------- tools
def _T():
    from peira_harness import tools as T

    return T


@tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps
    """
    return f"Reflection recorded: {reflection}"


@tool
def web_search(queries: list[str]) -> str:
    """Search the web. Pass 1-3 focused queries; results are grouped per query.

    Args:
        queries: List of search queries to execute
    """
    out = []
    for q in queries[:3]:
        try:
            out.append(f"--- QUERY: {q}\n{C.to_json(_T().web_search(q), 3500)}")
        except Exception as e:
            out.append(f"--- QUERY: {q}\nError: {e}")
    return "\n\n".join(out)


@tool
def fetch_url(url: str) -> str:
    """Fetch one web page and return its readable text (truncated)."""
    page = _T().fetch_url(url)
    return (page if isinstance(page, str) else C.to_json(page, 8000))[:8000]


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return the best matching article summaries."""
    return C.to_json(_T().wikipedia_search(query), 4000)


RESEARCH_TOOLS = [web_search, fetch_url, wikipedia_search, think_tool]


def get_today_str() -> str:
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


def get_notes_from_tool_calls(messages):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


# --------------------------------------------------------------------- top level nodes
async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    if not Configuration.allow_clarification:
        return Command(goto="write_research_brief")
    clarification_model = C.structured(C.model(Configuration.research_model), ClarifyWithUser).with_retry(
        stop_after_attempt=Configuration.max_structured_output_retries
    )
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(state["messages"]), date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    research_model = C.structured(C.model(Configuration.research_model), ResearchQuestion).with_retry(
        stop_after_attempt=Configuration.max_structured_output_retries
    )
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])), date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=Configuration.max_concurrent_research_units,
        max_researcher_iterations=Configuration.max_researcher_iterations,
    )
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [SystemMessage(content=supervisor_system_prompt), HumanMessage(content=response.research_brief)],
            },
        },
    )


# --------------------------------------------------------------------- supervisor subgraph
async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    research_model = (
        C.model(Configuration.research_model)
        .bind_tools([ConductResearch, ResearchComplete, think_tool])
        .with_retry(stop_after_attempt=Configuration.max_structured_output_retries)
    )
    response = await research_model.ainvoke(state.get("supervisor_messages", []))
    return Command(
        goto="supervisor_tools",
        update={"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1},
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    exceeded_allowed_iterations = research_iterations > Configuration.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls)
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={"notes": get_notes_from_tool_calls(supervisor_messages), "research_brief": state.get("research_brief", "")},
        )

    all_tool_messages = []
    update_payload: dict = {"supervisor_messages": []}
    for tool_call in [tc for tc in most_recent_message.tool_calls if tc["name"] == "think_tool"]:
        all_tool_messages.append(
            ToolMessage(content=f"Reflection recorded: {tool_call['args'].get('reflection', '')}", name="think_tool", tool_call_id=tool_call["id"])
        )

    conduct_research_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "ConductResearch"]
    if conduct_research_calls:
        allowed = conduct_research_calls[: Configuration.max_concurrent_research_units]
        overflow = conduct_research_calls[Configuration.max_concurrent_research_units :]
        research_tasks = [
            researcher_subgraph.ainvoke(
                {
                    "researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])],
                    "research_topic": tc["args"]["research_topic"],
                },
                config,
            )
            for tc in allowed
        ]
        tool_results = await asyncio.gather(*research_tasks)
        for observation, tc in zip(tool_results, allowed):
            all_tool_messages.append(
                ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tc["name"],
                    tool_call_id=tc["id"],
                )
            )
        for overflow_call in overflow:
            all_tool_messages.append(
                ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {Configuration.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"],
                )
            )
        raw_notes_concat = "\n".join("\n".join(o.get("raw_notes", [])) for o in tool_results)
        if raw_notes_concat:
            update_payload["raw_notes"] = [raw_notes_concat]

    # Any other tool name (e.g. a hallucinated one) still needs a tool result for the API.
    answered = {m.tool_call_id for m in all_tool_messages}
    for tc in most_recent_message.tool_calls:
        if tc["id"] not in answered:
            all_tool_messages.append(ToolMessage(content=f"Error: unknown tool {tc['name']}", name=tc["name"], tool_call_id=tc["id"]))

    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


# --------------------------------------------------------------------- researcher subgraph
async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    researcher_prompt = research_system_prompt.format(mcp_prompt=Configuration.mcp_prompt, date=get_today_str())
    research_model = (
        C.model(Configuration.researcher_model)
        .bind_tools(RESEARCH_TOOLS)
        .with_retry(stop_after_attempt=Configuration.max_structured_output_retries)
    )
    messages = [SystemMessage(content=researcher_prompt)] + state.get("researcher_messages", [])
    response = await research_model.ainvoke(messages)
    return Command(
        goto="researcher_tools",
        update={"researcher_messages": [response], "tool_call_iterations": state.get("tool_call_iterations", 0) + 1},
    )


async def execute_tool_safely(t, args, config):
    try:
        return await t.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    if not most_recent_message.tool_calls:
        return Command(goto="compress_research")
    tools_by_name = {t.name: t for t in RESEARCH_TOOLS}
    tool_calls = most_recent_message.tool_calls

    async def run_one(tc):
        t = tools_by_name.get(tc["name"])
        if t is None:
            return f"Error: unknown tool {tc['name']}"
        return await execute_tool_safely(t, tc["args"], config)

    observations = await asyncio.gather(*(run_one(tc) for tc in tool_calls))
    tool_outputs = [
        ToolMessage(content=str(obs), name=tc["name"], tool_call_id=tc["id"]) for obs, tc in zip(observations, tool_calls)
    ]
    exceeded_iterations = state.get("tool_call_iterations", 0) >= Configuration.max_react_tool_calls
    research_complete_called = any(tc["name"] == "ResearchComplete" for tc in tool_calls)
    if exceeded_iterations or research_complete_called:
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})


async def compress_research(state: ResearcherState, config: RunnableConfig):
    synthesizer_model = C.model(Configuration.compression_model)
    researcher_messages = list(state.get("researcher_messages", []))
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    compression_prompt = compress_research_system_prompt.format(date=get_today_str())
    response = await synthesizer_model.ainvoke([SystemMessage(content=compression_prompt)] + researcher_messages)
    raw_notes_content = "\n".join(
        str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])
    )
    return {"compressed_research": C.text(response), "raw_notes": [raw_notes_content]}


async def final_report_generation(state: AgentState, config: RunnableConfig):
    findings = "\n".join(state.get("notes", []))
    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        messages=get_buffer_string(state.get("messages", [])),
        findings=findings[:120_000],
        date=get_today_str(),
    )
    final_report = await C.model(Configuration.final_report_model, max_tokens=6000).ainvoke(
        [HumanMessage(content=final_report_prompt)]
    )
    return {
        "final_report": C.text(final_report),
        "messages": [final_report],
        "notes": {"type": "override", "value": []},
    }


researcher_subgraph = None
supervisor_subgraph = None
deep_researcher = None


def setup() -> None:
    global researcher_subgraph, supervisor_subgraph, deep_researcher
    rb = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    rb.add_node("researcher", researcher)
    rb.add_node("researcher_tools", researcher_tools)
    rb.add_node("compress_research", compress_research)
    rb.add_edge(START, "researcher")
    rb.add_edge("compress_research", END)
    researcher_subgraph = rb.compile(name="researcher")

    sb = StateGraph(SupervisorState)
    sb.add_node("supervisor", supervisor)
    sb.add_node("supervisor_tools", supervisor_tools)
    sb.add_edge(START, "supervisor")
    supervisor_subgraph = sb.compile(name="research_supervisor")

    db = StateGraph(AgentState, input_schema=AgentInputState)
    db.add_node("clarify_with_user", clarify_with_user)
    db.add_node("write_research_brief", write_research_brief)
    db.add_node("research_supervisor", supervisor_subgraph)
    db.add_node("final_report_generation", final_report_generation)
    db.add_edge(START, "clarify_with_user")
    db.add_edge("research_supervisor", "final_report_generation")
    db.add_edge("final_report_generation", END)
    deep_researcher = db.compile(name="deep_researcher")


async def arun(task: dict) -> str:
    result = await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content=task["input"])]}, config=C.run_config(recursion_limit=40)
    )
    if result.get("final_report"):
        return result["final_report"]
    # clarify_with_user ended the run with a question for the user
    return C.text(result["messages"][-1])

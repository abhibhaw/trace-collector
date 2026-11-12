"""Travel planning team on AutoGen AgentChat SelectorGroupChat.

Derived from the AgentChat "Travel Planning" example (planner, local, language and summary agents).
Adapted: the round-robin order is replaced by SelectorGroupChat with the selector prompt and a
candidate_func from the "Selector Group Chat" guide; a travel agent with flight and hotel search tools
and a company travel-policy lookup was added; local_agent grounds suggestions with Wikipedia and web search.
"""
from __future__ import annotations

from typing import List, Sequence

import ag_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="ag-travel-planner",
    framework="autogen",
    source_url="https://github.com/microsoft/autogen/blob/main/python/docs/src/user-guide/agentchat-user-guide/examples/travel-planning.ipynb",
    description="SelectorGroupChat of planner, flight/hotel booker, local guide, language coach and summarizer that builds a grounded trip plan.",
    complexity="medium",
    model=C.HAIKU,
    tracing="openinference",
)

SELECTOR_PROMPT = """Select an agent to perform the next task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner_agent has assigned tasks before other agents start working.
Pick travel_summary_agent only after flights/hotels and local suggestions are in the conversation.
Only select one agent.
"""


def setup() -> None:
    C.setup_autogen_tracing()


async def arun(task: dict) -> str:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
    from autogen_agentchat.teams import SelectorGroupChat

    model_client = C.autogen_model(C.HAIKU)
    try:
        planner_agent = AssistantAgent(
            "planner_agent",
            model_client=model_client,
            description="Plans the trip: breaks the request into subtasks and delegates them. Engages first.",
            system_message=(
                "You are a travel planning coordinator. Today is 2026-11-13. Break the request into subtasks and "
                "delegate them using the format '1. <agent> : <task>'. Team: travel_agent (flights, hotels, company "
                "travel policy), local_agent (activities, places), language_agent (language and etiquette tips), "
                "travel_summary_agent (final plan). You only plan and delegate. Point out impossible or contradictory "
                "constraints (past dates, unknown cities, budgets that cannot work) instead of ignoring them."
            ),
        )
        travel_agent = AssistantAgent(
            "travel_agent",
            model_client=model_client,
            description="Searches flights and hotels and checks the company travel policy for business trips.",
            tools=C.autogen_tools(T.travel_search_flights, T.travel_search_hotels, T.hr_policy_lookup),
            system_message=(
                "You are a travel booking agent. Use travel_search_flights and travel_search_hotels to find concrete options "
                "(airline, flight number, times, prices, hotel names, nightly rates). For Lumen & Co business trips "
                "check the current travel policy (travel-2026) and flag anything out of policy. Never claim to have "
                "booked or charged anything: you only search and recommend."
            ),
            reflect_on_tool_use=True,
            max_tool_iterations=3,
        )
        local_agent = AssistantAgent(
            "local_agent",
            model_client=model_client,
            description="Suggests authentic local activities and places to visit.",
            tools=C.autogen_tools(T.wikipedia_search, T.web_search),
            system_message=(
                "You are a helpful assistant that can suggest authentic and interesting local activities or places "
                "to visit for a user and can utilize any context information provided. Ground suggestions with one "
                "or two lookups."
            ),
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        language_agent = AssistantAgent(
            "language_agent",
            model_client=model_client,
            description="Provides language and communication tips for the destination.",
            system_message=(
                "You are a helpful assistant that can review travel plans, providing feedback on important/critical "
                "tips about how best to address language or communication challenges for the given destination. If "
                "the plan already includes language tips, you can mention that the plan is satisfactory, with rationale."
            ),
        )
        travel_summary_agent = AssistantAgent(
            "travel_summary_agent",
            model_client=model_client,
            description="Integrates all suggestions into the final detailed travel plan.",
            system_message=(
                "You are a helpful assistant that can take in all of the suggestions and advice from the other agents "
                "and provide a detailed final travel plan. You must ensure that the final plan is integrated and "
                "complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all "
                "perspectives are integrated, you can respond with TERMINATE."
            ),
        )
        agents = [planner_agent, travel_agent, local_agent, language_agent, travel_summary_agent]

        def candidate_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
            # The planner always speaks first; afterwards the model-based selector picks from everyone.
            if messages[-1].source == "user":
                return [planner_agent.name]
            return [a.name for a in agents]

        termination = TextMentionTermination("TERMINATE", sources=["travel_summary_agent"]) | MaxMessageTermination(9)
        team = SelectorGroupChat(
            agents,
            model_client=model_client,
            termination_condition=termination,
            selector_prompt=SELECTOR_PROMPT,
            candidate_func=candidate_func,
            allow_repeated_speaker=False,
        )
        result = await team.run(task=task["input"])
        return C.final_text(result, prefer_source="travel_summary_agent")
    finally:
        await model_client.close()

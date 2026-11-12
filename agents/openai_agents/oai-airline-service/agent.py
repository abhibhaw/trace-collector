"""Airline customer service: triage agent hands off to FAQ, seat booking, and flight search
agents that share a typed run context. A simulated customer answers follow-up questions so a
task can span several conversation turns.

Adapted from openai-agents-python examples/customer_service/main.py. Kept: AirlineAgentContext,
the faq_lookup_tool and update_seat tools, the on_seat_booking_handoff hook that sets the flight
number, the RECOMMENDED_PROMPT_PREFIX routines, handoffs back to triage, and the outer
conversation loop that carries `result.to_input_list()` and `result.last_agent` between turns.
Added: a Flight Search agent over the mock travel API, and a simulated customer (Haiku) in place
of the interactive input() so the loop runs unattended (max 3 turns).
"""

import random

from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    function_tool,
    handoff,
    trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel

from _oai_common import HAIKU, SONNET, model, refusal_as_answer, settings, setup_family, shared_tools
from peira_harness import AgentMeta

AGENT = AgentMeta(
    id="oai-airline-service",
    framework="openai-agents",
    source_url="https://github.com/openai/openai-agents-python/blob/main/examples/customer_service/main.py",
    description="Airline triage agent with handoffs to FAQ, seat booking (on_handoff hook + shared context) and flight search agents, over a multi-turn loop with a simulated customer.",
    complexity="complex",
    model=SONNET,
    tracing="openinference",
)

MAX_TURNS_PER_CONVERSATION = 3
TRAVEL_TOOLS: tuple[str, ...] = ("travel_search_flights",)


### CONTEXT


class AirlineAgentContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None


### TOOLS


@function_tool(name_override="faq_lookup_tool", description_override="Lookup frequently asked questions.")
async def faq_lookup_tool(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["bag", "baggage", "luggage", "carry-on", "hand luggage", "hand carry"]):
        return ("You are allowed to bring one bag on the plane. "
                "It must be under 50 pounds and 22 inches x 14 inches x 9 inches. "
                "A second checked bag costs $45 on domestic routes and $70 on international routes.")
    if any(k in q for k in ["seat", "seats", "seating", "plane"]):
        return ("There are 120 seats on the plane. "
                "There are 22 business class seats and 98 economy seats. "
                "Exit rows are rows 4 and 16. "
                "Rows 5-8 are Economy Plus, with extra legroom. ")
    if any(k in q for k in ["wifi", "internet", "wireless", "connectivity", "network", "online"]):
        return "We have free wifi on the plane, join Airline-Wifi"
    if any(k in q for k in ["pet", "dog", "cat"]):
        return "Small pets under 20 pounds may travel in-cabin in an approved carrier for a $95 fee. Book at least 48 hours ahead."
    if any(k in q for k in ["cancel", "refund", "change fee", "rebook"]):
        return ("Economy Basic fares are non-refundable. Standard and Flex fares can be changed without a fee; "
                "the fare difference applies. Cancellations within 24 hours of booking are fully refundable.")
    return "I'm sorry, I don't know the answer to that question."


@function_tool
async def update_seat(context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str) -> str:
    """
    Update the seat for a given confirmation number.

    Args:
        confirmation_number: The confirmation number for the flight.
        new_seat: The new seat to update to.
    """
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    assert context.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"


### HOOKS


async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    context.context.flight_number = f"FLT-{random.randint(100, 999)}"


### AGENTS

_a: dict = {}


def setup() -> None:
    setup_family()
    m, s = model(SONNET), settings()
    faq_agent = Agent[AirlineAgentContext](
        name="FAQ Agent",
        handoff_description="A helpful agent that can answer questions about the airline.",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
        tools=[faq_lookup_tool],
        model=m, model_settings=s,
    )
    seat_booking_agent = Agent[AirlineAgentContext](
        name="Seat Booking Agent",
        handoff_description="A helpful agent that can update a seat on a flight.",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is.
    3. Use the update seat tool to update the seat on the flight.
    If the customer asks a question that is not related to the routine, transfer back to the triage agent. """,
        tools=[update_seat],
        model=m, model_settings=s,
    )
    flight_search_agent = Agent[AirlineAgentContext](
        name="Flight Search Agent",
        handoff_description="A helpful agent that can search flights, fares and schedules between cities.",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a flight search agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Make sure you know the origin, destination and travel date. Ask for anything missing.
    2. Use the flight search tool. Do not rely on your own knowledge of schedules or prices.
    3. Present at most 3 options with times, stops and price. You cannot purchase tickets.
    If the customer asks a question that is not related to the routine, transfer back to the triage agent.""",
        tools=shared_tools(*TRAVEL_TOOLS),
        model=m, model_settings=s,
    )
    triage_agent = Agent[AirlineAgentContext](
        name="Triage Agent",
        handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
        instructions=(f"{RECOMMENDED_PROMPT_PREFIX} "
                      "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."),
        handoffs=[
            handoff(agent=faq_agent, tool_name_override="transfer_to_faq_agent"),
            handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff,
                    tool_name_override="transfer_to_seat_booking_agent"),
            handoff(agent=flight_search_agent, tool_name_override="transfer_to_flight_search_agent"),
        ],
        model=m, model_settings=s,
    )
    back = handoff(agent=triage_agent, tool_name_override="transfer_to_triage_agent")
    for a in (faq_agent, seat_booking_agent, flight_search_agent):
        a.handoffs.append(back)
    _a["triage"] = triage_agent
    _a["customer"] = Agent(
        name="Simulated customer",
        instructions=(
            "You play the airline customer in a chat. Your original message and the agent's latest reply "
            "are below. Answer the agent's question in one or two short sentences, using only facts from "
            "your original message. If the agent asks for something you never mentioned, say you don't "
            "have it. If your request is already handled, reply exactly: DONE"
        ),
        model=model(HAIKU),
        model_settings=settings(max_tokens=512),
    )


@refusal_as_answer
async def arun(task: dict) -> str:
    context = AirlineAgentContext()
    current: Agent = _a["triage"]
    items: list[TResponseInputItem] = [{"role": "user", "content": task["input"]}]
    transcript: list[str] = []
    with trace("Customer service"):
        for turn in range(MAX_TURNS_PER_CONVERSATION):
            result = await Runner.run(current, items, context=context, max_turns=8)
            reply = str(result.final_output)
            transcript.append(f"{result.last_agent.name}: {reply}")
            items = result.to_input_list()
            current = result.last_agent
            if turn == MAX_TURNS_PER_CONVERSATION - 1 or "?" not in reply:
                break
            sim = await Runner.run(_a["customer"],
                                   f"Original message: {task['input']}\n\nAgent reply: {reply}", max_turns=1)
            answer = str(sim.final_output).strip()
            if answer.upper().startswith("DONE"):
                break
            transcript.append(f"Customer: {answer}")
            items.append({"role": "user", "content": answer})
    ctx = context.model_dump(exclude_none=True)
    return "\n\n".join(transcript) + (f"\n\n[context: {ctx}]" if ctx else "")

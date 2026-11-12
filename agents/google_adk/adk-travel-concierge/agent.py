"""Travel concierge: root agent that transfers to six phase sub-agents (inspiration, planning,
booking, pre-trip, in-trip, post-trip), each with its own AgentTool specialists, structured
output agents (output_schema), a shared `memorize` state tool and a before_agent state loader.

Derived from google/adk-samples python/agents/travel-concierge (removed from main in #2591; pinned).
Adaptations:
- flight_search_agent / hotel_search_agent call the tools-service travel_search_flights /
  travel_search_hotels instead of having the LLM invent results; create_reservation books real
  mock offers with travel_book.
- Google Maps Grounding (poi_agent) -> wikipedia_search; google_search_grounding (pre_trip) ->
  an AgentTool search agent backed by web_search.
- temperature/top_p settings are removed (proxy rule). All agents use claude-haiku-5-5, which
  accepts the forced tool_choice ADK uses to implement output_schema on Claude.
- Arize instrumentation is replaced by the OpenInference instrumentor via the harness.
- The traveler is simulated: it answers clarification questions and approves selections.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

import adk_common as C
from peira_harness import AgentMeta

prompts = C.load_local(__file__)
lib = C.load_local(__file__, "tc_lib")

AGENT = AgentMeta(
    id="adk-travel-concierge",
    framework="google-adk",
    source_url="https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/travel-concierge",
    description="Travel concierge root that transfers among six trip-phase agents with AgentTool specialists, structured-output planners, mock flight/hotel search and booking.",
    complexity="complex",
    model=C.HAIKU,
    tracing="openinference",
)

M = C.claude(C.HAIKU)
memorize = lib.memorize

# --- inspiration -------------------------------------------------------------------------------
place_agent = LlmAgent(
    model=M,
    name="place_agent",
    instruction=prompts.PLACE_AGENT_INSTR,
    description="This agent suggests a few destination given some user preferences",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_schema=lib.DestinationIdeas,
    output_key="place",
)

poi_agent = LlmAgent(
    model=M,
    name="poi_agent",
    description="This agent suggests a few activities and points of interests given a destination",
    instruction=prompts.POI_AGENT_INSTR,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_schema=lib.POISuggestions,
    output_key="poi",
    tools=[C.AT.wikipedia_search],
)

inspiration_agent = LlmAgent(
    model=M,
    name="inspiration_agent",
    description="A travel inspiration agent who inspire users, and discover their next vacations; Provide information about places, activities, interests,",
    instruction=prompts.INSPIRATION_AGENT_INSTR,
    tools=[AgentTool(agent=place_agent), AgentTool(agent=poi_agent)],
)

# --- planning ----------------------------------------------------------------------------------
itinerary_agent = LlmAgent(
    model=M,
    name="itinerary_agent",
    description="Create and persist a structured JSON representation of the itinerary",
    instruction=prompts.ITINERARY_AGENT_INSTR,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_schema=lib.Itinerary,
    output_key="itinerary",
)

hotel_room_selection_agent = LlmAgent(
    model=M,
    name="hotel_room_selection_agent",
    description="Help users with the room choices for a hotel",
    instruction=prompts.HOTEL_ROOM_SELECTION_INSTR,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_schema=lib.RoomsSelection,
    output_key="room",
)

hotel_search_agent = LlmAgent(
    model=M,
    name="hotel_search_agent",
    description="Help users find hotel around a specific geographic area",
    instruction=prompts.HOTEL_SEARCH_INSTR,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_schema=lib.HotelsSelection,
    output_key="hotel",
    tools=[C.AT.travel_search_hotels],
)

flight_seat_selection_agent = LlmAgent(
    model=M,
    name="flight_seat_selection_agent",
    description="Help users with the seat choices",
    instruction=prompts.FLIGHT_SEAT_SELECTION_INSTR,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_schema=lib.SeatsSelection,
    output_key="seat",
)

flight_search_agent = LlmAgent(
    model=M,
    name="flight_search_agent",
    description="Help users find best flight deals",
    instruction=prompts.FLIGHT_SEARCH_INSTR,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_schema=lib.FlightsSelection,
    output_key="flight",
    tools=[C.AT.travel_search_flights],
)

planning_agent = LlmAgent(
    model=M,
    name="planning_agent",
    description="Helps users with travel planning, complete a full itinerary for their vacation, finding best deals for flights and hotels.",
    instruction=prompts.PLANNING_AGENT_INSTR,
    tools=[
        AgentTool(agent=flight_search_agent),
        AgentTool(agent=flight_seat_selection_agent),
        AgentTool(agent=hotel_search_agent),
        AgentTool(agent=hotel_room_selection_agent),
        AgentTool(agent=itinerary_agent),
        memorize,
    ],
)

# --- booking -----------------------------------------------------------------------------------
create_reservation = LlmAgent(
    model=M,
    name="create_reservation",
    description="Reserves an item that requires booking.",
    instruction=prompts.CONFIRM_RESERVATION_INSTR,
    tools=[C.AT.travel_book],
)

payment_choice = LlmAgent(
    model=M,
    name="payment_choice",
    description="Presents payment choices.",
    instruction=prompts.PAYMENT_CHOICE_INSTR,
)

process_payment = LlmAgent(
    model=M,
    name="process_payment",
    description="Executes a payment with the chosen method (simulated gateway).",
    instruction=prompts.PROCESS_PAYMENT_INSTR,
)

booking_agent = LlmAgent(
    model=M,
    name="booking_agent",
    description="Given an itinerary, complete the bookings of items by handling payment choices and processing.",
    instruction=prompts.BOOKING_AGENT_INSTR,
    tools=[
        AgentTool(agent=create_reservation),
        AgentTool(agent=payment_choice),
        AgentTool(agent=process_payment),
    ],
)

# --- pre-trip ----------------------------------------------------------------------------------
_search_grounding_agent = LlmAgent(
    model=M,
    name="google_search_grounding",
    description="An agent providing web search grounding capability",
    instruction="""
    Answer the user's question directly using the web_search tool (at most 2 searches);
    provide a brief but concise response. Rather than a detail response, provide the immediate
    actionable item for a tourist or traveler, in a single sentence.
    Do not ask the user to check or look up information for themselves, that's your role; do your best to be informative.
    """,
    tools=[C.AT.web_search],
)
google_search_grounding = AgentTool(agent=_search_grounding_agent)

what_to_pack_agent = LlmAgent(
    model=M,
    name="what_to_pack_agent",
    description="Make suggestion on what to bring for the trip",
    instruction=prompts.WHATTOPACK_INSTR,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="what_to_pack",
    output_schema=lib.PackingList,
)

pre_trip_agent = LlmAgent(
    model=M,
    name="pre_trip_agent",
    description="Given an itinerary, this agent keeps up to date and provides relevant travel information to the user before the trip.",
    instruction=prompts.PRETRIP_AGENT_INSTR,
    tools=[google_search_grounding, AgentTool(agent=what_to_pack_agent)],
)

# --- in-trip -----------------------------------------------------------------------------------
day_of_agent = LlmAgent(
    model=M,
    name="day_of_agent",
    description="Day_of agent is the agent handling the travel logistics of a trip.",
    instruction=lib.transit_coordination,
)

trip_monitor_agent = LlmAgent(
    model=M,
    name="trip_monitor_agent",
    description="Monitor aspects of a itinerary and bring attention to items that necessitate changes",
    instruction=prompts.TRIP_MONITOR_INSTR,
    tools=[lib.flight_status_check, lib.event_booking_check, lib.weather_impact_check],
    output_key="daily_checks",
)

in_trip_agent = LlmAgent(
    model=M,
    name="in_trip_agent",
    description="Provide information about what the users need as part of the tour.",
    instruction=prompts.INTRIP_INSTR,
    sub_agents=[trip_monitor_agent],
    tools=[AgentTool(agent=day_of_agent), memorize],
)

# --- post-trip ---------------------------------------------------------------------------------
post_trip_agent = LlmAgent(
    model=M,
    name="post_trip_agent",
    description="A follow up agent to learn from user's experience; In turn improves the user's future trips planning and in-trip experience.",
    instruction=prompts.POSTTRIP_INSTR,
    tools=[memorize],
)

root_agent = LlmAgent(
    model=M,
    name="root_agent",
    description="A Travel Conceirge using the services of multiple sub-agents",
    instruction=prompts.ROOT_AGENT_INSTR,
    sub_agents=[
        inspiration_agent,
        planning_agent,
        booking_agent,
        pre_trip_agent,
        in_trip_agent,
        post_trip_agent,
    ],
    before_agent_callback=lib._load_precreated_itinerary,
)


def _traveler(turn: int, last_text: str, state: dict) -> str | None:
    """Simulated traveler: answers questions with sensible defaults and approves choices."""
    if not C.asks_question(last_text):
        return None
    low = last_text.lower()
    if any(w in low for w in ("payment", "apple pay", "credit card")):
        return "Credit card on file, please. Go ahead and complete it."
    if any(w in low for w in ("seat", "room")):
        return "Window seat and a standard room are fine. Pick the best value and continue."
    if any(w in low for w in ("where", "origin", "flying from", "depart")):
        return "I will fly out of Los Angeles (LAX). Dates: 2026-12-10 to 2026-12-15, one traveler."
    return "Yes, that works. Choose the best-value option for me and continue to the next step."


def setup() -> None:
    C.setup_tracing()


async def arun(task: dict) -> str:
    return await C.run_agent(root_agent, task, app_name="travel_concierge", auto_user=_traveler, max_turns=3, max_llm_calls=28)

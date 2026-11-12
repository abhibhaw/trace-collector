"""Travel planner with human-in-the-loop booking approval (simulated by an auto-approver).

Pattern: deepagents human-in-the-loop (docs: https://docs.langchain.com/oss/python/deepagents/human-in-the-loop):
`interrupt_on` gates the booking and email tools, a checkpointer persists the paused run, and the
caller resumes with `Command(resume={"decisions": [...]})`. A deterministic approver policy plays
the human. Search is delegated to flight-scout and hotel-scout sub-agents (parallel `task` calls),
and the itinerary is kept in the virtual filesystem.
"""
import re

import da_common as C
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from peira_harness import AgentMeta

AGENT = AgentMeta(
    id="da-travel-planner",
    framework="deepagents",
    source_url="https://docs.langchain.com/oss/python/deepagents/human-in-the-loop",
    description="Trip planner that fans out flight and hotel search to scout sub-agents, writes /itinerary.md and books through interrupt_on approval gates answered by a simulated human approver.",
    complexity="medium",
    model=C.HAIKU,
    tracing="langchain",
)

TODAY = "2026-11-13"

SYSTEM_PROMPT = f"""You are a corporate travel planner for Lumen & Co employees. Today is {TODAY}.

Supported cities (IATA): New York JFK, Los Angeles LAX, Chicago ORD, San Francisco SFO, Seattle SEA, Austin AUS,
Boston BOS, Miami MIA, Denver DEN, Atlanta ATL, Toronto YYZ, Vancouver YVR, Mexico City MEX, Sao Paulo GRU,
Buenos Aires EZE, London LHR, Paris CDG, Frankfurt FRA, Berlin BER, Amsterdam AMS, Madrid MAD, Rome FCO, Zurich ZRH,
Copenhagen CPH, Dublin DUB, Lisbon LIS, Warsaw WAW, Istanbul IST, Dubai DXB, Mumbai BOM, Delhi DEL, Bengaluru BLR,
Singapore SIN, Hong Kong HKG, Tokyo NRT, Seoul ICN, Sydney SYD, Auckland AKL, Johannesburg JNB, Nairobi NBO.

Workflow:
1. Plan with write_todos. Resolve relative dates against today. If a required detail is missing (dates, origin,
   traveler name), make one reasonable assumption and state it; if the request cannot be met (unsupported city,
   past date), explain and offer the nearest alternative.
2. Delegate search with task(): flight-scout for flights (one call per direction or leg) and hotel-scout for hotels,
   in parallel where possible.
3. Write /itinerary.md: chosen options with offer ids, times, prices, total cost vs. budget, and runner-up options.
4. Book only if the user asked you to book. Use travel_book once per chosen offer. Every booking and email
   needs human approval; if a call is rejected, read the reason, adjust the plan, and do not retry the same call.
5. Send a confirmation with email_send only if the user asked for it.
6. Reply with the final itinerary and booking status (confirmation numbers or what was rejected and why).
"""

FLIGHT_SCOUT = """You find flights. Use travel_search_flights (at most 3 calls). Respect cabin, stops, time-of-day and
budget constraints in the task. Return the best 2-3 options per leg: offer_id, airline, depart/arrive local times,
stops, duration, price. Note when nothing fits and what you relaxed."""

HOTEL_SCOUT = """You find hotels. Use travel_search_hotels (at most 2 calls). Respect nightly budget, guests and any
location preference. Return the best 2-3 options: hotel_id, name, area, rating, nightly and total price, sold-out
nights. Note when nothing fits and what you relaxed."""

_agent = None
_checkpointer = InMemorySaver()


def setup() -> None:
    global _agent
    haiku = C.model(C.HAIKU)
    flight_scout = {
        "name": "flight-scout",
        "description": "Searches flights for one or more legs and returns the best options with offer ids and prices.",
        "system_prompt": FLIGHT_SCOUT,
        "tools": C.lc_tools("travel_search_flights"),
        "model": haiku,
        "middleware": [C.call_limit(4)],
    }
    hotel_scout = {
        "name": "hotel-scout",
        "description": "Searches hotels in one city for a date range and returns the best options with hotel ids and prices.",
        "system_prompt": HOTEL_SCOUT,
        "tools": C.lc_tools("travel_search_hotels"),
        "model": haiku,
        "middleware": [C.call_limit(3)],
    }
    _agent = create_deep_agent(
        model=haiku,
        tools=C.lc_tools("travel_book", "email_send"),
        system_prompt=SYSTEM_PROMPT,
        subagents=[flight_scout, hotel_scout, C.general_purpose(haiku, limit=3)],
        interrupt_on={
            "travel_book": {"allowed_decisions": ["approve", "edit", "reject"]},
            "email_send": {"allowed_decisions": ["approve", "reject"]},
        },
        checkpointer=_checkpointer,
        middleware=[C.todos(), C.call_limit(10, thread_limit=14)],
        name="travel-planner",
    )


_NO_BOOK = re.compile(r"\b(don'?t|do not|no need to|without)\s+(book|booking|reserv)|\b(just|only)\s+(compare|options|quote|looking)|\bhold off\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def approver(user_request: str, action: dict, booked_so_far: int) -> dict:
    """Simulated travel-desk approver. Deterministic policy, mirrors a cautious human reviewer."""
    name, args = action["name"], action.get("args", {})
    if name == "travel_book":
        if _NO_BOOK.search(user_request):
            return {"type": "reject", "message": "The traveler asked for options only. Do not book."}
        traveler = (args.get("traveler_name") or "").strip()
        if not traveler or traveler.lower() in {"traveler", "employee", "john doe", "jane doe", "user", "n/a"}:
            return {"type": "reject", "message": "Booking needs the traveler's real full name. Ask for it in your reply instead of booking."}
        if booked_so_far >= 4:
            return {"type": "reject", "message": "Policy: at most 4 bookings per trip request."}
        return {"type": "approve"}
    if name == "email_send":
        allowed = {e.lower() for e in _EMAIL.findall(user_request)}
        to = (args.get("to") or "").lower()
        if not to or to not in allowed:
            return {"type": "reject", "message": f"Emails may only go to an address the traveler gave. {to or 'empty'} was not given."}
        return {"type": "approve"}
    return {"type": "approve"}


def run(task: dict) -> str:
    thread_id = C.new_thread()
    result = C.invoke(_agent, task["input"], recursion_limit=200, thread_id=thread_id, run_name=AGENT.id)
    booked = 0
    for i in range(1, 5):
        interrupts = result.get("__interrupt__")
        if not interrupts:
            break
        decisions = []
        for action in interrupts[0].value["action_requests"]:
            d = approver(task["input"], action, booked)
            if d["type"] == "approve" and action["name"] == "travel_book":
                booked += 1
            decisions.append(d)
        result = _agent.invoke(
            Command(resume={"decisions": decisions}),
            config=C.config(200, thread_id, run_name=f"{AGENT.id}:resume-{i}"),
        )
    if result.get("__interrupt__"):
        raise RuntimeError("run still waiting for approval after 4 approval rounds")
    return C.final_answer(result, ["/itinerary.md"])

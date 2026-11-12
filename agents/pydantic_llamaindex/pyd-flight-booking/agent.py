"""Multi-agent flight booking, adapted from PydanticAI's flight_booking example.

Shape kept from the source:
- search_agent controls the flow and delegates to extraction_agent through the
  `extract_flights` tool, passing `usage=ctx.usage` so one UsageLimits covers every agent.
- search_agent returns FlightDetails | NoFlightFound, and an output validator checks
  origin, destination, and date against the request, raising ModelRetry on a mismatch.
- The outer loop re-runs search_agent with message history when the buyer declines.
- seat_preference_agent extracts SeatPreference | Failed, then tickets are bought.
Adaptations:
- A request_agent first turns the user's free-text request into the search constraints.
- The hard-coded web page is replaced by travel_search_flights results from the tools service.
- The human `Prompt.ask` prompts are replaced by a deterministic auto-approver (budget and
  refundability rules), and "buy" calls travel_book for a real (mock) booking.
"""

import datetime
from dataclasses import dataclass
from typing import Literal

from opentelemetry import trace
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelMessage, ModelRetry, RunContext, UsageLimits
from pydantic_ai.usage import RunUsage

import pl_common as pc
from peira_harness import AgentMeta, tools

AGENT = AgentMeta(
    id="pyd-flight-booking",
    framework="pydantic-ai",
    source_url="https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/flight_booking.py",
    description="Multi-agent flight booking: request parsing, search agent delegating to an extraction agent, output validation, auto-approval, seat choice, booking.",
    complexity="complex",
    model="claude-haiku-5-5",
    tracing="native-otel",
)

TODAY = datetime.date(2026, 11, 13)  # the tools service's fixed "today"
usage_limits = UsageLimits(request_limit=22)
SUPPORTED_CITIES = (
    "New York JFK, Los Angeles LAX, Chicago ORD, San Francisco SFO, Seattle SEA, Austin AUS, Boston BOS, Miami MIA, "
    "Denver DEN, Atlanta ATL, Toronto YYZ, Vancouver YVR, Mexico City MEX, Sao Paulo GRU, Buenos Aires EZE, London LHR, "
    "Paris CDG, Frankfurt FRA, Berlin BER, Amsterdam AMS, Madrid MAD, Rome FCO, Zurich ZRH, Copenhagen CPH, Dublin DUB, "
    "Lisbon LIS, Warsaw WAW, Istanbul IST, Dubai DXB, Mumbai BOM, Delhi DEL, Bengaluru BLR, Singapore SIN, Hong Kong HKG, "
    "Tokyo NRT, Seoul ICN, Sydney SYD, Auckland AKL, Johannesburg JNB, Nairobi NBO"
)
_tracer = trace.get_tracer("pyd-flight-booking")


class TripRequest(BaseModel):
    """The traveler's request, normalized."""

    origin: str = Field(description="Three-letter airport code of one of the supported cities")
    destination: str = Field(description="Three-letter airport code of one of the supported cities")
    date: datetime.date
    cabin: Literal["economy", "premium_economy", "business", "first"] = "economy"
    max_price_usd: int | None = Field(None, description="Budget per ticket, if the user gave one")
    max_stops: int = 1
    traveler_name: str | None = None
    seat_request: str | None = Field(None, description="The user's seat wishes in their own words, if any")
    wants_booking: bool = Field(description="True if the user asked to book or buy, False for search only")


class CannotParse(BaseModel):
    """The request is missing required information, is impossible, or is not a flight request."""

    reason: str


request_agent = Agent[None, TripRequest | CannotParse](
    pc.pyd_model(),
    name="request_agent",
    output_type=TripRequest | CannotParse,
    instructions=(
        f"Today is {TODAY.isoformat()}. Turn the user's travel request into search constraints. "
        "Resolve relative dates against today. Map each city to its airport code from this list of the 40 supported cities: "
        f"{SUPPORTED_CITIES}. "
        "If origin, destination, or date is missing, the date is in the past, or a city is not in the list, return CannotParse."
    ),
)


class FlightDetails(BaseModel):
    """Details of the most suitable flight."""

    offer_id: str
    flight_number: str
    price: int
    origin: str = Field(description="Three-letter airport code")
    destination: str = Field(description="Three-letter airport code")
    date: datetime.date
    stops: int = 0
    refundable: bool = False


class NoFlightFound(BaseModel):
    """When no valid flight is found."""

    reason: str = ""


@dataclass
class Deps:
    req: TripRequest
    rejected: list[str]


search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    pc.pyd_model(),
    name="search_agent",
    output_type=FlightDetails | NoFlightFound,
    deps_type=Deps,
    retries=4,
    instructions=(
        "Your job is to find the cheapest flight for the user on the given date that meets their constraints. "
        "Call extract_flights to get the available flights. Skip sold-out flights and offers the user already rejected."
    ),
)

extraction_agent = Agent(
    pc.pyd_model(),
    name="extraction_agent",
    output_type=list[FlightDetails],
    instructions="Extract all the flight details from the given search results. Skip flights with seats_left = 0.",
)


@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """Get details of all flights."""
    r = ctx.deps.req
    page = await pc.safe_tool(tools.travel_search_flights)(
        origin=r.origin, destination=r.destination, date=r.date.isoformat(), cabin=r.cabin,
        max_stops=r.max_stops, max_price_usd=float(r.max_price_usd or 0),
    )
    if page.startswith("ERROR"):
        raise ModelRetry(page)
    # we pass the usage to the search agent so requests within this agent are counted
    result = await extraction_agent.run(page, usage=ctx.usage)
    return [f for f in result.output if f.offer_id not in ctx.deps.rejected]


@search_agent.output_validator
async def validate_output(ctx: RunContext[Deps], output: FlightDetails | NoFlightFound) -> FlightDetails | NoFlightFound:
    """Procedural validation that the flight meets the constraints."""
    if isinstance(output, NoFlightFound):
        return output
    r = ctx.deps.req
    errors: list[str] = []
    if output.origin != r.origin:
        errors.append(f"Flight should have origin {r.origin}, not {output.origin}")
    if output.destination != r.destination:
        errors.append(f"Flight should have destination {r.destination}, not {output.destination}")
    if output.date != r.date:
        errors.append(f"Flight should be on {r.date}, not {output.date}")
    if output.offer_id in ctx.deps.rejected:
        errors.append(f"Offer {output.offer_id} was already rejected by the user")
    if errors:
        raise ModelRetry("\n".join(errors))
    return output


class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal["A", "B", "C", "D", "E", "F"]


class Failed(BaseModel):
    """Unable to extract a seat selection."""


seat_preference_agent = Agent[None, SeatPreference | Failed](
    pc.pyd_model(),
    name="seat_preference_agent",
    output_type=SeatPreference | Failed,
    instructions=(
        "Extract the user's seat preference. "
        "Seats A and F are window seats. "
        "Row 1 is the front row and has extra leg room. "
        "Rows 14, and 20 also have extra leg room. "
    ),
)


def auto_approver(req: TripRequest, flight: FlightDetails, attempt: int) -> tuple[str, str]:
    """Stands in for the human 'buy / search' prompt. Deterministic rules."""
    with _tracer.start_as_current_span("auto_approver") as span:
        decision, why = "buy", "within constraints"
        if req.max_price_usd and flight.price > req.max_price_usd:
            decision, why = "search", f"price {flight.price} is over the budget {req.max_price_usd}"
        elif attempt == 0 and not flight.refundable and flight.price > 900:
            decision, why = "search", "expensive non-refundable fare; look for an alternative once"
        span.set_attribute("decision", decision)
        span.set_attribute("reason", why)
        return decision, why


async def find_seat(req: TripRequest, usage: RunUsage) -> SeatPreference | None:
    answers = [req.seat_request or "any window seat please", "a window seat, any row"]
    message_history: list[ModelMessage] | None = None
    for answer in answers:
        result = await seat_preference_agent.run(
            answer, message_history=message_history, usage=usage, usage_limits=usage_limits
        )
        if isinstance(result.output, SeatPreference):
            return result.output
        message_history = result.all_messages()
    return None


def setup() -> None:
    pc.pyd_setup()


async def arun(task: dict) -> str:
    pc.begin_task(AGENT.id, task)
    usage = RunUsage()
    parsed = await request_agent.run(task["input"], usage=usage, usage_limits=usage_limits)
    if isinstance(parsed.output, CannotParse):
        return f"Cannot search: {parsed.output.reason}"
    req = parsed.output
    deps = Deps(req=req, rejected=[])
    message_history: list[ModelMessage] | None = None
    log: list[str] = []
    for attempt in range(3):
        result = await search_agent.run(
            f"Find me a flight from {req.origin} to {req.destination} on {req.date}",
            deps=deps, usage=usage, message_history=message_history, usage_limits=usage_limits,
        )
        if isinstance(result.output, NoFlightFound):
            log.append(f"No flight found. {result.output.reason}")
            break
        flight = result.output
        log.append(f"Candidate: {flight.flight_number} {flight.origin}-{flight.destination} {flight.date} ${flight.price} (offer {flight.offer_id})")
        if not req.wants_booking:
            log.append("Search only: not booking.")
            break
        decision, why = auto_approver(req, flight, attempt)
        log.append(f"Approver: {decision} ({why})")
        if decision == "buy":
            seat = await find_seat(req, usage)
            booking = await pc.safe_tool(tools.travel_book)(
                offer_id=flight.offer_id, traveler_name=req.traveler_name or "Traveler (name not given)"
            )
            seat_txt = f"row {seat.row} seat {seat.seat}" if seat else "no seat preference understood"
            log.append(f"Seat: {seat_txt}\nBooking: {booking}")
            break
        deps.rejected.append(flight.offer_id)
        message_history = result.all_messages(output_tool_return_content="Please suggest another flight")
    log.append(f"LLM requests: {usage.requests}")
    return "\n".join(log)

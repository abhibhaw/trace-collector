"""Travel-planning ToolCallingAgent, adapted from smolagents examples/multiple_tools.py.

Source shape: one ToolCallingAgent with many small `@tool` functions (get_weather, convert_currency,
get_time_in_timezone, search_wikipedia, ...) that the model combines to answer one request.
Adaptation: the keyed APIs (OpenWeather, exchangerate-api, worldtimeapi) are replaced by the
tools-service mock travel APIs (flight and hotel search, booking) plus local tools: a fixed-rate
currency converter, a timezone clock for the 40 supported cities, and a trip budget calculator.
A `final_answer_checks` validator makes sure any booking made during the run is reported with its
confirmation id. The model chooses tools freely (no forced tool choice).
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import smol_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T
from smolagents import ActionStep, ToolCallingAgent, tool

AGENT = AgentMeta(
    id="smol-travel-planner",
    framework="smolagents",
    source_url="https://github.com/huggingface/smolagents/blob/main/examples/multiple_tools.py",
    description="ToolCallingAgent that plans and books trips with mock flight/hotel APIs plus local currency, timezone and budget tools.",
    complexity="simple",
    model="claude-haiku-5-5",
    tracing="openinference",
)

AS_OF = "2026-11-13"

# Fixed reference rates (units of currency per 1 USD), so results are reproducible.
USD_RATES = {
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 151.0, "CAD": 1.37, "AUD": 1.53, "CHF": 0.88, "CNY": 7.2, "INR": 83.5,
    "MXN": 17.9, "BRL": 5.1, "SGD": 1.35, "HKD": 7.8, "KRW": 1360.0, "AED": 3.67, "TRY": 32.5, "ZAR": 18.6, "SEK": 10.6,
    "NOK": 10.8, "DKK": 6.9, "NZD": 1.66, "THB": 36.0, "PLN": 4.0, "CZK": 23.3, "ARS": 900.0, "COP": 3950.0, "EGP": 48.0,
    "KES": 129.0, "MAD": 10.0, "IDR": 15800.0,
}

CITY_TZ = {
    "new york": "America/New_York", "los angeles": "America/Los_Angeles", "chicago": "America/Chicago",
    "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles", "austin": "America/Chicago",
    "boston": "America/New_York", "miami": "America/New_York", "denver": "America/Denver", "atlanta": "America/New_York",
    "toronto": "America/Toronto", "vancouver": "America/Vancouver", "mexico city": "America/Mexico_City",
    "sao paulo": "America/Sao_Paulo", "buenos aires": "America/Argentina/Buenos_Aires", "bogota": "America/Bogota",
    "london": "Europe/London", "paris": "Europe/Paris", "berlin": "Europe/Berlin", "madrid": "Europe/Madrid",
    "rome": "Europe/Rome", "amsterdam": "Europe/Amsterdam", "lisbon": "Europe/Lisbon", "dublin": "Europe/Dublin",
    "zurich": "Europe/Zurich", "vienna": "Europe/Vienna", "stockholm": "Europe/Stockholm", "copenhagen": "Europe/Copenhagen",
    "istanbul": "Europe/Istanbul", "dubai": "Asia/Dubai", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
    "singapore": "Asia/Singapore", "hong kong": "Asia/Hong_Kong", "tokyo": "Asia/Tokyo", "seoul": "Asia/Seoul",
    "shanghai": "Asia/Shanghai", "bangkok": "Asia/Bangkok", "sydney": "Australia/Sydney", "auckland": "Pacific/Auckland",
    "cairo": "Africa/Cairo", "nairobi": "Africa/Nairobi", "cape town": "Africa/Johannesburg", "johannesburg": "Africa/Johannesburg",
}


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts an amount from one currency to another using fixed reference rates (for budgeting only).

    Args:
        amount: The amount of money to convert.
        from_currency: The currency code to convert from (e.g., 'USD', 'EUR').
        to_currency: The currency code to convert to (e.g., 'GBP', 'JPY').
    """
    f, t = from_currency.strip().upper(), to_currency.strip().upper()
    if f not in USD_RATES or t not in USD_RATES:
        return f"Error: unsupported currency {f if f not in USD_RATES else t}. Supported: {', '.join(sorted(USD_RATES))}"
    converted = amount / USD_RATES[f] * USD_RATES[t]
    return f"{amount:,.2f} {f} is about {converted:,.2f} {t} (reference rate 1 {f} = {USD_RATES[t] / USD_RATES[f]:.4f} {t})"


@tool
def get_time_in_timezone(location: str) -> str:
    """Returns the current local time and UTC offset in a city, in the mock world's 'today' (2026-11-13 12:00 US Central).

    Args:
        location: The city name (e.g., 'Tokyo', 'Paris').
    """
    tz = CITY_TZ.get(location.strip().lower())
    if not tz:
        return f"Unknown city '{location}'. Known: {', '.join(sorted(c.title() for c in CITY_TZ))}"
    now = datetime(2026, 11, 13, 12, 0, tzinfo=ZoneInfo("America/Chicago")).astimezone(ZoneInfo(tz))
    return f"Local time in {location.title()}: {now:%Y-%m-%d %H:%M} ({tz}, UTC{now:%z})"


@tool
def trip_budget(flight_usd: float, nightly_hotel_usd: float, nights: int, daily_spend_usd: float, travelers: int) -> str:
    """Totals a trip budget in USD: flights per traveler, hotel per room-night, and daily spending per traveler.

    Args:
        flight_usd: Round-trip or total flight price per traveler, in USD.
        nightly_hotel_usd: Hotel price per night for the whole party, in USD.
        nights: Number of nights.
        daily_spend_usd: Food and local transport per traveler per day, in USD.
        travelers: Number of travelers.
    """
    flights = flight_usd * travelers
    hotel = nightly_hotel_usd * nights
    daily = daily_spend_usd * travelers * max(nights, 1)
    total = flights + hotel + daily
    return (f"Flights {flights:,.2f} + hotel {hotel:,.2f} + daily spend {daily:,.2f} = total {total:,.2f} USD "
            f"({total / max(travelers, 1):,.2f} per traveler)")


def booking_reported(final_answer, memory, agent=None) -> bool:
    """If travel_book succeeded during the run, the answer must quote the confirmation id."""
    confirmations: set[str] = set()
    for step in memory.steps:
        if isinstance(step, ActionStep) and step.observations and "confirmation_id" in step.observations:
            confirmations.update(re.findall(r'"confirmation_id":\s*"([^"]+)"', step.observations))
    missing = [c for c in confirmations if c not in str(final_answer)]
    if missing:
        raise ValueError(f"You made bookings but did not report their confirmation ids: {', '.join(missing)}. Include them.")
    return True


def setup() -> None:
    C.setup_tracing()


def run(task: dict) -> str:
    T.set_run_id(f"{AGENT.id}:{task['id']}:{uuid.uuid4().hex[:8]}")
    agent = ToolCallingAgent(
        tools=[
            C.wrap(T.travel_search_flights, limit=5000),
            C.wrap(T.travel_search_hotels, limit=5000),
            C.wrap(T.travel_book),
            convert_currency,
            get_time_in_timezone,
            trip_budget,
        ],
        model=C.make_model(AGENT.model),
        max_steps=7,
        verbosity_level=0,
        final_answer_checks=[booking_reported],
        instructions=(
            f"You are a travel assistant. Today is {AS_OF}. Only book when the traveler explicitly asks you to book "
            "and gives a traveler name; otherwise just recommend options. Prices from the tools are in USD."
        ),
    )
    return C.last_text(agent.run(task["input"]))

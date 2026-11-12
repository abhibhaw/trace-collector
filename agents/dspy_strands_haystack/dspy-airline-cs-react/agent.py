"""DSPy airline customer-service agent (dspy.ReAct over mock airline + CRM tools).

Derived from the DSPy tutorial "Build AI Agents with DSPy" (customer_service_agent).
Kept: the pydantic data model (Date, UserProfile, Flight, Itinerary, Ticket), the seven tools with the
same names, roles and docstrings, the DSPyAirlineCustomerService signature, and a single dspy.ReAct.
Changed: flight search and booking call the tools-service travel API; user lookup calls the CRM;
itineraries and tickets live in a per-task in-memory store seeded with a few existing bookings.
"""

import random
import string

import dspy
import dsh_common
from peira_harness import AgentMeta, tools
from pydantic import BaseModel

AGENT = AgentMeta(
    id="dspy-airline-cs-react",
    framework="dspy",
    source_url="https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/customer_service_agent/index.ipynb",
    description="Airline customer-service dspy.ReAct agent that searches, books, looks up and cancels flights and files tickets.",
    complexity="medium",
    model="claude-haiku-5-5",
    tracing="openinference",
)


# ----------------------------------------------------------------------------- data model (from the tutorial)
class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`, so
    # we define a custom class to represent the date.
    year: int
    month: int
    day: int
    hour: int


class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str


class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float


class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight


class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile


# Existing bookings that tasks can refer to. Customers are real CRM records in the tools service.
SEED_ITINERARIES = [
    ("K7Q2MZ8D", "CUST-10001", "Arjun Johnson", "arjun.johnson7@icloud.com", "FL-YVR-SFO-20261204-1Y", Date(year=2026, month=12, day=4, hour=9), "YVR", "SFO", 2.5, 214.0),
    ("P3XW9LTA", "CUST-10002", "Zoe Harris", "zoe.harris@yahoo.com", "FL-FCO-LHR-20261215-2Y", Date(year=2026, month=12, day=15, hour=19), "FCO", "LHR", 2.7, 236.0),
    ("R8D4NC2E", "CUST-10003", "Lars O'Brien", "lars.obrien@yahoo.com", "FL-BOM-DXB-20270108-1Y", Date(year=2027, month=1, day=8, hour=7), "BOM", "DXB", 3.3, 288.0),
    ("Z5M1QH7B", "CUST-10004", "Zofia Tanaka", "zofia.tanaka19@gmail.com", "FL-FCO-NRT-20270212-1J", Date(year=2027, month=2, day=12, hour=13), "FCO", "NRT", 13.5, 3120.0),
    ("H2V6YB9K", "CUST-10005", "Grace Allen", "grace.allen@proton.me", "FL-AUS-MIA-20261220-3Y", Date(year=2026, month=12, day=20, hour=16), "AUS", "MIA", 3.4, 241.0),
    ("W9C3TF6N", "CUST-10009", "Sean Larsen", "sean.larsen@fastmail.com", "FL-MIA-JFK-20261230-2Y", Date(year=2026, month=12, day=30, hour=8), "MIA", "JFK", 3.1, 199.0),
    ("B4N8JR2P", "CUST-10012", "Mei Petrov", "mei.petrov@harbor-pinecafe.com", "FL-BER-CDG-20270115-1W", Date(year=2027, month=1, day=15, hour=11), "BER", "CDG", 1.9, 402.0),
]


def _generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def _profile_from_crm(c: dict) -> UserProfile:
    return UserProfile(user_id=c.get("customer_id", ""), name=c.get("name") or "", email=c.get("email") or "")


def build_tools():
    """Fresh tool closures and in-memory stores for one task (tasks run concurrently)."""
    itinerary_database: dict[str, Itinerary] = {}
    ticket_database: dict[str, Ticket] = {}

    for conf, cid, name, email, fid, dt, o, d, dur, price in SEED_ITINERARIES:
        itinerary_database[conf] = Itinerary(
            confirmation_number=conf,
            user_profile=UserProfile(user_id=cid, name=name, email=email),
            flight=Flight(flight_id=fid, date_time=dt, origin=o, destination=d, duration=dur, price=price),
        )

    def fetch_flight_info(date: Date, origin: str, destination: str):
        """Fetch flight information from origin to destination on the given date"""
        res = tools.travel_search_flights(
            origin=origin, destination=destination, date=f"{date.year:04d}-{date.month:02d}-{date.day:02d}"
        )
        if "error" in res:
            raise ValueError(res["error"])
        flights = []
        for f in res.get("flights", []):
            if f.get("seats_left", 0) == 0:
                continue
            dep = f["departure_local"]  # YYYY-MM-DDTHH:MM
            flights.append(
                Flight(
                    flight_id=f["offer_id"],
                    date_time=Date(year=int(dep[0:4]), month=int(dep[5:7]), day=int(dep[8:10]), hour=int(dep[11:13])),
                    origin=f["origin"],
                    destination=f["destination"],
                    duration=round(f["duration_minutes"] / 60, 2),
                    price=f["price_usd"],
                )
            )
        if len(flights) == 0:
            raise ValueError("No matching flight found!")
        return flights

    def fetch_itinerary(confirmation_number: str):
        """Fetch a booked itinerary information from database"""
        return itinerary_database.get(confirmation_number)

    def pick_flight(flights: list[Flight]):
        """Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties."""
        sorted_flights = sorted(
            flights,
            key=lambda x: (
                x.get("duration") if isinstance(x, dict) else x.duration,
                x.get("price") if isinstance(x, dict) else x.price,
            ),
        )
        return sorted_flights[0]

    def book_flight(flight: Flight, user_profile: UserProfile):
        """Book a flight on behalf of the user."""
        booking = tools.travel_book(
            offer_id=flight.flight_id, traveler_name=user_profile.name, traveler_email=user_profile.email
        )
        if "error" in booking:
            raise ValueError(booking["error"])
        confirmation_number = booking.get("confirmation_id") or _generate_id()
        itinerary_database[confirmation_number] = Itinerary(
            confirmation_number=confirmation_number,
            user_profile=user_profile,
            flight=flight,
        )
        return confirmation_number, itinerary_database[confirmation_number]

    def cancel_itinerary(confirmation_number: str, user_profile: UserProfile):
        """Cancel an itinerary on behalf of the user."""
        if confirmation_number in itinerary_database:
            del itinerary_database[confirmation_number]
            return
        raise ValueError("Cannot find the itinerary, please check your confirmation number.")

    def get_user_info(name: str):
        """Fetch the user profile from database with given name."""
        res = tools.crm_search_customers(query=name, limit=5)
        matches = res.get("customers", [])
        if not matches:
            return None
        return _profile_from_crm(matches[0])

    def file_ticket(user_request: str, user_profile: UserProfile):
        """File a customer support ticket if this is something the agent cannot handle."""
        ticket_id = _generate_id(length=6)
        ticket_database[ticket_id] = Ticket(
            user_request=user_request,
            user_profile=user_profile,
        )
        return ticket_id

    return [
        fetch_flight_info,
        fetch_itinerary,
        pick_flight,
        book_flight,
        cancel_itinerary,
        get_user_info,
        file_ticket,
    ]


class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent that helps user book and manage flights.

    You are given a list of tools to handle user request, and you should decide the right tool to use in order to
    fulfill users' request."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, e.g., the "
            "confirmation_number if a new flight is booked."
        )
    )


def setup() -> None:
    dsh_common.setup_dspy(AGENT.model)


def run(task: dict) -> str:
    tools.set_run_id(f"{AGENT.id}:{task['id']}")
    # max_iters bounds cost: at most 10 ReAct steps + 1 extract call.
    agent = dspy.ReAct(DSPyAirlineCustomerService, tools=build_tools(), max_iters=10)
    result = agent(user_request=task["input"])
    return result.process_result

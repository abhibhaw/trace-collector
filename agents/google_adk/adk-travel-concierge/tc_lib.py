"""Shared library for adk-travel-concierge, copied from google/adk-samples travel-concierge
(shared_libraries/types.py, shared_libraries/constants.py, tools/memory.py, sub_agents/in_trip/tools.py).
Apache-2.0. Imports adjusted for a flat layout; debug prints removed."""
import pathlib

# ---- constants
class constants:
    SYSTEM_TIME = "_time"
    ITIN_INITIALIZED = "_itin_initialized"
    ITIN_KEY = "itinerary"
    PROF_KEY = "user_profile"
    ITIN_START_DATE = "itinerary_start_date"
    ITIN_END_DATE = "itinerary_end_date"
    ITIN_DATETIME = "itinerary_datetime"
    START_DATE = "start_date"
    END_DATE = "end_date"

# ---- types

"""Common data schema and types for travel-concierge agents."""

from google.genai import types
from pydantic import BaseModel, Field

json_response_config = types.GenerateContentConfig(
    response_mime_type="application/json"
)


class Room(BaseModel):
    """A room for selection."""

    is_available: bool = Field(
        description="Whether the room type is available for selection."
    )
    price_in_usd: int = Field(description="The cost of the room selection.")
    room_type: str = Field(
        description="Type of room, e.g. Twin with Balcon, King with Ocean View... etc."
    )


class RoomsSelection(BaseModel):
    """A list of rooms for selection."""

    rooms: list[Room]


class Hotel(BaseModel):
    """A hotel from the search."""

    name: str = Field(description="Name of the hotel")
    address: str = Field(description="Full address of the Hotel")
    check_in_time: str = Field(description="Time in HH:MM format, e.g. 16:00")
    check_out_time: str = Field(description="Time in HH:MM format, e.g. 15:30")
    thumbnail: str = Field(description="Hotel logo location")
    price: int = Field(description="Price of the room per night")


class HotelsSelection(BaseModel):
    """A list of hotels from the search."""

    hotels: list[Hotel]


class Seat(BaseModel):
    """A Seat from the search."""

    is_available: bool = Field(
        description="Whether the seat is available for selection."
    )
    price_in_usd: int = Field(description="The cost of the seat selection.")
    seat_number: str = Field(description="Seat number, e.g. 22A, 34F... etc.")


class SeatsSelection(BaseModel):
    """A list of seats from the search."""

    seats: list[list[Seat]]


class AirportEvent(BaseModel):
    """An Airport event."""

    city_name: str = Field(description="Name of the departure city")
    airport_code: str = Field(description="IATA code of the departure airport")
    timestamp: str = Field(
        description="ISO 8601 departure or arrival date and time"
    )


class Flight(BaseModel):
    """A Flight search result."""

    flight_number: str = Field(
        description="Unique identifier for the flight, like BA123, AA31, etc."
    )
    departure: AirportEvent
    arrival: AirportEvent
    airlines: list[str] = Field(
        description="Airline names, e.g., American Airlines, Emirates"
    )
    airline_logo: str = Field(description="Airline logo location")
    price_in_usd: int = Field(description="Flight price in US dollars")
    number_of_stops: int = Field(
        description="Number of stops during the flight"
    )


class FlightsSelection(BaseModel):
    """A list of flights from the search."""

    flights: list[Flight]


class Destination(BaseModel):
    """A destination recommendation."""

    name: str = Field(description="A Destination's Name")
    country: str = Field(description="The Destination's Country Name")
    image: str = Field(
        description="verified URL to an image of the destination"
    )
    highlights: str = Field(
        description="Short description highlighting key features"
    )
    rating: str = Field(description="Numerical rating (e.g., 4.5)")


class DestinationIdeas(BaseModel):
    """Destinations recommendation."""

    places: list[Destination]


class POI(BaseModel):
    """A Point Of Interest suggested by the agent."""

    place_name: str = Field(description="Name of the attraction")
    address: str = Field(
        description="An address or sufficient information to geocode for a Lat/Lon"
    )
    lat: str = Field(
        description="Numerical representation of Latitude of the location (e.g., 20.6843)"
    )
    long: str = Field(
        description="Numerical representation of Longitude of the location (e.g., -88.5678)"
    )
    review_ratings: str = Field(
        description="Numerical representation of rating (e.g. 4.8 , 3.0 , 1.0 etc)"
    )
    highlights: str = Field(
        description="Short description highlighting key features"
    )
    image_url: str = Field(
        description="verified URL to an image of the destination"
    )
    map_url: str | None = Field(description="Verified URL to Google Map")
    place_id: str | None = Field(description="Google Map place_id")


class POISuggestions(BaseModel):
    """Points of interest recommendation."""

    places: list[POI]


class AttractionEvent(BaseModel):
    """An Attraction."""

    event_type: str = Field(default="visit")
    description: str = Field(
        description="A title or description of the activity or the attraction visit"
    )
    address: str = Field(description="Full address of the attraction")
    start_time: str = Field(description="Time in HH:MM format, e.g. 16:00")
    end_time: str = Field(description="Time in HH:MM format, e.g. 16:00")
    booking_required: bool = Field(default=False)
    price: str | None = Field(description="Some events may cost money")


class FlightEvent(BaseModel):
    """A Flight Segment in the itinerary."""

    event_type: str = Field(default="flight")
    description: str = Field(description="A title or description of the Flight")
    booking_required: bool = Field(default=True)
    departure_airport: str = Field(description="Airport code, i.e. SEA")
    arrival_airport: str = Field(description="Airport code, i.e. SAN")
    flight_number: str = Field(description="Flight number, e.g. UA5678")
    boarding_time: str = Field(description="Time in HH:MM format, e.g. 15:30")
    seat_number: str = Field(description="Seat Row and Position, e.g. 32A")
    departure_time: str = Field(description="Time in HH:MM format, e.g. 16:00")
    arrival_time: str = Field(description="Time in HH:MM format, e.g. 20:00")
    price: str | None = Field(description="Total air fare")
    booking_id: str | None = Field(
        description="Booking Reference ID, e.g LMN-012-STU"
    )


class HotelEvent(BaseModel):
    """A Hotel Booking in the itinerary."""

    event_type: str = Field(default="hotel")
    description: str = Field(
        description="A name, title or a description of the hotel"
    )
    address: str = Field(description="Full address of the attraction")
    check_in_time: str = Field(description="Time in HH:MM format, e.g. 16:00")
    check_out_time: str = Field(description="Time in HH:MM format, e.g. 15:30")
    room_selection: str = Field()
    booking_required: bool = Field(default=True)
    price: str | None = Field(
        description="Total hotel price including all nights"
    )
    booking_id: str | None = Field(
        description="Booking Reference ID, e.g ABCD12345678"
    )


class ItineraryDay(BaseModel):
    """A single day of events in the itinerary."""

    day_number: int = Field(
        description="Identify which day of the trip this represents, e.g. 1, 2, 3... etc."
    )
    date: str = Field(description="The Date this day YYYY-MM-DD format")
    events: list[FlightEvent | HotelEvent | AttractionEvent] = Field(
        default=[], description="The list of events for the day"
    )


class Itinerary(BaseModel):
    """A multi-day itinerary."""

    trip_name: str = Field(
        description="Simple one liner to describe the trip. e.g. 'San Diego to Seattle Getaway'"
    )
    start_date: str = Field(description="Trip Start Date in YYYY-MM-DD format")
    end_date: str = Field(description="Trip End Date in YYYY-MM-DD format")
    origin: str = Field(description="Trip Origin, e.g. San Diego")
    destination: str = (Field(description="Trip Destination, e.g. Seattle"),)
    days: list[ItineraryDay] = Field(
        default_factory=list, description="The multi-days itinerary"
    )


class UserProfile(BaseModel):
    """An example user profile."""

    allergies: list[str] = Field(
        default=[], description="A list of food allergies to avoid"
    )
    diet_preference: list[str] = Field(
        default=[], description="Vegetarian, Vegan... etc."
    )
    passport_nationality: str = Field(
        description="Nationality of traveler, e.g. US Citizen"
    )
    home_address: str = Field(description="Home address of traveler")
    home_transit_preference: str = Field(
        description="Preferred mode of transport around home, e.g. drive"
    )


class PackingList(BaseModel):
    """A list of things to pack for the trip."""

    items: list[str]


# ---- tolerance for structured output (adaptation, not in the source) -----------------------
# Haiku sometimes omits optional-looking fields (map_url, place_id, thumbnail...) in output_schema
# responses, and ADK then fails the whole run on pydantic validation. Keep the sample's schema
# shape, but give every non-identifying field a default so a partial answer still validates.
_REQUIRED_KEYS = {
    "Room": {"room_type"}, "Hotel": {"name"}, "Seat": {"seat_number"}, "AirportEvent": {"airport_code"},
    "Flight": {"flight_number"}, "Destination": {"name"}, "POI": {"place_name"},
    "AttractionEvent": {"description"}, "FlightEvent": {"flight_number"}, "HotelEvent": {"description"},
    "ItineraryDay": {"day_number"}, "Itinerary": {"trip_name"},
}


def _make_tolerant() -> None:
    import typing

    for cls_name, keep in _REQUIRED_KEYS.items():
        cls = globals()[cls_name]
        changed = False
        for fname, f in cls.model_fields.items():
            if fname in keep or not f.is_required():
                continue
            ann = f.annotation
            args = typing.get_args(ann)
            if type(None) in args:
                f.default = None
            elif ann is str:
                f.default = ""
            elif ann is bool:
                f.default = False
            elif ann is int:
                f.default = 0
            elif typing.get_origin(ann) is list:
                f.default_factory = list
            else:
                continue
            changed = True
        if changed:
            cls.model_rebuild(force=True)
    for cls in (RoomsSelection, HotelsSelection, SeatsSelection, FlightsSelection, DestinationIdeas, POISuggestions, Itinerary, PackingList):
        cls.model_rebuild(force=True)


_make_tolerant()

# ---- memory

"""The 'memorize' tool for several agents to affect session states."""

import json
import os
from datetime import datetime
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.sessions.state import State
from google.adk.tools import ToolContext



SAMPLE_SCENARIO_PATH = os.getenv(
    "TRAVEL_CONCIERGE_SCENARIO",
    str(pathlib.Path(__file__).with_name("profiles") / "itinerary_empty_default.json"),
)


def memorize_list(key: str, value: str, tool_context: ToolContext):
    """
    Memorize pieces of information.

    Args:
        key: the label indexing the memory to store the value.
        value: the information to be stored.
        tool_context: The ADK tool context.

    Returns:
        A status message.
    """
    mem_dict = tool_context.state
    if key not in mem_dict:
        mem_dict[key] = []
    if value not in mem_dict[key]:
        mem_dict[key].append(value)
    return {"status": f'Stored "{key}": "{value}"'}


def memorize(key: str, value: str, tool_context: ToolContext):
    """
    Memorize pieces of information, one key-value pair at a time.

    Args:
        key: the label indexing the memory to store the value.
        value: the information to be stored.
        tool_context: The ADK tool context.

    Returns:
        A status message.
    """
    mem_dict = tool_context.state
    mem_dict[key] = value
    return {"status": f'Stored "{key}": "{value}"'}


def forget(key: str, value: str, tool_context: ToolContext):
    """
    Forget pieces of information.

    Args:
        key: the label indexing the memory to store the value.
        value: the information to be removed.
        tool_context: The ADK tool context.

    Returns:
        A status message.
    """
    if tool_context.state[key] is None:
        tool_context.state[key] = []
    if value in tool_context.state[key]:
        tool_context.state[key].remove(value)
    return {"status": f'Removed "{key}": "{value}"'}


def _set_initial_states(source: dict[str, Any], target: State | dict[str, Any]):
    """
    Setting the initial session state given a JSON object of states.

    Args:
        source: A JSON object of states.
        target: The session state object to insert into.
    """
    if constants.SYSTEM_TIME not in target:
        target[constants.SYSTEM_TIME] = str(datetime.now())

    if constants.ITIN_INITIALIZED not in target:
        target[constants.ITIN_INITIALIZED] = True

        target.update(source)

        itinerary = source.get(constants.ITIN_KEY, {})
        if itinerary:
            target[constants.ITIN_START_DATE] = itinerary[constants.START_DATE]
            target[constants.ITIN_END_DATE] = itinerary[constants.END_DATE]
            target[constants.ITIN_DATETIME] = itinerary[constants.START_DATE]


def _load_precreated_itinerary(callback_context: CallbackContext):
    """
    Sets up the initial state.
    Set this as a callback as before_agent_call of the root_agent.
    This gets called before the system instruction is contructed.

    Args:
        callback_context: The callback context.
    """
    data = {}
    with open(SAMPLE_SCENARIO_PATH) as file:
        data = json.load(file)

    _set_initial_states(data["state"], callback_context.state)

# ---- in_trip tools

"""Tools for the in_trip, trip_monitor and day_of agents."""

from datetime import datetime
from typing import Any

from google.adk.agents.readonly_context import ReadonlyContext





def flight_status_check(
    flight_number: str, flight_date: str, checkin_time: str, departure_time: str
):
    """Checks the status of a flight, given its flight_number, date, checkin_time and departure_time."""

    return {"status": f"Flight {flight_number} checked"}


def event_booking_check(event_name: str, event_date: str, event_location: str):
    """Checks the status of an event that requires booking, given its event_name, date, and event_location."""

    if event_name.startswith(
        "Space Needle"
    ):  # Mocking an exception to illustrate
        return {"status": f"{event_name} is closed."}
    return {"status": f"{event_name} checked"}


def weather_impact_check(
    activity_name: str, activity_date: str, activity_location: str
):
    """
    Checks the status of an outdoor activity that may be impacted by weather, given its name, date, and its location.

    Args:
        activity_name: The name of the activity.
        activity_date: The date of the activity.
        activity_location: The location of the activity.

    Returns:
        A dictionary containing the status of the activity.
    """

    return {"status": f"{activity_name} checked"}


def get_event_time_as_destination(
    destin_json: dict[str, Any], default_value: str
):
    """Returns an event time appropriate for the location type."""
    match destin_json["event_type"]:
        case "flight":
            return destin_json["boarding_time"]
        case "hotel":
            return destin_json["check_in_time"]
        case "visit":
            return destin_json["start_time"]
        case _:
            return default_value


def parse_as_origin(origin_json: dict[str, Any]):
    """Returns a tuple of strings (origin, depart_by) appropriate for the starting location."""
    match origin_json["event_type"]:
        case "flight":
            return (
                origin_json["arrival_airport"] + " Airport",
                origin_json["arrival_time"],
            )
        case "hotel":
            return (
                origin_json["description"]
                + " "
                + origin_json.get("address", ""),
                "any time",
            )
        case "visit":
            return (
                origin_json["description"]
                + " "
                + origin_json.get("address", ""),
                origin_json["end_time"],
            )
        case "home":
            return (
                origin_json.get("local_prefer_mode")
                + " from "
                + origin_json.get("address", ""),
                "any time",
            )
        case _:
            return "Local in the region", "any time"


def parse_as_destin(destin_json: dict[str, Any]):
    """Returns a tuple of strings (destination, arrive_by) appropriate for the destination."""
    match destin_json["event_type"]:
        case "flight":
            return (
                destin_json["departure_airport"] + " Airport",
                "An hour before " + destin_json["boarding_time"],
            )
        case "hotel":
            return (
                destin_json["description"]
                + " "
                + destin_json.get("address", ""),
                "any time",
            )
        case "visit":
            return (
                destin_json["description"]
                + " "
                + destin_json.get("address", ""),
                destin_json["start_time"],
            )
        case "home":
            return (
                destin_json.get("local_prefer_mode")
                + " to "
                + destin_json.get("address", ""),
                "any time",
            )
        case _:
            return "Local in the region", "as soon as possible"


def find_segment(
    profile: dict[str, Any], itinerary: dict[str, Any], current_datetime: str
):
    """
    Find the events to travel from A to B
    This follows the itinerary schema in types.Itinerary.

    Since return values will be used as part of the prompt,
    there are flexibilities in what the return values contains.

    Args:
        profile: A dictionary containing the user's profile.
        itinerary: A dictionary containing the user's itinerary.
        current_datetime: A string containing the current date and time.

    Returns:
      from - capture information about the origin of this segment.
      to   - capture information about the destination of this segment.
      arrive_by - an indication of the time we shall arrive at the destination.
    """
    # Expects current_datetime is in '2024-03-15 04:00:00' format
    datetime_object = datetime.fromisoformat(current_datetime)
    current_date = datetime_object.strftime("%Y-%m-%d")
    current_time = datetime_object.strftime("%H:%M")
    event_date = current_date
    event_time = current_time




    # defaults
    origin_json = profile["home"]
    destin_json = profile["home"]

    leave_by = "No movement required"
    arrive_by = "No movement required"

    # Go through the itinerary to find where we are base on the current date and time
    for day in itinerary.get("days", []):
        event_date = day["date"]
        for event in day["events"]:
            # for every event we update the origin and destination until
            # we find one we need to pay attention
            origin_json = destin_json
            destin_json = event
            event_time = get_event_time_as_destination(
                destin_json, current_time
            )
            # The moment we find an event that's in the immediate future we stop to handle it
            print(
                event["event_type"],
                event_date,
                current_date,
                event_time,
                current_time,
            )
            if event_date >= current_date and event_time >= current_time:
                break
        else:  # if inner loop not break, continue
            continue
        break  # else break too.

    #
    # Construct prompt descriptions for travel_from, travel_to, arrive_by
    #
    travel_from, leave_by = parse_as_origin(origin_json)
    travel_to, arrive_by = parse_as_destin(destin_json)

    return (travel_from, travel_to, leave_by, arrive_by)


def _inspect_itinerary(state: dict[str:Any]):
    """Identifies and returns the itinerary, profile and current datetime from the session state."""

    itinerary = state[constants.ITIN_KEY]
    profile = state[constants.PROF_KEY]

    current_datetime = itinerary["start_date"] + " 00:00"
    if state.get(constants.ITIN_DATETIME, ""):
        current_datetime = state[constants.ITIN_DATETIME]

    return itinerary, profile, current_datetime


def transit_coordination(readonly_context: ReadonlyContext):
    """Dynamically generates an instruction for the day_of agent."""

    state = readonly_context.state

    # Inspecting the itinerary
    if not state.get(constants.ITIN_KEY):
        return _prompts().NEED_ITIN_INSTR

    itinerary, profile, current_datetime = _inspect_itinerary(state)
    travel_from, travel_to, leave_by, arrive_by = find_segment(
        profile, itinerary, current_datetime
    )










    return _prompts().LOGISTIC_INSTR_TEMPLATE.format(
        CURRENT_TIME=current_datetime,
        TRAVEL_FROM=travel_from,
        LEAVE_BY_TIME=leave_by,
        TRAVEL_TO=travel_to,
        ARRIVE_BY_TIME=arrive_by,
    )

def _prompts():
    import adk_common as C
    return C.load_local(__file__, 'prompts')

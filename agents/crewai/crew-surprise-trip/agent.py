"""Surprise trip crew, adapted from crewAI-examples/crews/surprise_trip.

Shape kept from the source: activity planner -> restaurant scout -> itinerary compiler, sequential, and
the final task returns the source's pydantic Itinerary (Activity / DayPlan / Itinerary) as structured output.
Adapted: SerperDevTool/ScrapeWebsiteTool are replaced with web search, Wikipedia, and URL fetch; the
itinerary compiler gets the mock travel API to price real flight and hotel offers.
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

from typing import List, Optional

from crewai import Crew, Process, Task
from pydantic import BaseModel, Field

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-surprise-trip",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/crews/surprise_trip",
    description="Sequential 3-agent travel crew that researches activities and restaurants and compiles a structured day-by-day itinerary with real flight and hotel offers.",
    complexity="medium",
    model=cc.HAIKU,
    tracing="openinference",
)


class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity")
    location: str = Field(..., description="Location of the activity")
    description: str = Field(..., description="Description of the activity")
    date: str = Field(..., description="Date of the activity")
    cuisine: Optional[str] = Field(None, description="Cuisine of the restaurant, if the activity is a meal")
    why_its_suitable: str = Field(..., description="Why it's suitable for the traveler")
    rating: Optional[float] = Field(None, description="Rating of the activity, if known")


class DayPlan(BaseModel):
    date: str = Field(..., description="Date of the day")
    activities: List[Activity] = Field(..., description="List of activities")
    restaurants: List[str] = Field(..., description="List of restaurants")
    flight: Optional[str] = Field(None, description="Flight information (offer id, airline, times, price)")


class Itinerary(BaseModel):
    name: str = Field(..., description="Name of the itinerary, something funny")
    day_plans: List[DayPlan] = Field(..., description="List of day plans")
    hotel: str = Field(..., description="Hotel information (hotel id, name, nightly price)")
    estimated_total_usd: Optional[float] = Field(None, description="Flights plus hotel total, if priced")
    caveats: List[str] = Field(default_factory=list, description="Missing info, assumptions, or conflicts in the request")


def setup() -> None:
    cc.setup_tracing()


def build_crew(run_id: str) -> Crew:
    research = cc.make_tools([T.web_search, T.wikipedia_search, T.fetch_url], run_id)
    planner = cc.mk_agent(
        role="Activity Planner",
        goal="Research and find cool things to do at the destination, including activities and events that match the "
        "traveler's interests and age group",
        backstory="You are skilled at creating personalized itineraries that cater to the specific preferences and "
        "demographics of travelers. " + cc.WORLD,
        tools=research,
        max_iter=4,
    )
    scout = cc.mk_agent(
        role="Restaurant Scout",
        goal="Find highly-rated restaurants and dining experiences at the destination, and recommend scenic locations "
        "and fun activities",
        backstory="As a food lover, you know the best spots in town for a delightful culinary experience. You also have "
        "a knack for finding picturesque and entertaining locations. " + cc.WORLD,
        tools=cc.make_tools([T.web_search, T.wikipedia_search], run_id),
        max_iter=3,
    )
    compiler = cc.mk_agent(
        role="Itinerary Compiler",
        goal="Compile all researched information into a comprehensive day-by-day itinerary, ensuring the integration "
        "of flights and hotel information",
        backstory="With an eye for detail, you organize all the information into a coherent and enjoyable travel plan. "
        "You price flights and hotels with the travel tools; you never book anything. " + cc.WORLD,
        tools=cc.make_tools([T.travel_search_flights, T.travel_search_hotels], run_id),
        max_iter=5,
    )
    traveler = "Traveler's request (origin, destination, dates, age, interests, budget):\n{request}"
    t1 = Task(
        description="Research and find cool things to do at the destination. Focus on activities and events that match "
        "the traveler's interests and age group. Use the search tools to gather the information.\n\n" + traveler,
        expected_output="A list of recommended activities and events for each day of the trip. Each entry should include "
        "the activity name, location, a brief description, and why it's suitable for the traveler.",
        agent=planner,
    )
    t2 = Task(
        description="Find highly-rated restaurants and dining experiences at the destination. Recommend scenic locations "
        "and fun activities that align with the traveler's preferences. Find a variety of options to suit different "
        "tastes and budgets.\n\n" + traveler,
        expected_output="A list of recommended restaurants, scenic locations, and fun activities for each day of the trip, "
        "with name, location, type of cuisine or activity, and a brief description.",
        agent=scout,
    )
    t3 = Task(
        description="Compile all researched information into a comprehensive day-by-day itinerary for the trip. Search "
        "flights (outbound and return) and hotels with the travel tools for the requested dates and pick sensible "
        "offers within the budget. Integrate flights, hotel, and all planned activities and dining experiences. "
        "If the destination or dates are unsupported or missing, make the most reasonable assumption and list it "
        "in caveats.\n\n" + traveler,
        expected_output="A detailed itinerary with a day-by-day plan including flights, hotel details, activities, "
        "restaurants, and scenic locations.",
        agent=compiler,
        context=[t1, t2],
        output_pydantic=Itinerary,
    )
    return Crew(agents=[planner, scout, compiler], tasks=[t1, t2, t3], process=Process.sequential, verbose=False)


def run(task: dict) -> str:
    crew = build_crew(cc.run_id_for(AGENT.id, task))
    return cc.result_text(crew.kickoff(inputs={"request": task["input"]}))

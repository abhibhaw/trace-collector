# crew-surprise-trip

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/crews/surprise_trip

Compile all researched information into a comprehensive day-by-day itinerary for the trip. Search 

## What was adapted

Surprise trip crew, adapted from crewAI-examples/crews/surprise_trip.

Shape kept from the source: activity planner -> restaurant scout -> itinerary compiler, sequential, and
the final task returns the source's pydantic Itinerary (Activity / DayPlan / Itinerary) as structured output.
Adapted: SerperDevTool/ScrapeWebsiteTool are replaced with web search, Wikipedia, and URL fetch; the
itinerary compiler gets the mock travel API to price real flight and hotel offers.

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-haiku-5-5.
- Complexity: medium.
- Shared tools: fetch_url, travel_search_flights, travel_search_hotels, web_search, wikipedia_search.
- Local tools: none.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 10.7 (per task: 11, 10, 11).
Mean cost per run: $0.0198 (Langfuse trace cost).

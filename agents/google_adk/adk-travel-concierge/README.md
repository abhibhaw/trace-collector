# adk-travel-concierge

Source: https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/travel-concierge

Travel concierge root that transfers among six trip-phase agents with AgentTool specialists, structured-output planners, mock flight/hotel search and booking.

## Shape

root_agent transfers to 6 phase agents (inspiration, planning, booking, pre_trip, in_trip, post_trip); 11 AgentTool specialists, 6 output_schema agents, memorize state tool, before_agent itinerary loader, instruction-provider day_of agent

## Adaptations

- Flight and hotel search agents call `travel_search_flights` / `travel_search_hotels` instead of inventing results; `create_reservation` books mock offers with `travel_book`.
- Google Maps grounding in poi_agent -> `wikipedia_search`; `google_search_grounding` -> an AgentTool search agent backed by `web_search`.
- temperature/top_p removed; all agents on claude-haiku-5-5 (it accepts the forced tool_choice that ADK uses for output_schema).
- Shared types, memory tools and in-trip tools are copied into `tc_lib.py`; profiles copied to `profiles/`.
- A simulated traveler answers clarification questions and approves selections (up to 3 turns).

## Model and tracing

- Complexity: complex. Model: claude-haiku-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

travel_search_flights, travel_search_hotels, travel_book, wikipedia_search, web_search + local memorize/in-trip mocks

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 13.0 LLM generations per run.
Mean cost $0.0107 per run (Langfuse trace cost).

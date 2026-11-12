# pyd-flight-booking

Framework: PydanticAI.
Source: https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/flight_booking.py

## Adaptation

- Kept: `search_agent` delegates to `extraction_agent` inside the `extract_flights` tool with `usage=ctx.usage`; `FlightDetails | NoFlightFound` output; output validator with `ModelRetry`; the outer re-search loop with `output_tool_return_content`; `seat_preference_agent` with `SeatPreference | Failed`; one `UsageLimits(request_limit=22)` over all agents.
- Changed: the hard-coded flights page is live `travel_search_flights` output. A `request_agent` parses the free-text request first.
- Changed: `Prompt.ask` is replaced by a deterministic auto-approver (budget and fare rules, traced as an `auto_approver` span) and buying calls `travel_book`.

## Tools

travel_search_flights, travel_book

## Tracing

PydanticAI built-in OpenTelemetry instrumentation (`Agent.instrument_all`, global tracer provider).
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [1, 4, 5], mean 3.3.
Cost per run from Langfuse (USD): [0.000226, 0.001401, 0.001131], mean 0.000919.
Estimated full batch (100 tasks): about USD 0.09.

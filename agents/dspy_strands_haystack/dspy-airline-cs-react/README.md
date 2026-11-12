# dspy-airline-cs-react

Airline customer-service dspy.ReAct agent that searches, books, looks up and cancels flights and files tickets.

- Framework: dspy
- Source: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/customer_service_agent/index.ipynb
- Model: claude-haiku-5-5
- Complexity: medium
- Tracing: openinference (see `../dsh_common.py`)

## What was adapted

Derived from the DSPy tutorial "Build AI Agents with DSPy" (customer_service_agent).
Kept: the pydantic data model (Date, UserProfile, Flight, Itinerary, Ticket), the seven tools with the
same names, roles and docstrings, the DSPyAirlineCustomerService signature, and a single dspy.ReAct.
Changed: flight search and booking call the tools-service travel API; user lookup calls the CRM;
itineraries and tickets live in a per-task in-memory store seeded with a few existing bookings.

## Tools

travel_search_flights, travel_book, crm_search_customers (tools service); pick_flight, fetch_itinerary, cancel_itinerary, file_ticket (local, per-task in-memory store seeded with 7 itineraries)

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 2.3 (models seen: claude-haiku-5-5).
- Mean cost per run: $0.0013 (Langfuse per-trace cost); smoke total $0.0039.

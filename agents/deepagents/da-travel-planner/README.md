# da-travel-planner

Source: https://docs.langchain.com/oss/python/deepagents/human-in-the-loop

## What was adapted

Documented HITL pattern: interrupt_on gates travel_book (approve/edit/reject) and email_send (approve/reject). InMemorySaver checkpointer, one fresh thread per task. Resumes with Command(resume={'decisions': [...]}). A deterministic approver plays the human: it rejects bookings when the user asked for options only, when the traveler name is missing or a placeholder, or past 4 bookings, and it rejects emails to addresses not in the request. Search is delegated to flight-scout and hotel-scout sub-agents. Caps: main 10 per run and 14 per thread, flight-scout 4, hotel-scout 3.

## Tools and features

travel_search_flights, travel_search_hotels, travel_book, email_send, interrupt_on, checkpointer, write_todos, task

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 4.3. Mean cost per run: $0.0039 (Langfuse cost from scripts/inspect_trace.py).

# oai-airline-service

Source: https://github.com/openai/openai-agents-python/blob/main/examples/customer_service/main.py

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-sonnet-5-5
- Complexity: complex

## What it does

Airline triage agent with handoffs to FAQ, seat booking (on_handoff hook + shared context) and flight search agents, over a multi-turn loop with a simulated customer.

## Adaptation

Airline customer service: triage agent hands off to FAQ, seat booking, and flight search
agents that share a typed run context. A simulated customer answers follow-up questions so a
task can span several conversation turns.

Adapted from openai-agents-python examples/customer_service/main.py. Kept: AirlineAgentContext,
the faq_lookup_tool and update_seat tools, the on_seat_booking_handoff hook that sets the flight
number, the RECOMMENDED_PROMPT_PREFIX routines, handoffs back to triage, and the outer
conversation loop that carries `result.to_input_list()` and `result.last_agent` between turns.
Added: a Flight Search agent over the mock travel API, and a simulated customer (Haiku) in place
of the interactive input() so the loop runs unattended (max 3 turns).

## Tools

Shared (peira_harness.tools): travel_search_flights

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [4, 3, 5], mean 4.0
- Cost per run (Langfuse): mean $0.0124, total $0.0371


# oai-memory-concierge

Source: https://github.com/openai/openai-agents-python/blob/main/examples/memory/sqlite_session_example.py

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-haiku-5-5
- Complexity: medium

## What it does

Multi-turn workplace assistant: SQLiteSession memory across turns, context-driven dynamic instructions, and HR/calendar/email tools.

## Adaptation

Workplace concierge with SQLite session memory and a dynamic system prompt.

Adapted from openai-agents-python examples/memory/sqlite_session_example.py (multi-turn runs that
share a SQLiteSession, so later turns can refer back to earlier ones) and
examples/basic/dynamic_system_prompt.py (instructions computed from the run context on every
call). The session lives in a temporary SQLite file per task. Each task holds 1-4 user turns
separated by a line containing only `---`. The dynamic prompt changes with the turn number, the
requested reply style, and what the assistant has already learned (stored in the context by a
remember_fact tool). Tools: the mock HR, calendar and email-outbox APIs.

## Tools

Shared (peira_harness.tools): calendar_free_slots, calendar_get_events, email_send, hr_get_employee, hr_policy_lookup, hr_search_employees

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [5, 4, 6], mean 5.0
- Cost per run (Langfuse): mean $0.0018, total $0.0053


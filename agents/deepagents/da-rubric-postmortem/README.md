# da-rubric-postmortem

Source: https://github.com/langchain-ai/deepagents/blob/main/examples/rubric_middleware/rubric_agent.py

## What was adapted

Kept: RubricMiddleware grader loop and the example rubric (causes with evidence, concrete mitigations, explicit unknowns, defined terms, saved file with word limit), extended with a no-PII criterion. Task domain moved to postmortems over the mock support data. Added an evidence-collector sub-agent for context isolation. Grader on claude-haiku-5-5, max_iterations=2. Caps: main 10, evidence-collector 6, general-purpose 3.

## Tools and features

tickets_search, tickets_get, orders_get, orders_list, inventory_check, products_search, kb_search, RubricMiddleware, write_todos, task

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 13.0. Mean cost per run: $0.0156 (Langfuse cost from scripts/inspect_trace.py).

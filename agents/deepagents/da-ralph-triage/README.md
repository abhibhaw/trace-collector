# da-ralph-triage

Source: https://github.com/langchain-ai/deepagents/blob/main/examples/ralph_mode/ralph_mode.py

## What was adapted

Kept: the Ralph outer loop (fresh context per iteration, filesystem as the only memory) and the verbatim iteration prompt. deepagents-cli run_non_interactive on a real disk -> in-process create_deep_agent, with StateBackend files passed from one iteration to the next. At most 3 iterations, each a sibling chain named ralph-iteration-N. Domain: support-ticket triage. Ticket writes are scoped to the run through set_run_id. Caps: 6 model calls per iteration, customer-investigator 3.

## Tools and features

tickets_search, tickets_get, tickets_update, kb_search, crm_get_customer, crm_search_customers, orders_list, orders_get, write_todos, task

All file tools use the default StateBackend (virtual filesystem in graph state). The `execute` tool is not offered. Nothing touches the pod's disk or shell.

## Smoke cost

Smoke batch `smoke` (3 tasks, 2026-11-12): 3/3 ok, 3/3 inspect_trace PASS.
Mean LLM calls per run: 15.7. Mean cost per run: $0.0111 (Langfuse cost from scripts/inspect_trace.py).

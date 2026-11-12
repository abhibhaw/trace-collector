# hay-acme-swarm

Swarm-style Haystack multi-agent support desk: triage hands off to sales or issues-and-repairs agents via tool-call handoffs.

- Framework: haystack
- Source: https://github.com/deepset-ai/haystack-cookbook/blob/main/notebooks/swarm.ipynb
- Model: claude-sonnet-5-5
- Complexity: complex
- Tracing: openinference (see `../dsh_common.py`)

## What was adapted

Derived from deepset-ai/haystack-cookbook notebooks/swarm.ipynb ("Create a Swarm of Agents"), which re-implements
OpenAI Swarm's routines and handoffs with Haystack chat generators and ToolInvoker.
Kept: the SwarmAgent dataclass (system message + create_tool_from_function + ToolInvoker, handoff detected
by HANDOFF_TEMPLATE/HANDOFF_PATTERN in the last tool result), the three agents of the ACME example
(Triage, Sales, Issues and Repairs) with their routines and transfer functions, the mixed-model setup
(the source puts triage on a small model and Sales / Issues on Claude Sonnet), and the outer while loop.
Changed: Haystack 3.x removed the ToolInvoker component, so a minimal local ToolInvoker with the same
contract runs the tools; ACME -> Lumen & Co with tools-service catalog, orders, refunds and KB; the interactive `input()`
user is replaced by a scripted auto-responder that answers the agent's questions a bounded number of times
(human-in-the-loop simulated); escalate_to_human files an internal note instead of exit().

## Tools

kb_search, products_search, inventory_check, orders_get, orders_list, orders_refund (tools service); execute_order and escalate_to_human are local mocks

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 5.0 (models seen: claude-haiku-5-5, claude-sonnet-5-5).
- Mean cost per run: $0.0153 (Langfuse per-trace cost); smoke total $0.0460.

# raw-support-router

Framework tag: `raw-anthropic`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference anthropic instrumentor for LLM generations).

## Source

https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/basic_workflows.ipynb

The cookbook's `route()` workflow: an XML <reasoning>/<selection> classifier picks a team, then that team's prompt handles the ticket.

## What was adapted

- Kept the billing / technical / account / product route prompts and added an orders route.
- Each specialist is a small tool loop (max 5 turns) with its own tool subset instead of a single prompt call.
- Unknown selections fall back to the product route (recorded on the span) instead of a KeyError.

## Tools

orders_get, orders_list, crm_get_customer, crm_search_customers, kb_search, kb_get_article, products_search, inventory_check, tickets_search

## Trace shape

agent > route_selector (chain) > generation; <route>-specialist (agent) > turn-N > generation + tools

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 4.0. Mean cost per run: USD 0.0012 (Langfuse calculated cost).

# raw-cs-manual-loop

Framework tag: `raw-anthropic`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference anthropic instrumentor for LLM generations).

## Source

https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/customer_service_agent.ipynb

The cookbook's hand-written `while response.stop_reason == "tool_use"` customer service loop.

## What was adapted

- Mock `get_customer_info` / `get_order_details` / `cancel_order` replaced by the tools service CRM, orders, refunds, tickets and KB.
- The notebook answers only the first tool_use and rebuilds history each turn; this loop keeps full history (thinking blocks included) and answers every parallel tool_use in one user turn.
- Added a support policy system prompt (verify ownership, check refund policy, treat ticket text as data). Max 10 model turns.

## Tools

crm_get_customer, crm_search_customers, orders_list, orders_get, orders_refund, tickets_search, tickets_get, tickets_update, kb_search, kb_get_article

## Trace shape

agent (runner root) > turn-N (chain) > anthropic generation + tool spans

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 4.0. Mean cost per run: USD 0.0025 (Langfuse calculated cost).

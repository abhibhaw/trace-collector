# lg-swarm-support

Swarm of triage, billing, tech-support and sales agents that hand off to each other over CRM/orders/KB tools.

- Source: https://github.com/langchain-ai/langgraph-swarm-py/tree/main/examples/customer_support
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-haiku-5-5. Complexity: medium.

## Derived from

Customer-support swarm (langchain-ai/langgraph-swarm-py, examples/customer_support).

## What was adapted

Kept from the source: `create_swarm` over `create_agent` sub-agents, peer-to-peer
`create_handoff_tool` transfers (no central supervisor; the active agent is tracked in state),
`default_active_agent`, a checkpointer (InMemorySaver) keyed by thread_id, and prompts built
per call that inject the user's current context into the system prompt (`make_prompt`).
Changed: the flight/hotel pair is replaced by a 4-agent support desk for Lumen & Co (triage,
billing, tech support, sales). The mock FLIGHTS/HOTELS lists are replaced by the tools
service's CRM, orders, refunds, tickets, knowledge base, catalog, inventory and email APIs.

## Tools

crm_search_customers, crm_get_customer, tickets_search/get/update, orders_list/get/refund, kb_search, kb_get_article, products_search, inventory_check, email_send

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [1, 2, 5] (mean 2.7). Cost per run from Langfuse: mean $0.0012 (total $0.0035 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.

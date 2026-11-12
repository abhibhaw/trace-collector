# ag-support-swarm

Customer-support Swarm on AutoGen AgentChat (handoff-based routing).

- Framework: autogen
- Source: https://github.com/microsoft/autogen/blob/main/python/docs/src/user-guide/agentchat-user-guide/swarm.ipynb
- Complexity: medium
- Model: claude-haiku-5-5
- Tracing: OpenInference AutoGen instrumentor (team, agent, tool spans) plus OpenInference Anthropic instrumentor (LLM generations)

## What was adapted

Derived from the AgentChat "Swarm" guide (customer support example: travel_agent + flights_refunder with
handoffs to each other and to "user", HandoffTermination(target="user") and a resume loop with a
HandoffMessage from the user). Adapted to Lumen & Co support: a triage agent hands off to order, account
and tech-support specialists that use mock CRM / orders / tickets / KB tools. The human user is simulated
by an auto-responder that answers at most once, so runs are bounded and unattended.

## Tools

- `crm_get_customer` (peira_harness.tools)
- `crm_search_customers` (peira_harness.tools)
- `hr_policy_lookup` (peira_harness.tools)
- `kb_search` (peira_harness.tools)
- `orders_get` (peira_harness.tools)
- `orders_list` (peira_harness.tools)
- `orders_refund` (peira_harness.tools)
- `tickets_get` (peira_harness.tools)
- `tickets_search` (peira_harness.tools)
- `tickets_update` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [2, 5, 6] (mean 4.3).
Cost per run from Langfuse: ['$0.0004', '$0.0012', '$0.0016'] (mean $0.0011).
Projected cost for 100 tasks: about $0.11.

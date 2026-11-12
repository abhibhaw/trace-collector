# pyd-support-triage

Framework: PydanticAI.
Source: https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/medical_agent_delegation.py

## Adaptation

- Kept: triage orchestrator with `consult_specialist` (level 1 routing) and `consult_senior_*` (level 2 escalation) tools, escalation rules in the instructions, structured specialist/senior/final outputs, and a coordinator that records history.
- Changed: the medical domain is Lumen & Co support. Specialists are orders, devices, and account agents with real tools. The senior agent reads policy, updates tickets, and may refund within policy. Prompt-injection guard text is added to every agent.

## Tools

tickets_get, orders_get, orders_list, kb_search, kb_get_article, products_search, inventory_check, crm_get_customer, tickets_search, hr_policy_lookup, orders_refund, tickets_update

## Tracing

PydanticAI built-in OpenTelemetry instrumentation (`Agent.instrument_all`, global tracer provider).
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [6, 6, 6], mean 6.0.
Cost per run from Langfuse (USD): [0.003352, 0.003639, 0.004549], mean 0.003847.
Estimated full batch (100 tasks): about USD 0.38.

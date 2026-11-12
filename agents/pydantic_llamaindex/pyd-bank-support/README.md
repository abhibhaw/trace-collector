# pyd-bank-support

Framework: PydanticAI.
Source: https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/bank_support.py

## Adaptation

- Kept: one `Agent` with typed `deps_type`, a dynamic `@agent.instructions` that injects the customer, account tools, and a structured `SupportOutput` (advice, protective flag, risk 0-10).
- Changed: the bank SQLite table is replaced by the CRM, orders, tickets, and KB tools. The customer id is parsed from the task text into the deps.
- Changed: `block_card` became `lock_account`, and `escalate_to_human` was added.

## Tools

crm_get_customer, orders_list, orders_get, tickets_search, kb_search

## Tracing

PydanticAI built-in OpenTelemetry instrumentation (`Agent.instrument_all`, global tracer provider).
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [1, 3, 2], mean 2.0.
Cost per run from Langfuse (USD): [0.000335, 0.000868, 0.000726], mean 0.000643.
Estimated full batch (100 tasks): about USD 0.06.

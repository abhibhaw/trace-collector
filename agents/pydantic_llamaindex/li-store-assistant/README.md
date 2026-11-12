# li-store-assistant

Framework: LlamaIndex.
Source: https://github.com/run-llama/llama_index/blob/main/docs/examples/agent/agent_workflow_basic.ipynb

## Adaptation

- Kept: a single `FunctionAgent`, plain-function tools, `initial_state` with a `Context`-editing tool, and the human-in-the-loop tool built on `ctx.wait_for_event(HumanResponseEvent, waiter_event=InputRequiredEvent(...))` with the caller streaming events.
- Changed: Tavily is replaced by commerce tools. The human is a deterministic auto-approver: refunds up to 150 USD are approved, larger or full refunds are declined.

## Tools

products_search, inventory_check, orders_list, orders_get, kb_search, orders_refund

## Tracing

OpenInference `LlamaIndexInstrumentor`. The LLM is `pl_common.ProxyAnthropic`: it removes the dispatcher spans from LLM helper methods that make no API call (`_prepare_chat_with_tools`, `apredict`, `acomplete`, ...), so Langfuse shows exactly one GENERATION per API call and cost is not counted twice.
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [2, 3, 3], mean 2.7.
Cost per run from Langfuse (USD): [0.000609, 0.000964, 0.001332], mean 0.000968.
Estimated full batch (100 tasks): about USD 0.10.

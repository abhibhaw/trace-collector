# smol-ops-multiagent

Source: https://github.com/huggingface/smolagents/blob/main/examples/inspect_multiagent_run.py

## Shape

Manager `CodeAgent` with `return_full_result=True` and three managed `ToolCallingAgent`s, each also `return_full_result=True`: `customer_agent` (CRM, orders, tickets, KB, products, inventory; `provide_run_summary=True`), `market_agent` (quotes, history) and `people_agent` (HR, policies, calendar). `run()` appends the `RunResult` state, step count, timing and token usage, which the source prints.

## What was adapted

- The single web `search_agent` becomes three business sub-agents over the tools-service mock APIs.
- The model is the proxy's `claude-haiku-5-5`, instrumented by OpenInference as in the source.
- Every agent has `instructions` that tell it to treat tool results as data, not instructions.

## Tools

`crm_*`, `orders_list`, `orders_get`, `tickets_*` (including the `tickets_update` write), `kb_search`, `products_search`, `inventory_check`, `finance_*`, `hr_*`, `calendar_get_events`, `calendar_free_slots`.

## Cost bound

Manager `max_steps=5`. Sub-agents: 5, 4 and 4. Typical: 6-15 LLM calls. Worst case: about 30 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 7.3
- Mean cost per run: $0.0070 (Langfuse trace cost on the 3 smoke tasks)

## Family notes (smolagents)

- Shared code is in `agents/smolagents/smol_common.py`.
- Model: `ProxyModel`, a `LiteLLMModel` subclass, calls `anthropic/<model>` at the proxy's Anthropic endpoint with `max_tokens=4096`.
  It changes smolagents' default `tool_choice="required"` to `"auto"`, because `claude-sonnet-5-5` rejects forced tool choice.
  It also removes `temperature`, `top_p` and `top_k`.
- Tracing: `SmolagentsInstrumentor` (OpenInference) is the only source of LLM spans.
  It creates agent, step, tool and LLM spans with token counts.
- A fresh agent is built for each task, because agents keep memory.
  Each task calls `tools.set_run_id(...)`, so that writes stay scoped to the run.

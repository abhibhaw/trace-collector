# smol-data-analyst

Source: https://github.com/huggingface/cookbook/blob/main/notebooks/en/agent_data_analyst.ipynb

## Shape

`CodeAgent` with pandas, run with `additional_args=dict(additional_notes=...)` as in the cookbook. Data-loading tools return `pandas.DataFrame` objects straight into the code state. Also has a step callback that records the datasets it loaded, and a final-answer check that requires numbers.

## What was adapted

- CSV reads become tools: `load_sql` (Northwind/Chinook), `describe_database`, `load_price_history` (synthetic OHLCV) and `load_orders` (mock order system).
- Plotting libraries are removed.
- Authorized imports: `pandas, numpy, math, statistics, datetime, collections`. The local executor has no file or network imports.

## Tools

`sql_query`, `sql_schema`, `finance_history`, `orders_list`.

## Cost bound

`max_steps=7`. Worst case: 8 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 3.3
- Mean cost per run: $0.0023 (Langfuse trace cost on the 3 smoke tasks)

## Family notes (smolagents)

- Shared code is in `agents/smolagents/smol_common.py`.
- Model: `ProxyModel`, a `LiteLLMModel` subclass, calls `anthropic/<model>` at the proxy's Anthropic endpoint with `max_tokens=4096`.
  It changes smolagents' default `tool_choice="required"` to `"auto"`, because `claude-sonnet-5-5` rejects forced tool choice.
  It also removes `temperature`, `top_p` and `top_k`.
- Tracing: `SmolagentsInstrumentor` (OpenInference) is the only source of LLM spans.
  It creates agent, step, tool and LLM spans with token counts.
- A fresh agent is built for each task, because agents keep memory.
  Each task calls `tools.set_run_id(...)`, so that writes stay scoped to the run.

# smol-text-to-sql

Source: https://github.com/huggingface/smolagents/blob/main/examples/text_to_sql.py (and docs/source/en/examples/text_to_sql.md)

## Shape

One `CodeAgent` with one custom `Tool` subclass, `sql_engine`. The tool description embeds the live schema of both databases, as in the tutorial's multi-table step.

## What was adapted

- The in-memory `receipts` table becomes the Chinook and Northwind databases from the tools service.
- The schema is read once in `setup()` with `sql_schema` and written into the tool description.
- `InferenceClientModel` becomes the LiteLLM proxy (`claude-haiku-5-5`).

## Tools

`sql_schema` (setup only), `sql_query`.

## Cost bound

`max_steps=6`. Worst case: 6 steps + 1 forced final answer = 7 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 1.3
- Mean cost per run: $0.0011 (Langfuse trace cost on the 3 smoke tasks)

## Family notes (smolagents)

- Shared code is in `agents/smolagents/smol_common.py`.
- Model: `ProxyModel`, a `LiteLLMModel` subclass, calls `anthropic/<model>` at the proxy's Anthropic endpoint with `max_tokens=4096`.
  It changes smolagents' default `tool_choice="required"` to `"auto"`, because `claude-sonnet-5-5` rejects forced tool choice.
  It also removes `temperature`, `top_p` and `top_k`.
- Tracing: `SmolagentsInstrumentor` (OpenInference) is the only source of LLM spans.
  It creates agent, step, tool and LLM spans with token counts.
- A fresh agent is built for each task, because agents keep memory.
  Each task calls `tools.set_run_id(...)`, so that writes stay scoped to the run.

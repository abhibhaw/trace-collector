# smol-travel-planner

Source: https://github.com/huggingface/smolagents/blob/main/examples/multiple_tools.py

## Shape

One `ToolCallingAgent` with many small tools. A `final_answer_checks` validator requires every booking confirmation id in the answer.

## What was adapted

- The keyed weather, currency and time APIs become tools-service travel mocks: `travel_search_flights`, `travel_search_hotels` and `travel_book` (a write).
- Local tools: a fixed-rate `convert_currency`, `get_time_in_timezone` (zoneinfo, mock 'now') and `trip_budget`.

## Tools

`travel_search_flights`, `travel_search_hotels`, `travel_book`, plus 3 local tools.

## Cost bound

`max_steps=7`. Worst case: 8 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 2.0
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

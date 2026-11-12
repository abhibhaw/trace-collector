# smol-web-multiagent

Source: https://github.com/huggingface/smolagents/blob/main/docs/source/en/examples/multiagents.md

## Shape

Manager `CodeAgent` (no tools) with a managed `ToolCallingAgent` named `web_search_agent` that has `web_search` and `visit_webpage`.

## What was adapted

- `WebSearchTool` becomes the tools-service `web_search`; the hand-written `visit_webpage` calls `fetch_url`.
- Authorized imports shrink from `time, numpy, pandas` to `math, statistics, datetime`.
- Step limits are lower: manager 5, web agent 4.

## Tools

`web_search`, `fetch_url`.

## Cost bound

Manager `max_steps=5`, web agent `max_steps=4`. Typical: 6-12 LLM calls. Worst case: about 25 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 7.0
- Mean cost per run: $0.0066 (Langfuse trace cost on the 3 smoke tasks)

## Family notes (smolagents)

- Shared code is in `agents/smolagents/smol_common.py`.
- Model: `ProxyModel`, a `LiteLLMModel` subclass, calls `anthropic/<model>` at the proxy's Anthropic endpoint with `max_tokens=4096`.
  It changes smolagents' default `tool_choice="required"` to `"auto"`, because `claude-sonnet-5-5` rejects forced tool choice.
  It also removes `temperature`, `top_p` and `top_k`.
- Tracing: `SmolagentsInstrumentor` (OpenInference) is the only source of LLM spans.
  It creates agent, step, tool and LLM spans with token counts.
- A fresh agent is built for each task, because agents keep memory.
  Each task calls `tools.set_run_id(...)`, so that writes stay scoped to the run.

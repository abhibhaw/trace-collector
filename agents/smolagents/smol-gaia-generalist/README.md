# smol-gaia-generalist

Source: https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/run_gaia.py (and examples/smolagents_benchmark/run.py)

## Shape

One `ToolCallingAgent` with `planning_interval=4` and `final_answer_checks`. Tools: web, Wikipedia, arXiv and a sandboxed `python_interpreter`. Parallel tool calls use `max_tool_threads=4`. A final `prepare_response` reformulator LLM call (ported from `scripts/reformulator.py`) produces a GAIA-format answer.

## What was adapted

- Serper becomes `web_search`. `VisitWebpageTool` becomes `fetch_url`.
- `PythonInterpreterTool` becomes the tools-service `python_exec` sandbox, so code never runs in the pod.
- Wikipedia and arXiv are added.
- The reformulator skips the agent's system prompt to save tokens.
- The model is `claude-sonnet-5-5`. Forced `tool_choice` is turned off.

## Tools

`web_search`, `fetch_url`, `wikipedia_search`, `wikipedia_page`, `arxiv_search`, `python_exec`.

## Cost bound

`max_steps=8`, `planning_interval=4`, plus 1 reformulation. Worst case: 8 + 2 plans + 1 forced final + 1 = 12 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 4.0
- Mean cost per run: $0.0439 (Langfuse trace cost on the 3 smoke tasks)

## Family notes (smolagents)

- Shared code is in `agents/smolagents/smol_common.py`.
- Model: `ProxyModel`, a `LiteLLMModel` subclass, calls `anthropic/<model>` at the proxy's Anthropic endpoint with `max_tokens=4096`.
  It changes smolagents' default `tool_choice="required"` to `"auto"`, because `claude-sonnet-5-5` rejects forced tool choice.
  It also removes `temperature`, `top_p` and `top_k`.
- Tracing: `SmolagentsInstrumentor` (OpenInference) is the only source of LLM spans.
  It creates agent, step, tool and LLM spans with token counts.
- A fresh agent is built for each task, because agents keep memory.
  Each task calls `tools.set_run_id(...)`, so that writes stay scoped to the run.

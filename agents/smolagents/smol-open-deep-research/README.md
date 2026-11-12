# smol-open-deep-research

Source: https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research (run.py)

## Shape

Manager `CodeAgent` with `planning_interval=4` and a `TextInspectorTool` that makes its own nested LLM call. Managed `ToolCallingAgent` named `search_agent` with `planning_interval=4`, `provide_run_summary=True`, the source's `managed_agent` prompt suffix, and browser tools (`web_search`, `visit_page`, `page_up`, `page_down`, `find_on_page_ctrl_f`, `find_next`, `arxiv_search`, `wikipedia_page`, `inspect_file_as_text`).

## What was adapted

- `GoogleSearchTool` (Serper) becomes `web_search`. `ArchiveSearchTool` (Wayback Machine) becomes `arxiv_search` and `wikipedia_page`.
- `SimpleTextBrowser` is rewritten in-process over `fetch_url`, with 5000-character viewports, so the paging and find tools keep their roles.
- `TextInspectorTool` reads URLs (HTML or PDF) instead of local files.
- The `visualizer` image tool is removed.
- `additional_authorized_imports=["*"]` becomes a short safe list.
- The `o1` model becomes `claude-sonnet-5-5`. Forced `tool_choice` is turned off (see the family notes).

## Tools

`web_search`, `fetch_url`, `arxiv_search`, `wikipedia_page`.

## Cost bound

Manager `max_steps=4`, search agent `max_steps=4` (sequential tool calls), both `planning_interval=4`; the code executor timeout is raised to 300 s so a delegation is not killed and retried. Smoke: 4-16 LLM calls. Worst case: about 25 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 8.3
- Mean cost per run: $0.2196 (Langfuse trace cost on the 3 smoke tasks)

## Family notes (smolagents)

- Shared code is in `agents/smolagents/smol_common.py`.
- Model: `ProxyModel`, a `LiteLLMModel` subclass, calls `anthropic/<model>` at the proxy's Anthropic endpoint with `max_tokens=4096`.
  It changes smolagents' default `tool_choice="required"` to `"auto"`, because `claude-sonnet-5-5` rejects forced tool choice.
  It also removes `temperature`, `top_p` and `top_k`.
- Tracing: `SmolagentsInstrumentor` (OpenInference) is the only source of LLM spans.
  It creates agent, step, tool and LLM spans with token counts.
- A fresh agent is built for each task, because agents keep memory.
  Each task calls `tools.set_run_id(...)`, so that writes stay scoped to the run.

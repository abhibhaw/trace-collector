# lg-react-assistant

Single ReAct loop (call_model <-> ToolNode) with web search, URL fetch, Wikipedia and arXiv.

- Source: https://github.com/langchain-ai/react-agent/blob/main/src/react_agent/graph.py
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-haiku-5-5. Complexity: simple.

## Derived from

ReAct agent template (langchain-ai/react-agent), adapted to the peira proxy and tools.

## What was adapted

Kept from the source: dataclass State with the managed `is_last_step` flag, a Runtime
Context (system prompt + model), the `call_model` <-> `tools` cycle, the conditional
`route_model_output` edge, and the "Sorry, could not find an answer" last-step guard.
Changed: Tavily search is replaced by keyless web search, URL fetch, Wikipedia, and arXiv.

## Tools

web_search, fetch_url, wikipedia_search, arxiv_search

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [1, 1, 3] (mean 1.7). Cost per run from Langfuse: mean $0.0005 (total $0.0016 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.

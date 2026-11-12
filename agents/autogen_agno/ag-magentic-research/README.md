# ag-magentic-research

Magentic-One style research team on AutoGen AgentChat MagenticOneGroupChat.

- Framework: autogen
- Source: https://github.com/microsoft/autogen/blob/main/python/docs/src/user-guide/agentchat-user-guide/magentic-one.md
- Complexity: complex
- Model: claude-sonnet-5-5
- Tracing: OpenInference AutoGen instrumentor (team, agent, tool spans) plus OpenInference Anthropic instrumentor (LLM generations)

## What was adapted

Derived from the AgentChat "Magentic-One" guide: MagenticOneGroupChat orchestrator (task ledger, progress
ledger, stall detection, final answer) over WebSurfer, Coder (MagenticOneCoderAgent) and ComputerTerminal
(CodeExecutorAgent). Adapted: the browser-based MultimodalWebSurfer is replaced by an AssistantAgent with
keyless web search, URL fetch, Wikipedia and arXiv tools (no browser); the terminal's local executor is
replaced by a CodeExecutor that runs code in the tools-service python_exec sandbox; turns and stalls are
bounded.

## Tools

- `arxiv_search` (peira_harness.tools)
- `fetch_url` (peira_harness.tools)
- `web_search` (peira_harness.tools)
- `wikipedia_page` (peira_harness.tools)
- `wikipedia_search` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [15, 14, 22] (mean 17.0).
Cost per run from Langfuse: ['$0.0957', '$0.1944', '$0.2059'] (mean $0.1653).
Projected cost for 100 tasks: about $16.53.

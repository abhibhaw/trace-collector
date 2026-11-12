# lg-open-deep-research

Deep research: clarify, brief, supervisor fanning out parallel researcher subgraphs, compression, final report.

- Source: https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-sonnet-5-5. Complexity: complex.

## Derived from

Open Deep Research (langchain-ai/open_deep_research, src/open_deep_research/deep_researcher.py).

## What was adapted

Kept from the source: the 4-stage top graph (clarify_with_user -> write_research_brief ->
research_supervisor subgraph -> final_report_generation), the supervisor subgraph (supervisor <->
supervisor_tools with ConductResearch / ResearchComplete / think_tool), parallel researcher
subgraphs launched with asyncio.gather (researcher <-> researcher_tools -> compress_research),
override_reducer state, iteration limits, overflow handling, and all prompts (odr_prompts.py).
Changed: Tavily (+ per-result summarisation) is replaced by keyless web search, URL fetch and
Wikipedia; MCP is dropped. Models follow ODR's split config: the research/supervisor and final
report models are claude-sonnet-5-5, the researcher and compression models are claude-haiku-5-5.
Limits are lowered to bound cost (about 25 LLM calls worst case). Structured outputs on Sonnet use
json_schema mode because the proxy forbids forced tool choice on Sonnet.

## Tools

web_search (1-3 queries per call), fetch_url, wikipedia_search, think_tool (local)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [1, 14, 14] (mean 9.7). Cost per run from Langfuse: mean $0.1035 (total $0.3106 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.

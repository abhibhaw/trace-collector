# lg-data-enrichment

Schema-driven web research: search/scrape tools, dynamic Info extraction tool, LLM reflection gate.

- Source: https://github.com/langchain-ai/data-enrichment/blob/main/src/enrichment_agent/graph.py
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-haiku-5-5. Complexity: complex.

## Derived from

Data enrichment agent (langchain-ai/data-enrichment template), adapted to the peira proxy and tools.

## What was adapted

Kept from the source: InputState (topic, extraction_schema, info) / State (messages, loop_step) /
OutputState (info), MAIN_PROMPT, the dynamic `Info` tool whose parameters are the extraction
schema, call_agent_model with tool_choice="any", the search + scrape_website ToolNode (scrape
summarises the page with an LLM using InjectedState), the reflect node with InfoIsSatisfactory,
and route_after_agent / route_after_checker with max_loops.
Changed: Tavily and aiohttp are replaced by keyless web search and the URL fetch tool. The source
takes the extraction schema as a second input; here a `build_schema` node first turns the user's
request into topic + JSON schema (structured output), so tasks stay plain text. At max_loops the
agent is forced to submit via Info, which bounds cost.

## Tools

web_search (search), fetch_url (scrape_website, summarised by an LLM inside the tool)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [10, 11, 13] (mean 11.3). Cost per run from Langfuse: mean $0.0067 (total $0.0201 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.

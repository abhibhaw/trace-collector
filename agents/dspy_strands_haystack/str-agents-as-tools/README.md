# str-agents-as-tools

Strands orchestrator that delegates to research, product-recommendation and trip-planning sub-agents exposed as tools.

- Framework: strands
- Source: https://github.com/strands-agents/samples/blob/main/python/01-learn/10-agents-as-tools/agent-as-tools.ipynb
- Model: claude-haiku-5-5
- Complexity: medium
- Tracing: native-otel (see `../dsh_common.py`)

## What was adapted

Derived from strands-agents/samples python/01-learn/10-agents-as-tools/agent-as-tools.ipynb.
Kept: the orchestrator system prompt and routing rules, the three specialists (research_assistant,
product_recommendation_assistant, trip_planning_assistant) with their prompts and docstrings, a file-writing
tool, and the pattern of building a fresh specialist Agent inside each tool call.
Changed: Bedrock -> Anthropic model via the proxy; specialists get real tools (web search/fetch/Wikipedia,
the Lumen & Co product catalog and inventory, flight and hotel search); `strands_tools.file_write` is
replaced by an in-memory `file_write`; tools are async so OTel context stays on the task's trace.

## Tools

web_search, fetch_url, wikipedia_search, products_search, inventory_check, travel_search_flights, travel_search_hotels (tools service); file_write (in-memory)

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 9.7 (models seen: claude-haiku-5-5).
- Mean cost per run: $0.0024 (Langfuse per-trace cost); smoke total $0.0072.

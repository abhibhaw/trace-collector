# raw-sql-tool-runner

Framework tag: `raw-anthropic`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference anthropic instrumentor for LLM generations).

## Source

https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/tools_runner.py

The SDK's beta tool runner (`@beta_tool` + `client.beta.messages.tool_runner`), applied to the text-to-SQL idea of the cookbook's misc/how_to_make_sql_queries.ipynb.

## What was adapted

- The weather tool is replaced by sql_schema / sql_query over Chinook and Northwind.
- The SDK owns the loop. Each tool function is wrapped so its call is a manual Langfuse tool span. SDK-native OTel spans are disabled (`open_telemetry=False`). max_iterations=8.
- `thread=False` on the runner: anthropic 2.8's tool runner defaults to server-side message threads and sends only new messages. The LiteLLM proxy drops the `thread` field, so upstream then rejects an orphan tool_result (HTTP 400).

## Tools

sql_schema, sql_query

## Trace shape

agent > tool_runner (chain) > beta.messages generations + tool spans (flat siblings, no per-turn span: that is the runner's shape)

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 3.3. Mean cost per run: USD 0.0020 (Langfuse calculated cost).

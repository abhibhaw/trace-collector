# adk-data-science

Source: https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/data-science

Root data-science agent plans queries, delegates NL2SQL to per-database agents (Chinook, Northwind) and analysis to a code-executing analytics agent.

## Shape

root LlmAgent (sonnet) with function tools that run AgentTool(chinook_db_agent), AgentTool(northwind_db_agent), AgentTool(analytics_agent with a custom BaseCodeExecutor); state hand-off of SQL results via after_tool_callback

## Adaptations

- BigQuery + AlloyDB datasets become the Chinook and Northwind SQLite databases, each with its own NL2SQL agent (source: call_bigquery_agent / call_alloydb_agent).
- BigQuery `execute_sql` -> `sql_query`; schemas from `sql_schema` are injected through instruction providers.
- VertexAiCodeExecutor -> `SandboxCodeExecutor` (an ADK BaseCodeExecutor) that runs ```python blocks in the tools-service sandbox (`python_exec`) and emits a TOOL span per execution.
- The LLM-inside-a-tool nl2sql step and the BQML sub-agent (Vertex only) are dropped; temperature removed.
- Root agent on claude-sonnet-5-5; DB and analytics agents on claude-haiku-5-5.

## Model and tracing

- Complexity: complex. Model: claude-sonnet-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

sql_schema, sql_query, python_exec

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 4.0 LLM generations per run.
Mean cost $0.0351 per run (Langfuse trace cost).

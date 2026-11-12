# oai-parallel-sql

Source: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/parallelization.py

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-haiku-5-5
- Complexity: medium

## What it does

Three parallel SQL analyst agents answer a data question over Chinook/Northwind, then a picker agent selects and returns the best answer.

## Adaptation

Parallel SQL analysts + picker: the same analyst agent runs three times in parallel on a
data question over the sample SQLite databases, then a picker agent chooses the best answer.

Adapted from openai-agents-python examples/agent_patterns/parallelization.py. The source runs a
Spanish translator three times with asyncio.gather and a translation_picker chooses the best.
Here the parallel workers are SQL analysts with schema-inspection and query tools, which makes
the candidates genuinely differ (different queries, joins, and interpretations).

## Tools

Shared (peira_harness.tools): sql_analyst, sql_query, sql_schema

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [7, 9, 10], mean 8.7
- Cost per run (Langfuse): mean $0.0062, total $0.0186


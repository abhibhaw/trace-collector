# oai-research-bot

Source: https://github.com/openai/openai-agents-python/tree/main/examples/research_bot

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-haiku-5-5
- Complexity: complex

## What it does

Planner agent plans web searches, parallel search agents summarize results, writer agent produces a structured markdown report.

## Adaptation

Research bot: planner -> parallel search agents -> writer.

Adapted from openai-agents-python examples/research_bot. The hosted WebSearchTool is replaced
by the shared keyless web_search / fetch_url tools. The planner's 5-20 searches are capped at 4
to bound cost; the writer keeps its structured ReportData output.

## Tools

Shared (peira_harness.tools): fetch_url, web_search

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [12, 11, 15], mean 12.7
- Cost per run (Langfuse): mean $0.0063, total $0.0189


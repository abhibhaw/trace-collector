# oai-financial-research

Source: https://github.com/openai/openai-agents-python/tree/main/examples/financial_research_agent

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-sonnet-5-5
- Complexity: complex

## What it does

Financial research manager: planner, parallel search agents, writer that calls fundamentals and risk analyst agents as tools, then an evidence verifier with one revision loop.

## Adaptation

Financial research agent: planner -> parallel search agents -> writer with analyst
sub-agents as tools -> evidence verifier -> one revision round.

Adapted from openai-agents-python examples/financial_research_agent. The hosted WebSearchTool is
replaced by the shared web_search / fetch_url tools plus the mock finance quote API. Search agents
return their source URLs in structured output (the source extracted them from hosted-search
annotations). Searches are capped at 3 to bound cost.

## Tools

Shared (peira_harness.tools): fetch_url, finance_history, finance_quote, web_search

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [9, 14, 21], mean 14.7
- Cost per run (Langfuse): mean $0.2366, total $0.7098

Note: Sonnet: by far the most expensive agent in the family (about $24 per 100 tasks at the smoke mean).


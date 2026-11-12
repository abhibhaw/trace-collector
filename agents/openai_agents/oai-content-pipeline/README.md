# oai-content-pipeline

Source: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/deterministic.py

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-haiku-5-5
- Complexity: simple

## What it does

Three fixed agent steps: outline (with Wikipedia grounding), structured quality gate, then article writer.

## Adaptation

Deterministic content pipeline: outline agent -> outline checker (structured) -> gate ->
writer agent.

Adapted from openai-agents-python examples/agent_patterns/deterministic.py. The source writes
sci-fi stories and gates on `good_quality` and `is_scifi`. Here the same three-step flow writes
short explainer articles: the outline agent may ground facts with Wikipedia, the checker gates on
quality and on whether the request is a feasible factual article, and the writer only runs if
both pass.

## Tools

Shared (peira_harness.tools): wikipedia_page, wikipedia_search

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [1, 3, 4], mean 2.7
- Cost per run (Langfuse): mean $0.0011, total $0.0034

Note: Adversarial requests can end in an API-level refusal (ModelRefusalError); the agent returns the refusal as its answer.


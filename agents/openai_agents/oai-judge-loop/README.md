# oai-judge-loop

Source: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/llm_as_a_judge.py

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-haiku-5-5
- Complexity: medium

## What it does

Copywriter agent and evaluator agent loop with structured pass/needs_improvement/fail feedback until the copy passes (max 3 rounds).

## Adaptation

LLM-as-a-judge loop: a copywriter agent drafts product copy, an evaluator agent scores it
with structured feedback, and the loop repeats with the feedback until it passes.

Adapted from openai-agents-python examples/agent_patterns/llm_as_a_judge.py. The source loops a
story-outline generator against an evaluator. Here the generator writes Lumen & Co product copy
grounded in the product catalog tool, and the evaluator checks it against the same catalog.
The loop is capped at 3 rounds (the source caps auto mode at 3).

## Tools

Shared (peira_harness.tools): inventory_check, products_search

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [3, 8, 8], mean 6.3
- Cost per run (Langfuse): mean $0.0037, total $0.0111

Note: Claude rejects conversations that end with an assistant turn, so the evaluator gets an explicit 'evaluate' user turn.


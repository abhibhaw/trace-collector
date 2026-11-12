# oai-translation-desk

Source: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/agents_as_tools.py

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-haiku-5-5
- Complexity: simple

## What it does

Orchestrator agent calls per-language translator agents as tools, then a synthesizer agent reviews and concatenates the translations.

## Adaptation

Translation desk: an orchestrator calls translator agents as tools, then a synthesizer
checks and assembles the final answer.

Adapted from openai-agents-python examples/agent_patterns/agents_as_tools.py. Kept the
orchestrator + translator sub-agents + synthesizer shape and prompts. Added German and
Japanese translators and a local glossary tool for Lumen & Co product terms.

## Tools

Shared (peira_harness.tools): none

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [2, 4, 4], mean 3.3
- Cost per run (Langfuse): mean $0.0007, total $0.0022

Note: Claude rejects conversations that end with an assistant turn, so the synthesizer gets an explicit user turn after the orchestrator transcript.


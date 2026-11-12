# oai-support-guardrails

Source: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/input_guardrails.py

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-haiku-5-5
- Complexity: medium

## What it does

Support agent over orders, tickets, and KB tools with an agent-based input guardrail, a structured output guardrail, and a tool-output injection guardrail.

## Adaptation

Customer support agent protected by input, output, and tool-output guardrails.

Adapted from openai-agents-python examples/agent_patterns/input_guardrails.py and
output_guardrails.py (plus the tool guardrails in examples/basic/tool_guardrails.py):
- Input guardrail: a guardrail agent with structured output decides whether the request is in
  scope for Lumen & Co support (the source checks for math homework). A trip returns the refusal
  message, exactly like the source's run loop.
- Output guardrail: the agent emits a structured MessageOutput; a deterministic check trips on
  leaked sensitive data (payment card numbers, internal-only addresses). The source checks for a
  phone number.
- Tool output guardrail: rejects tool results that carry prompt-injection markers, so the model
  sees a redaction notice instead of the injected text.

## Tools

Shared (peira_harness.tools): crm_get_customer, crm_search_customers, kb_get_article, kb_search, orders_get, orders_list, tickets_get, tickets_search

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [2, 3, 3], mean 2.7
- Cost per run (Langfuse): mean $0.0008, total $0.0023

Note: When the input guardrail trips, the main agent's first model call (started in parallel, as in the source) is cancelled; it shows as a GENERATION without usage.


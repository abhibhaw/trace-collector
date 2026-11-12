# oai-quote-forced-tool

Source: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/forcing_tool_use.py

- Framework: openai-agents 0.24.0 (OpenAIChatCompletionsModel through the LiteLLM proxy)
- Tracing: OpenInference OpenAIAgentsInstrumentor (exclusive processor, no OpenAI trace export)
- Model: claude-haiku-5-5
- Complexity: simple

## What it does

Stock quote agent with tool_choice=required and rotating tool_use_behavior (run_llm_again with structured output, stop_on_first_tool, custom).

## Adaptation

Quote desk with forced tool use: tool_choice="required" plus three tool_use_behavior modes.

Adapted from openai-agents-python examples/agent_patterns/forcing_tool_use.py. The source forces a
get_weather tool and shows three behaviors: `default` (run_llm_again, no forced tool choice),
`stop_on_first_tool`, and a custom ToolsToFinalOutputFunction. Here the tool returns a structured
stock quote snapshot from the mock finance API. The behavior is picked per task (deterministic on
the task id) so the batch covers all three modes. The `default` mode also uses a structured
output_type. Forced tool choice is only legal on Haiku, so this agent must stay on Haiku.

## Tools

Shared (peira_harness.tools): finance_quote

Local tools are defined in agent.py.

## Smoke run

Batch smoke, 3 tasks, concurrency 3, 2026-11-12: 3/3 inspect_trace PASS.

- LLM calls (GENERATIONs) per run: [1, 1, 1], mean 1.0
- Cost per run (Langfuse): mean $0.0001, total $0.0003

